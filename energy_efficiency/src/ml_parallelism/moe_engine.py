from copy import deepcopy
from typing import Any, List, Optional, Tuple, Type
import src.ml_parallelism.utils as utils
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

class MoEDistConfig:

    def __init__(self,
                 num_experts : int,
                 num_local_experts : int,
                 rank : int,
                 world_size : int,
                 device : torch.device,
                 expert_capacity_coefficient : float = 1.0,
                 multi_stream : bool = False,
                 num_streams : int = 4,
                 handle_by_capacity : bool = False,
                 track_capacity : bool = False) -> None:
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.expert_capacity_coefficient = expert_capacity_coefficient
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.multi_stream = multi_stream
        self.num_streams = num_streams
        self.handle_by_capacity = handle_by_capacity
        self.track_capacity = track_capacity
        self.eventDtoH = torch.cuda.Event()

class CapacityHeuristic:

    def __init__(self, config : MoEDistConfig, warmup_steps : int = 3) -> None:
        self.device = config.device
        self.eventDtoH = config.eventDtoH
        self.warmup_steps = warmup_steps
        self.num_local_experts = config.num_local_experts

        self.ready = False
        
        self.start_host_timestamp = 0
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_recorded = False

        self.end_host_timestamp = 0
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.delay_set = False

    def _measurement(self) -> None:
        assert self.end_event.query()
        self.host_forward = (self.end_host_timestamp - self.start_host_timestamp) / 1e6 # Divide to get in ms, same unit as CUDA Events
        self.device_forward = self.start_event.elapsed_time(self.end_event)
        # Either we estimate the delay by measuring the time between when the event was recorded and event.query()==True, or the delay is more than the host forward time.
        self._set_delay(self.delay_estimate if self.delay_set else self.host_forward)
        self.ready = True

    def _heuristic(self) -> bool:
        cur = time.perf_counter_ns()
        since_start = (cur - self.start_host_timestamp) / 1e6
        query_eventDtoH = self.eventDtoH.query()
        query_start_event = self.start_event.query()
        if query_eventDtoH:
            return True
        elif not query_start_event:
            if (self.host_forward - since_start) < self.device_forward + self.delay_estimate:
                if dist.get_rank() == 0:
                    print(self.host_forward, self.device_forward, self.delay_estimate, since_start)
                self.eventDtoH.synchronize()
                return True
            return False
            
        return False

    def _set_delay(self, val : float) -> None:
        self.delay_estimate = val
        self.delay_set = True

    def _check_delay(self):
        if self.start_recorded and not self.delay_set and self.start_event.query():
            self._set_delay((time.perf_counter_ns() - self.start_host_timestamp) / 1e6)

    def query(self) -> bool:
        if not self.delay_set:
            self._check_delay()
        return self.ready and self._heuristic()

    def start(self) -> None:
        if self.warmup_steps > 0:
            return
        if self.start_recorded and not self.ready:
            self._measurement()
        self.start_host_timestamp = time.perf_counter_ns()
        self.start_event.record(torch.cuda.current_stream(self.device))
        self.start_recorded = True

    def stop(self) -> None:
        if self.warmup_steps > 0:
            self.warmup_steps -= 1
            return
        self.end_host_timestamp = time.perf_counter_ns()
        self.end_event.record(torch.cuda.current_stream(self.device))

class CH2:

    def __init__(self, config : MoEDistConfig) -> None:
        self.num_local_experts = config.num_local_experts
        self.eventDtoH = config.eventDtoH
        self.device = config.device

        self.cur_expert = 0
        self.synchronize_threshold = self.num_local_experts * 0.75

        self.has_synced = False
        self.before_last_event = torch.cuda.Event()

    def _is_before_last(self) -> bool:
        return self.cur_expert == self.num_local_experts - 2

    def _record_before_last(self) -> None:
        self.before_last_event.record(torch.cuda.current_stream(self.device))

    def query(self) -> bool:
        ret = False
        if self.eventDtoH.query():
            ret = True
        elif self.cur_expert > self.synchronize_threshold:
            self.eventDtoH.synchronize()
            self.has_synced = True
            ret = True
        if self._is_before_last() and self.has_synced:
            self._record_before_last()
        self.cur_expert += 1
        return ret

    def start(self) -> None:
        self.cur_expert = 0

    def stop(self) -> None:
        if self.has_synced:
            if self.before_last_event.query():
                # If the before last event has executed before we scheduled the last one, then we synchronized too early.
                self.synchronize_threshold += 1
            else:
                # Otherwise, we might be able to synchronize earlier.
                self.synchronize_threshold -= 1


class LocalExperts(nn.Module):

    def __init__(self, config : MoEDistConfig, expert: nn.Module) -> None:
        super().__init__()
        self.experts = nn.ModuleList([deepcopy(expert) for _ in range(config.num_local_experts)])
        self.num_local_experts = config.num_local_experts
        self.world_size = config.world_size
        self.handle_by_capacity = config.handle_by_capacity
        self.eventDtoH = config.eventDtoH
        # self.capacity_heuristic = CapacityHeuristic(config)
        self.capacity_heuristic = CH2(config)

    def forward(self, dispatched_hidden_states, experts_capacity_usage) -> torch.Tensor:
        if self.handle_by_capacity:
            chunks = dispatched_hidden_states.chunk(self.num_local_experts, dim=0)
            experts_capacity_usage_chunks = experts_capacity_usage.chunk(self.num_local_experts, dim=0)
            expert_index = 0
            out = torch.zeros(dispatched_hidden_states.shape, dtype=dispatched_hidden_states.dtype, device=dispatched_hidden_states.device)
            self.capacity_heuristic.start()
            for expert, hidden_states, expert_capacity_usage in zip(self.experts, chunks, experts_capacity_usage_chunks):
                if self.capacity_heuristic.query():
                    cap = expert_capacity_usage.max()
                    out[expert_index,:,0:cap] = expert(hidden_states[:,:,0:cap])
                else: 
                    out[expert_index] = expert(hidden_states)
                expert_index += 1
            self.capacity_heuristic.stop()
            return out
        else:
            chunks = dispatched_hidden_states.chunk(self.num_local_experts, dim=0)
            experts_hidden_states = []
            for expert, hidden_states in zip(self.experts, chunks):
                expert_hidden_states = expert(hidden_states)
                experts_hidden_states.append(expert_hidden_states)
            return torch.cat(experts_hidden_states)

class MultiStreamLocalExperts(nn.Module):

    def __init__(self, config : MoEDistConfig, expert : nn.Module) -> None:
        super().__init__()
        self.experts = nn.ModuleList([deepcopy(expert) for _ in range(config.num_local_experts)])
        self.num_local_experts = config.num_local_experts
        self.world_size = config.world_size
        self.device = config.device
        self.num_streams = config.num_streams
        self.default_stream = torch.cuda.current_stream(device=self.device)
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_streams)]
        self.stream_schedule = self._make_stream_schedule(config)

    def _make_stream_schedule(self, config : MoEDistConfig) -> List[torch.cuda.Stream]:
        schedule = []
        for i in range(self.num_local_experts):
            schedule.append(self.streams[i % self.num_streams])
        return schedule

    def _wait_default(self) -> None:
        for stream in self.streams:
            stream.wait_stream(self.default_stream)

    def _wait_streams(self) -> None:
        for stream in self.streams:
            self.default_stream.wait_stream(stream)

    def forward(self, dispatched_hidden_states, experts_capacity_usage) -> torch.Tensor:
        chunks = dispatched_hidden_states.chunk(self.num_local_experts, dim=0)
        experts_hidden_states = []
        self._wait_default()
        for expert, hidden_states, stream in zip(self.experts, chunks, self.stream_schedule):
            with torch.cuda.stream(stream):
                expert_hidden_states = expert(hidden_states)
                experts_hidden_states.append(expert_hidden_states)
        self._wait_streams()
        return torch.cat(experts_hidden_states)

class AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, t: torch.Tensor) -> torch.Tensor:
        # ctx.group = group
        t = t.contiguous()
        output = torch.empty_like(t)
        # dist.all_to_all_single(output, t, group=group)
        dist.all_to_all_single(output, t)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:
        return AllToAll.apply(*grad_output)
        # Only need to return (None, grad) if the first input to forward was a ProcessGroup or something like this.
        # return (None, AllToAll.apply(*grad_output))

class AllToAllUtils():

    def __init__(self, config : MoEDistConfig) -> None:
        self.num_experts = config.num_experts
        self.num_local_experts = config.num_local_experts
        self.expert_capacity_coefficient = config.expert_capacity_coefficient
        self.world_size = config.world_size
        self.rank = config.rank
        self.device = config.device
        self.eventDtoH = config.eventDtoH

    def _capacity(self, nb_tokens : int) -> int:
        return int(self.expert_capacity_coefficient * (nb_tokens / self.num_experts))

    def dispatch_inputs(self, hidden_states : torch.Tensor, expert_assignments : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.Tensor]:
        # b = batch size
        # s = sequence length
        # m = hidden size
        # l = batch size * sequence length
        # e = number of experts
        # c = expert capacity
        
        # Expert assignments is a one hot encoding of the assignments and has shape (b,s,e). Tokens can be pre-dropped by 
        # settings its one-hot-encoding (last dimension) to all zeros.
        # Reshaped input has shape (l,e)
        reshaped_one_hot = expert_assignments.reshape(-1,*expert_assignments.shape[2:])

        # The capacity is dynamically computed from the number of tokens (denoted by l), and the capacity coefficient.
        expert_capacity = self._capacity(hidden_states.shape[0] * hidden_states.shape[1])

        # Has shape (e). Number of tokens per experts.
        nb_tokens_per_expert = reshaped_one_hot.detach().sum(dim=0).minimum(torch.tensor(expert_capacity))
        # Has shape (w,e) where w is the world size.
        global_nb_tokens_per_expert = torch.zeros([self.world_size, self.num_experts], dtype=nb_tokens_per_expert.dtype, device=nb_tokens_per_expert.device)
        # This needs to happens as early as possible so the GPU has enough time to send the tensor from GPU to CPU.
        dist.all_gather_into_tensor(global_nb_tokens_per_expert, nb_tokens_per_expert)
        global_nb_tokens_per_expert = global_nb_tokens_per_expert.to(torch.device("cpu"), non_blocking=True)
        self.eventDtoH.record(stream=torch.cuda.current_stream(self.device))

        # hidden_states has shape (b,s,m)
        # reshaped hidden states has shape (l,m)
        reshaped_hidden_states = hidden_states.reshape(-1, *hidden_states.shape[2:])

        # Has shape (l,e). Cumulative sum on each expert to compute the capacity index that each token has for its given index.
        index_from_cumsum = (torch.cumsum(reshaped_one_hot, dim=0) - 1) * reshaped_one_hot
        # Has shape (l). Given j=token_capacity_index[i], it means that token "i" has is at index "j" of its expert's capacity.
        token_capacity_index = torch.sum(index_from_cumsum, dim=1).to(torch.int64)
        # Here we compute a drop mask. Token that would overflow the capacity of an expert are dropped. The mask has shape (l)
        drop_mask = torch.lt(token_capacity_index, expert_capacity)
        # We drop tokens twice. First to do the one-hot-encoding, but since dropped tokens are automatically assigned to expert 0, 
        # then we drop them again after the one-hot-encoding.
        token_capacity_index *= drop_mask
        # Contains the position of each token in its expert's capacity. Has shape (l,c). For example, given a=token_to_expert[i], 
        # then the non-zero entry in "a" is where token "i" would be in the tokens assigned to the expert.
        token_to_expert = F.one_hot(token_capacity_index, num_classes=expert_capacity)
        # We remove the dropped tokens from expert 0's assignments.
        token_to_expert *= drop_mask.reshape(*token_capacity_index.shape, 1).expand(*token_to_expert.shape)

        # DEBUG
        # print("---- dispatch_inputs debug ----")
        # print("hidden_states.shape:", hidden_states.shape)
        # print("expert_assignments.shape:", expert_assignments.shape)
        # print("reshaped_one_hot.shape:", reshaped_one_hot.shape)
        # print("token_to_expert.shape:", token_to_expert.shape)
        # print("nb_tokens_per_expert:", nb_tokens_per_expert)


        # Has shape (l,e,c).
        dispatch_mask = torch.einsum('le,lc->lec', reshaped_one_hot, token_to_expert)

        # Has shape (e,c,m). Each entry dispatched_tokens[i] contains the tokens assigned to expert "i". An entry 
        # dispatched_tokens[i][j] contains the hidden states of some token assigned to expert "i".
        dispatched_tokens = torch.einsum('lec,lm->ecm', dispatch_mask.to(hidden_states.dtype), reshaped_hidden_states)

        return dispatched_tokens, dispatch_mask, hidden_states.shape, global_nb_tokens_per_expert

    # Converts from (w * n, c, m) to (n, w, c, m) where w is world size and n is number of local experts (rest as above)
    def prep_for_experts(self, hidden_states : torch.Tensor) -> torch.Tensor:
        return hidden_states.reshape(self.world_size, self.num_local_experts, *hidden_states.shape[1:]).permute(1,0,2,3)

    # From the input with shape (w,e), it takes the capacity usage of all it's local experts. The output has shape (n,w)
    def get_local_expert_capacity_usage(self, experts_capacity_usage : torch.Tensor) -> torch.Tensor:
        # This will return a view of the tensor. At this point, it is possible the data is not yet available in the tensor.
        return experts_capacity_usage[:,self.rank * self.num_local_experts : self.rank * self.num_local_experts + self.num_local_experts].permute(1,0)

    # Converts from (n, w, c, m) to (w * n, c, m)
    def dispatch_outputs(self, hidden_states : torch.Tensor) -> torch.Tensor:
        return hidden_states.permute(1,0,2,3).reshape(-1, *hidden_states.shape[2:])

    # Converts from (e, c, m) to (b, s, m)
    def combine(self, dispatched_experts_hidden_states : torch.Tensor, dispatch_mask : torch.Tensor, combine_shape : torch.Size) -> torch.Tensor:
        return torch.einsum('lec,ecm->lm', dispatch_mask.to(dispatched_experts_hidden_states.dtype), dispatched_experts_hidden_states).reshape(*combine_shape[0:2], *dispatched_experts_hidden_states.shape[2:])

class MoEDist(nn.Module):
    
    def __init__(self,
                 config: MoEDistConfig,
                 expert: nn.Module,
                 all_to_all: Type[AllToAll] = AllToAll,
                 transform: Optional[AllToAllUtils] = None) -> None:
        super().__init__()
        if transform is None:
            transform = AllToAllUtils(config)
        if config.multi_stream:
            self.experts = MultiStreamLocalExperts(config, expert)
        else:
            self.experts = LocalExperts(config, expert)
        self.all_to_all = all_to_all
        self.transform = transform
        self.capacity_stats = utils.MoECapacityStatsNOOP()
        if config.track_capacity:
            self.capacity_stats = utils.MoECapacityStatsSimple(config.device)

    def forward(self, hidden_states, expert_assignments) -> torch.Tensor:
        # Prepare tokens to send to their assigned expert. 
        # dispatched_hidden_states has shape (e,c,m), and it contains the tokens assigned to each expert.
        # dispatch_mask has shape (l,e,c) and it is the mask used to dispatch and regroup tokens before/after all-to-all.
        # combine_shape contains the shape of the tensor that will be regrouped.
        # global_experts_capacity_usage has shape (w,e), and it contains how many tokens are assigned to each expert by each rank.
        dispatched_hidden_states, dispatch_mask, combine_shape, global_experts_capacity_usage = self.transform.dispatch_inputs(hidden_states, expert_assignments)

        # All-to-all step to distribute tokens to the rank that has the expert they are assigned to. Receives the tokens assigned to the local experts.
        # dispatched_hidden_states has shape (w * n,c,m)
        dispatched_hidden_states = self.all_to_all.apply(dispatched_hidden_states)

        self.capacity_stats.start()
        
        # Reshape the dispatched tokens to shape (n,w,c,m).
        dispatched_hidden_states = self.transform.prep_for_experts(dispatched_hidden_states)
        # From the from the matrix of how many tokens are assigned to each expert by each rank, take only how many are assigned 
        # to the local experts by each rank and reshape into (n,w).
        local_experts_capacity_usage = self.transform.get_local_expert_capacity_usage(global_experts_capacity_usage)

        # Process the tokens through their assigned expert.
        local_experts_hidden_states = self.experts(dispatched_hidden_states, local_experts_capacity_usage)
        
        # Reshape the output for the all-to-all communication.
        local_experts_hidden_states = self.transform.dispatch_outputs(local_experts_hidden_states)
        
        self.capacity_stats.stop(local_experts_capacity_usage.sum())

        # All-to-all step to send the tokens back to their respective rank.
        dispatched_experts_hidden_states = self.all_to_all.apply(local_experts_hidden_states)
        
        # From the tokens received back from other ranks, reconstruct the sequences.
        experts_hidden_states = self.transform.combine(dispatched_experts_hidden_states, dispatch_mask, combine_shape)
        return experts_hidden_states


