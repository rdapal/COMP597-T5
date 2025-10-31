from typing import Any, Optional, Tuple
import src.hardware_management.scheduler as scheduler
import torch

class ThrottledComputation(torch.autograd.Function):

    @staticmethod
    def forward(ctx : Any, freq_scheduler : scheduler.Scheduler, frequency : int, *t : torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.scheduler = freq_scheduler
        ctx.frequency = frequency
        freq_scheduler.schedule_throttling(frequency)
        return t

    @staticmethod
    def backward(ctx : Any, *grad_outputs : Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        ctx.scheduler.schedule_reset()
        return (None, None, *grad_outputs)

class UnrestrictedComputation(torch.autograd.Function):

    @staticmethod
    def forward(ctx : Any, freq_scheduler : scheduler.Scheduler, frequency : int, *t : torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.scheduler = freq_scheduler
        ctx.frequency = frequency
        freq_scheduler.schedule_reset()
        return t

    @staticmethod
    def backward(ctx : Any, *grad_outputs : Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        ctx.scheduler.schedule_throttling(ctx.frequency)
        return (None, None, *grad_outputs)

class ThrottledWorkloadComputation(torch.autograd.Function):

    @staticmethod
    def forward(ctx : Any, frequency_scheduler : scheduler.Scheduler, forward_workload : str, backward_workload : str, *args : torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.scheduler = frequency_scheduler
        ctx.backward_workload = backward_workload
        frequency_scheduler.schedule_throttling(workload_name=forward_workload)
        return args

    @staticmethod
    def backward(ctx : Any, *grad_outputs : Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        ctx.scheduler.schedule_reset(workload_name=ctx.backward_workload)
        print(f"throttled backward {ctx.backward_workload}")
        return (None, None, None, *grad_outputs)

class UnrestrictedWorkloadComputation(torch.autograd.Function):

    @staticmethod
    def forward(ctx : Any, frequency_scheduler : scheduler.Scheduler, forward_workload : str, backward_workload : str, *args : torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.scheduler = frequency_scheduler
        ctx.backward_workload = backward_workload
        frequency_scheduler.schedule_reset(workload_name=forward_workload)
        return args

    @staticmethod
    def backward(ctx : Any, *grad_outputs : Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        ctx.scheduler.schedule_throttling(workload_name=ctx.backward_workload)
        print(f"unrestricted backward {ctx.backward_workload}")
        return (None, None, None, *grad_outputs)
