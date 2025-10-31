from abc import ABC, abstractmethod
from datetime import timedelta

from torch._C import device
from src.trainer.stats.utils import RunningEnergy, RunningStat, RunningTimer
from typing import Callable, List, Tuple
import argparse
import enum
import os
import pandas as pd
import pynvml
import torch
import torch.distributed as dist

class TaskType(enum.Enum):
    ALL_REDUCE = "all_reduce"
    ALL_TO_ALL_SINGLE = "a2a_single"
    SEND_RCV = "sr"
    MATRIX_MULTIPLY = "mm"
    TENSOR_ADD = "tadd"
    TENSOR_COPY = "tcopy"
    TENSOR_SCALAR_ADD = "tsa"
    TENSOR_SCALAR_SUB = "tss"
    TENSOR_SCALAR_MUL = "tsm"
    TENSOR_SCALAR_DIV = "tsd"
    ACTIVATION_FUNCTION = "act_fn"
    HOST_TO_DEVICE = "h2d"
    DEVICE_TO_HOST = "d2h"
    # TODO (greta) add new task!

    @classmethod
    def from_str(cls, task : str):
        return cls(task)

    def __str__(self) -> str:
        return self.value

    def is_distributed(self):
        if self.value == "all_reduce":
            return True
        elif self.value == "a2a_single":
            return True
        elif self.value == "sr":
            return True
        return False

class Config:

    def __init__(self, args : argparse.Namespace) -> None:
        self.task = TaskType.from_str(args.task)
        self.world_size = int(os.environ.get("WORLD_SIZE", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.is_distributed = self.task.is_distributed()
        self.device = torch.device("cuda:0") if self.rank == -1 else torch.device(f"cuda:{self.rank}")
        self.gpu_freqs = args.gpu_freqs

        # All-Reduce configs
        self.all_reduce_num_warmup_iterations = args.all_reduce_num_warmup_iterations
        self.all_reduce_num_iterations = args.all_reduce_num_iterations
        self.all_reduce_num_comm_per_iteration = args.all_reduce_num_comm_per_iteration
        self.all_reduce_log_rank = args.all_reduce_log_rank
        self.all_reduce_numel_input = args.all_reduce_numel_input

        # All-to-all single configs
        self.a2a_single_num_warmup_iterations = args.a2a_single_num_warmup_iterations
        self.a2a_single_num_iterations = args.a2a_single_num_iterations
        self.a2a_single_num_comm_per_iteration = args.a2a_single_num_comm_per_iteration
        self.a2a_single_log_rank = args.a2a_single_log_rank
        self.a2a_single_numel_per_rank = args.a2a_single_numel_per_rank

        # Send-receive configs
        self.sr_num_warmup_iterations = args.sr_num_warmup_iterations
        self.sr_num_iterations = args.sr_num_iterations
        self.sr_num_comm_per_iteration = args.sr_num_comm_per_iteration
        self.sr_numel = args.sr_numel

        # Matrix multiplication configs
        self.mm_num_warmup_iterations = args.mm_num_warmup_iterations
        self.mm_num_iterations = args.mm_num_iterations
        self.mm_num_op_per_iteration = args.mm_num_op_per_iteration
        self.mm_in_dimension = args.mm_in_dimension
        self.mm_intermediate_dimension = args.mm_intermediate_dimension
        self.mm_out_dimension = args.mm_out_dimension
        self.mm_transpose = args.mm_transpose

        # Tensor addition config
        self.tadd_num_warmup_iterations = args.tadd_num_warmup_iterations
        self.tadd_num_iterations = args.tadd_num_iterations
        self.tadd_num_op_per_iteration = args.tadd_num_op_per_iteration
        self.tadd_matrix_shape = tuple(args.tadd_matrix_shape)
        self.tadd_num_data_reuse = args.tadd_num_data_reuse
    
        # Tensor copy config
        self.tcopy_num_warmup_iterations = args.tcopy_num_warmup_iterations
        self.tcopy_num_iterations = args.tcopy_num_iterations
        self.tcopy_num_op_per_iteration = args.tcopy_num_op_per_iteration
        self.tcopy_num_data_reuse = args.tcopy_num_data_reuse
        self.tcopy_numel = args.tcopy_numel

        # Tensor scalar addition config
        self.tsa_num_warmup_iterations = args.tsa_num_warmup_iterations
        self.tsa_num_iterations = args.tsa_num_iterations
        self.tsa_num_op_per_iteration = args.tsa_num_op_per_iteration
        self.tsa_num_data_reuse = args.tsa_num_data_reuse
        self.tsa_numel = args.tsa_numel
        self.tsa_scalar = args.tsa_scalar

        # Tensor scalar subtraction config
        self.tss_num_warmup_iterations = args.tss_num_warmup_iterations
        self.tss_num_iterations = args.tss_num_iterations
        self.tss_num_op_per_iteration = args.tss_num_op_per_iteration
        self.tss_num_data_reuse = args.tss_num_data_reuse
        self.tss_numel = args.tss_numel
        self.tss_scalar = args.tss_scalar

        # Tensor scalar multiplication config
        self.tsm_num_warmup_iterations = args.tsm_num_warmup_iterations
        self.tsm_num_iterations = args.tsm_num_iterations
        self.tsm_num_op_per_iteration = args.tsm_num_op_per_iteration
        self.tsm_num_data_reuse = args.tsm_num_data_reuse
        self.tsm_numel = args.tsm_numel
        self.tsm_scalar = args.tsm_scalar

        # Tensor scalar division config
        self.tsd_num_warmup_iterations = args.tsd_num_warmup_iterations
        self.tsd_num_iterations = args.tsd_num_iterations
        self.tsd_num_op_per_iteration = args.tsd_num_op_per_iteration
        self.tsd_num_data_reuse = args.tsd_num_data_reuse
        self.tsd_numel = args.tsd_numel
        self.tsd_scalar = args.tsd_scalar

        # Activation function config
        self.act_fn_num_warmup_iterations = args.act_fn_num_warmup_iterations
        self.act_fn_num_iterations = args.act_fn_num_iterations
        self.act_fn_num_op_per_iteration = args.act_fn_num_op_per_iteration
        self.act_fn_num_data_reuse = args.act_fn_num_data_reuse
        self.act_fn_numel = args.act_fn_numel
        self.act_fn_scaling = args.act_fn_scaling
        self.act_fn_function = args.act_fn_function

        # Host to device transfer config
        self.h2d_num_warmup_iterations = args.h2d_num_warmup_iterations
        self.h2d_num_iterations = args.h2d_num_iterations
        self.h2d_num_ops_per_iterations = args.h2d_num_ops_per_iterations
        self.h2d_numel = args.h2d_numel

        # Device to host transfer config
        self.d2h_num_warmup_iterations = args.d2h_num_warmup_iterations
        self.d2h_num_iterations = args.d2h_num_iterations
        self.d2h_num_ops_per_iterations = args.d2h_num_ops_per_iterations
        self.d2h_numel = args.d2h_numel

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_freqs", action="extend", nargs="+", type=int, help="GPU frequencies on which to run the task. The task is repeated for each frequency. A frequency of -1 will run the task with the default GPU configuration.")
    parser.add_argument("--task", type=str, help="Which task to perform. A task's configuration flags are prefixed with the task name.", choices=[TaskType.ALL_REDUCE.value, TaskType.ALL_TO_ALL_SINGLE.value, TaskType.SEND_RCV.value, TaskType.MATRIX_MULTIPLY.value, TaskType.TENSOR_ADD.value, TaskType.TENSOR_COPY.value, TaskType.TENSOR_SCALAR_ADD.value, TaskType.TENSOR_SCALAR_SUB.value, TaskType.TENSOR_SCALAR_MUL.value, TaskType.TENSOR_SCALAR_DIV.value, TaskType.ACTIVATION_FUNCTION.value, TaskType.HOST_TO_DEVICE.value, TaskType.DEVICE_TO_HOST.value])

    # All-reduce args
    parser.add_argument("--all_reduce_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--all_reduce_num_comm_per_iteration", type=int, help="Number of all-reduce communications per iteration.", default=10)
    parser.add_argument("--all_reduce_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--all_reduce_log_rank", type=int, help="Which rank will log it stats", default=0)
    parser.add_argument("--all_reduce_numel_input", type=int, help="The number of elements in the input. (default=16777216 (64MB))", default=16777216)

    # All-to-all single args
    parser.add_argument("--a2a_single_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--a2a_single_num_comm_per_iteration", type=int, help="Number of all-to-all communications per iteration.", default=10)
    parser.add_argument("--a2a_single_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--a2a_single_log_rank", type=int, help="Which rank will log it stats", default=0)
    parser.add_argument("--a2a_single_numel_per_rank", type=int, help="The number of elements each rank send to each rank. The input tensor will be world_size times this number. (default=16777216 (16MB))", default=4194304)

    # Send-Receive args
    parser.add_argument("--sr_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--sr_num_comm_per_iteration", type=int, help="Number of send-receive communications per iteration.", default=10)
    parser.add_argument("--sr_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--sr_numel", type=int, help="The number of elements in the tensor sent. (default=16777216 (64MB))", default=16777216)

    # Matrix multiplication
    parser.add_argument("--mm_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--mm_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--mm_num_op_per_iteration", type=int, help="Number of matrix multiplication per iteration.", default=10)
    parser.add_argument("--mm_in_dimension", type=int, help="Input dimension of the multiplication. Given matrices A, B with dim n x p and p x m, the output has dim n x m. The input dimension is n.", default=512)
    parser.add_argument("--mm_intermediate_dimension", type=int, help="Intermediate dimension of the multiplication. Given matrices A, B with dim n x p and p x m, the output has dim n x m. The intermediate dimension is p.", default=512)
    parser.add_argument("--mm_out_dimension", type=int, help="Output dimension of the multiplication. Given matrices A, B with dim n x p and p x m, the output has dim n x m. The output dimension is m.", default=512)
    parser.add_argument("--mm_transpose", action="store_true", help="Where to store the second matrix transposed. Given matrices A, B with dim n x p and p x m. B will be stored with dim m x p, and the multiplication with be A(t(B)) where t(B) is the transpose of B instead of performing AB.")

    # Tensor addition
    parser.add_argument("--tadd_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--tadd_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--tadd_num_op_per_iteration", type=int, help="Number of matrix multiplication per iteration.", default=10)
    parser.add_argument("--tadd_num_data_reuse", type=int, help="Number times to reuse the data.", default=1)
    parser.add_argument("--tadd_matrix_shape", action="extend", nargs="+", type=int, help="The shape of the tensors to add. This is a list, the length of the list is the number of dimensions and each value is the size of the corresponding dimension.", default=[])
    
    # Tensor copy
    parser.add_argument("--tcopy_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--tcopy_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--tcopy_num_op_per_iteration", type=int, help="Number of matrix multiplication per iteration.", default=10)
    parser.add_argument("--tcopy_num_data_reuse", type=int, help="Number times to reuse the data.", default=1)
    parser.add_argument("--tcopy_numel", type=int, help="Number of elements per tensor getting copied. (default=16777216 (64MB))", default=16777216)

    # Tensor scalar addition
    parser.add_argument("--tsa_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--tsa_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--tsa_num_op_per_iteration", type=int, help="Number of tensor scalar additions per iteration.", default=10)
    parser.add_argument("--tsa_num_data_reuse", type=int, help="Number times to reuse the data.", default=1)
    parser.add_argument("--tsa_numel", type=int, help="Number of elements in the tensors used. (default=16777216 (64MB))", default=16777216)
    parser.add_argument("--tsa_scalar", type=float, help="Scalar to add to tensors.", default=1.0e-5)

    # Tensor scalar addition
    parser.add_argument("--tss_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--tss_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--tss_num_op_per_iteration", type=int, help="Number of tensor scalar additions per iteration.", default=10)
    parser.add_argument("--tss_num_data_reuse", type=int, help="Number times to reuse the data.", default=1)
    parser.add_argument("--tss_numel", type=int, help="Number of elements in the tensors used. (default=16777216 (64MB))", default=16777216)
    parser.add_argument("--tss_scalar", type=float, help="Scalar to subtract to tensors.", default=1.0e-5)

    # Tensor scalar addition
    parser.add_argument("--tsm_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--tsm_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--tsm_num_op_per_iteration", type=int, help="Number of tensor scalar additions per iteration.", default=10)
    parser.add_argument("--tsm_num_data_reuse", type=int, help="Number times to reuse the data.", default=1)
    parser.add_argument("--tsm_numel", type=int, help="Number of elements in the tensors used. (default=16777216 (64MB))", default=16777216)
    parser.add_argument("--tsm_scalar", type=float, help="Scalar to multiply tensors.", default=1.0e-5)

    # Tensor scalar division
    parser.add_argument("--tsd_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--tsd_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--tsd_num_op_per_iteration", type=int, help="Number of tensor scalar additions per iteration.", default=10)
    parser.add_argument("--tsd_num_data_reuse", type=int, help="Number times to reuse the data.", default=1)
    parser.add_argument("--tsd_numel", type=int, help="Number of elements in the tensors used. (default=16777216 (64MB))", default=16777216)
    parser.add_argument("--tsd_scalar", type=float, help="Scalar to divide tensors.", default=1.0e-5)

    # Tensor scalar division
    parser.add_argument("--act_fn_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements", default=5)
    parser.add_argument("--act_fn_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--act_fn_num_op_per_iteration", type=int, help="Number of calls to the activation function per iteration.", default=10)
    parser.add_argument("--act_fn_num_data_reuse", type=int, help="Number times to reuse the data.", default=1)
    parser.add_argument("--act_fn_numel", type=int, help="Number of elements in the tensors used. (default=16777216 (64MB))", default=16777216)
    parser.add_argument("--act_fn_scaling", type=float, help="Scales the values in the tensors to have range (-act_fn_scaling,act_fn_scaling).", default=5.0)
    parser.add_argument("--act_fn_function", type=str, help="Name of the activation function to use.", default="relu", choices=["relu", "gelu", "silu"])

    # Host to device transfer
    parser.add_argument("--h2d_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements.", default=5)
    parser.add_argument("--h2d_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--h2d_num_ops_per_iterations", type=int, help="Number of transfers per iteration.", default=10)
    parser.add_argument("--h2d_numel", type=int, help="Number of elements per tensor. (default=16777216 (64MB))", default=16777216)

    # Device to host transfer
    parser.add_argument("--d2h_num_warmup_iterations", type=int, help="Number of warmup iterations before doing measurements.", default=5)
    parser.add_argument("--d2h_num_iterations", type=int, help="How many times to run inputs.", default=25)
    parser.add_argument("--d2h_num_ops_per_iterations", type=int, help="Number of transfers per iteration.", default=10)
    parser.add_argument("--d2h_numel", type=int, help="Number of elements per tensor. (default=16777216 (64MB))", default=16777216)

    args = parser.parse_args()
    return args

class InputGenerator:
    def __init__(self, numel : int, device : torch.device, pin_memory : bool = False) -> None:
        self.numel_input = numel
        self.device = device
        self.pin_memory = pin_memory

    def gen(self):
        return torch.rand(self.numel_input, device=self.device, pin_memory=self.pin_memory)

    def gen_n(self, n: int):
        values = []
        for _ in range(n):
            values.append(self.gen())
        return values

class GPUModulator:

    def __init__(self, config : Config) -> None:
        self.rank = config.rank
        self.world_size = config.world_size
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)

    def reset_gpu(self):
        pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)

    def log_gpu_clocks(self):
        print(f"GRAPHICS {pynvml.nvmlDeviceGetClock(self.handle, pynvml.NVML_CLOCK_GRAPHICS, pynvml.NVML_CLOCK_ID_CURRENT)} --- MEM {pynvml.nvmlDeviceGetClock(self.handle, pynvml.NVML_CLOCK_MEM, pynvml.NVML_CLOCK_ID_CURRENT)} --- SM {pynvml.nvmlDeviceGetClock(self.handle, pynvml.NVML_CLOCK_SM, pynvml.NVML_CLOCK_ID_CURRENT)} -- VIDEO {pynvml.nvmlDeviceGetClock(self.handle, pynvml.NVML_CLOCK_VIDEO, pynvml.NVML_CLOCK_ID_CURRENT)}")

    def set_gpu_freq(self, f : int):
        if f == -1:
            self.reset_gpu()
            return
        pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, 0, f)

class Task(ABC):

    @abstractmethod
    def run(self, gpu_freq : int):
        pass

    @abstractmethod
    def log(self):
        pass

class AllReduceTask(Task):
    device : torch.device
    rank : int
    world_size : int
    num_warmup_iterations : int
    num_iterations : int
    num_comm_per_iteration : int
    numel : int
    generator : InputGenerator
    comm_time : RunningTimer
    comm_energy : RunningEnergy
    comm_gpu_freq : RunningStat
    should_log : bool

    def __init__(self, config : Config) -> None:
        super().__init__()
        self.device = config.device
        self.rank = config.rank
        self.world_size = config.world_size
        self.num_warmup_iterations = config.all_reduce_num_warmup_iterations
        self.num_iterations = config.all_reduce_num_iterations
        self.num_comm_per_iteration = config.all_reduce_num_comm_per_iteration
        self.numel = config.all_reduce_numel_input
        self.generator = InputGenerator(config.all_reduce_numel_input, config.device)
        self.comm_time = RunningTimer()
        self.comm_energy = RunningEnergy(self.device.index)
        self.comm_gpu_freq = RunningStat()
        self.should_log = config.all_reduce_log_rank == self.rank

    def perform_comms(self, inputs : List[torch.Tensor]):
        for input in inputs:
            dist.all_reduce(input, op=dist.ReduceOp.SUM)

    def warmup(self):
        for _ in range(self.num_warmup_iterations):
            self.perform_comms(self.generator.gen_n(self.num_comm_per_iteration))

    def comm_iterations(self, i : int, gpu_freq : int) -> None:
        inputs = self.generator.gen_n(self.num_comm_per_iteration)

        self.comm_gpu_freq.update(gpu_freq)
        torch.cuda.synchronize(device=self.device)
        self.comm_time.start()
        self.comm_energy.start()
        self.perform_comms(inputs)
        torch.cuda.synchronize(device=self.device)
        self.comm_time.stop()
        self.comm_energy.stop()

        if self.should_log:
            print(f"iter({i}): {self.comm_time.get_last() / 1000000}")

    def run(self, gpu_freq : int):
        self.warmup()
        for i in range(self.num_iterations):
            self.comm_iterations(i, gpu_freq)

    def log(self):
        num_rows = len(self.comm_time.stat.history)
        data_dict = {
            "gpu_freq": self.comm_gpu_freq.history,
            "num_warmup_iterations": [self.num_warmup_iterations] * num_rows,
            "num_iterations": [self.num_iterations] * num_rows,
            "num_op_per_iteration": [self.num_comm_per_iteration] * num_rows,
            "numel": [self.numel] * num_rows,
            "time_ns": self.comm_time.stat.history,
            "energy_uj": self.comm_energy.stat.history,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv(f"all_reduce-{self.rank}.csv", index=False)

class AllToAllSingleTask(Task):
    device : torch.device
    rank : int
    world_size : int
    num_warmup_iterations : int
    num_iterations : int
    num_comm_per_iteration : int
    numel : int
    generator : InputGenerator
    comm_time : RunningTimer
    comm_energy : RunningEnergy
    comm_gpu_freq : RunningStat
    should_log : bool

    def __init__(self, config : Config) -> None:
        super().__init__()
        self.device = config.device
        self.rank = config.rank
        self.world_size = config.world_size
        self.num_warmup_iterations = config.a2a_single_num_warmup_iterations
        self.num_iterations = config.a2a_single_num_iterations
        self.num_comm_per_iteration = config.a2a_single_num_comm_per_iteration
        self.numel = self.world_size * config.a2a_single_numel_per_rank
        self.generator = InputGenerator(self.numel, config.device)
        self.comm_time = RunningTimer()
        self.comm_energy = RunningEnergy(self.device.index)
        self.comm_gpu_freq = RunningStat()
        self.should_log = config.a2a_single_log_rank == self.rank

    def perform_comms(self, inputs : List[torch.Tensor], outputs : List[torch.Tensor]):
        for input, output in zip(inputs, outputs):
            dist.all_to_all_single(output, input)

    def warmup(self):
        for _ in range(self.num_warmup_iterations):
            self.perform_comms(self.generator.gen_n(self.num_comm_per_iteration), self.generator.gen_n(self.num_comm_per_iteration))

    def comm_iterations(self, i : int, gpu_freq : int) -> None:
        inputs = self.generator.gen_n(self.num_comm_per_iteration)
        outputs = self.generator.gen_n(self.num_comm_per_iteration)

        self.comm_gpu_freq.update(gpu_freq)
        torch.cuda.synchronize(device=self.device)
        self.comm_time.start()
        self.comm_energy.start()
        self.perform_comms(inputs, outputs)
        torch.cuda.synchronize(device=self.device)
        self.comm_time.stop()
        self.comm_energy.stop()

        if self.should_log:
            print(f"iter({i}): {self.comm_time.get_last() / 1000000}")

    def run(self, gpu_freq : int):
        self.warmup()
        for i in range(self.num_iterations):
            self.comm_iterations(i, gpu_freq)

    def log(self):
        num_rows = len(self.comm_time.stat.history)
        data_dict = {
            "gpu_freq": self.comm_gpu_freq.history,
            "num_warmup_iterations": [self.num_warmup_iterations] * num_rows,
            "num_iterations": [self.num_iterations] * num_rows,
            "num_op_per_iteration": [self.num_comm_per_iteration] * num_rows,
            "numel": [self.numel] * num_rows,
            "time_ns": self.comm_time.stat.history,
            "energy_uj": self.comm_energy.stat.history,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv(f"a2a_single-{self.rank}.csv", index=False)

class SendReceiveTask(Task):
    device : torch.device
    rank : int
    world_size : int
    num_warmup_iterations : int
    num_iterations : int
    num_comm_per_iteration : int
    numel : int
    generator : InputGenerator
    comm_time : RunningTimer
    comm_energy : RunningEnergy
    comm_gpu_freq : RunningStat
    should_log : bool

    def __init__(self, config : Config) -> None:
        super().__init__()
        assert config.world_size == 2 # Only support one sender and one receiver
        self.device = config.device
        self.rank = config.rank
        self.world_size = config.world_size
        self.num_warmup_iterations = config.sr_num_warmup_iterations
        self.num_iterations = config.sr_num_iterations
        self.num_comm_per_iteration = config.sr_num_comm_per_iteration
        self.numel = config.sr_numel
        self.generator = InputGenerator(self.numel, config.device)
        self.comm_time = RunningTimer()
        self.comm_energy = RunningEnergy(self.device.index)
        self.comm_gpu_freq = RunningStat()

    def perform_comms(self, inputs : List[torch.Tensor]):
        for t in inputs:
            if self.rank == 0:
                dist.send(t, dst=1)
            else:
                dist.recv(t, src=0)

    def warmup(self):
        for _ in range(self.num_warmup_iterations):
            self.perform_comms(self.generator.gen_n(self.num_comm_per_iteration))

    def comm_iterations(self, i : int, gpu_freq : int) -> None:
        inputs = self.generator.gen_n(self.num_comm_per_iteration)

        self.comm_gpu_freq.update(gpu_freq)
        torch.cuda.synchronize(device=self.device)
        self.comm_time.start()
        self.comm_energy.start()
        self.perform_comms(inputs)
        torch.cuda.synchronize(device=self.device)
        self.comm_time.stop()
        self.comm_energy.stop()

        if self.rank == 0:
            print(f"iter({i}): {self.comm_time.get_last() / 1000000}")

    def run(self, gpu_freq : int):
        self.warmup()
        for i in range(self.num_iterations):
            self.comm_iterations(i, gpu_freq)

    def log(self):
        num_rows = len(self.comm_time.stat.history)
        data_dict = {
            "gpu_freq": self.comm_gpu_freq.history,
            "num_warmup_iterations": [self.num_warmup_iterations] * num_rows,
            "num_iterations": [self.num_iterations] * num_rows,
            "num_op_per_iteration": [self.num_comm_per_iteration] * num_rows,
            "numel": [self.numel] * num_rows,
            "time_ns": self.comm_time.stat.history,
            "energy_uj": self.comm_energy.stat.history,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv(f"sr-{self.rank}.csv", index=False)

class MatrixMultiplyTask(Task):
    device : torch.device
    num_warmup_iterations : int
    num_iterations : int
    num_op_per_iteration : int
    in_dim : int
    intermediate_dim : int
    out_dim : int
    numel_A : int
    numel_B : int
    generator_A : InputGenerator
    generator_B : InputGenerator
    transpose : bool
    op_time : RunningTimer
    op_energy : RunningEnergy
    op_gpu_freq : RunningStat

    def __init__(self, config : Config) -> None:
        super().__init__()
        self.device = config.device
        self.num_warmup_iterations = config.mm_num_warmup_iterations
        self.num_iterations = config.mm_num_iterations
        self.num_op_per_iteration = config.mm_num_op_per_iteration

        self.in_dim = config.mm_in_dimension
        self.intermediate_dim = config.mm_intermediate_dimension
        self.out_dim = config.mm_out_dimension
        self.numel_A = self.in_dim * self.intermediate_dim
        self.numel_B = self.intermediate_dim * self.out_dim

        self.generator_A = InputGenerator(self.numel_A, self.device)
        self.generator_B = InputGenerator(self.numel_B, self.device)

        self.transpose = config.mm_transpose

        self.op_time = RunningTimer()
        self.op_energy = RunningEnergy(self.device.index)
        self.op_gpu_freq = RunningStat()

    def op(self, A : torch.Tensor, B : torch.Tensor):
        A = A.view(self.in_dim, self.intermediate_dim)
        if self.transpose:
            B = B.view(self.out_dim, self.intermediate_dim)
            B = B.t()
        else:
            B = B.view(self.intermediate_dim, self.out_dim)
        torch.mm(A, B)

    def perform_ops(self, As : List[torch.Tensor], Bs : List[torch.Tensor]):
        for A, B in zip(As, Bs):
            self.op(A, B)

    def warmup(self):
        for _ in range(self.num_warmup_iterations):
            self.perform_ops(self.generator_A.gen_n(self.num_op_per_iteration), self.generator_B.gen_n(self.num_op_per_iteration))

    def iteration(self, i : int, gpu_freq : int):
        A_inputs = self.generator_A.gen_n(self.num_op_per_iteration)
        B_inputs = self.generator_B.gen_n(self.num_op_per_iteration)

        self.op_gpu_freq.update(gpu_freq)
        torch.cuda.synchronize(device=self.device)
        self.op_time.start()
        self.op_energy.start()
        self.perform_ops(A_inputs, B_inputs)
        torch.cuda.synchronize(device=self.device)
        self.op_time.stop()
        self.op_energy.stop()
        print(f"iter({i}): {self.op_time.get_last() / 1000000}ms")

    def run(self, gpu_freq : int):
        self.warmup()
        for i in range(self.num_iterations):
            self.iteration(i, gpu_freq)

    def log(self):
        num_rows = len(self.op_time.stat.history)
        data_dict = {
            "gpu_freq": self.op_gpu_freq.history,
            "num_warmup_iterations": [self.num_warmup_iterations] * num_rows,
            "num_iterations": [self.num_iterations] * num_rows,
            "num_op_per_iteration": [self.num_op_per_iteration] * num_rows,
            "in_dim": [self.in_dim] * num_rows,
            "intermediate_dim": [self.intermediate_dim] * num_rows,
            "out_dim": [self.out_dim] * num_rows,
            "transpose": [self.transpose] * num_rows,
            "time_ns": self.op_time.stat.history,
            "energy_uj": self.op_energy.stat.history,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv("mm.csv", index=False)

class TensorAdditionTask(Task):
    device : torch.device
    num_warmup_iterations : int
    num_iterations : int
    num_data_reuse : int
    num_op_per_iteration : int
    shape : Tuple[int, ...]
    numel : int
    generator_A : InputGenerator
    generator_B : InputGenerator
    op_time : RunningTimer
    op_energy : RunningEnergy
    op_gpu_freq : RunningStat

    def __init__(self, config : Config) -> None:
        super().__init__()

        self.device = config.device
        self.num_warmup_iterations = config.tadd_num_warmup_iterations
        self.num_iterations = config.tadd_num_iterations
        self.num_data_reuse = config.tadd_num_data_reuse
        self.num_op_per_iteration = config.tadd_num_op_per_iteration
        
        self.shape = config.tadd_matrix_shape
        self.numel = 1
        for dim in self.shape:
            self.numel *= dim

        self.generator_A = InputGenerator(self.numel, self.device)
        self.generator_B = InputGenerator(self.numel, self.device)

        self.op_time = RunningTimer()
        self.op_energy = RunningEnergy(self.device.index)
        self.op_gpu_freq = RunningStat()

    def op(self, A : torch.Tensor, B : torch.Tensor):
        A = A.view(self.shape)
        B = B.view(self.shape)
        _ = A + B

    def perform_ops(self, As : List[torch.Tensor], Bs : List[torch.Tensor]):
        for _ in range(self.num_data_reuse):
            for A, B in zip(As, Bs):
                self.op(A, B)

    def warmup(self):
        for _ in range(self.num_warmup_iterations):
            self.perform_ops(self.generator_A.gen_n(self.num_op_per_iteration), self.generator_B.gen_n(self.num_op_per_iteration))

    def iteration(self, i : int, gpu_freq : int):
        A_inputs = self.generator_A.gen_n(self.num_op_per_iteration)
        B_inputs = self.generator_B.gen_n(self.num_op_per_iteration)

        self.op_gpu_freq.update(gpu_freq)
        torch.cuda.synchronize(device=self.device)
        self.op_time.start()
        self.op_energy.start()
        self.perform_ops(A_inputs, B_inputs)
        torch.cuda.synchronize(device=self.device)
        self.op_time.stop()
        self.op_energy.stop()
        print(f"iter({i}): {self.op_time.get_last() / 1000000}ms")

    def run(self, gpu_freq : int):
        self.warmup()
        for i in range(self.num_iterations):
            self.iteration(i, gpu_freq)

    def log(self):
        num_rows = len(self.op_time.stat.history)
        data_dict = {
            "gpu_freq": self.op_gpu_freq.history,
            "num_warmup_iterations": [self.num_warmup_iterations] * num_rows,
            "num_iterations": [self.num_iterations] * num_rows,
            "num_op_per_iteration": [self.num_op_per_iteration] * num_rows,
            "num_data_reuse": [self.num_data_reuse] * num_rows,
            "shape": [self.shape.__str__()] * num_rows,
            "time_ns": self.op_time.stat.history,
            "energy_uj": self.op_energy.stat.history,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv("tadd.csv", index=False)

class TensorCopyTask(Task):
    device : torch.device
    num_warmup_iterations : int
    num_iterations : int
    num_data_reuse : int
    num_op_per_iteration : int
    numel : int
    generator : InputGenerator
    op_time : RunningTimer
    op_energy : RunningEnergy
    op_gpu_freq : RunningStat

    def __init__(self, config : Config) -> None:
        super().__init__()

        self.device = config.device
        self.num_warmup_iterations = config.tcopy_num_warmup_iterations
        self.num_iterations = config.tcopy_num_iterations
        self.num_data_reuse = config.tcopy_num_data_reuse
        self.num_op_per_iteration = config.tcopy_num_op_per_iteration
        self.numel = config.tcopy_numel

        self.generator = InputGenerator(self.numel, self.device)

        self.op_time = RunningTimer()
        self.op_energy = RunningEnergy(self.device.index)
        self.op_gpu_freq = RunningStat()

    def op(self, src : torch.Tensor, dest : torch.Tensor):
        dest.copy_(src)

    def perform_ops(self, sources : List[torch.Tensor], destinations : List[torch.Tensor]):
        for _ in range(self.num_data_reuse):
            for src, dst in zip(sources, destinations):
                self.op(src, dst)

    def warmup(self):
        for _ in range(self.num_warmup_iterations):
            self.perform_ops(self.generator.gen_n(self.num_op_per_iteration), self.generator.gen_n(self.num_op_per_iteration))

    def iteration(self, i : int, gpu_freq : int):
        sources = self.generator.gen_n(self.num_op_per_iteration)
        destinations = self.generator.gen_n(self.num_op_per_iteration)

        self.op_gpu_freq.update(gpu_freq)
        torch.cuda.synchronize(device=self.device)
        self.op_time.start()
        self.op_energy.start()
        self.perform_ops(sources, destinations)
        torch.cuda.synchronize(device=self.device)
        self.op_time.stop()
        self.op_energy.stop()
        print(f"iter({i}): {self.op_time.get_last() / 1000000}ms")

    def run(self, gpu_freq : int):
        self.warmup()
        for i in range(self.num_iterations):
            self.iteration(i, gpu_freq)

    def log(self):
        num_rows = len(self.op_time.stat.history)
        data_dict = {
            "gpu_freq": self.op_gpu_freq.history,
            "num_warmup_iterations": [self.num_warmup_iterations] * num_rows,
            "num_iterations": [self.num_iterations] * num_rows,
            "num_op_per_iteration": [self.num_op_per_iteration] * num_rows,
            "num_data_reuse": [self.num_data_reuse] * num_rows,
            "numel": [self.numel] * num_rows,
            "time_ns": self.op_time.stat.history,
            "energy_uj": self.op_energy.stat.history,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv("tcopy.csv", index=False)

class TensorScalarOpTask(Task):
    device : torch.device
    num_warmup_iterations : int
    num_iterations : int
    num_data_reuse : int
    num_op_per_iteration : int
    numel : int
    generator_A : InputGenerator
    scalar : float
    op_time : RunningTimer
    op_energy : RunningEnergy
    op_gpu_freq : RunningStat
    out_name : str

    def __init__(self, 
                 device : torch.device,
                 num_warmup_iterations : int,
                 num_iterations : int,
                 num_data_reuse : int,
                 num_op_per_iteration : int,
                 numel : int,
                 scalar : float,
                 out_name : str) -> None:
        super().__init__()

        self.device = device
        self.num_warmup_iterations = num_warmup_iterations
        self.num_iterations = num_iterations
        self.num_data_reuse = num_data_reuse
        self.num_op_per_iteration = num_op_per_iteration
        
        self.numel = numel

        self.scalar = scalar

        self.generator_A = InputGenerator(self.numel, self.device)

        self.op_time = RunningTimer()
        self.op_energy = RunningEnergy(self.device.index)
        self.op_gpu_freq = RunningStat()

        self.out_name = out_name

    @abstractmethod
    def op(self, A : torch.Tensor, x : float):
        pass

    def perform_ops(self, As : List[torch.Tensor], x : float):
        for _ in range(self.num_data_reuse):
            for A in As:
                self.op(A, x)

    def warmup(self):
        for _ in range(self.num_warmup_iterations):
            self.perform_ops(self.generator_A.gen_n(self.num_op_per_iteration), self.scalar)

    def iteration(self, i : int, gpu_freq : int):
        A_inputs = self.generator_A.gen_n(self.num_op_per_iteration)

        self.op_gpu_freq.update(gpu_freq)
        torch.cuda.synchronize(device=self.device)
        self.op_time.start()
        self.op_energy.start()
        self.perform_ops(A_inputs, self.scalar)
        torch.cuda.synchronize(device=self.device)
        self.op_time.stop()
        self.op_energy.stop()
        print(f"iter({i}): {self.op_time.get_last() / 1000000}ms")

    def run(self, gpu_freq : int):
        self.warmup()
        for i in range(self.num_iterations):
            self.iteration(i, gpu_freq)

    def log(self):
        num_rows = len(self.op_time.stat.history)
        data_dict = {
            "gpu_freq": self.op_gpu_freq.history,
            "num_warmup_iterations": [self.num_warmup_iterations] * num_rows,
            "num_iterations": [self.num_iterations] * num_rows,
            "num_op_per_iteration": [self.num_op_per_iteration] * num_rows,
            "num_data_reuse": [self.num_data_reuse] * num_rows,
            "numel": [self.numel] * num_rows,
            "scalar": [self.scalar] * num_rows,
            "time_ns": self.op_time.stat.history,
            "energy_uj": self.op_energy.stat.history,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv(self.out_name, index=False)

class TensorScalarAddTask(TensorScalarOpTask):

    def __init__(self, config : Config) -> None:
        super().__init__(config.device, config.tsa_num_warmup_iterations, config.tsa_num_iterations, config.tsa_num_data_reuse, config.tsa_num_op_per_iteration, config.tsa_numel, config.tsa_scalar, "tsa.csv")

    def op(self, A : torch.Tensor, x : float):
        _ = A + x

class TensorScalarSubTask(TensorScalarOpTask):

    def __init__(self, config : Config) -> None:
        super().__init__(config.device, config.tss_num_warmup_iterations, config.tss_num_iterations, config.tss_num_data_reuse, config.tss_num_op_per_iteration, config.tss_numel, config.tss_scalar, "tss.csv")

    def op(self, A : torch.Tensor, x : float):
        _ = A - x


class TensorScalarMulTask(TensorScalarOpTask):

    def __init__(self, config : Config) -> None:
        super().__init__(config.device, config.tsm_num_warmup_iterations, config.tsm_num_iterations, config.tsm_num_data_reuse, config.tsm_num_op_per_iteration, config.tsm_numel, config.tsm_scalar, "tsm.csv")

    def op(self, A : torch.Tensor, x : float):
        _ = A * x

class TensorScalarDivTask(TensorScalarOpTask):

    def __init__(self, config : Config) -> None:
        super().__init__(config.device, config.tsd_num_warmup_iterations, config.tsd_num_iterations, config.tsd_num_data_reuse, config.tsd_num_op_per_iteration, config.tsd_numel, config.tsd_scalar, "tsd.csv")

    def op(self, A : torch.Tensor, x : float):
        _ = A / x

class ActivationFunctionTask(Task):
    device : torch.device
    num_warmup_iterations : int
    num_iterations : int
    num_data_reuse : int
    num_op_per_iteration : int
    numel : int
    scaling : float
    generator : InputGenerator
    act_fn : Callable
    op_time : RunningTimer
    op_energy : RunningEnergy
    op_gpu_freq : RunningStat
    out_name : str

    def __init__(self, config : Config) -> None:
        super().__init__()

        self.device = config.device
        self.num_warmup_iterations = config.act_fn_num_warmup_iterations
        self.num_iterations = config.act_fn_num_iterations
        self.num_data_reuse = config.act_fn_num_data_reuse
        self.num_op_per_iteration = config.act_fn_num_op_per_iteration
        
        self.numel = config.act_fn_numel

        self.scaling = config.act_fn_scaling

        self.generator = InputGenerator(self.numel, self.device)

        self.act_fn_name = config.act_fn_function
        if self.act_fn_name == "relu":
            self.act_fn = torch.nn.ReLU()
        elif self.act_fn_name == "gelu":
            self.act_fn = torch.nn.GELU()
        elif self.act_fn_name == "silu":
            self.act_fn = torch.nn.SiLU()
        else:
            raise(Exception(f"Unknown activation function {self.act_fn_name}"))

        self.op_time = RunningTimer()
        self.op_energy = RunningEnergy(self.device.index)
        self.op_gpu_freq = RunningStat()

    def op(self, A : torch.Tensor):
        _ = self.act_fn(A)

    def perform_ops(self, As : List[torch.Tensor]):
        for _ in range(self.num_data_reuse):
            for A in As:
                self.op(A)

    def gen_data(self) -> List[torch.Tensor]:
        # The generator generates uniform values in the range [0,1). By generating pairs and doing subtraction, 
        # we generate data in the range [-0.5,0.5). Then we scale by 2x to have a range of [-x,x)
        A = self.generator.gen_n(self.num_op_per_iteration)
        data = [(a - 0.5) * (2 * self.scaling) for a in A]
        return data

    def warmup(self):
        for _ in range(self.num_warmup_iterations):
            self.perform_ops(self.gen_data())

    def iteration(self, i : int, gpu_freq : int):
        A_inputs = self.gen_data()

        self.op_gpu_freq.update(gpu_freq)
        torch.cuda.synchronize(device=self.device)
        self.op_time.start()
        self.op_energy.start()
        self.perform_ops(A_inputs)
        torch.cuda.synchronize(device=self.device)
        self.op_time.stop()
        self.op_energy.stop()
        print(f"iter({i}): {self.op_time.get_last() / 1000000}ms")

    def run(self, gpu_freq : int):
        self.warmup()
        for i in range(self.num_iterations):
            self.iteration(i, gpu_freq)

    def log(self):
        num_rows = len(self.op_time.stat.history)
        data_dict = {
            "gpu_freq": self.op_gpu_freq.history,
            "num_warmup_iterations": [self.num_warmup_iterations] * num_rows,
            "num_iterations": [self.num_iterations] * num_rows,
            "num_op_per_iteration": [self.num_op_per_iteration] * num_rows,
            "num_data_reuse": [self.num_data_reuse] * num_rows,
            "numel": [self.numel] * num_rows,
            "scaling": [self.scaling] * num_rows,
            "act_fn_name": [self.act_fn_name] * num_rows,
            "time_ns": self.op_time.stat.history,
            "energy_uj": self.op_energy.stat.history,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv(f"act_fn_{self.act_fn_name}.csv", index=False)

class HostDeviceTransferTask(Task):
    src : torch.device
    dst : torch.device
    num_warmup_iterations : int
    num_iterations : int
    num_data_reuse : int
    num_ops_per_iteration : int
    numel : int
    generator : InputGenerator
    op_time : RunningTimer
    op_energy : RunningEnergy
    op_gpu_freq : RunningStat
    out_name : str

    def __init__(self, src : torch.device, dst : torch.device, num_warmup_iterations : int, num_iterations : int, num_data_reuse : int, num_ops_per_iteration : int, numel : int, pin_memory : bool, op_energy : RunningEnergy, out_name : str) -> None:
        super().__init__()
        self.src = src
        self.dst = dst
        self.num_warmup_iterations = num_warmup_iterations
        self.num_iterations = num_iterations
        self.num_data_reuse = num_data_reuse
        self.num_ops_per_iteration = num_ops_per_iteration
        self.numel = numel
        self.out_name = out_name

        self.generator = InputGenerator(self.numel, self.src, pin_memory=pin_memory)
        
        self.op_time = RunningTimer()
        self.op_energy = op_energy
        self.op_gpu_freq = RunningStat()

    @abstractmethod
    def synchronize(self):
        pass

    def op(self, t : torch.Tensor):
        t.to(device=self.dst, non_blocking=True)
    
    def perform_ops(self, inputs : List[torch.Tensor]):
        for _ in range(self.num_data_reuse):
            for t in inputs:
                self.op(t)

    def warmup(self):
        for _ in range(self.num_warmup_iterations):
            self.perform_ops(self.generator.gen_n(self.num_ops_per_iteration))

    def iteration(self, i : int, gpu_freq : int):
        inputs = self.generator.gen_n(self.num_ops_per_iteration)

        self.op_gpu_freq.update(gpu_freq)
        self.synchronize()
        self.op_time.start()
        self.op_energy.start()
        self.perform_ops(inputs)
        self.synchronize()
        self.op_time.stop()
        self.op_energy.stop()
        print(f"iter({i}): {self.op_time.get_last() / 1000000}ms")

    def run(self, gpu_freq : int):
        self.warmup()
        for i in range(self.num_iterations):
            self.iteration(i, gpu_freq)

    def log(self):
        num_rows = len(self.op_time.stat.history)
        data_dict = {
            "gpu_freq": self.op_gpu_freq.history,
            "num_warmup_iterations": [self.num_warmup_iterations] * num_rows,
            "num_iterations": [self.num_iterations] * num_rows,
            "num_ops_per_iteration": [self.num_ops_per_iteration] * num_rows,
            "numel": [self.numel] * num_rows,
            "time_ns": self.op_time.stat.history,
            "energy_uj": self.op_energy.stat.history,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv(self.out_name, index=False)

class HostToDeviceTask(HostDeviceTransferTask):

    def __init__(self, config : Config) -> None:
        # CPU side memory allocation is expensive, so we reuse it. tensor.to(gpu) allocates everytime it is called so this should be enough.
        # Also, this is why we pin memory for this task: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        super().__init__(torch.device("cpu"), config.device, config.h2d_num_warmup_iterations, config.h2d_num_iterations, config.h2d_num_ops_per_iterations, 1, config.h2d_numel, True, RunningEnergy(config.device.index), "h2d.csv")

    def synchronize(self):
        torch.cuda.synchronize(device=self.dst)

class DeviceToHost(HostDeviceTransferTask):

    def __init__(self, config : Config) -> None:
        super().__init__(config.device, torch.device("cpu"), config.d2h_num_warmup_iterations, config.d2h_num_iterations, 1, config.d2h_num_ops_per_iterations, config.d2h_numel, False, RunningEnergy(config.device.index), "d2h.csv")

    def synchronize(self):
        torch.cuda.synchronize(device=self.src)

def task_factory(config : Config) -> Task:
    if config.task == TaskType.ALL_REDUCE:
        return AllReduceTask(config)
    elif config.task == TaskType.ALL_TO_ALL_SINGLE:
        return AllToAllSingleTask(config)
    elif config.task == TaskType.SEND_RCV:
        return SendReceiveTask(config)
    elif config.task == TaskType.MATRIX_MULTIPLY:
        return MatrixMultiplyTask(config)
    elif config.task == TaskType.TENSOR_ADD:
        return TensorAdditionTask(config)
    elif config.task == TaskType.TENSOR_COPY:
        return TensorCopyTask(config)
    elif config.task == TaskType.TENSOR_SCALAR_ADD:
        return TensorScalarAddTask(config)
    elif config.task == TaskType.TENSOR_SCALAR_SUB:
        return TensorScalarSubTask(config)
    elif config.task == TaskType.TENSOR_SCALAR_MUL:
        return TensorScalarMulTask(config)
    elif config.task == TaskType.TENSOR_SCALAR_DIV:
        return TensorScalarDivTask(config)
    elif config.task == TaskType.ACTIVATION_FUNCTION:
        return ActivationFunctionTask(config)
    elif config.task == TaskType.HOST_TO_DEVICE:
        return HostToDeviceTask(config)
    elif config.task == TaskType.DEVICE_TO_HOST:
        return DeviceToHost(config)
    raise(Exception(f"Unknown task {config.task}"))

def init_dist(config : Config) -> None:
    dist.init_process_group(
        backend='nccl',
        rank=config.rank,
        world_size=config.world_size,
        timeout=timedelta(seconds=20),
    )

def init(config : Config) -> Tuple[Task, GPUModulator]:
    pynvml.nvmlInit()
    task = task_factory(config)
    modulator = GPUModulator(config)
    if config.is_distributed:
        init_dist(config)

    return task, modulator

def process_args(args : argparse.Namespace):
    config = Config(args)

    return config


if __name__ == "__main__":
    args = get_args()
    
    config = process_args(args)

    task, modulator = init(config)

    for freq in config.gpu_freqs:
        modulator.set_gpu_freq(freq)
        modulator.log_gpu_clocks()
        task.run(freq)
    task.log()
    modulator.reset_gpu()
