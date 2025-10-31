from src.hardware_management.gpu import GPU
from src.hardware_management.functions import ThrottledComputation, UnrestrictedComputation, ThrottledWorkloadComputation, UnrestrictedWorkloadComputation
from src.hardware_management.modules import ThrottledModule
from src.hardware_management.scheduler import (
    SchedulerConfig,
    SchedulerProcess,
    Scheduler,
    NOOPScheduler,
    FrequencyScheduler,
    LearnedFrequencyScheduler,
)
import pynvml
import src.config as config
import torch

# Initialize NVML to make the module standalone so the user does not have to initalize it himself.
try:
    pynvml.nvmlInit()
# The type checker might complain about NVMLError_AlreadyInitialized not being 
# a class of pynvml. It might also propose pynvml.NVML_ERROR_ALREADY_INITIALIZED 
# instead. However, pynvml.NVMLError_AlreadyInitialized is the appropriate class 
# to use. pynvml.NVML_ERROR_ALREADY_INITIALIZED is the integer error code 
# returned by the C code.
except pynvml.NVMLError_AlreadyInitialized: 
    pass

def init_scheduler_from_conf(conf : config.Config, device : torch.device) -> Scheduler:
    if conf.enable_throttling:
        return LearnedFrequencyScheduler(SchedulerConfig(device))
    else:
        return NOOPScheduler()
