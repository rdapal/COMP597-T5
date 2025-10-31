import ctypes
from typing import List
import pynvml
import src.hardware_management.utils as utils

class GPU:
    handle : "ctypes._Pointer[pynvml.struct_c_nvmlDevice_t]" # NVML device handle
    cuda_index : int
    index : int

    # This constructor makes it compatible with torch.device.index
    def __init__(self, cuda_index : int) -> None:
        self.cuda_index = cuda_index
        self.handle = utils.get_nvml_handle_by_cuda_index(self.cuda_index)
        self.index = pynvml.nvmlDeviceGetIndex(self.handle)

    def get_sm_frequency(self) -> int:
        return pynvml.nvmlDeviceGetClock(self.handle, pynvml.NVML_CLOCK_SM, pynvml.NVML_CLOCK_ID_CURRENT)

    def get_memory_frequency(self) -> int:
        return pynvml.nvmlDeviceGetClock(self.handle, pynvml.NVML_CLOCK_MEM, pynvml.NVML_CLOCK_ID_CURRENT)

    def get_supported_graphics_clock(self, mem_frequency : int) -> List[int]:
        return pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.handle, mem_frequency)
    
    def get_min_supported_graphics_clock(self, mem_frequency : int) -> int:
        clock_frequencies = self.get_supported_graphics_clock(mem_frequency)
        min = clock_frequencies[0]
        for f in clock_frequencies:
            if f < min:
                min = f
        return min

    def get_min_currently_supported_graphics_clock(self) -> int:
        return self.get_min_supported_graphics_clock(self.get_memory_frequency())

    def get_max_supported_graphics_clock(self, mem_frequency : int) -> int:
        clock_frequencies = self.get_supported_graphics_clock(mem_frequency)
        max = clock_frequencies[0]
        for f in clock_frequencies:
            if f > max:
                max = f
        return max

    def get_max_currently_supported_graphics_clock(self) -> int:
        return self.get_max_supported_graphics_clock(self.get_memory_frequency())

    def get_currently_supported_graphics_clock(self) -> List[int]:
        return self.get_supported_graphics_clock(self.get_memory_frequency())

    def set_gpu_locked_clocks(self, min_frequency : int, max_frequency : int) -> None:
        pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, min_frequency, max_frequency)

    def set_gpu_locked_clocks_force(self, frequency) -> None:
        self.set_gpu_locked_clocks(frequency, frequency)

    def reset_gpu_locked_clocks(self):
        pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)
