import ctypes
# To import this, you need to install the appropriate module using 
# "pip install cuda-python==VERSION" where you set VERSION to an appropriate 
# version of the package, matching your environment. You can find more about 
# the module here: https://nvidia.github.io/cuda-python/cuda-bindings/latest/index.html
# This was designed for version 12.6, other versions may not work.
import cuda.bindings.runtime as cudart
import pynvml


# import os
# import torch
# print("Compiled CUDA version:", torch.version.cuda)
# print("Runtime CUDA available:", torch.cuda.is_available())
# print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
# print(f"PyTorch sees {torch.cuda.device_count()} GPUs")
# for i in range(torch.cuda.device_count()):
#     print(f"[CUDA {i}] {torch.cuda.get_device_name(i)}")
# pynvml.nvmlInit()
# n = pynvml.nvmlDeviceGetCount()
# for i in range(n):
#     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#     pci = pynvml.nvmlDeviceGetPciInfo(handle)
#     print(f"[NVML {i}] PCI Bus ID: {pci.busId.encode()}")
# for i in range(torch.cuda.device_count()):
#     print(f"CUDA {i} → {torch.cuda.get_device_properties(i).pci_bus_id}")


# This is required, because the cuda index can be different than the ones 
# provides by pynvml/nvidia-smi. It depends on whether CUDA_VISIBLE_DEVICES and 
# CUDA_DEVICE_ORDER are set. The PCI bus ID provides a way to translate between 
# the two. Environment variables are documented here:
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-environment-variables
def get_pci_bus_id_by_cuda_index(index : int) -> str:
    # Set length long enough to accomodate any length. (doc: https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html#cuda.bindings.runtime.cudaDeviceGetPCIBusId)
    bus_id_max_str_length = 1024

    print(f"Getting PCI bus ID for CUDA device with index {index}")
    err, bus_id = cudart.cudaDeviceGetPCIBusId(bus_id_max_str_length, index)
    if err != cudart.cudaError_t.cudaSuccess:
        raise Exception(f"Failed to obtain PCI bus id for device with index {index}, error: {err.name}")
    # Use strip to remove unused bytes from the string.
    return bus_id.strip()

def get_nvml_handle_by_cuda_index(index : int) -> "ctypes._Pointer[pynvml.struct_c_nvmlDevice_t]":
    # return pynvml.nvmlDeviceGetHandleByPciBusId(get_pci_bus_id_by_cuda_index(index))
    if index is None:
        index = 0
    return pynvml.nvmlDeviceGetHandleByIndex(index)