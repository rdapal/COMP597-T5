import ctypes
import multiprocessing as mp
import os
from typing import List
import pynvml
import random
import statistics
import time

class GPU:
    handle : "ctypes._Pointer[pynvml.struct_c_nvmlDevice_t]" # NVML device handle
    cuda_index : int
    index : int

    # This constructor makes it compatible with torch.device.index
    def __init__(self, cuda_index : int) -> None:
        self.cuda_index = cuda_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_index)
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

def rand_sleep():
    return (random.randrange(0, 7, 1) + 1) / 1000

def func1(ready, q, s1, s2, v, c1, c2, num_iter):
    ready.wait()
    for _ in range(num_iter):
        q.put(time.perf_counter_ns())
        time.sleep(rand_sleep())
        s2.acquire()

    for _ in range(num_iter):
        v.value = time.perf_counter_ns()
        s1.release()
        time.sleep(rand_sleep())
        s2.acquire()

    for _ in range(num_iter):
        v.value = time.perf_counter_ns()
        c1.value = True
        time.sleep(rand_sleep())
        while not c2.value:
            pass
        c2.value = False

def func2(ready, q, s1, s2, v, c1, c2, num_iter):
    pynvml.nvmlInit()
    q_delays = []
    sem_delays = []
    shared_var_delays = []

    q_nvml = []
    sem_nvml = []
    shared_var_nvml = []

    device = GPU(int(os.environ.get("LOCAL_RANK", 0)))

    ready.set()
    for _ in range(num_iter):
        t = q.get()
        q_delays.append(time.perf_counter_ns() - t)
        # start = time.perf_counter_ns()
        s = time.perf_counter_ns()
        # device.set_gpu_locked_clocks(300, 1500)
        # device.reset_gpu_locked_clocks()
        # device.set_gpu_locked_clocks(300, 1500)
        device.set_gpu_locked_clocks(0, 1500)
        e = time.perf_counter_ns()
        q_nvml.append(e - s)
        # end = time.perf_counter_ns()
        # print(f"{os.environ.get("LOCAL_RANK", 0)} {(end - start) / 1000000 :> 12.4f}")
        # time.sleep((random.randrange(0, 50000, 1) + 50) / 1000000)
        # time.sleep(0.015)
        s2.release()

    for _ in range(num_iter):
        s1.acquire()
        sem_delays.append(time.perf_counter_ns() - v.value)
        s = time.perf_counter_ns()
        # device.set_gpu_locked_clocks(300, 1500)
        # device.reset_gpu_locked_clocks()
        # device.set_gpu_locked_clocks(300, 1500)
        device.set_gpu_locked_clocks(0, 1500)
        e = time.perf_counter_ns()
        sem_nvml.append(e - s)
        # time.sleep((random.randrange(0, 50000, 1) + 50) / 1000000)
        # time.sleep(rand_sleep())
        s2.release()

    for _ in range(num_iter):
        while not c1.value:
            time.sleep(0.001)
        c1.value = False
        shared_var_delays.append(time.perf_counter_ns() - v.value)
        s = time.perf_counter_ns()
        # device.set_gpu_locked_clocks(300, 1500)
        # device.reset_gpu_locked_clocks()
        # device.set_gpu_locked_clocks(300, 1500)
        device.set_gpu_locked_clocks(0, 1500)
        e = time.perf_counter_ns()
        shared_var_nvml.append(e - s)
        # time.sleep((random.randrange(0, 50000, 1) + 50) / 1000000)
        # time.sleep(rand_sleep())
        c2.value = True
    device.reset_gpu_locked_clocks()

    q_delays = q_delays[5:]
    sem_delays = sem_delays[5:]
    shared_var_delays = shared_var_delays[5:]
    
    print(f"queue          : {statistics.mean(q_delays) / 1e6 :> 12.4f} {statistics.stdev(q_delays) / 1e6 :> 12.4f} {min(q_delays) / 1e6 :> 12.4f} {max(q_delays) / 1e6 :> 12.4f} {statistics.quantiles(q_delays, n=100)[-1] / 1e6 :> 12.4f}")
    print(f"queue nvml     : {statistics.mean(q_nvml) / 1e6 :> 12.4f} {statistics.stdev(q_nvml) / 1e6 :> 12.4f} {min(q_nvml) / 1e6 :> 12.4f} {max(q_nvml) / 1e6 :> 12.4f} {statistics.quantiles(q_nvml, n=100)[-1] / 1e6 :> 12.4f}")
    print(f"sem            : {statistics.mean(sem_delays) / 1e6 :> 12.4f} {statistics.stdev(sem_delays) / 1e6 :> 12.4f} {min(sem_delays) / 1e6 :> 12.4f} {max(sem_delays) / 1e6 :> 12.4f} {statistics.quantiles(sem_delays, n=100)[-1] / 1e6 :> 12.4f}")
    print(f"sem nvml       : {statistics.mean(sem_nvml) / 1e6 :> 12.4f} {statistics.stdev(sem_nvml) / 1e6 :> 12.4f} {min(sem_nvml) / 1e6 :> 12.4f} {max(sem_nvml) / 1e6 :> 12.4f} {statistics.quantiles(sem_nvml, n=100)[-1] / 1e6 :> 12.4f}")
    print(f"shared var     : {statistics.mean(shared_var_delays) / 1e6 : 12.4f} {statistics.stdev(shared_var_delays) / 1e6 :> 12.4f} {min(shared_var_delays) / 1e6 :> 12.4f} {max(shared_var_delays) / 1e6 :> 12.4f} {statistics.quantiles(shared_var_delays, n=100)[-1] / 1e6 :> 12.4f}")
    print(f"shared var nvml: {statistics.mean(shared_var_nvml) / 1e6 : 12.4f} {statistics.stdev(shared_var_nvml) / 1e6 :> 12.4f} {min(shared_var_nvml) / 1e6 :> 12.4f} {max(shared_var_nvml) / 1e6 :> 12.4f} {statistics.quantiles(shared_var_nvml, n=100)[-1] / 1e6 :> 12.4f}")

def main():
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    q = ctx.Queue()
    s1 = ctx.Semaphore(value=0)
    s2 = ctx.Semaphore(value=0)
    v = ctx.Value(ctypes.c_int64)
    c1 = ctx.Value(ctypes.c_bool)
    c2 = ctx.Value(ctypes.c_bool)
    num_iter = 505
    p = ctx.Process(target=func2, args=(ready, q, s1, s2, v, c1, c2, num_iter))
    p.start()
    func1(ready, q, s1, s2, v, c1, c2, num_iter)
    p.join()

if __name__ == "__main__":
    main()
