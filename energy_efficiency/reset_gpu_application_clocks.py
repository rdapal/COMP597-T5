import pynvml

def reset_gpu(gpu_index : int):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    pynvml.nvmlDeviceResetApplicationsClocks(handle)

def main():
    pynvml.nvmlInit()
    nb_gpu = pynvml.nvmlDeviceGetCount()
    for i in range(nb_gpu):
        reset_gpu(i)

if __name__ == "__main__":
    main()
