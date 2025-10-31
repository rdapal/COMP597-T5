import os
import pandas as pd
import pynvml
import src.trainer.stats.base as base
import time
import torch

class AveragedEnergy(base.TrainerStats):

    def __init__(self, device : torch.device, num_steps_required : int = 5) -> None:
        super().__init__()
        self.device = device
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
        self.start_time = 0
        self.start_energy = 0
        self.end_time = 0
        self.end_energy = 0
        self.num_steps_required = num_steps_required
        self.num_steps_done = 0
        self.total_num_steps_done = 0
        self.rank = os.environ.get("RANK", 0)
        self.computation_time = []
        self.energy = []
        # self.temps = []
        self.steps = []

    def _get_energy(self) -> int:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
    
    # def _get_temp(self) -> int:
    #     # TODO This function is deprecated in newer versions of NVML (newer than currently used), will need to upgrade to nvmlDeviceGetTemperatureV
    #     return pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

    def start_train(self) -> None:
        pass

    def stop_train(self) -> None:
        data = pd.DataFrame({
            "rank": [self.rank] * len(self.energy),
            "step": self.steps,
            "time": self.computation_time,
            "energy": self.energy,
            # "temp": self.temps,
        })
        data.to_csv(f"average-energy-{self.rank}.csv", index=False)

    def start_step(self) -> None:
        if self.num_steps_done == 0:
            torch.cuda.synchronize(device=self.device)
            self.start_time = time.perf_counter_ns()
            self.start_energy = self._get_energy()

    def stop_step(self) -> None:
        self.num_steps_done += 1
        self.total_num_steps_done += 1
        if self.num_steps_done == self.num_steps_required:
            torch.cuda.synchronize(device=self.device)
            self.end_time = time.perf_counter_ns()
            self.end_energy = self._get_energy()
            measurement = self.end_energy - self.start_energy
            self.computation_time.append(self.end_time - self.start_time)
            self.energy.append(measurement)
            # self.temps.append(self._get_temp())
            self.steps.append(self.total_num_steps_done)
            print(f"Energy over {self.num_steps_required} steps : {measurement / 1e3 : .3f}J")
            self.num_steps_done = 0

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
        pass

    def log_loss(self, loss: float, rank: int) -> None:
        pass

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        pass

