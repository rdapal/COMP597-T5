from abc import ABC, abstractmethod
from typing import List, Tuple
import time
import torch

class MoECapacityStats(ABC):

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self, nb_tokens : torch.Tensor) -> None:
        pass

    @abstractmethod
    def log_history(self):
        pass

class MoECapacityStatsNOOP(MoECapacityStats):
    
    def __init__(self) -> None:
        super().__init__()

    def start(self) -> None:
        pass

    def stop(self, nb_tokens: torch.Tensor) -> None:
        pass

    def log_history(self):
        pass

class MoECapacityStatsSimple(MoECapacityStats):

    def __init__(self, device : torch.device) -> None:
        super().__init__()
        self.history : List[Tuple[torch.Tensor, int]] = []
        self.start_ts = 0
        self.device = device

    def start(self) -> None:
        torch.cuda.synchronize(self.device)
        self.start_ts = time.perf_counter_ns()

    def stop(self, nb_tokens : torch.Tensor) -> None: 
        torch.cuda.synchronize(self.device)
        self.history.append((nb_tokens, time.perf_counter_ns() - self.start_ts))

    def log_history(self) -> None:
        for nb_tokens, t in self.history:
            print(f"{nb_tokens},{t / 1e6 : .4f}")
