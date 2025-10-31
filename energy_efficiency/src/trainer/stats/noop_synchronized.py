import src.trainer.stats.base as base
import torch

class NOOPSynchronizedTrainerStats(base.TrainerStats):
    """Synchronized NOOP Trainer stats to ignore data accumulation but still synchronise the Python code 
    execution with asynchronous GPU.

    This class implements the `TrainerStats` interface. All the methods are 
    NOOP so that training can be done with accumulating statistics.

    """

    def __init__(self, device : torch.device) -> None:
        super().__init__()
        self.device = device

    def start_train(self) -> None:
        torch.cuda.synchronize(self.device)

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)

    def start_step(self) -> None:
        torch.cuda.synchronize(self.device)
    
    def stop_step(self) -> None:
        torch.cuda.synchronize(self.device)

    def start_optimizer_step(self) -> None:
        torch.cuda.synchronize(self.device)

    def stop_optimizer_step(self) -> None:
        torch.cuda.synchronize(self.device)

    def start_forward(self) -> None:
        torch.cuda.synchronize(self.device)

    def stop_forward(self) -> None:
        torch.cuda.synchronize(self.device)

    def start_backward(self) -> None:
        torch.cuda.synchronize(self.device)

    def stop_backward(self) -> None:
        torch.cuda.synchronize(self.device)

    def start_save_checkpoint(self) -> None:
        torch.cuda.synchronize(self.device)

    def stop_save_checkpoint(self) -> None:
        torch.cuda.synchronize(self.device)

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        pass

    def log_loss(self, loss: float, rank: int) -> None:
        pass
