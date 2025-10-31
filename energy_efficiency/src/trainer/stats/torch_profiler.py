import os
import src.trainer.stats.base as base
import torch.profiler

class TorchProfilerStats(base.TrainerStats):
    """Used to profile training with the PyTorch profiler.

    This class is used to profile the training of a `Trainer` using the PyTorch 
    profiler. It only records the training steps.

    Parameters
    ----------
    pr
        A PyTorch profiler object used for profiling.

    Attributes
    ----------
    pr : torch.profiler.profile
        The PyTorch profiler object used for profiling as provided to the 
        constructor.

    """

    def __init__(self, pr : torch.profiler.profile) -> None:
        super().__init__()
        self.pr = pr

    def start_train(self) -> None:
        """Start the profiler.
        """
        self.pr.start()

    def stop_train(self) -> None:
        """Stop the profiler.
        
        The trace is outputted as a chrome trace in the file 
        `training-trace.json`. This trace can be visualized at [ui.perfetto.dev](ui.perfetto.dev). 

        Notes
        -----
            If training is distributed, only rank 0 will output its trace. If 
            you expect that ranks will have varying traces, this should not be 
            used.

        """
        self.pr.stop()
        if "RANK" not in os.environ:
            self.pr.export_chrome_trace("training-trace.json")
        else:
            self.pr.export_chrome_trace(f'training-trace-{os.environ["RANK"]}.json')

    def start_step(self) -> None:
        """Start a profiler step.
        """
        self.pr.step()

    def stop_step(self) -> None:
        pass

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
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

    def log_loss(self, loss: float, rank: int) -> None:
        pass
