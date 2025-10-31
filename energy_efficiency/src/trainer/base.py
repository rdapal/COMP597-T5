from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from src.trainer.stats.utils import RunningEnergy
import src.hardware_management as hardware_management
import src.trainer.stats as stats
import src.trainer.utils as utils
import torch
import torch.nn as nn
import torch.utils.data as data
import tqdm.auto
import src.config as config

class Trainer(ABC):
    """Base class implemented by all trainer objects.

    This abstract class defines the methods and basic attributes of a trainer 
    class.

    Parameters
    ----------
    model
        The model to train.
    loader
        A PyTorch dataloader that will be used to obtain the data at each step.
    device
        The device on which the model resides and where batches will be moved to.
    stats
        An object to gather statistics during training. It is an optional 
        parameter. The default `NOOPTrainerStats` will simply no-op on every 
        call used to gather statistical data.
    frequency_scheduler
        The `Scheduler` object used to modulate the GPU's frequency.
    checkpoint_frequency
        The after how many steps a checkpoint is saved.

    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    loader : torch.utils.data.DataLoader
        The object used to load data during training.
    device : torch.device
        The device on which the model resides and where batches will be moved to.
    stats : src.trainer.stats.TrainerStats
        The `TrainerStats` object used to gather statistics.
    frequency_scheduler : src.hardware_management.Scheduler
        The `Scheduler` object used to modulate the GPU's frequency.
    checkpoint_frequency : int
        The after how many steps a checkpoint is saved.

    """

    def __init__(self, 
                 model : nn.Module, 
                 loader : data.DataLoader, 
                 device : torch.device, 
                 stats : stats.TrainerStats = stats.NOOPTrainerStats(), 
                 frequency_scheduler : hardware_management.Scheduler = hardware_management.NOOPScheduler(),
                 enable_checkpointing : bool = False,
                 checkpoint_frequency : int = 1,
                 conf: Optional[config.Config] = None):
        self.model = model
        self.loader = loader
        self.device = device
        self.stats = stats
        self.frequency_scheduler = frequency_scheduler
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_frequency = checkpoint_frequency
        self.conf = conf

    # TODO(Olivier): Consider adding parameters to allow configurations of the workloads
    def register_frequency_scheduler_workloads(self) -> None:
        """Register the frequency scheduler that will be used.

        The purpose is to provide and extendable method to register all the 
        frequency scheduler workloads that will be used. Children class can 
        override it to add workloads.
        """
        self.frequency_scheduler.register_workload(utils.TrainingComponents.OPTIMIZER_STEP.value)
        self.frequency_scheduler.register_workload(utils.TrainingComponents.SAVE_CHECKPOINT.value)

    def should_save_checkpoint(self, i : int) -> bool:
        """Condition to device when to save a checkpoint.

        Implements a condition to check if it is time to save a checkpoint 
        based on the iteration number. This allows customizing the behaviour 
        of when to save a checkpoint.

        Parameters
        ----------
        i
            This is the iteration number.

        """
        return (i + 1) % self.checkpoint_frequency == 0

    def checkpoint_path(self, i : int) -> str:
        """Generates the checkpoint path.

        This generates the path and filename of where to save the checkpoint. 
        This allows easy customization of how to name checkpoints and where to 
        save them.

        Parameters
        ----------
        i
            This is the iteration number.

        """
        return "checkpoint.tar"

    # TODO(Olivier) Consider adding the loss to the checkpoint.
    def checkpoint_dict(self, i : int) -> Dict[str,Any]:
        """Generate the dictionary of contents to save in the checkpoint.
        
        This generates a dictionary containing all the information that should 
        be saved in order to be able to restore from the checkpoint.

        Parameters
        ----------
        i
            This is the iteration number.

        Returns
        -------
        dict
            The dictionary that will be saved.
        """
        return {
            "step": i,
            "model_state_dict": self.model.state_dict(),
        }

    def save_checkpoint(self, i : int) -> None:
        """Save a checkpoint of the current training task.

        This is designed to save a snapshot of the current training task. It 
        should save the model state, optimizer state along with anything else 
        required to restart training from the checkpoint.

        Parameters
        ----------
        i 
            This is the iteration number.

        """
        path = self.checkpoint_path(i)
        checkpoint_dict = self.checkpoint_dict(i)
        torch.save(checkpoint_dict, path)
    
    def process_batch(self, i : int, batch : Any) -> Any:
        return {k: v.to(self.device) for k, v in batch.items()}
    
    @abstractmethod
    def forward(self, i : int, batch : Any, model_kwargs : Dict[str, Any]) -> torch.Tensor:
        """Execution of the model's forward pass.

        Parameters
        ----------
        i
            This is the iteration number.
        batch
            The data of the current batch. The type is defined by the 
            `DataLoader` provided when the object was constructed. It must be a 
            dictionary or similar as it is unpacked to the model using 
            `**batch`.
        model_kwargs
            Additional arguments that need to be provided to the model during 
            the forward pass.
        
        Returns
        -------
        torch.Tensor
            The loss computed during the iteration.
        
        """
        pass

    @abstractmethod
    def backward(self, i : int, loss : torch.Tensor) -> None:
        """Execution of the model's backward pass.

        Parameters
        ----------
        i
            The iteration number.
        loss
            The loss computed during the forward pass.
        
        """
        pass

    @abstractmethod
    def optimizer_step(self, i : int) -> None:
        """Execution of the optimizer step.

        Parameters
        ----------
        i
            The iteration number.
        """
        pass

    def step(self, i : int, batch : Any, model_kwargs : Optional[Dict[str, Any]]) -> Tuple[torch.Tensor, Optional[str]]:
        """Execution of a single training step.

        Parameters
        ----------
        i
            This is the iteration number.
        batch
            The data of the current batch. The type is defined by the 
            `DataLoader` provided when the object was constructed. It must be a 
            dictionary or similar as it is unpacked to the model using 
            `**batch`.
        model_kwargs
            Additional arguments that need to be provided to the model during 
            the forward pass.

        Returns
        -------
        torch.Tensor
            The loss computed during the iteration.
        str, optional
            Any additional information to print before updating the progress 
            bar. This allows allows properly printing an update without 
            breaking, overwriting or duplicating the progress bar.

        """
        if model_kwargs is None:
            model_kwargs = {}
        batch = self.process_batch(i, batch)

        throttle_type = self.conf.throttle_type if self.conf is not None and self.conf.enable_throttling else None
        throttle_frequency = self.conf.throttle_frequency if self.conf is not None and self.conf.enable_throttling else None

        # (greta) throttling tests for unet3d 
        if throttle_type == "all_fixed":
            print(f"[GRETA LOG] Setting fixed frequency for all training steps")
            self.frequency_scheduler.schedule_throttling(frequency=throttle_frequency, workload_name=utils.TrainingComponents.ALL_FIXED.value) 

        self.stats.start_forward()
        if throttle_type == "forward":
            print(f"[GRETA LOG] Throttling during forward pass")
            self.frequency_scheduler.schedule_throttling(workload_name=utils.TrainingComponents.FORWARD.value)
        if throttle_type == "dym_best":
            print(f"[GRETA LOG] Throttling during forward pass - DYM BEST")
            self.frequency_scheduler.schedule_throttling(frequency=1335, workload_name=utils.TrainingComponents.FORWARD.value)
        loss = self.forward(i, batch, model_kwargs)
        if throttle_type == "forward" or throttle_type == "dym_best":
            self.frequency_scheduler.schedule_reset(workload_name=utils.TrainingComponents.FORWARD.value)
        self.stats.stop_forward()

        self.stats.start_backward()
        if throttle_type == "backward":
            print(f"[GRETA LOG] Throttling during backward pass")
            self.frequency_scheduler.schedule_throttling(workload_name=utils.TrainingComponents.BACKWARD.value)
            if throttle_type == "dym_best":
                self.frequency_scheduler.schedule_throttling(frequency=1305, workload_name=utils.TrainingComponents.BACKWARD.value)
        self.backward(i, loss)
        if throttle_type == "backward" or throttle_type == "dym_best":
            self.frequency_scheduler.schedule_reset(workload_name=utils.TrainingComponents.BACKWARD.value)
        self.stats.stop_backward()

        self.stats.start_optimizer_step()
        if throttle_type == "optimizer":
            print(f"[GRETA LOG] Throttling during optimizer step")
            self.frequency_scheduler.schedule_throttling(workload_name=utils.TrainingComponents.OPTIMIZER_STEP.value)
        if throttle_type == "dym_best":
            self.frequency_scheduler.schedule_throttling(frequency=1417, workload_name=utils.TrainingComponents.OPTIMIZER_STEP.value)
        self.optimizer_step(i)
        if throttle_type == "optimizer" or throttle_type == "dym_best":
            self.frequency_scheduler.schedule_reset(workload_name=utils.TrainingComponents.OPTIMIZER_STEP.value)
        self.stats.stop_optimizer_step()

        if throttle_type == "all_fixed":
            self.frequency_scheduler.schedule_reset(workload_name=utils.TrainingComponents.ALL_FIXED.value)
        
        return loss, None

    def train(self, model_kwargs : Optional[Dict[str, Any]]) -> None:
        """Training loop for the model.
        
        This will execute a training step on each batch provided by the 
        dataloader. The number of iterations is defined by the dataloader 
        provided when the object was constructed. 

        A progress bar is updated after every iteration with the iteration 
        number and the most recent loss.

        Training statistics will be logged after every iteration if the `stats` 
        attribute implements the method `log_step`. Additionally, more 
        statistics will be displayed at the end of training if the `stats` 
        attribute implements the method `log_stats`.

        Parameters
        ----------
        model_kwargs
            Additional arguments that need to be provided to the model during 
            the forward pass.

        Notes
        -----
            This does not support multi-epoch training. If you need training on 
            multiple epochs, you should implement a class that inherits 
            `Trainer` and overrides the `train` method.

        """
        progress_bar = tqdm.auto.tqdm(range(len(self.loader)), desc="loss: N/A")

        checkpoint_energy = RunningEnergy(self.device.index)

        self.stats.start_train()
        for i, batch in enumerate(self.loader):
            self.stats.start_step()
            loss, descr = self.step(i, batch, model_kwargs)
            self.stats.stop_step()

            if self.enable_checkpointing and self.should_save_checkpoint(i):
                self.stats.start_save_checkpoint()
                checkpoint_energy.start()
                self.frequency_scheduler.schedule_throttling(workload_name=utils.TrainingComponents.SAVE_CHECKPOINT.value)
                self.save_checkpoint(i)
                self.frequency_scheduler.schedule_reset(workload_name=utils.TrainingComponents.SAVE_CHECKPOINT.value)
                checkpoint_energy.stop()
                self.stats.stop_save_checkpoint()
                print(f"checkpoint energy consumption: {checkpoint_energy.get_last()}")

            # for every rank, log the loss
            self.stats.log_loss(loss, 0) #rank 0 for single GPU
            self.stats.log_step()

            if descr is not None:
                progress_bar.clear()
                print(descr)
            progress_bar.clear()
            
            progress_bar.set_description(f'loss: {loss : .4f}')
            progress_bar.update(1)

        self.stats.stop_train()
        progress_bar.close()
        self.stats.log_stats()
        self.frequency_scheduler.stop()
