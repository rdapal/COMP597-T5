from typing import Any, List
import torch
import torch.nn as nn
import src.hardware_management.functions as functions
import src.hardware_management.scheduler as scheduler

class ThrottledModule(nn.Module):
    _class_number_instances = 0

    def __init__(self, child : nn.Module, frequency_scheduler : scheduler.Scheduler, num_warmup_steps : int = 5, num_measurement_steps : int = 5, num_communication_warmup_steps : int = 5, time_threshold : float = 0.05):
        super().__init__()
        id = self.__class__._new_id()
        self._forward_workload_name = self.__class__.__name__ + id.__str__() + "Forward"
        self._backward_workload_name = self.__class__.__name__ + id.__str__() + "Backward"
        self.frequency_scheduler = frequency_scheduler
        self.child = child
        self.frequency_scheduler.register_workload(self._forward_workload_name, num_warmup_steps, num_measurement_steps, num_communication_warmup_steps, time_threshold)
        self.frequency_scheduler.register_workload(self._backward_workload_name, num_warmup_steps, num_measurement_steps, num_communication_warmup_steps, time_threshold)

    @classmethod
    def _new_id(cls) -> int:
        id = cls._class_number_instances
        cls._class_number_instances += 1
        return id

    def _extract_tensors(self, *args, **kwargs) -> List[torch.Tensor]:
        tensors : List[torch.Tensor] = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensors.append(arg)
        for kwarg in kwargs.values():
            if isinstance(kwarg, torch.Tensor):
                tensors.append(kwarg)
        return tensors

    # TODO(Olivier) The backward pass will only be called if the output of `apply` is used.
    def forward(self, *args, **kwargs) -> Any:
        functions.ThrottledWorkloadComputation.apply(self.frequency_scheduler, self._forward_workload_name, self._backward_workload_name, *self._extract_tensors(*args))
        out = self.child(*args, **kwargs)
        if isinstance(out, tuple):
            functions.UnrestrictedWorkloadComputation.apply(self.frequency_scheduler, self._forward_workload_name, self._backward_workload_name, *self._extract_tensors(*out))
        else:
            functions.UnrestrictedWorkloadComputation.apply(self.frequency_scheduler, self._forward_workload_name, self._backward_workload_name, out)
        return out
