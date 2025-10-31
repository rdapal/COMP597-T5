from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import ctypes
import enum
import multiprocessing
import os
import queue
import src.hardware_management.gpu as gpu
import time
import torch

class _Task:
    device : torch.device
    event : torch.cuda.Event

    def __init__(self, device : torch.device, notes : str = "") -> None:
        self.device = device
        self.event = torch.cuda.Event(enable_timing=False, interprocess=True)
        self.notes = notes
        self.ts = 0

    def __getstate__(self):
        return {"device": self.device.__str__(),
                "event": self.event.ipc_handle(),
                "notes": self.notes,
                "ts": time.perf_counter_ns()}

    def __setstate__(self, state):
        self.device = torch.device(state["device"])
        self.event = torch.cuda.Event.from_ipc_handle(self.device, state["event"])
        self.notes = state["notes"]
        self.ts = state["ts"]

    def __str__(self) -> str:
        return (
            f"Task:\n"
            f"  type: {self.__class__.__name__}\n"
            f"  device: {self.device}\n"
            f"  notes: {self.notes}"
        )

    def record(self) -> None:
        self.event.record(torch.cuda.current_stream(device=self.device))

    def record_with_stream(self, stream) -> None:
        assert stream.device == self.device
        self.event.record(stream)

    @abstractmethod
    def _execute(self, gpu_device : gpu.GPU):
        pass

    def execute(self, gpu_device : gpu.GPU):
        self.event.synchronize()
        s = time.perf_counter_ns()
        self._execute(gpu_device)
        e = time.perf_counter_ns()
        print(f"task execution {os.environ.get('RANK', 0)} {(e - s) / 1e6}")

class ThrottleTask(_Task):

    def __init__(self, device : torch.device, frequency : int, notes : str = "") -> None:
        super().__init__(device, notes)
        self.frequency = frequency

    def __getstate__(self):
        state = super().__getstate__()
        state["frequency"] = self.frequency
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.frequency = state["frequency"]

    def __str__(self) -> str:
        return (
            f"{super().__str__()}\n"
            f"  frequency: {self.frequency}"
        )

    def _execute(self, gpu_device : gpu.GPU):
        gpu_device.set_gpu_locked_clocks(0, self.frequency)

class ResetTask(_Task):

    def __init__(self, device : torch.device, notes : str = "") -> None:
        super().__init__(device, notes)

    def _execute(self, gpu_device : gpu.GPU):
        gpu_device.reset_gpu_locked_clocks()

class SchedulerConfig:
    device : torch.device
    task_queue : multiprocessing.Queue
    termination_signal : multiprocessing.Value
    process_ready : multiprocessing.Event
    queue_timeout_seconds : float

    def __init__(self, device : torch.device, queue_timeout_seconds : float = 0.5) -> None:
        self.device = device
        self.queue_timeout_seconds = queue_timeout_seconds
        self.ctx = multiprocessing.get_context("spawn")
        self.task_queue = self.ctx.Queue()
        self.termination_signal = self.ctx.Value(ctypes.c_bool)
        self.process_ready = self.ctx.Event()

class SchedulerProcess:
    gpu_device : gpu.GPU
    task_queue : multiprocessing.Queue
    termination_signal : multiprocessing.Value
    queue_timeout_seconds : float

    def __init__(self, gpu_device : gpu.GPU, task_queue : multiprocessing.Queue, ready : multiprocessing.Event, termination_signal : multiprocessing.Value, queue_timeout_seconds : float = 0.5) -> None:
        self.gpu_device = gpu_device
        self.task_queue = task_queue
        self.ready = ready
        self.termination_signal = termination_signal
        self.queue_timeout_seconds = queue_timeout_seconds

    @classmethod
    def _start(cls, device_index : int, task_queue : multiprocessing.Queue, ready : multiprocessing.Event, termination_signal : multiprocessing.Value, queue_timeout_seconds : float = 0.5) -> None:
        device_index = torch.cuda.current_device() # Ensure the correct device is set in this process.
        gpu_device = gpu.GPU(device_index)
        p = cls(gpu_device, task_queue, ready, termination_signal, queue_timeout_seconds)
        p.run()

    @classmethod
    def create_and_start(cls, config : SchedulerConfig) -> multiprocessing.Process:
        p = config.ctx.Process(target=cls._start, args=(config.device.index, config.task_queue, config.process_ready, config.termination_signal, config.queue_timeout_seconds))
        p.start()
        config.process_ready.wait()
        return p

    def _get_next_task(self) -> Optional[_Task]:
        try:
            return self.task_queue.get(True, self.queue_timeout_seconds)
        except queue.Empty:
            return None

    def _execute_task(self, task : _Task) -> None:
        task.execute(self.gpu_device)

    def _loop(self):
        self.ready.set()
        # prev = 0
        while not self.termination_signal.value:
            task = self._get_next_task()
            if task is not None:
                # t = time.perf_counter_ns()
                self._execute_task(task)
                # now = time.perf_counter_ns()
                # print(f"delay {(t - task.ts) / 1000000} {self.gpu_device.index} {self.gpu_device.cuda_index} {task.device}")
                # if task.__class__ == ResetTask:
                #     print(f"{task.__class__.__name__} {(now - prev) / 1000000}")
                # prev = now
                # print(task)
        time.sleep(0.5)
        self.gpu_device.reset_gpu_locked_clocks()

    def run(self):
        self._loop()

class Scheduler:

    @abstractmethod
    def register_workload(self, workload_name : str, num_warmup_steps : int = 5, num_measurement_steps : int = 5, num_communication_warmup_steps : int = 5, time_threshold : float = 0.05) -> None:
        pass
    
    @abstractmethod
    def schedule_throttling(self, frequency : int = 0, workload_name : Optional[str] = None) -> None:
        pass

    @abstractmethod
    def schedule_reset(self, workload_name : Optional[str] = None) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

class NOOPScheduler(Scheduler):

    def __init__(self) -> None:
        super().__init__()

    def register_workload(self, workload_name : str, num_warmup_steps : int = 5, num_measurement_steps : int = 5, num_communication_warmup_steps : int = 5, time_threshold : float = 0.05) -> None:
        pass
    
    def schedule_throttling(self, frequency : int = 0, workload_name : Optional[str] = None) -> None:
        pass

    def schedule_reset(self, workload_name : Optional[str] = None) -> None:
        pass

    def stop(self) -> None:
        pass

class FrequencyScheduler(Scheduler):
    
    def __init__(self, config : SchedulerConfig, process : Optional[multiprocessing.Process] = None) -> None:
        super().__init__()
        self.scheduler_process_queue = config.task_queue
        self.device = config.device
        self.termination_signal = config.termination_signal
        if process is None:
            self.process = SchedulerProcess.create_and_start(config)
        else:
            self.process = process

    def stop(self) -> None:
        # Let the scheduler process know to terminate.
        self.termination_signal.value = True
        self.process.join()

    def __del__(self):
        self.stop()

    def register_workload(self, workload_name : str, num_warmup_steps : int = 5, num_measurement_steps : int = 5, num_communication_warmup_steps : int = 5, time_threshold : float = 0.05) -> None:
        pass

    def _schedule_task(self, task : _Task) -> None:
        task.record()
        self.scheduler_process_queue.put(task)

    def schedule_throttling(self, frequency : int = 0, workload_name : Optional[str] = None) -> None:
        assert frequency is not None
        if frequency <= 0:
            print(f"received non-positive frequency {frequency}, ignoring")
            return
        task = ThrottleTask(self.device, frequency)
        self._schedule_task(task)

    def schedule_reset(self, workload_name : Optional[str] = None) -> None:
        task = ResetTask(self.device)
        self._schedule_task(task)

class RunningAverage:
    """Implements a running average.

    Attributes
    ----------
    average : float
        Current average.
    n : int
        Number of samples.
    """

    def __init__(self) -> None:
        self.average = 0.0
        self.n = 0

    def update(self, value : float) -> None:
        """Update the average with a new value.

        This will perform an inplace update on the average.

        Parameters
        ----------
        value
            The value to include in the running average.

        """
        self.average = (self.average * self.n + value) / (self.n + 1)
        self.n += 1

    def get(self) -> float:
        """The current average.
        """
        return self.average

    def get_num_samples(self) -> int:
        """The number of samples included in the average.
        """
        return self.n

    def reset(self) -> None:
        """Reset the average to its state when it was instantiated
        """
        self.average = 0.0
        self.n = 0

class AveragedCudaTimingMeasurement:
    _device : torch.device
    _num_measurements_required : int
    _average : RunningAverage
    _start_event : torch.cuda.Event
    _start_event_recorded : bool
    _end_event : torch.cuda.Event
    _end_event_recorded : bool

    def __init__(self, device : torch.device) -> None:
        self._device = device
        self._average = RunningAverage()
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._start_event_recorded = False
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._end_event_recorded = False

    def measurement_done(self) -> bool:
        return self._start_event_recorded and self._end_event_recorded

    def record(self) -> None:
        assert self.measurement_done()
        self._end_event.synchronize()
        self._average.update(self._start_event.elapsed_time(self._end_event))
        self._start_event_recorded = False
        self._end_event_recorded = False

    def start(self) -> None:
        if self.measurement_done():
            self.record()
        self._start_event.record(torch.cuda.current_stream(self._device))
        self._start_event_recorded = True

    def stop(self) -> None:
        self._end_event.record(torch.cuda.current_stream(self._device))
        self._end_event_recorded = True

    def get(self) -> float:
        return self._average.get()

    def get_num_measurements(self) -> int:
        return self._average.get_num_samples()

    def reset(self) -> None:
        self._start_event_recorded = False
        self._end_event_recorded = False
        self._average.reset()

class RangeSearch(ABC):
    
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def done(self) -> bool:
        pass

    @abstractmethod
    def move_left(self) -> None:
        pass

    @abstractmethod
    def move_right(self) -> None:
        pass

    @abstractmethod
    def get_min(self) -> int:
        pass

    @abstractmethod
    def get_max(self) -> int:
        pass

    @abstractmethod
    def get_index(self) -> int:
        pass

class PseudoBinarySearch(RangeSearch):
    _original_min : int
    _original_max : int
    _min : int
    _max : int
    _index : int
    _reduction_coefficient : float

    def __init__(self, min : int, max : int, reduction_coefficient : float = 0.5) -> None:
        assert 0.0 < reduction_coefficient and reduction_coefficient < 1.0
        super().__init__()
        self._original_min = min
        self._original_max = max
        self._reduction_coefficient = reduction_coefficient
        self._min = -1
        self._max = -1
        self._index = -1
        self.reset()

    def reset(self) -> None:
        self._min = self._original_min
        self._max = self._original_max
        self._compute_index()

    def done(self) -> bool:
        if abs(self._max - self._min) <= 2:
            return True
        return False

    def _compute_index(self) -> None:
        self._index = int((self._min + self._max) / 2)

    def _update(self, min, max) -> None:
        assert min < max
        self._min = min
        self._max = max
        self._compute_index()

    def _compute_offset(self) -> int:
        delta = self._max - self._min
        offset = int(delta * self._reduction_coefficient)
        if offset <= 0:
            return 1
        return offset

    def move_left(self) -> None:
        if self.done():
            return
        offset = self._compute_offset()
        self._update(self._min, self._min + offset)

    def move_right(self) -> None:
        if self.done():
            return
        offset = self._compute_offset()
        self._update(self._max - offset, self._max)

    def get_min(self) -> int:
        return self._min

    def get_max(self) -> int:
        return self._max

    def get_index(self) -> int:
        return self._index

class PointUpdateSearch(RangeSearch):
    _original_index : int
    _min : int
    _max : int
    _index : int
    
    def __init__(self, min : int, max : int, index : int) -> None:
        assert 0 <= min and min <= index and index < max
        super().__init__()
        self._original_index = index
        self._min = min
        self._max = max
        self.reset()

    def reset(self) -> None:
        self._index = self._original_index

    def done(self) -> bool:
        return False
    
    def move_left(self) -> None:
        if self._min <= self._index - 1:
            self._index -= 1

    def move_right(self) -> None:
        if self._index + 1 < self._max:
            self._index += 1

    def get_min(self) -> int:
        return self._min

    def get_max(self) -> int:
        return self._max

    def get_index(self) -> int:
        return self._index

class _LearnedWorkloadModes(enum.Enum):
    WARMUP = enum.auto()
    INITIAL_MEASUREMENT = enum.auto()
    COMMUNICATION_WARMUP = enum.auto()
    SEARCH = enum.auto()
    TWEAKING = enum.auto()

# Alright, this is a little chunky, here's a quick breakdown:
# - Waits a certain number of steps before doing measurements.
# - Warmups for a certain number of steps. During that, it does measurements to 
#   have an average time the workload takes.
# - Then it starts doing binary search on the supported GPU frequencies. It 
#   decreases the frequency if within the time threshold. A time threshold of 
#   0.05 means it tolerates an increase in computation time of 5%.
# - Once binary search is done, it keeps checking the performance. It does 
#   point updates in either direction depending on whether the threshold is 
#   respected (move to the next supported frequency in either direction).
class LearnedWorkload:

    def __init__(self, name : str, device : torch.device, num_warmup_steps : int = 5, num_measurement_steps : int = 5, num_communication_warmup_steps : int = 5, time_threshold : float = 0.05) -> None:
        self.name = name
        self.device = device
        self.gpu_device = gpu.GPU(self.device.index)

        self._mode = _LearnedWorkloadModes.WARMUP

        self._num_warmup_steps = num_warmup_steps
        self._num_measurement_steps = num_measurement_steps
        self._num_communication_warmup_steps = num_communication_warmup_steps

        self._reference_timing = -1.0
        self._time_threshold = time_threshold

        self._measurement = AveragedCudaTimingMeasurement(self.device)

        self._supported_frequencies = self.gpu_device.get_currently_supported_graphics_clock()
        self._supported_frequencies.sort()
        self._current_frequency = self.gpu_device.get_max_currently_supported_graphics_clock()

        self._search = PseudoBinarySearch(0, len(self._supported_frequencies))

    def _update_warmup(self) -> None:
        if self._measurement.get_num_measurements() >= self._num_warmup_steps:
            self._mode = _LearnedWorkloadModes.INITIAL_MEASUREMENT
            self._measurement.reset()

    def _update_initial_measurement(self) -> None:
        if self._measurement.get_num_measurements() >= self._num_measurement_steps:
            self._mode = _LearnedWorkloadModes.COMMUNICATION_WARMUP
            self._reference_timing = self._measurement.get()
            self._measurement.reset()

    def _update_communication_warmup(self) -> None:
        if self._measurement.get_num_measurements() >= self._num_communication_warmup_steps:
            self._mode = _LearnedWorkloadModes.SEARCH
            self._search_update_frequency()
            self._measurement.reset()

    def _update_search(self) -> None:
        if self._search.done():
            self._mode = _LearnedWorkloadModes.TWEAKING
            self._search = PointUpdateSearch(0, len(self._supported_frequencies), self._search.get_index())
            self._measurement.reset()
        elif self._measurement.get_num_measurements() >= self._num_measurement_steps:
            self._search_update()

    def _update_tweaking(self) -> None:
        if self._measurement.get_num_measurements() >= self._num_measurement_steps:
            self._search_update()

    def _within_threshold(self, cur_timing : float) -> bool:
        diff = abs(cur_timing - self._reference_timing)
        return (diff / self._reference_timing) < self._time_threshold

    def _search_update_frequency(self) -> None:
        self._current_frequency = self._supported_frequencies[self._search.get_index()]

    def _search_update(self) -> None:
        cur_timing = self._measurement.get()
        go_left = True
        if cur_timing < self._reference_timing or self._within_threshold(cur_timing):
            self._search.move_left()
        else:
            self._search.move_right()
            go_left = False
        print(f"\n Search update: {self.name} {self._reference_timing} {cur_timing} {go_left} {self._search.get_index()} {self._search.get_min()} {self._search.get_max()} {self._current_frequency}")
        self._search_update_frequency()
        self._measurement.reset()

    def start(self) -> None:
        if self._measurement.measurement_done():
            self._measurement.record()

        if self._mode == _LearnedWorkloadModes.WARMUP:
            self._update_warmup()
        elif self._mode == _LearnedWorkloadModes.INITIAL_MEASUREMENT:
            self._update_initial_measurement()
        elif self._mode == _LearnedWorkloadModes.COMMUNICATION_WARMUP:
            self._update_communication_warmup()
        elif self._mode == _LearnedWorkloadModes.SEARCH:
            self._update_search()
        elif self._mode == _LearnedWorkloadModes.TWEAKING:
            self._update_tweaking()
        else:
            raise Exception(f"Unknown learned workload mode : {self._mode}")

        self._measurement.start()
        
    def stop(self) -> None:
        self._measurement.stop()
        
    def is_ready(self) -> bool:
        return self._mode == _LearnedWorkloadModes.COMMUNICATION_WARMUP or self._mode == _LearnedWorkloadModes.SEARCH or self._mode == _LearnedWorkloadModes.TWEAKING

    def get_frequency(self) -> int:
        return self._current_frequency

class LearnedFrequencyScheduler(Scheduler):

    def __init__(self, config : SchedulerConfig, process : Optional[multiprocessing.Process] = None) -> None:
        super().__init__()
        self.scheduler_process_queue = config.task_queue
        self.device = config.device
        self.termination_signal = config.termination_signal
        if process is None:
            self.process = SchedulerProcess.create_and_start(config)
        else:
            self.process = process
        self.workloads : Dict[str,LearnedWorkload] = {}
        self.scheduled_frequencies : List[int] = []

    def stop(self):
        # Let the scheduler process know to terminate.
        self.termination_signal.value = True
        print("potato")
        self.process.join()

    def __del__(self):
        self.stop()

    def register_workload(self, workload_name : str, num_warmup_steps : int = 5, num_measurement_steps : int = 5, num_communication_warmup_steps : int = 5, time_threshold : float = 0.05) -> None:
        if workload_name in self.workloads:
            raise Exception(f"Workload {workload_name} is already registered, cannot register again.")
        self.workloads[workload_name] = LearnedWorkload(
                name=workload_name, 
                device=self.device, 
                num_warmup_steps=num_warmup_steps, 
                num_measurement_steps=num_measurement_steps, 
                num_communication_warmup_steps=num_communication_warmup_steps, 
                time_threshold=time_threshold)

    def _create_workload_if_not_exists(self, workload_name : str) -> None:
        if workload_name not in self.workloads:
            # self.workloads[workload_name] = LearnedWorkload(workload_name, self.device)
            self.register_workload(workload_name)

    def _get_workload(self, workload_name : str) -> LearnedWorkload:
        self._create_workload_if_not_exists(workload_name)
        return self.workloads[workload_name]

    def _schedule_task(self, task : _Task) -> None:
        task.record()
        self.scheduler_process_queue.put(task)

    def schedule_throttling(self, frequency: int = 0, workload_name: Optional[str] = None) -> None:
        if workload_name is None:
            print("WARN: received no workload_name, ignoring")
            return
        elif frequency > 0:
            # TODO(Olivier): Need to add frequency to scheduled_frequencies
            print(f"WARN: overriding workload {workload_name}'s frequency with {frequency}")
            task = ThrottleTask(self.device, frequency)
            self._schedule_task(task)
            return
        workload = self._get_workload(workload_name)
        workload.start()
        if workload.is_ready():
            task = ThrottleTask(self.device, workload.get_frequency(), notes=workload_name)
            self._schedule_task(task)
            # print(f"{workload_name}: {workload.get_frequency()}")
            self.scheduled_frequencies.append(workload.get_frequency())

    def schedule_reset(self, workload_name: Optional[str] = None) -> None:
        if workload_name is None:
            print("WARN: received no workload_name, ignoring")
            return
        workload = self._get_workload(workload_name)
        workload.stop()
        if workload.is_ready():
            if len(self.scheduled_frequencies) > 0:
                self.scheduled_frequencies.pop()
            if len(self.scheduled_frequencies) > 0:
                task = ThrottleTask(self.device, self.scheduled_frequencies[-1], notes=workload_name)
                self._schedule_task(task)
            else:
                task = ResetTask(self.device, notes=workload_name)
                self._schedule_task(task)

