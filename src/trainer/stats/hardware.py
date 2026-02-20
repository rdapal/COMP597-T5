"""
Hardware Monitoring TrainerStats

Collects per-phase:
- Timing (ms)
- GPU power draw (W) via NVML
- Energy consumed (J) = avg_power × duration
- GPU memory (MB) via PyTorch
- GPU utilization (%) via NVML
- GPU temperature (°C) via NVML
- CPU utilization (%) via psutil
Reference: Based on the interface defined in src/trainer/stats/base.py
Following patterns from src/trainer/stats/simple.py
"""

import logging
import os
import csv
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict, fields
from typing import List, Optional

import torch

import src.config as config
import src.trainer.stats.base as base

logger = logging.getLogger(__name__)

trainer_stats_name = "hardware"

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.debug("pynvml not available — GPU power/utilization will not be tracked")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available — CPU utilization will not be tracked")


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    """
    Factory function for the auto-discovery system.
    Called when --trainer_stats hardware is specified.
    """
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to hardware trainer stats. Using CUDA device 0")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_train_steps = kwargs.get("num_train_steps", 100)

    output_dir = "/home/2021/rdapal/COMP597/COMP597-starter-code/hardware_stats"
    run_id = None

    hw_conf = getattr(conf.trainer_stats_configs, 'hardware', None)
    if hw_conf:
        output_dir = getattr(hw_conf, 'output_dir', output_dir)
        run_id = getattr(hw_conf, 'run_id', run_id)

    return HardwareTrainerStats(
        device=device,
        output_dir=output_dir,
        run_id=run_id,
        num_train_steps=num_train_steps,
    )


@dataclass
class StepRecord:
    """All data recorded for a single training step."""
    step_num: int

    # ---- Timing (milliseconds) ----
    step_time_ms: float
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float

    # ---- GPU Memory via PyTorch (MB) ----
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    gpu_memory_peak_mb: float

    # ---- GPU Power via NVML (Watts) ----
    # Sampled at start and end of each phase
    gpu_power_forward_start_w: float
    gpu_power_forward_end_w: float
    gpu_power_backward_start_w: float
    gpu_power_backward_end_w: float
    gpu_power_optimizer_start_w: float
    gpu_power_optimizer_end_w: float

    # ---- Per-Phase Energy (Joules) ----
    # Energy = avg_power(W) × duration(s)
    energy_forward_j: float
    energy_backward_j: float
    energy_optimizer_j: float
    energy_step_j: float          # sum of phase energies

    # ---- GPU Temperature (*C) via NVML ----
    gpu_temperature_c: float

    # ---- Utilization (%) ----
    gpu_utilization: float        # NVML compute utilization
    gpu_memory_utilization: float # NVML memory bandwidth utilization
    cpu_utilization: float        # psutil

    # ---- Timestamp ----
    timestamp: str


class HardwareTrainerStats(base.TrainerStats):
    """
    TrainerStats with hardware monitoring including power & energy.

    Power measurement approach:
    - Sample GPU power (NVML nvmlDeviceGetPowerUsage) at start and end of each phase
    - Compute per-phase energy:
        E_phase = ((P_start + P_end) / 2) × duration_seconds
    - This should be accurate for phases >~50ms where power is relatively stable to avoid innacurate power data

    Parameters
    ----------
    device : torch.device
    The PyTorch device used for training

    output_dir : str
    Directory to save CSV and JSON output files

    run_id : str, optional
    Unique identifier for this run. Auto-generated if not provided.

    num_train_steps : int
    number of trianing steps for logging
    """

    def __init__(
        self,
        device: torch.device,
        output_dir: str = "./hardware_stats",
        run_id: Optional[str] = None,
        num_train_steps: int = 100,
    ):
        super().__init__()
        self.device = device
        self.output_dir = output_dir
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.num_train_steps = num_train_steps

        os.makedirs(output_dir, exist_ok=True)

        # Initialize NVML
        self.nvml_handle = None
        self.gpu_name = "unknown"
        self.gpu_power_limit_w = 0.0
        if PYNVML_AVAILABLE and device.type == 'cuda':
            try:
                pynvml.nvmlInit()
                device_index = device.index if device.index is not None else 0
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.nvml_handle)
                if isinstance(self.gpu_name, bytes):
                    self.gpu_name = self.gpu_name.decode('utf-8')
                # Power limit in watts (NVML returns milliwatts)
                try:
                    self.gpu_power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(
                        self.nvml_handle
                    ) / 1000.0
                except pynvml.NVMLError:
                    self.gpu_power_limit_w = 0.0
                    logger.warning("Could not query GPU power management limit")

                logger.info(
                    f"NVML initialized: {self.gpu_name}, "
                    f"power limit={self.gpu_power_limit_w:.0f}W"
                )
            except Exception as e:
                logger.warning(f"Could not initialize NVML: {e}")

        # Storage
        self.step_records: List[StepRecord] = []
        self.current_step = 0

        # Timing accumulators (nanoseconds)
        self._step_start_ns = 0
        self._forward_start_ns = 0
        self._backward_start_ns = 0
        self._optimizer_start_ns = 0
        self._checkpoint_start_ns = 0

        self._forward_time_ns = 0
        self._backward_time_ns = 0
        self._optimizer_time_ns = 0

        # Power accumulators (watts)
        self._power_forward_start_w = 0.0
        self._power_forward_end_w = 0.0
        self._power_backward_start_w = 0.0
        self._power_backward_end_w = 0.0
        self._power_optimizer_start_w = 0.0
        self._power_optimizer_end_w = 0.0

        logger.info(
            f"HardwareTrainerStats initialized: "
            f"output_dir={output_dir}, run_id={self.run_id}, "
            f"gpu={self.gpu_name}, "
            f"nvml={'yes' if self.nvml_handle else 'no'}, "
            f"psutil={'yes' if PSUTIL_AVAILABLE else 'no'}"
        )

    # ==================== Helpers ====================

    def _sync_cuda(self) -> None:
        """Synchronize CUDA for accurate timing"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _time_ns(self) -> int:
        return time.perf_counter_ns()

    def _get_gpu_memory(self) -> tuple:
        """Returns (allocated_bytes, reserved_bytes, peak_bytes) via PyTorch"""
        if self.device.type != 'cuda':
            return (0, 0, 0)
        try:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            return (allocated, reserved, peak)
        except Exception:
            return (0, 0, 0)

    def _get_gpu_power_w(self) -> float:
        """Get instantaneous GPU power draw in Watts via NVML"""
        if self.nvml_handle:
            try:
                # nvmlDeviceGetPowerUsage returns milliwatts
                return pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
            except Exception:
                pass
        return 0.0

    def _get_gpu_temperature(self) -> float:
        """Get GPU temperature in Celsius via NVML"""
        if self.nvml_handle:
            try:
                return float(pynvml.nvmlDeviceGetTemperature(
                    self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                ))
            except Exception:
                pass
        return 0.0

    def _get_gpu_utilization(self) -> tuple:
        """
        Returns (compute_util%, memory_bandwidth_util%) via NVML
        
        Note: 'memory' here is memory BANDWIDTH utilization,
        NOT memory capacity utilization. This is the proportion of time
        the memory subsystem was actively reading/writing to purposely avoid capturing unecessary data.
        """
        if self.nvml_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                return (float(util.gpu), float(util.memory))
            except Exception:
                pass
        return (0.0, 0.0)

    def _get_cpu_utilization(self) -> float:
        if PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_percent(interval=None)
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _compute_energy_j(power_start_w: float, power_end_w: float,
                          duration_ns: int) -> float:
        """
        Compute energy in Joules using trapezoid approximation
        E = ((P_start + P_end) / 2) × duration_seconds
        """
        avg_power_w = (power_start_w + power_end_w) / 2.0
        duration_s = duration_ns / 1_000_000_000.0
        return avg_power_w * duration_s

    # ==================== TrainerStats Interface ====================

    def start_train(self) -> None:
        logger.info(
            f"Starting training with hardware+energy monitoring "
            f"(run_id={self.run_id}, gpu={self.gpu_name})"
        )
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)  # Initialize

    def stop_train(self) -> None:
        logger.info("Training complete. Saving hardware+energy stats...")
        self._save_results()
        if self.nvml_handle:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def start_step(self) -> None:
        self._sync_cuda()
        self._step_start_ns = self._time_ns()

        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        # Reset accumulators
        self._forward_time_ns = 0
        self._backward_time_ns = 0
        self._optimizer_time_ns = 0
        self._power_forward_start_w = 0.0
        self._power_forward_end_w = 0.0
        self._power_backward_start_w = 0.0
        self._power_backward_end_w = 0.0
        self._power_optimizer_start_w = 0.0
        self._power_optimizer_end_w = 0.0

    def stop_step(self) -> None:
        self._sync_cuda()
        step_time_ns = self._time_ns() - self._step_start_ns

        # Collect hardware metrics
        gpu_mem = self._get_gpu_memory()
        gpu_util, mem_util = self._get_gpu_utilization()
        cpu_util = self._get_cpu_utilization()
        gpu_temp = self._get_gpu_temperature()

        # Compute per-phase energy (Joules)
        energy_forward = self._compute_energy_j(
            self._power_forward_start_w, self._power_forward_end_w,
            self._forward_time_ns
        )
        energy_backward = self._compute_energy_j(
            self._power_backward_start_w, self._power_backward_end_w,
            self._backward_time_ns
        )
        energy_optimizer = self._compute_energy_j(
            self._power_optimizer_start_w, self._power_optimizer_end_w,
            self._optimizer_time_ns
        )
        energy_step = energy_forward + energy_backward + energy_optimizer

        record = StepRecord(
            step_num=self.current_step,
            step_time_ms=step_time_ns / 1_000_000,
            forward_time_ms=self._forward_time_ns / 1_000_000,
            backward_time_ms=self._backward_time_ns / 1_000_000,
            optimizer_time_ms=self._optimizer_time_ns / 1_000_000,
            gpu_memory_allocated_mb=gpu_mem[0] / (1024 * 1024),
            gpu_memory_reserved_mb=gpu_mem[1] / (1024 * 1024),
            gpu_memory_peak_mb=gpu_mem[2] / (1024 * 1024),
            gpu_power_forward_start_w=self._power_forward_start_w,
            gpu_power_forward_end_w=self._power_forward_end_w,
            gpu_power_backward_start_w=self._power_backward_start_w,
            gpu_power_backward_end_w=self._power_backward_end_w,
            gpu_power_optimizer_start_w=self._power_optimizer_start_w,
            gpu_power_optimizer_end_w=self._power_optimizer_end_w,
            energy_forward_j=energy_forward,
            energy_backward_j=energy_backward,
            energy_optimizer_j=energy_optimizer,
            energy_step_j=energy_step,
            gpu_temperature_c=gpu_temp,
            gpu_utilization=gpu_util,
            gpu_memory_utilization=mem_util,
            cpu_utilization=cpu_util,
            timestamp=datetime.now().isoformat(),
        )
        self.step_records.append(record)
        self.current_step += 1

    def start_forward(self) -> None:
        self._sync_cuda()
        self._power_forward_start_w = self._get_gpu_power_w()
        self._forward_start_ns = self._time_ns()

    def stop_forward(self) -> None:
        self._sync_cuda()
        self._forward_time_ns = self._time_ns() - self._forward_start_ns
        self._power_forward_end_w = self._get_gpu_power_w()

    def start_backward(self) -> None:
        self._sync_cuda()
        self._power_backward_start_w = self._get_gpu_power_w()
        self._backward_start_ns = self._time_ns()

    def stop_backward(self) -> None:
        self._sync_cuda()
        self._backward_time_ns = self._time_ns() - self._backward_start_ns
        self._power_backward_end_w = self._get_gpu_power_w()

    def start_optimizer_step(self) -> None:
        self._sync_cuda()
        self._power_optimizer_start_w = self._get_gpu_power_w()
        self._optimizer_start_ns = self._time_ns()

    def stop_optimizer_step(self) -> None:
        self._sync_cuda()
        self._optimizer_time_ns = self._time_ns() - self._optimizer_start_ns
        self._power_optimizer_end_w = self._get_gpu_power_w()

    def start_save_checkpoint(self) -> None:
        self._sync_cuda()
        self._checkpoint_start_ns = self._time_ns()

    def stop_save_checkpoint(self) -> None:
        self._sync_cuda()
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def log_step(self) -> None:
        """Log the previous step's measurements to console"""
        if not self.step_records:
            return

        rec = self.step_records[-1]
        avg_power = (
            (rec.gpu_power_forward_start_w + rec.gpu_power_forward_end_w +
             rec.gpu_power_backward_start_w + rec.gpu_power_backward_end_w +
             rec.gpu_power_optimizer_start_w + rec.gpu_power_optimizer_end_w) / 6.0
        )

        print(
            f"[Step {rec.step_num:4d}] "
            f"total={rec.step_time_ms:.1f}ms | "
            f"fwd={rec.forward_time_ms:.1f}ms | "
            f"bwd={rec.backward_time_ms:.1f}ms | "
            f"opt={rec.optimizer_time_ms:.1f}ms | "
            f"power={avg_power:.1f}W | "
            f"energy={rec.energy_step_j*1000:.2f}mJ | "
            f"temp={rec.gpu_temperature_c:.0f}°C | "
            f"mem={rec.gpu_memory_allocated_mb:.0f}MB | "
            f"util={rec.gpu_utilization:.0f}%"
        )

    def log_stats(self) -> None:
        """Log summary statistics at end of training."""
        if len(self.step_records) < 2:
            logger.warning("Not enough steps for statistics")
            return

        # Exclude warmup (step 0) for steady-state stats
        steady = self.step_records[1:]

        avg_step = sum(r.step_time_ms for r in steady) / len(steady)
        avg_fwd = sum(r.forward_time_ms for r in steady) / len(steady)
        avg_bwd = sum(r.backward_time_ms for r in steady) / len(steady)
        avg_opt = sum(r.optimizer_time_ms for r in steady) / len(steady)
        avg_energy = sum(r.energy_step_j for r in steady) / len(steady)
        total_energy = sum(r.energy_step_j for r in self.step_records)
        avg_temp = sum(r.gpu_temperature_c for r in steady) / len(steady)

        # Average power per phase
        avg_power_fwd = sum(
            (r.gpu_power_forward_start_w + r.gpu_power_forward_end_w) / 2
            for r in steady
        ) / len(steady)
        avg_power_bwd = sum(
            (r.gpu_power_backward_start_w + r.gpu_power_backward_end_w) / 2
            for r in steady
        ) / len(steady)
        avg_power_opt = sum(
            (r.gpu_power_optimizer_start_w + r.gpu_power_optimizer_end_w) / 2
            for r in steady
        ) / len(steady)

        print("\n" + "=" * 70)
        print("HARDWARE + ENERGY SUMMARY (Steady State, excluding warmup)")
        print("=" * 70)
        print(f"GPU: {self.gpu_name}")
        print(f"Power Limit: {self.gpu_power_limit_w:.0f}W")
        print(f"Steps: {len(self.step_records)} total, {len(steady)} steady-state")
        print()
        print(f"--- Timing ---")
        print(f"  Avg step:      {avg_step:.2f} ms")
        print(f"  Avg forward:   {avg_fwd:.2f} ms ({avg_fwd/avg_step*100:.1f}%)")
        print(f"  Avg backward:  {avg_bwd:.2f} ms ({avg_bwd/avg_step*100:.1f}%)")
        print(f"  Avg optimizer: {avg_opt:.2f} ms ({avg_opt/avg_step*100:.1f}%)")
        print()
        print(f"--- Power (steady state average) ---")
        print(f"  Forward:   {avg_power_fwd:.1f} W")
        print(f"  Backward:  {avg_power_bwd:.1f} W")
        print(f"  Optimizer: {avg_power_opt:.1f} W")
        print()
        print(f"--- Energy ---")
        print(f"  Avg per step:  {avg_energy*1000:.2f} mJ")
        print(f"  Total training: {total_energy:.3f} J")
        print(f"  Total training: {total_energy/3600*1000:.4f} mWh")
        print()
        print(f"--- Thermal ---")
        print(f"  Avg GPU temp:  {avg_temp:.1f} °C")
        print("=" * 70)

    # ==================== Output ====================

    def _save_results(self) -> None:
        """Save all step records to CSV and metadata to JSON"""
        if not self.step_records:
            logger.warning("No step records to save")
            return

        # --- CSV ---
        csv_path = os.path.join(
            self.output_dir, f"hardware_stats_{self.run_id}.csv"
        )
        fieldnames = [f.name for f in fields(StepRecord)]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.step_records:
                writer.writerow(asdict(record))

        logger.info(f"Saved {len(self.step_records)} step records to {csv_path}")

        # --- Metadata JSON ---
        steady = self.step_records[1:] if len(self.step_records) > 1 else self.step_records
        total_energy = sum(r.energy_step_j for r in self.step_records)

        metadata = {
            "run_id": self.run_id,
            "gpu_name": self.gpu_name,
            "gpu_power_limit_w": self.gpu_power_limit_w,
            "device": str(self.device),
            "num_steps": len(self.step_records),
            "num_steady_state_steps": len(steady),
            "nvml_available": self.nvml_handle is not None,
            "psutil_available": PSUTIL_AVAILABLE,
            "total_energy_j": total_energy,
            "avg_step_energy_mj": (
                sum(r.energy_step_j for r in steady) / len(steady) * 1000
                if steady else 0
            ),
            "avg_step_time_ms": (
                sum(r.step_time_ms for r in steady) / len(steady)
                if steady else 0
            ),
            "avg_gpu_power_w": (
                sum(
                    (r.gpu_power_forward_start_w + r.gpu_power_forward_end_w +
                     r.gpu_power_backward_start_w + r.gpu_power_backward_end_w +
                     r.gpu_power_optimizer_start_w + r.gpu_power_optimizer_end_w) / 6.0
                    for r in steady
                ) / len(steady) if steady else 0
            ),
            "timestamp_start": self.step_records[0].timestamp,
            "timestamp_end": self.step_records[-1].timestamp,
        }

        json_path = os.path.join(
            self.output_dir, f"metadata_{self.run_id}.json"
        )
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {json_path}")

        # Print summary
        self.log_stats()
