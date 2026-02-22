"""
Hardware Monitoring TrainerStats v.2

Collects per-phase (data_transfer, forward, backward, optimizer):
  - Timing (ms) via time.perf_counter_ns() with torch.cuda.synchronize()
  - GPU power (W) via NVML nvmlDeviceGetPowerUsage at phase start/end
  - Energy (J) via trapezoidal approx.

Collects per-step:
  - GPU memory (MB) via torch.cuda (allocated, reserved, peak)
  - GPU compute utilization (%) via NVML
  - GPU temperature (°C) via NVML
  - CPU utilization (%) via psutil
  - Overhead time/energy (step total minus sum of 4 phases)
  - CO2 emissions (mg) = energy_kWh × carbon_intensity

Known Limitations
-----------------
1. NVML power refresh is ~50-100ms on Ada Lovelace GPUs. Phases shorter
   than this (data_transfer ~1ms, optimizer ~38ms) will have power
   readings where start and end return the same cached value.
   Identical power across phases may partly reflect measurement limitation
   rather than true physical energy invariance across phases.
2. GPU utilization from nvmlDeviceGetUtilizationRates is averaged over
   the driver's sampling window (~1s), so each reading reflects ~4 steps.
3. DataLoader iteration time (batch fetching from CPU memory) occurs
   before start_step() in the training loop and is NOT captured.
   For synthetic data with num_workers=0 this should be <0.1ms
4. CPU utilization from psutil.cpu_percent(interval=None) reflects
   usage since the previous call, not a true instantaneous snapshot.

Reference
---------
Interface defined in src/trainer/stats/base.py
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
    logger.debug("pynvml not available — GPU power/utilization/temperature disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available — CPU utilization disabled")

# ============================================================
# Number of warmup steps to exclude from steady-state statistics
# Power stabilises by step 5-6 on Ada GPUs; timing by step 1.
# Using 6 for conservative power/energy analysis
# ============================================================
WARMUP_STEPS = 6


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    """Factory function called by the auto-discovery system
    when --trainer_stats hardware is specified.
    """
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided — defaulting to cuda:0")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_train_steps = kwargs.get("num_train_steps", 100)

    # Defaults
    output_dir = "./hardware_stats"
    run_id = None
    carbon_intensity = 30.0  # Quebec (Hydro-Québec) default

    hw_conf = getattr(conf.trainer_stats_configs, "hardware", None)
    if hw_conf:
        output_dir = getattr(hw_conf, "output_dir", output_dir)
        run_id = getattr(hw_conf, "run_id", run_id)
        carbon_intensity = getattr(hw_conf, "carbon_intensity", carbon_intensity)

    return HardwareTrainerStats(
        device=device,
        output_dir=output_dir,
        run_id=run_id,
        num_train_steps=num_train_steps,
        carbon_intensity=carbon_intensity,
    )


# ============================================================
# StepRecord — one row per training step in the output CSV
# ============================================================


@dataclass
class StepRecord:
    """All data recorded for a single training step

        energy_step_j = energy_data_transfer_j
                      + energy_forward_j
                      + energy_backward_j
                      + energy_optimizer_j
                      + energy_overhead_j
    """

    step_num: int

    # ---- Timing (milliseconds) ----
    step_time_ms: float
    data_transfer_time_ms: float
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float
    overhead_time_ms: float  # step − (dt + fwd + bwd + opt)

    # ---- GPU Memory via PyTorch (MB) ----
    gpu_memory_allocated_mb: float   # end-of-step allocated tensors
    gpu_memory_reserved_mb: float    # caching allocator pool
    gpu_memory_peak_mb: float        # max allocated during step

    # ---- GPU Power via NVML (Watts) — sampled at phase boundaries ----
    gpu_power_step_start_w: float
    gpu_power_step_end_w: float
    gpu_power_data_transfer_start_w: float
    gpu_power_data_transfer_end_w: float
    gpu_power_forward_start_w: float
    gpu_power_forward_end_w: float
    gpu_power_backward_start_w: float
    gpu_power_backward_end_w: float
    gpu_power_optimizer_start_w: float
    gpu_power_optimizer_end_w: float

    # ---- Per-Phase Energy (Joules) via trapezoidal approx. ----
    energy_data_transfer_j: float
    energy_forward_j: float
    energy_backward_j: float
    energy_optimizer_j: float
    energy_overhead_j: float  # avg_step_power × overhead_duration
    energy_step_j: float      # closed sum of the 5 components above

    # ---- CO2 Emissions ----
    co2_step_mg: float  # milligrams CO2eq for this step

    # ---- GPU Temperature (°C) via NVML ----
    gpu_temperature_c: float

    # ---- Utilization (%) ----
    gpu_utilization: float          # NVML compute (~1 s window average)
    gpu_memory_utilization: float   # NVML memory bandwidth utilisation
    cpu_utilization: float          # psutil (since last call)

    # ---- Timestamp ----
    timestamp: str


# ============================================================
# Main class
# ============================================================


class HardwareTrainerStats(base.TrainerStats):
    """TrainerStats implementation with per-phase hardware monitoring
    energy accounting, and CO2 estimation.

    Power measurement approach
    --------------------------
    Sample GPU power via NVML "nvmlDeviceGetPowerUsage" at the start
    and end of each phase.  Per-phase energy is computed with the
    trapezoidal rule:

        E_phase = ((P_start + P_end) / 2) × duration_seconds

    *NOTE:* NVML refreshes power readings every ~50–100 ms on Ada
    Lovelace GPUs.  Phases shorter than one refresh interval
    (data_transfer ≈ 1 ms, optimizer ≈ 38 ms) will likely return the
    same cached value for both samples, so the per-phase power
    differentiation is bound to this limitation

    Parameters
    ----------
    device : torch.device
        The PyTorch device used for training
    output_dir : str
        Directory to save CSV and JSON output files
    run_id : str, optional
        Unique identifier for this run.  Auto-generated if none
    num_train_steps : int
        Expected number of training steps
    carbon_intensity : float
        Grid carbon intensity in gCO2eq / kWh
        Quebec (Hydro-Québec) ≈ 30.  Canada avg ≈ 110.  US avg ≈ 390
    """

    def __init__(
        self,
        device: torch.device,
        output_dir: str = "./hardware_stats",
        run_id: Optional[str] = None,
        num_train_steps: int = 100,
        carbon_intensity: float = 30.0,
    ):
        super().__init__()
        self.device = device
        self.output_dir = output_dir
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.num_train_steps = num_train_steps
        self.carbon_intensity = carbon_intensity

        os.makedirs(output_dir, exist_ok=True)

        # ---- NVML initialisation ----
        self.nvml_handle = None
        self.gpu_name = "unknown"
        self.gpu_power_limit_w = 0.0
        if PYNVML_AVAILABLE and device.type == "cuda":
            try:
                pynvml.nvmlInit()
                idx = device.index if device.index is not None else 0
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = pynvml.nvmlDeviceGetName(self.nvml_handle)
                self.gpu_name = name.decode("utf-8") if isinstance(name, bytes) else name
                try:
                    self.gpu_power_limit_w = (
                        pynvml.nvmlDeviceGetPowerManagementLimit(self.nvml_handle) / 1000.0
                    )
                except pynvml.NVMLError:
                    logger.warning("Could not query GPU power management limit")
                logger.info(
                    f"NVML ready: {self.gpu_name}, "
                    f"power limit = {self.gpu_power_limit_w:.0f} W"
                )
            except Exception as exc:
                logger.warning(f"Could not initialise NVML: {exc}")

        # ---- Storage ----
        self.step_records: List[StepRecord] = []
        self.current_step = 0

        # ---- Step-level accumulators ----
        self._step_start_ns: int = 0
        self._step_start_power_w: float = 0.0

        # Phase timing (nanoseconds)
        self._dt_start_ns: int = 0
        self._fwd_start_ns: int = 0
        self._bwd_start_ns: int = 0
        self._opt_start_ns: int = 0

        self._dt_time_ns: int = 0
        self._fwd_time_ns: int = 0
        self._bwd_time_ns: int = 0
        self._opt_time_ns: int = 0

        # Phase power (watts)
        self._power_dt_start: float = 0.0
        self._power_dt_end: float = 0.0
        self._power_fwd_start: float = 0.0
        self._power_fwd_end: float = 0.0
        self._power_bwd_start: float = 0.0
        self._power_bwd_end: float = 0.0
        self._power_opt_start: float = 0.0
        self._power_opt_end: float = 0.0

        logger.info(
            f"HardwareTrainerStats v2 initialised — "
            f"output_dir={output_dir}, run_id={self.run_id}, "
            f"gpu={self.gpu_name}, "
            f"carbon_intensity={carbon_intensity} gCO2eq/kWh, "
            f"nvml={'yes' if self.nvml_handle else 'no'}, "
            f"psutil={'yes' if PSUTIL_AVAILABLE else 'no'}"
        )

    # ==========================================================
    # Helpers
    # ==========================================================

    def _sync_cuda(self) -> None:
        """Ensure all pending GPU kernels have completed"""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _time_ns() -> int:
        return time.perf_counter_ns()

    def _get_gpu_memory(self) -> tuple:
        """Return (allocated_bytes, reserved_bytes, peak_bytes)"""
        if self.device.type != "cuda":
            return (0, 0, 0)
        try:
            return (
                torch.cuda.memory_allocated(self.device),
                torch.cuda.memory_reserved(self.device),
                torch.cuda.max_memory_allocated(self.device),
            )
        except Exception:
            return (0, 0, 0)

    def _get_gpu_power_w(self) -> float:
        """Instantaneous GPU power in Watts via NVML

        *NOTE:* NVML refreshes every ~50–100 ms on Ada GPUs
        Back-to-back reads closer than that return a cached value
        """
        if self.nvml_handle is not None:
            try:
                return pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
            except Exception:
                pass
        return 0.0

    def _get_gpu_temperature(self) -> float:
        if self.nvml_handle is not None:
            try:
                return float(
                    pynvml.nvmlDeviceGetTemperature(
                        self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass
        return 0.0

    def _get_gpu_utilization(self) -> tuple:
        """Return (compute_util%, mem_bw_util%).

        *NOTE:* this is averaged over the driver's sampling window (~1 s).
        """
        if self.nvml_handle is not None:
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                return (float(u.gpu), float(u.memory))
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
    def _compute_energy_j(
        power_start_w: float, power_end_w: float, duration_ns: int
    ) -> float:
        """Trapezoidal energy: E = ((P0 + P1) / 2) × Δt"""
        return (power_start_w + power_end_w) / 2.0 * (duration_ns / 1e9)

    def _compute_co2_mg(self, energy_j: float) -> float:
        """Convert energy (J) to CO2 (milligrams)

        CO2_mg = energy_kWh × intensity_g/kWh × 1000
        which simplifies to energy_J × intensity / 3600
        """
        return energy_j * self.carbon_intensity / 3600.0

    # ==========================================================
    # TrainerStats interface — training lifecycle
    # ==========================================================

    def start_train(self) -> None:
        logger.info(
            f"Training start — hardware+energy v2 "
            f"(run_id={self.run_id}, gpu={self.gpu_name}, "
            f"carbon={self.carbon_intensity} gCO2eq/kWh)"
        )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)  # prime the counter

    def stop_train(self) -> None:
        logger.info("Training complete — saving hardware+energy stats...")
        self._save_results()
        if self.nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ==========================================================
    # TrainerStats interface — step lifecycle
    # ==========================================================

    def start_step(self) -> None:
        self._sync_cuda()
        self._step_start_power_w = self._get_gpu_power_w()
        self._step_start_ns = self._time_ns()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # Reset all phase accumulators
        self._dt_time_ns = 0
        self._fwd_time_ns = 0
        self._bwd_time_ns = 0
        self._opt_time_ns = 0
        self._power_dt_start = 0.0
        self._power_dt_end = 0.0
        self._power_fwd_start = 0.0
        self._power_fwd_end = 0.0
        self._power_bwd_start = 0.0
        self._power_bwd_end = 0.0
        self._power_opt_start = 0.0
        self._power_opt_end = 0.0

    def stop_step(self) -> None:
        self._sync_cuda()
        step_end_ns = self._time_ns()
        step_end_power_w = self._get_gpu_power_w()

        step_time_ns = step_end_ns - self._step_start_ns

        # ---- Hardware snapshots ----
        gpu_mem = self._get_gpu_memory()
        gpu_util, mem_util = self._get_gpu_utilization()
        cpu_util = self._get_cpu_utilization()
        gpu_temp = self._get_gpu_temperature()

        # ---- Per-phase energy (trapezoidal) ----
        e_dt = self._compute_energy_j(
            self._power_dt_start, self._power_dt_end, self._dt_time_ns
        )
        e_fwd = self._compute_energy_j(
            self._power_fwd_start, self._power_fwd_end, self._fwd_time_ns
        )
        e_bwd = self._compute_energy_j(
            self._power_bwd_start, self._power_bwd_end, self._bwd_time_ns
        )
        e_opt = self._compute_energy_j(
            self._power_opt_start, self._power_opt_end, self._opt_time_ns
        )

        # ---- Overhead ----
        phase_sum_ns = (
            self._dt_time_ns
            + self._fwd_time_ns
            + self._bwd_time_ns
            + self._opt_time_ns
        )
        overhead_ns = max(0, step_time_ns - phase_sum_ns)
        avg_step_power = (self._step_start_power_w + step_end_power_w) / 2.0
        e_overhead = avg_step_power * (overhead_ns / 1e9)

        # ---- Closed energy sum ----
        e_step = e_dt + e_fwd + e_bwd + e_opt + e_overhead

        # ---- CO2 ----
        co2_mg = self._compute_co2_mg(e_step)

        record = StepRecord(
            step_num=self.current_step,
            # timing
            step_time_ms=step_time_ns / 1e6,
            data_transfer_time_ms=self._dt_time_ns / 1e6,
            forward_time_ms=self._fwd_time_ns / 1e6,
            backward_time_ms=self._bwd_time_ns / 1e6,
            optimizer_time_ms=self._opt_time_ns
                        optimizer_time_ms=self._opt_time_ns / 1e6,
            overhead_time_ms=overhead_ns / 1e6,
            # memory
            gpu_memory_allocated_mb=gpu_mem[0] / (1024 * 1024),
            gpu_memory_reserved_mb=gpu_mem[1] / (1024 * 1024),
            gpu_memory_peak_mb=gpu_mem[2] / (1024 * 1024),
            # power
            gpu_power_step_start_w=self._step_start_power_w,
            gpu_power_step_end_w=step_end_power_w,
            gpu_power_data_transfer_start_w=self._power_dt_start,
            gpu_power_data_transfer_end_w=self._power_dt_end,
            gpu_power_forward_start_w=self._power_fwd_start,
            gpu_power_forward_end_w=self._power_fwd_end,
            gpu_power_backward_start_w=self._power_bwd_start,
            gpu_power_backward_end_w=self._power_bwd_end,
            gpu_power_optimizer_start_w=self._power_opt_start,
            gpu_power_optimizer_end_w=self._power_opt_end,
            # energy
            energy_data_transfer_j=e_dt,
            energy_forward_j=e_fwd,
            energy_backward_j=e_bwd,
            energy_optimizer_j=e_opt,
            energy_overhead_j=e_overhead,
            energy_step_j=e_step,
            # co2
            co2_step_mg=co2_mg,
            # temperature
            gpu_temperature_c=gpu_temp,
            # utilization
            gpu_utilization=gpu_util,
            gpu_memory_utilization=mem_util,
            cpu_utilization=cpu_util,
            # timestamp
            timestamp=datetime.now().isoformat(),
        )
        self.step_records.append(record)
        self.current_step += 1

    # ==========================================================
    # TrainerStats interface — phase hooks
    # ==========================================================

    # ---- Data Transfer (CPU to GPU) ----

    def start_data_transfer(self) -> None:
        self._sync_cuda()
        self._power_dt_start = self._get_gpu_power_w()
        self._dt_start_ns = self._time_ns()

    def stop_data_transfer(self) -> None:
        self._sync_cuda()
        self._dt_time_ns = self._time_ns() - self._dt_start_ns
        self._power_dt_end = self._get_gpu_power_w()

    # ---- Forward ----

    def start_forward(self) -> None:
        self._sync_cuda()
        self._power_fwd_start = self._get_gpu_power_w()
        self._fwd_start_ns = self._time_ns()

    def stop_forward(self) -> None:
        self._sync_cuda()
        self._fwd_time_ns = self._time_ns() - self._fwd_start_ns
        self._power_fwd_end = self._get_gpu_power_w()

    # ---- Backward ----

    def start_backward(self) -> None:
        self._sync_cuda()
        self._power_bwd_start = self._get_gpu_power_w()
        self._bwd_start_ns = self._time_ns()

    def stop_backward(self) -> None:
        self._sync_cuda()
        self._bwd_time_ns = self._time_ns() - self._bwd_start_ns
        self._power_bwd_end = self._get_gpu_power_w()

    # ---- Optimizer ----

    def start_optimizer_step(self) -> None:
        self._sync_cuda()
        self._power_opt_start = self._get_gpu_power_w()
        self._opt_start_ns = self._time_ns()

    def stop_optimizer_step(self) -> None:
        self._sync_cuda()
        self._opt_time_ns = self._time_ns() - self._opt_start_ns
        self._power_opt_end = self._get_gpu_power_w()

    # ---- Checkpointing (no-op) ----

    def start_save_checkpoint(self) -> None:
        self._sync_cuda()

    def stop_save_checkpoint(self) -> None:
        self._sync_cuda()

    # ---- Loss / Logging ----

    def log_loss(self, loss: torch.Tensor) -> None:
        pass

    def log_step(self) -> None:
        """Log the most recent step to console."""
        if not self.step_records:
            return

        rec = self.step_records[-1]
        # Average power across all phase boundary readings
        power_readings = [
            rec.gpu_power_forward_start_w, rec.gpu_power_forward_end_w,
            rec.gpu_power_backward_start_w, rec.gpu_power_backward_end_w,
            rec.gpu_power_optimizer_start_w, rec.gpu_power_optimizer_end_w,
        ]
        avg_power = sum(power_readings) / len(power_readings)

        print(
            f"[Step {rec.step_num:4d}] "
            f"total={rec.step_time_ms:.1f}ms "
            f"(dt={rec.data_transfer_time_ms:.1f} "
            f"fwd={rec.forward_time_ms:.1f} "
            f"bwd={rec.backward_time_ms:.1f} "
            f"opt={rec.optimizer_time_ms:.1f} "
            f"oh={rec.overhead_time_ms:.1f}) | "
            f"pwr={avg_power:.0f}W | "
            f"E={rec.energy_step_j * 1000:.1f}mJ | "
            f"CO2={rec.co2_step_mg:.3f}mg | "
            f"T={rec.gpu_temperature_c:.0f}°C | "
            f"mem={rec.gpu_memory_allocated_mb:.0f}MB | "
            f"util={rec.gpu_utilization:.0f}%"
        )

    def log_stats(self) -> None:
        """Print summary statistics at end of training"""
        if len(self.step_records) < WARMUP_STEPS + 1:
            logger.warning(
                f"Only {len(self.step_records)} steps recorded, "
                f"need > {WARMUP_STEPS} for steady-state stats"
            )
            return

        steady = self.step_records[WARMUP_STEPS:]
        all_recs = self.step_records

        print("\n" + "=" * 65)
        print("HARDWARE + ENERGY SUMMARY  (v2)")
        print("=" * 65)
        print(f"Total steps: {len(all_recs)} "
              f"({WARMUP_STEPS} warmup, {len(steady)} steady-state)")

        # ---- Timing ----
        print(f"\n--- Timing (steady state, ms) ---")
        for label, attr in [
            ("Data xfer", "data_transfer_time_ms"),
            ("Forward",   "forward_time_ms"),
            ("Backward",  "backward_time_ms"),
            ("Optimizer", "optimizer_time_ms"),
            ("Overhead",  "overhead_time_ms"),
            ("Total step","step_time_ms"),
        ]:
            vals = [getattr(r, attr) for r in steady]
            avg = sum(vals) / len(vals)
            std = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"  {label:12s}: {avg:8.2f} ± {std:.2f} ms")

        # ---- Phase proportions ----
        print(f"\n--- Phase Proportions (steady state) ---")
        avg_dt  = sum(r.data_transfer_time_ms for r in steady) / len(steady)
        avg_fwd = sum(r.forward_time_ms for r in steady) / len(steady)
        avg_bwd = sum(r.backward_time_ms for r in steady) / len(steady)
        avg_opt = sum(r.optimizer_time_ms for r in steady) / len(steady)
        avg_oh  = sum(r.overhead_time_ms for r in steady) / len(steady)
        avg_total = sum(r.step_time_ms for r in steady) / len(steady)
        for label, val in [("Data xfer", avg_dt), ("Forward", avg_fwd),
                           ("Backward", avg_bwd), ("Optimizer", avg_opt),
                           ("Overhead", avg_oh)]:
            pct = val / avg_total * 100 if avg_total > 0 else 0
            print(f"  {label:12s}: {pct:5.1f}%  ({val:.2f} ms)")

        # ---- Memory ----
        print(f"\n--- GPU Memory (steady state) ---")
        print(f"  Allocated: {sum(r.gpu_memory_allocated_mb for r in steady)/len(steady):.0f} MB")
        print(f"  Reserved:  {sum(r.gpu_memory_reserved_mb for r in steady)/len(steady):.0f} MB")
        print(f"  Peak:      {sum(r.gpu_memory_peak_mb for r in steady)/len(steady):.0f} MB")

        # ---- Power ----
        print(f"\n--- GPU Power (steady state, W) ---")
        for label, s_col, e_col in [
            ("Forward",
             "gpu_power_forward_start_w", "gpu_power_forward_end_w"),
            ("Backward",
             "gpu_power_backward_start_w", "gpu_power_backward_end_w"),
            ("Optimizer",
             "gpu_power_optimizer_start_w", "gpu_power_optimizer_end_w"),
        ]:
            avgs = [(getattr(r, s_col) + getattr(r, e_col)) / 2 for r in steady]
            mean_p = sum(avgs) / len(avgs)
            std_p = (sum((v - mean_p) ** 2 for v in avgs) / len(avgs)) ** 0.5
            print(f"  {label:12s}: {mean_p:6.1f} ± {std_p:.1f} W")

        # ---- Energy ----
        print(f"\n--- Energy ---")
        total_e = sum(r.energy_step_j for r in all_recs)
        avg_e_mj = sum(r.energy_step_j for r in steady) / len(steady) * 1000
        warmup_e_mj = all_recs[0].energy_step_j * 1000
        print(f"  Total training:    {total_e:.1f} J ({total_e / 3600:.4f} Wh)")
        print(f"  Avg steady step:   {avg_e_mj:.1f} mJ")
        print(f"  Warmup step 0:     {warmup_e_mj:.1f} mJ")

        # ---- CO2 ----
        print(f"\n--- CO2 Emissions ({self.carbon_intensity} gCO2eq/kWh) ---")
        total_co2 = sum(r.co2_step_mg for r in all_recs)
        avg_co2 = sum(r.co2_step_mg for r in steady) / len(steady)
        print(f"  Total training:    {total_co2:.3f} mg ({total_co2/1000:.6f} g)")
        print(f"  Avg steady step:   {avg_co2:.4f} mg")

        # ---- Temperature ----
        print(f"\n--- Temperature ---")
        temps = [r.gpu_temperature_c for r in steady]
        print(f"  Range: {min(temps):.0f}°C → {max(temps):.0f}°C")
        print(f"  Avg:   {sum(temps)/len(temps):.1f}°C")

        # ---- Utilization ----
        print(f"\n--- GPU Utilization (steady state) ---")
        print(f"  Compute: {sum(r.gpu_utilization for r in steady)/len(steady):.1f}%")
        print(f"  Mem BW:  {sum(r.gpu_memory_utilization for r in steady)/len(steady):.1f}%")

        # ---- Warmup comparison ----
        print(f"\n--- Warmup (Step 0) vs Steady State ---")
        w = all_recs[0]
        for label, w_attr, s_attr in [
            ("Total time", "step_time_ms", "step_time_ms"),
            ("Forward",    "forward_time_ms", "forward_time_ms"),
            ("Backward",   "backward_time_ms", "backward_time_ms"),
            ("Energy",     "energy_step_j", "energy_step_j"),
        ]:
            w_val = getattr(w, w_attr)
            s_val = sum(getattr(r, s_attr) for r in steady) / len(steady)
            if s_val > 0:
                ratio = w_val / s_val
                print(f"  {label:12s}: {w_val:.2f} vs {s_val:.2f} ({ratio:.2f}x)")

        print("=" * 65 + "\n")

    # ==========================================================
    # Persistence
    # ==========================================================

    def _save_results(self) -> None:
        """Write CSV + JSON metadata"""
        if not self.step_records:
            logger.warning("No step records to save")
            return

        # ---- CSV ----
        csv_path = os.path.join(
            self.output_dir, f"hardware_stats_{self.run_id}.csv"
        )
        field_names = [f.name for f in fields(StepRecord)]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for rec in self.step_records:
                writer.writerow(asdict(rec))
        logger.info(f"CSV saved: {csv_path} ({len(self.step_records)} rows)")

        # ---- JSON metadata ----
        steady = self.step_records[WARMUP_STEPS:]
        total_energy = sum(r.energy_step_j for r in self.step_records)
        total_co2 = sum(r.co2_step_mg for r in self.step_records)

        metadata = {
            "version": 2,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "gpu": {
                "name": self.gpu_name,
                "power_limit_w": self.gpu_power_limit_w,
            },
            "config": {
                "num_steps": len(self.step_records),
                "warmup_steps": WARMUP_STEPS,
                "steady_state_steps": len(steady),
                "carbon_intensity_gco2eq_per_kwh": self.carbon_intensity,
            },
            "totals": {
                "energy_j": round(total_energy, 3),
                "energy_wh": round(total_energy / 3600, 6),
                "co2_mg": round(total_co2, 4),
                "co2_g": round(total_co2 / 1000, 7),
            },
            "steady_state_averages": {
                "step_time_ms": round(
                    sum(r.step_time_ms for r in steady) / len(steady), 2
                ) if steady else 0,
                "energy_step_mj": round(
                    sum(r.energy_step_j for r in steady) / len(steady) * 1000, 1
                ) if steady else 0,
                "gpu_power_w": round(
                    sum(
                        (r.gpu_power_forward_start_w + r.gpu_power_forward_end_w
                         + r.gpu_power_backward_start_w + r.gpu_power_backward_end_w
                         + r.gpu_power_optimizer_start_w + r.gpu_power_optimizer_end_w
                         ) / 6.0
                        for r in steady
                    ) / len(steady), 1
                ) if steady else 0,
                "gpu_utilization_pct": round(
                    sum(r.gpu_utilization for r in steady) / len(steady), 1
                ) if steady else 0,
                "gpu_temperature_c": round(
                    sum(r.gpu_temperature_c for r in steady) / len(steady), 1
                ) if steady else 0,
            },
            "known_limitations": [
                "NVML power refresh ~50-100ms; phases <50ms have aliased readings",
                "GPU utilization averaged over ~1s driver window",
                "DataLoader iteration time not captured (before start_step)",
                "CPU utilization from psutil reflects only inter-call average",
            ],
        }

        json_path = os.path.join(
            self.output_dir, f"metadata_{self.run_id}.json"
        )
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {json_path}")
