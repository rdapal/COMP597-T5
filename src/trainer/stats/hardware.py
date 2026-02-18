"""
Hardware Monitoring TrainerStats

Add hardware metrics:
- GPU utilization (%)
- GPU memory: allocated, reserved, peak (MB)
- CPU utilization (%)
- Per-phase time series data

Outputs CSV files for plotting and analysis

Based on the interface defined in src/trainer/stats/base.py
Following patterns from src/trainer/stats/simple.py
"""

import logging
import os
import csv
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch

import src.config as config
import src.trainer.stats.base as base

logger = logging.getLogger(__name__)

# Module-level name for auto-discovery
trainer_stats_name = "hardware"

# Optional GPU monitoring via pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.debug("pynvml not available - GPU utilization will be estimated")

# Optional CPU monitoring via psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available - CPU utilization will not be tracked")


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    """
    Factory function for the auto-discovery system
    
    Called by the starter code when --trainer_stats hardware is specified
    """
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to hardware trainer stats. Using CUDA device 0")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_train_steps = kwargs.get("num_train_steps", 100)
    
    # Get config if available
    output_dir = "./hardware_stats"
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
    """Data recorded for each training step."""
    step_num: int
    
    # Timing (nanoseconds, converted to ms for output)
    step_time_ns: int
    forward_time_ns: int
    backward_time_ns: int
    optimizer_time_ns: int
    
    # GPU Memory (bytes, converted to MB for output)
    gpu_memory_allocated: int
    gpu_memory_reserved: int
    gpu_memory_peak: int
    
    # Utilization (percentage)
    gpu_utilization: float
    cpu_utilization: float
    
    # Timestamp
    timestamp: str


class HardwareTrainerStats(base.TrainerStats):
    """
    TrainerStats implementation with hardware monitoring
    
    Collects timing (like SimpleTrainerStats) plus the metrics:
    - GPU memory usage (allocated, reserved, peak)
    - GPU utilization %
    - CPU utilization %
    
    Exports all data to CSV for plotting.
    
    Params
    ----------
    device : torch.device
    The PyTorch device used for training

    output_dir : str
    Directory to save CSV and JSON output files

    run_id : str, optional
    Unique identifier for this run. Auto-generated if not provided.

    num_train_steps : int
    number of training steps for logging
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
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize NVML for GPU monitoring
        self.nvml_handle = None
        if PYNVML_AVAILABLE and device.type == 'cuda':
            try:
                pynvml.nvmlInit()
                device_index = device.index if device.index is not None else 0
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                logger.info(f"NVML initialized for GPU {device_index}")
            except Exception as e:
                logger.warning(f"Could not initialize NVML: {e}")
        
        # Storage for all step records
        self.step_records: List[StepRecord] = []
        
        # Current step tracking
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
        
        logger.info(
            f"HardwareTrainerStats initialized: "
            f"output_dir={output_dir}, run_id={self.run_id}, "
            f"nvml={'yes' if self.nvml_handle else 'no'}, "
            f"psutil={'yes' if PSUTIL_AVAILABLE else 'no'}"
        )
    
    def _sync_cuda(self) -> None:
        """Synchronize CUDA for accurate timing."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
    
    def _time_ns(self) -> int:
        """Get current time in nanoseconds."""
        return time.perf_counter_ns()
    
    def _get_gpu_memory(self) -> tuple:
        """
        Get GPU memory stats
        
        Returns:
            (allocated_bytes, reserved_bytes, peak_bytes)
        """
        if self.device.type != 'cuda':
            return (0, 0, 0)
        
        try:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            return (allocated, reserved, peak)
        except Exception:
            return (0, 0, 0)
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage (0-100)."""
        if self.nvml_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                return float(util.gpu)
            except Exception:
                pass
        return 0.0
    
    def _get_cpu_utilization(self) -> float:
        """Get CPU utilization percentage (0-100)."""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_percent(interval=None)
            except Exception:
                pass
        return 0.0
    
    # ==================== TrainerStats Interface ====================
    
    def start_train(self) -> None:
        """Called at the start of training."""
        logger.info(f"Starting training with hardware monitoring (run_id={self.run_id})")
        
        # Reset peak memory tracking
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Initialize CPU monitoring
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)  # First call initializes
    
    def stop_train(self) -> None:
        """Called at the end of training."""
        logger.info("Training complete. Saving hardware stats...")
        self._save_results()
        
        # Cleanup NVML
        if self.nvml_handle:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    
    def start_step(self) -> None:
        """Called at the start of each training step."""
        self._sync_cuda()
        self._step_start_ns = self._time_ns()
        
        # Reset peak memory for this step
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Reset phase accumulators
        self._forward_time_ns = 0
        self._backward_time_ns = 0
        self._optimizer_time_ns = 0
    
    def stop_step(self) -> None:
        """Called at the end of each training step."""
        self._sync_cuda()
        step_time_ns = self._time_ns() - self._step_start_ns
        
        # Collect hardware metrics
        gpu_mem = self._get_gpu_memory()
        gpu_util = self._get_gpu_utilization()
        cpu_util = self._get_cpu_utilization()
        
        # Create record
        record = StepRecord(
            step_num=self.current_step,
            step_time_ns=step_time_ns,
            forward_time_ns=self._forward_time_ns,
            backward_time_ns=self._backward_time_ns,
            optimizer_time_ns=self._optimizer_time_ns,
            gpu_memory_allocated=gpu_mem[0],
            gpu_memory_reserved=gpu_mem[1],
            gpu_memory_peak=gpu_mem[2],
            gpu_utilization=gpu_util,
            cpu_utilization=cpu_util,
            timestamp=datetime.now().isoformat(),
        )
        self.step_records.append(record)
        
        self.current_step += 1
    
    def start_forward(self) -> None:
        """Called at the start of forward pass."""
        self._sync_cuda()
        self._forward_start_ns = self._time_ns()
    
    def stop_forward(self) -> None:
        """Called at the end of forward pass."""
        self._sync_cuda()
        self._forward_time_ns = self._time_ns() - self._forward_start_ns
    
    def start_backward(self) -> None:
        """Called at the start of backward pass."""
        self._sync_cuda()
        self._backward_start_ns = self._time_ns()
    
    def stop_backward(self) -> None:
        """Called at the end of backward pass."""
        self._sync_cuda()
        self._backward_time_ns = self._time_ns() - self._backward_start_ns
    
    def start_optimizer_step(self) -> None:
        """Called at the start of optimizer step."""
        self._sync_cuda()
        self._optimizer_start_ns = self._time_ns()
    
    def stop_optimizer_step(self) -> None:
        """Called at the end of optimizer step."""
        self._sync_cuda()
        self._optimizer_time_ns = self._time_ns() - self._optimizer_start_ns
    
    def start_save_checkpoint(self) -> None:
        """Called at the start of checkpointing."""
        self._sync_cuda()
        self._checkpoint_start_ns = self._time_ns()
    
    def stop_save_checkpoint(self) -> None:
        """Called at the end of checkpointing."""
        self._sync_cuda()
        # Currently not tracking checkpoint time in records
        pass
    
    def log_loss(self, loss: torch.Tensor) -> None:
        """Log loss value (NOTE currently not stored)"""
        pass
    
    def log_step(self) -> None:
        """Log the previous step's measurements."""
        if not self.step_records:
            return
        
        rec = self.step_records[-1]
        step_ms = rec.step_time_ns / 1_000_000
        fwd_ms = rec.forward_time_ns / 1_000_000
        bwd_ms = rec.backward_time_ns / 1_000_000
        opt_ms = rec.optimizer_time_ns / 1_000_000
        mem_mb = rec.gpu_memory_allocated / (1024 * 1024)
        
        print(
            f"step {step_ms:.2f} -- "
            f"forward {fwd_ms:.2f} -- "
            f"backward {bwd_ms:.2f} -- "
            f"optimizer step {opt_ms:.2f} -- "
            f"GPU mem {mem_mb:.0f}MB -- "
            f"GPU util {rec.gpu_utilization:.0f}%"
        )
    
    def log_stats(self) -> None:
        """Log summary statistics"""
        if not self.step_records:
            print("No step records collected")
            return
        
        # Separate warmup (first step) from steady state
        warmup = self.step_records[0] if self.step_records else None
        steady = self.step_records[1:] if len(self.step_records) > 1 else self.step_records
        
        def avg(records, field):
            values = [getattr(r, field) for r in records]
            return sum(values) / len(values) if values else 0
        
        def to_ms(ns):
            return ns / 1_000_000
        
        def to_mb(b):
            return b / (1024 * 1024)
        
        print("\n" + "=" * 80)
        print("                    HARDWARE MONITORING RESULTS")
        print("=" * 80)
        print(f"Run ID: {self.run_id}")
        print(f"Total Steps: {len(self.step_records)}")
        print(f"Output Directory: {self.output_dir}")
        
        if warmup:
            print("\n" + "-" * 40)
            print("WARMUP (Step 0)")
            print("-" * 40)
            print(f"  Step Time:          {to_ms(warmup.step_time_ns):.2f} ms")
            print(f"  Forward:            {to_ms(warmup.forward_time_ns):.2f} ms")
            print(f"  Backward:           {to_ms(warmup.backward_time_ns):.2f} ms")
            print(f"  Optimizer:          {to_ms(warmup.optimizer_time_ns):.2f} ms")
            print(f"  GPU Memory:         {to_mb(warmup.gpu_memory_allocated):.1f} MB")
            print(f"  GPU Peak Memory:    {to_mb(warmup.gpu_memory_peak):.1f} MB")
            print(f"  GPU Utilization:    {warmup.gpu_utilization:.1f}%")
        
        if steady:
            print("\n" + "-" * 40)
            print("STEADY STATE (Excluding Step 0)")
            print("-" * 40)
            
            avg_step = to_ms(avg(steady, 'step_time_ns'))
            avg_fwd = to_ms(avg(steady, 'forward_time_ns'))
            avg_bwd = to_ms(avg(steady, 'backward_time_ns'))
            avg_opt = to_ms(avg(steady, 'optimizer_time_ns'))
            avg_mem = to_mb(avg(steady, 'gpu_memory_allocated'))
            peak_mem = to_mb(max(r.gpu_memory_peak for r in steady))
            avg_gpu_util = avg(steady, 'gpu_utilization')
            avg_cpu_util = avg(steady, 'cpu_utilization')
            
            print(f"  Avg Step Time:      {avg_step:.2f} ms")
            print(f"  Avg Forward:        {avg_fwd:.2f} ms ({avg_fwd/avg_step*100:.1f}%)")
            print(f"  Avg Backward:       {avg_bwd:.2f} ms ({avg_bwd/avg_step*100:.1f}%)")
            print(f"  Avg Optimizer:      {avg_opt:.2f} ms ({avg_opt/avg_step*100:.1f}%)")
            print(f"  Avg GPU Memory:     {avg_mem:.1f} MB")
            print(f"  Peak GPU Memory:    {peak_mem:.1f} MB")
            print(f"  Avg GPU Util:       {avg_gpu_util:.1f}%")
            print(f"  Avg CPU Util:       {avg_cpu_util:.1f}%")
            print(f"  Throughput:         {1000/avg_step:.2f} steps/sec")
            
            if warmup:
                overhead = to_ms(warmup.step_time_ns) / avg_step
                print(f"  Warmup Overhead:    {overhead:.2f}x")
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80 + "\n")
    
    def _save_results(self) -> None:
        """Save all results to CSV and JSON files."""
        if not self.step_records:
            logger.warning("No records to save")
            return
        
        # Save step-level CSV
        csv_path = os.path.join(self.output_dir, f"hardware_stats_{self.run_id}.csv")
        
        fieldnames = [
            'step_num',
            'step_time_ms', 'forward_time_ms', 'backward_time_ms', 'optimizer_time_ms',
            'gpu_memory_allocated_mb', 'gpu_memory_reserved_mb', 'gpu_memory_peak_mb',
            'gpu_utilization', 'cpu_utilization',
            'timestamp'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for rec in self.step_records:
                row = {
                    'step_num': rec.step_num,
                    'step_time_ms': rec.step_time_ns / 1_000_000,
                    'forward_time_ms': rec.forward_time_ns / 1_000_000,
                    'backward_time_ms': rec.backward_time_ns / 1_000_000,
                    'optimizer_time_ms': rec.optimizer_time_ns / 1_000_000,
                    'gpu_memory_allocated_mb': rec.gpu_memory_allocated / (1024 * 1024),
                    'gpu_memory_reserved_mb': rec.gpu_memory_reserved / (1024 * 1024),
                    'gpu_memory_peak_mb': rec.gpu_memory_peak / (1024 * 1024),
                    'gpu_utilization': rec.gpu_utilization,
                    'cpu_utilization': rec.cpu_utilization,
                    'timestamp': rec.timestamp,
                }
                writer.writerow(row)
        
        logger.info(f"Step metrics saved to: {csv_path}")
        
        # Save metadata JSON
        steady = self.step_records[1:] if len(self.step_records) > 1 else self.step_records
        
        def avg(records, field):
            values = [getattr(r, field) for r in records]
            return sum(values) / len(values) if values else 0
        
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "num_steps": len(self.step_records),
            "nvml_available": self.nvml_handle is not None,
            "psutil_available": PSUTIL_AVAILABLE,
            "summary": {
                "avg_step_time_ms": avg(steady, 'step_time_ns') / 1_000_000,
                "avg_forward_time_ms": avg(steady, 'forward_time_ns') / 1_000_000,
                "avg_backward_time_ms": avg(steady, 'backward_time_ns') / 1_000_000,
                "avg_optimizer_time_ms": avg(steady, 'optimizer_time_ns') / 1_000_000,
                "avg_gpu_memory_mb": avg(steady, 'gpu_memory_allocated') / (1024 * 1024),
                "peak_gpu_memory_mb": max(r.gpu_memory_peak for r in steady) / (1024 * 1024) if steady else 0,
                "avg_gpu_utilization": avg(steady, 'gpu_utilization'),
                "avg_cpu_utilization": avg(steady, 'cpu_utilization'),
                "warmup_step_time_ms": self.step_records[0].step_time_ns / 1_000_000 if self.step_records else 0,
            }
        }
        
        json_path = os.path.join(self.output_dir, f"metadata_{self.run_id}.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {json_path}")
