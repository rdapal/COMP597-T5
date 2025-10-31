from typing import Any, Optional, Type, TypeVar, cast
import argparse
import enum
import os

T = TypeVar("T")

def _missing_arg(name : str) -> None:
    raise Exception(f"missing argument {name}")

def _wrong_arg_type(name : str, expected : Type[Any], actual : Type[Any]) -> None:
    raise Exception(f"argument {name} expected to have type {expected} but got {actual}")

def _get_arg(args : argparse.Namespace, name : str, arg_type : Type[T]) -> T:
    if not hasattr(args, name):
        _missing_arg(name)
    elif not isinstance(getattr(args, name), arg_type):
        _wrong_arg_type("model", str, type(args.model))
    return getattr(args, name)

@enum.unique
class ConfigArgs(enum.Enum):
    MODEL = "model"
    TRAINER = "trainer"
    DATASET = "dataset"
    DATASET_TRAIN_FILES = "dataset_train_files"
    DATASET_SPLIT = "dataset_split"
    DATASET_LOAD_NUM_PROC = "dataset_load_num_proc"
    TOKENIZE_NUM_PROCESS = "tokenize_num_process"
    BATCH_SIZE = "batch_size"
    TRAIN_STATS = "train_stats"
    SWITCH_TRANSFORMER_NUM_EXPERTS = "switch_transformer_num_experts"
    QWEN_NUM_EXPERTS = "qwen_num_experts" # number of experts for Qwen model
    RUN_NUM = "run_num"  # number of the run used for codecarbon file tracking
    PROJECT_NAME = "project_name"  # name of the project used for codecarbon file tracking
    LEARNING_RATE = "learning_rate"  # learning rate for training
    ENABLE_THROTTLING = "enable_throttling"
    EXPERT_THROTTLING_PERFORMANCE_THRESHOLD = "expert_throttling_performance_threshold"
    THROTTLE_TYPE = "throttle_type"  # (greta) testing unet3d and throttling
    THROTTLE_FREQUENCY = "throttle_frequency"  # (greta) frequency to set the GPU to during throttling

    def to_arg(self) -> str:
        return f"--{self.value}"

class _TorchDistributedEnvAutoName(enum.Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name

@enum.unique
class _TorchDistributedEnv(_TorchDistributedEnvAutoName):
    LOCAL_RANK = enum.auto()
    RANK = enum.auto()
    GROUP_RANK = enum.auto()
    ROLE_RANK = enum.auto()
    LOCAL_WORLD_SIZE = enum.auto()
    WORLD_SIZE = enum.auto()
    GROUP_WORLD_SIZE = enum.auto()
    ROLE_WORLD_SIZE = enum.auto()

    def is_present(self) -> bool:
        return self.value in os.environ
    
    def get(self) -> Optional[str]:
        if self.is_present():
            return os.environ[self.value]
        return None

    def get_int(self, default : int) -> int:
        val = self.get()
        if val is not None:
            val = int(val)
        else:
            val = default
        return val

    def get_str(self, default : str) -> str:
        val = self.get()
        if val is None:
            val = default
        return val

    def get_float(self, default : float) -> float:
        val = self.get()
        if val is not None:
            val = float(val)
        else:
            val = default
        return val

class ConfigDistributed:
    """Torch distributed config from environment variables.

    This config builds itself by looking up standard PyTorch distributed 
    environment variables. When the variables are not present, the fields are 
    `None`.

    """

    def __init__(self) -> None:
        self.local_rank : int = _TorchDistributedEnv.LOCAL_RANK.get_int(-1)
        """Local rank of the process."""
        self.rank : int = _TorchDistributedEnv.RANK.get_int(-1)
        """Rank of the process."""
        self.world_size : int = _TorchDistributedEnv.WORLD_SIZE.get_int(-1)
        """World size of the process."""

    def is_set_up(self) -> bool:
        return self.local_rank >= 0 and self.rank >= 0 and self.world_size >= 1

class Config:
    """Configuration of the program.

    Provides the configuration for the whole training program.

    """

    def __init__(self, args : argparse.Namespace) -> None:
        self.model : str = _get_arg(args, ConfigArgs.MODEL.value, str)
        """The name of the model to generate and train."""
        self.trainer : str = _get_arg(args, ConfigArgs.TRAINER.value, str)
        """The name of the training technique."""
        self.dataset : str = _get_arg(args, ConfigArgs.DATASET.value, str)
        """The name of the dataset to use for training."""
        self.dataset_train_files : str = _get_arg(args, ConfigArgs.DATASET_TRAIN_FILES.value, str)
        """Which files of the dataset to use for training."""
        self.dataset_split : str = _get_arg(args, ConfigArgs.DATASET_SPLIT.value, str)
        """How to split the dataset (ex: train[:100])."""
        self.dataset_load_num_proc : int = _get_arg(args, ConfigArgs.DATASET_LOAD_NUM_PROC.value, int)
        """Number of threads used to load the dataset."""
        self.tokenize_num_process : int = _get_arg(args, ConfigArgs.TOKENIZE_NUM_PROCESS.value, int)
        """Number of threads used to tokenize the dataset."""
        self.batch_size : int = _get_arg(args, ConfigArgs.BATCH_SIZE.value, int)
        """Size of batches."""
        self.train_stats : str = _get_arg(args, ConfigArgs.TRAIN_STATS.value, str)
        """Type of statistics to gather. By default it is set to no-op, which 
        ignores everything."""
        self.switch_transformers_num_experts : int = _get_arg(args, ConfigArgs.SWITCH_TRANSFORMER_NUM_EXPERTS.value, int)
        """When the selected model is switch-base-n, sets the number of experts 
        per sparse layer. It is recommended to only use powers of two."""
        self.config_distributed : ConfigDistributed = ConfigDistributed()
        """Configuration from torch distributed environment variables. Fields 
        are `None` is the variables are not set"""

        self.qwen_num_experts : int = _get_arg(args, ConfigArgs.QWEN_NUM_EXPERTS.value, int)
        """When the selected model is qwen, sets the number of experts per sparse layer. 
        It is recommended to only use powers of two."""

        self.run_num : int = _get_arg(args, ConfigArgs.RUN_NUM.value, int)
        """The run number used for codecarbon file tracking."""
        self.project_name : str = _get_arg(args, ConfigArgs.PROJECT_NAME.value, str)
        """The name of the project used for codecarbon file tracking."""

        self.learning_rate : float = _get_arg(args, ConfigArgs.LEARNING_RATE.value, float)
        """The learning rate for training. It is used by the optimizer for both Switch Transformers and Qwen models."""

        self.enable_throttling : bool = _get_arg(args, ConfigArgs.ENABLE_THROTTLING.value, bool)
        """Enables GPU frequency throttling at various pre selected points during training."""
        self.expert_throttling_performance_threshold : float = _get_arg(args, ConfigArgs.EXPERT_THROTTLING_PERFORMANCE_THRESHOLD.value, float)
        """How much performance can be reduced by throttling."""

        self.throttle_type : str = _get_arg(args, ConfigArgs.THROTTLE_TYPE.value, str)
        """Which pass to throttle."""
        self.throttle_frequency : int = _get_arg(args, ConfigArgs.THROTTLE_FREQUENCY.value, int)
        """The frequency to set the GPU to during the fixed pass when throttling is enabled."""

