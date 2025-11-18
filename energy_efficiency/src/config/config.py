"""This file contains the configurations for the project.
"""
from typing import Any, Type, TypeVar
import argparse
import enum

# Used for templated typing.
T = TypeVar("T")

def _missing_arg(name : str) -> None:
    """Raise an exception for a missing argument.
    """
    raise Exception(f"missing argument {name}")

def _wrong_arg_type(name : str, expected : Type[Any], actual : Type[Any]) -> None:
    """Raise an exception for an argument that has the wrong type.
    """
    raise Exception(f"argument {name} expected to have type {expected} but got {actual}")

def _get_arg(args : argparse.Namespace, name : str, arg_type : Type[T]) -> T:
    """Extract an argument from a namespace.

    This function verifies that the requested argument is present and has the correct type. If the checks pass, the argument's value is returned.

    Parameters
    ----------
    args
        Namespace containing the programs arguments.
    name
        Name of the argument to extract from the namespace.
    arg_type
        Expected type of the argument.

    Returns
    -------
    T
        The value of requested argument.

    """
    if not hasattr(args, name):
        _missing_arg(name)
    elif not isinstance(getattr(args, name), arg_type):
        _wrong_arg_type("model", str, type(args.model))
    return getattr(args, name)

@enum.unique
class ConfigArgs(enum.Enum):
    """List of arguments that can be provided to the process.
    """
    MODEL = "model"
    TRAINER = "trainer"
    DATASET = "dataset"
    DATASET_TRAIN_FILES = "dataset_train_files"
    DATASET_SPLIT = "dataset_split"
    DATASET_LOAD_NUM_PROC = "dataset_load_num_proc"
    TOKENIZE_NUM_PROCESS = "tokenize_num_process"
    BATCH_SIZE = "batch_size"
    TRAIN_STATS = "train_stats"
    RUN_NUM = "run_num"  # number of the run used for codecarbon file tracking
    PROJECT_NAME = "project_name"  # name of the project used for codecarbon file tracking
    LEARNING_RATE = "learning_rate"  # learning rate for training

    def to_arg(self) -> str:
        return f"--{self.value}"


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

        self.run_num : int = _get_arg(args, ConfigArgs.RUN_NUM.value, int)
        """The run number used for codecarbon file tracking."""
        self.project_name : str = _get_arg(args, ConfigArgs.PROJECT_NAME.value, str)
        """The name of the project used for codecarbon file tracking."""

        self.learning_rate : float = _get_arg(args, ConfigArgs.LEARNING_RATE.value, float)
        """The learning rate for training. It is used by the optimizer for both Switch Transformers and Qwen models."""

