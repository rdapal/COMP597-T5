"""Statistical tools to use along trainers.

Provides classes to accumulate data from trainers and provide basic analysis on 
the data.

"""
from src.trainer.stats.base import TrainerStats
from src.trainer.stats.noop import NOOPTrainerStats
from src.trainer.stats.noop_synchronized import NOOPSynchronizedTrainerStats
from src.trainer.stats.simple import SimpleTrainerStats
from src.trainer.stats.torch_profiler import TorchProfilerStats
from src.trainer.stats.codecarbon import CodeCarbonStats
from src.trainer.stats.averaged_energy import AveragedEnergy
from src.trainer.stats.utils import *
import src.config as config
import torch.profiler

def init_from_conf(conf : config.Config, **kwargs):
    """Factory for initialize a `TrainerStats`.

    This is a factory that initializes a `TrainerStats` objects using a 
    configuration object. 

    Parameters
    ----------
    conf
        A configuration object.
    **kwargs
        This is used for when additional configurations are need for the 
        specified trainer type. 

        If `args.train_stats == "no-op", then `kwargs` must contain a device 
        field. When it is not provided, it uses the default PyTorch and logs a 
        warning.

        If `args.train_stats == "torch-profiler"`, then `kwargs` must contain 
        `num_train_steps`, an integer which must be greater than 3.

    """
    if conf.train_stats == "no-op":
        return NOOPTrainerStats()
    elif conf.train_stats == "no-op-sync":
        if "device" in kwargs:
            device = kwargs["device"]
        else:
            print("[WARN] No device provided to simple train stats. Using default PyTorch device")
            device = torch.get_default_device()
        return NOOPSynchronizedTrainerStats(device)
    elif conf.train_stats == "simple":
        if "device" in kwargs:
            device = kwargs["device"]
        else:
            print("[WARN] No device provided to simple train stats. Using default PyTorch device")
            device = torch.get_default_device()
        return SimpleTrainerStats(device=device)
    elif conf.train_stats == "torch-profiler":
        # TODO check that kwargs are present
        pr = torch.profiler.profile(

            schedule=torch.profiler.schedule(wait=1, warmup=2, active=kwargs["num_train_steps"] - 3, repeat=0),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )                
        return TorchProfilerStats(pr=pr)
    elif conf.train_stats == "codecarbon":
        if "device" in kwargs:
            device = kwargs["device"]
        else:
            print("[WARN] No device provided to simple train stats. Using default PyTorch device")
            device = torch.get_default_device() 
        return CodeCarbonStats(device, conf.run_num, conf.project_name)
    elif conf.train_stats == "averaged-energy":
        if "device" in kwargs:
            device = kwargs["device"]
        else:
            print("[WARN] No device provided to averaged-energy train stats. Using default PyTorch device")
            device = torch.get_default_device() 
        return AveragedEnergy(device)
    else:
        raise Exception(f"Unknown trainer stats format: {conf.train_stats}")
