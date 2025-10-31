import torch
import torch.nn as nn
import torch.utils.data as data
from typing import Dict, Optional, Tuple

# MLCommons modules
from src.models.unet3d_mlcommons.pytorch.model.unet3d import Unet3D
from src.models.unet3d_mlcommons.pytorch.model.losses import DiceCELoss, DiceScore
from src.models.unet3d_mlcommons.pytorch.data_loading.data_loader import get_data_loaders

import src.trainer as trainer
import src.trainer.stats as trainer_stats
import src.hardware_management as hardware_management
import src.config as config
import transformers

################################################################################
################################    Wrappers    ################################
################################################################################


class DictDataLoaderWrapper:
    """Wraps a DataLoader to output dicts instead of tuples/lists."""
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, y = batch
                yield {"input": x, "target": y}
            elif isinstance(batch, dict):
                yield batch
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

    def __len__(self):
        return len(self.loader)

class UNet3DWrapper(nn.Module):
    """Wraps the MLCommons UNet3D model to output HuggingFace-style losses."""
    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: torch.device):
        super().__init__()
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Simulate a HuggingFace-like output dict with .loss
        """
        preds = self.model(input)
        loss = self.loss_fn(preds, target)

        class Output:
            pass
        out = Output()
        out.loss = loss
        out.logits = preds
        return out
    
################################################################################
###############################    Pre-init    ##################################
################################################################################


def pre_init_unet_mlcommons(
    conf: config.Config, data_dir: Optional[str] = None
) -> Tuple[nn.Module, data.DataLoader, data.DataLoader, nn.Module, nn.Module]:
    """
    Initialize the MLCommons Unet3D model, loss, and dataloaders.
    """

    # === Ensure all MLCommons-style config fields exist ===
    defaults = {
        # Data loading & reproducibility
        "loader": "pytorch",                # or "synthetic" if using synthetic data or pytorch data loader
        "input_shape": (128, 128, 128),
        "oversampling": 0.4,
        "seed": 20,
        "val_split": 0.1,
        "num_workers": 4,
        "batch_size": 4,
        "prefetch": 2,
        "data_dir": data_dir or "/raid/data/imseg/raw-data/kits19/preproc-data",
        "benchmark": True,                  
        "deterministic": True,
        "distributed": False,
        "amp": False,                        
        "epochs": 2,                         # keep small for quick test
        "learning_rate": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "dist": False,
        "local_rank": 0,
        "world_size": 1,  
        "rank": 0,
        "resume": False,
    }

    for key, value in defaults.items():
        if not hasattr(conf, key):
            setattr(conf, key, value)

    model = Unet3D(
        in_channels=1,
        n_class=3,
        normalization=getattr(conf, "normalization", "instancenorm"),
        activation=getattr(conf, "activation", "relu"),
    )

    print(f"[INFO] Initialized UNet3D model with {sum(p.numel() for p in model.parameters())} parameters.")

    # Get data loaders using MLCommons utility
    num_shards = getattr(conf, "num_shards", 1)
    global_rank = getattr(conf, "global_rank", 0)
    train_loader, val_loader = get_data_loaders(conf, num_shards=num_shards, global_rank=global_rank)
    train_loader = DictDataLoaderWrapper(train_loader)
    val_loader = DictDataLoaderWrapper(val_loader)
    print(f"[INFO] Train loader batches: {len(train_loader)}, Validation loader batches: {len(val_loader)}")


    # Loss + evaluation metrics from MLCommons
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout="NCDHW", include_background=True)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout="NCDHW", include_background=True)
    print("Initialized MLCommons UNet3D model, loss, and dataloaders.")

    return model, train_loader, val_loader, loss_fn, score_fn


def init_unet_optim(conf, model):
    lr = getattr(conf, "lr", 1e-4)
    weight_decay = getattr(conf, "weight_decay", 1e-5)
    print(f"Using AdamW optimizer with lr={lr}, weight_decay={weight_decay}")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


################################################################################
#################################    Simple    #################################
################################################################################


def unet3d_mlcommons_trainer(
    conf: config.Config,
    model: nn.Module,
    train_loader: data.DataLoader,
    loss_fn: nn.Module,
) -> Tuple[trainer.Trainer, Optional[Dict]]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapped_model = UNet3DWrapper(model, loss_fn, device)

    optimizer = init_unet_optim(conf, wrapped_model)
    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader),
    )

    frequency_scheduler = hardware_management.init_scheduler_from_conf(conf, device)
    stats = trainer_stats.init_from_conf(conf=conf, device=device, num_train_steps=len(train_loader))

    return trainer.SimpleTrainer(
        loader=train_loader,
        model=wrapped_model,       
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        stats=stats,
        frequency_scheduler=frequency_scheduler,
        conf=conf,
    ), None


################################################################################
##################################    Init    ##################################
################################################################################

def unet3d_mlcommons_init(conf: config.Config, data_dir: Optional[str] = None) -> Tuple[trainer.Trainer, Optional[Dict]]:
    model, train_loader, val_loader, loss_fn, score_fn = pre_init_unet_mlcommons(conf, data_dir)

    if conf.trainer == "simple":
        return unet3d_mlcommons_trainer(conf, model, train_loader, loss_fn)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")