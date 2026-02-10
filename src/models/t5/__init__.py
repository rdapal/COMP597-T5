# === import necessary modules ===
from src.models.t5.model import t5_init
import src.config as config  # Configurations
import src.trainer as trainer  # Trainer base class

# === import necessary external modules ===
from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "t5"

def init_model(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    return t5_init(conf, dataset)
