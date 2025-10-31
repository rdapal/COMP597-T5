from typing import Dict, Optional, Tuple
import src.config as config
import src.models.switch_transformer.switch_base_n as switch_base_n
import src.trainer as trainer
import torch.utils.data as data

################################################################################
##################################    Init    ##################################
################################################################################

def switch_base_8_init(conf : config.Config, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    conf.switch_transformers_num_experts = 8
    return switch_base_n.switch_base_n_init(conf, dataset)
