from typing import Dict, Optional, Tuple
import deepspeed.moe.utils
import src.config as config
import src.hardware_management as hardware_management
import src.trainer as trainer
import src.trainer.stats as trainer_stats
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers

"""
This file contains the code to train a GPT-2 model using Simple trainer.
It is based on the GPT-2 model from HuggingFace Transformers.
https://huggingface.co/docs/transformers/en/model_doc/gpt2
"""

"""
Find a tokenizer from the model and initialize it. make sure the pad
"""
def init_gpt2_tokenizer():
    """
    Initializes the GPT-2 tokenizer.
    Returns:
        transformers.PreTrainedTokenizer: The GPT-2 tokenizer.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
    # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def process_dataset(conf: config.Config, tokenizer: transformers.PreTrainedTokenizer, dataset: data.Dataset) -> data.Dataset:
    """
    Processes the dataset for training.
    Args:
        conf (config.Config): The configuration object.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        dataset (data.Dataset): The dataset to process.
    Returns:
        data.Dataset: The processed dataset.
    """
    def tokenize(examples):
        return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt") 
    dataset = dataset.map(tokenize, batched=True, num_proc=conf.tokenize_num_process)
    dataset = dataset.remove_columns(column_names=["text", "url", "timestamp"])
    return dataset

def init_gpt2_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    """
    Initializes the optimizer for the GPT-2 model.
    Args:
        conf (config.Config): The configuration object.
        model (nn.Module): The GPT-2 model.
    Returns:
        optim.Optimizer: The initialized optimizer.
    """
    # This is a simple AdamW optimizer with weight decay.
    return optim.AdamW(model.parameters(), lr=conf.learning_rate)

def pre_init_gpt2(conf: config.Config, dataset: data.Dataset) -> Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForLanguageModeling]:
    """
    Prepares the GPT-2 model, dataset, tokenizer and data collator for training.
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForLanguageModeling]: The GPT-2 model, dataset, tokenizer and data collator.
    """
    tokenizer = init_gpt2_tokenizer()
    dataset = process_dataset(conf, tokenizer, dataset)
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # mlm_probability=0.15

    model_config = transformers.GPT2Config()# (n_layer=3, loss_type="ForCausalLMLoss") # default is 12 layers
    # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config
    model = transformers.GPT2LMHeadModel(config=model_config)
    # https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2LMHeadModel

    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, dataset, tokenizer, data_collator

################################################################################
#################################    Simple    #################################
################################################################################

def simple_trainer(conf : config.Config, model : transformers.GPT2LMHeadModel, dataset : data.Dataset, tokenizer : transformers.PreTrainedTokenizer, data_collator : transformers.DataCollatorForLanguageModeling) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for GPT-2 model.
    Args:
        conf (config.Config): The configuration object.
        model (transformers.GPT2LMHeadModel): The GPT-2 model to train.
        dataset (data.Dataset): The dataset to train on.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        data_collator (transformers.DataCollatorForLanguageModeling): The data collator to use.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The simple trainer and a dictionary with additional options.
        The dictionary can contain options like "output_router_logits" to control the output of the router logits.
    """
    loader = data.DataLoader(dataset, batch_size=conf.batch_size, collate_fn=data_collator)
    model = model.cuda()
    optimizer = init_gpt2_optim(conf, model)
    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader), 
    )
    frequency_scheduler = hardware_management.init_scheduler_from_conf(conf, model.device)

    return trainer.SimpleTrainer(loader=loader, model=model, optimizer=optimizer, lr_scheduler=scheduler, device=model.device, stats=trainer_stats.init_from_conf(conf=conf, device=model.device, num_train_steps=len(loader)), frequency_scheduler=frequency_scheduler), None

################################################################################
##################################    Init    ##################################
################################################################################

def gpt2_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initializes the GPT-2 model and returns the appropriate trainer based on the configuration.
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The initialized trainer and a dictionary with additional options.
        The dictionary can contain options like "output_router_logits" to control the output of the router logits.
    """
    model, dataset, tokenizer, data_collator = pre_init_gpt2(conf, dataset)
    if conf.trainer == "simple": 
        return simple_trainer(conf, model, dataset, tokenizer, data_collator)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")
