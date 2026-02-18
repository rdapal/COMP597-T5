"""
T5 Model Implementation for MilaBench Energy Benchmarking

Supports:
1. Synthetic data (MilaBench style - random tokens)
2. Text datasets (HuggingFace - requires tokenization)

For T5 benchmark, synthetic data is standard approach.
"""

import src.config as config
import src.trainer as trainer
import src.trainer.stats as trainer_stats

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers
import logging

logger = logging.getLogger(__name__)


def init_t5_tokenizer():
    """Initialize T5 tokenizer (for text datasets not synthetic)."""
    return transformers.AutoTokenizer.from_pretrained("t5-base")


def is_synthetic_dataset(dataset) -> bool:
    """
    Check if dataset is synthetic (has tensor input_ids).
    
    Synthetic data has pre-generated torch tensors
    """
    try:
        sample = dataset[0]
        if isinstance(sample, dict) and 'input_ids' in sample:
            if isinstance(sample['input_ids'], torch.Tensor):
                return True
    except Exception:
        pass
    return False


def process_dataset(
    conf: config.Config,
    tokenizer: transformers.PreTrainedTokenizer,
    dataset: data.Dataset
) -> data.Dataset:
    """
    Process dataset for T5 training.
    Text data needs tokenization
    Synthetic data already has tensors
    """
    if is_synthetic_dataset(dataset):
        logger.info("Using synthetic dataset (pre-generated tokens)")
        return dataset
    
    # Text dataset
    logger.info("Processing text dataset with T5 tokenizer")
    max_length = 512
    
    def tokenize(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    num_proc = 1
    t5_config = getattr(conf.model_configs, 't5', None)
    if t5_config and hasattr(t5_config, 'tokenize_num_process'):
        num_proc = t5_config.tokenize_num_process
    
    dataset = dataset.map(tokenize, batched=True, num_proc=num_proc)
    
    columns_to_remove = [
        col for col in dataset.column_names 
        if col not in ["input_ids", "attention_mask", "labels"]
    ]
    if columns_to_remove:
        dataset = dataset.remove_columns(column_names=columns_to_remove)
    
    return dataset


def init_t5_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    """Initialize AdamW optimizer"""
    return optim.AdamW(model.parameters(), lr=conf.learning_rate)


def synthetic_collate_fn(batch):
    """
    Collate function for synthetic dataset 
    Stacks individual sample tensors into batched tensors.
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }


def pre_init_t5(
    conf: config.Config,
    dataset: data.Dataset
) -> Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, any]:
    """
    Prepare T5 model, dataset, tokenizer, and collate function.
    """
    tokenizer = init_t5_tokenizer()
    is_synthetic = is_synthetic_dataset(dataset)
    
    dataset = process_dataset(conf, tokenizer, dataset)
    
    # Choose collate function based on dataset type
    if is_synthetic:
        collate_fn = synthetic_collate_fn
        logger.info("Using synthetic collate function")
    else:
        collate_fn = transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            return_tensors="pt"
        )
        logger.info("Using DataCollatorForSeq2Seq")
    
    # Initialize T5 model with random weights (standard for benchmarking)
    model_config = transformers.T5Config.from_pretrained("t5-base")
    model = transformers.T5ForConditionalGeneration(config=model_config)
    
    logger.info(f"T5 model initialized: {model.num_parameters():,} parameters")
    
    return model, dataset, tokenizer, collate_fn


def simple_trainer(
    conf: config.Config,
    model: transformers.T5ForConditionalGeneration,
    dataset: data.Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    collate_fn
) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """Create SimpleTrainer for T5."""
    
    loader = data.DataLoader(
        dataset,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        shuffle=False,  # Deterministic for benchmarking
    )
    
    model = model.cuda()
    optimizer = init_t5_optim(conf, model)
    
    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader),
    )
    
    stats = trainer_stats.init_from_conf(
        conf=conf,
        device=model.device,
        num_train_steps=len(loader)
    )
    
    return trainer.SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=model.device,
        stats=stats
    ), None


def t5_init(
    conf: config.Config,
    dataset: data.Dataset
) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initialize T5 and return trainer

    Entry point is called by auto-discovery.
    """
    model, dataset, tokenizer, collate_fn = pre_init_t5(conf, dataset)
    
    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, tokenizer, collate_fn)
    else:
        raise ValueError(f"Unknown trainer: {conf.trainer}")
