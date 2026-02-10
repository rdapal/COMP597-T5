# === necessary modules ===
import src.config as config  # Configurations
import src.trainer as trainer  # Trainer base class
import src.trainer.stats as trainer_stats  # Trainer statistics module

# === necessary external modules ===
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers
import logging

logger = logging.getLogger(__name__)

"""
This file contains the code to train a T5 model using Simple trainer (src/trainer/simple.py).
It is based on the T5 model from HuggingFace Transformers.
https://huggingface.co/docs/transformers/en/model_doc/t5

T5 (Text-to-Text Transfer Transformer) reframes all NLP tasks as text-to-text problems.
T5 is an encoder-decoder model designed for sequence-to-sequence tasks like translation, summarization, and question answering.
"""


def init_t5_tokenizer():
    """
    Initializes the T5 tokenizer. This tokenizer is found in the HuggingFace Transformers library.
    
    Returns:
        transformers.PreTrainedTokenizer: The T5 tokenizer.
    """
    # https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer
    # Using t5-base (220M parameters) as specified in MilaBench
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")
    return tokenizer


def process_dataset(conf: config.Config, tokenizer: transformers.PreTrainedTokenizer, dataset: data.Dataset) -> data.Dataset:
    """
    Processes the dataset for T5 training.
    
    T5 is a seq2seq model, so we need both input and target sequences.
    For energy benchmarking purposes, we treat the text as both input and target
    
    Args:
        conf (config.Config): The configuration object.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        dataset (data.Dataset): The dataset to process.
        
    Returns:
        data.Dataset: The processed dataset with input_ids and labels.
    """
    max_length = 512  # Standard max length for T5-base
    
    def tokenize(examples):
        # T5 expects a text-to-text format
        # For benchmarking, we'll use the same text as both input and target
        # In practice, T5 tasks would have different input/output (e.g., "translate: ..." -> translation)
        
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For T5, labels are the target sequences
        # Using the same text as target for benchmarking purposes
        labels = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    # Get number of processes from config if available, otherwise default to 1
    num_proc = getattr(conf.model_configs, 't5', None)
    if num_proc and hasattr(num_proc, 'tokenize_num_process'):
        num_proc = num_proc.tokenize_num_process
    else:
        num_proc = 1
    
    dataset = dataset.map(tokenize, batched=True, num_proc=num_proc)
    
    # Remove unnecessary columns (keeping only input_ids, attention_mask, labels)
    columns_to_remove = [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
    if columns_to_remove:
        dataset = dataset.remove_columns(column_names=columns_to_remove)
    
    return dataset


def init_t5_optim(conf: config.Config, model: nn.Module) -> optim.Optimizer:
    """
    Initializes the optimizer for the T5 model.
    
    Args:
        conf (config.Config): The configuration object.
        model (nn.Module): The T5 model.
        
    Returns:
        optim.Optimizer: The initialized optimizer.
    """
    # AdamW optimizer with weight decay, commonly used for transformer models
    return optim.AdamW(model.parameters(), lr=conf.learning_rate)


def pre_init_t5(conf: config.Config, dataset: data.Dataset) -> Tuple[transformers.PreTrainedModel, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForSeq2Seq]:
    """
    Prepares the T5 model, dataset, tokenizer and data collator for training.
    
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
        
    Returns:
        Tuple containing:
            - T5 model (T5ForConditionalGeneration)
            - Processed dataset
            - Tokenizer
            - Data collator for seq2seq
    """
    tokenizer = init_t5_tokenizer()
    dataset = process_dataset(conf, tokenizer, dataset)
    
    # T5 uses DataCollatorForSeq2Seq for proper handling of encoder-decoder inputs
    # https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForSeq2Seq
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,  # Will be set after model creation if needed
        padding=True,
        return_tensors="pt"
    )
    
    # Initialize T5 model configuration
    # https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Config
    # Using t5-base configuration (220M parameters)
    model_config = transformers.T5Config.from_pretrained("t5-base")
    
    # Initialize model from config (random weights for benchmarking)
    # https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5ForConditionalGeneration
    model = transformers.T5ForConditionalGeneration(config=model_config)
    
    logger.info(f"T5 model initialized with {model.num_parameters():,} parameters")
    
    return model, dataset, tokenizer, data_collator


################################################################################
#################################    Simple    #################################
################################################################################

def simple_trainer(
    conf: config.Config,
    model: transformers.T5ForConditionalGeneration,
    dataset: data.Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    data_collator: transformers.DataCollatorForSeq2Seq
) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Simple trainer for T5 model. Uses the SimpleTrainer from src/trainer/simple.py.
    
    Args:
        conf (config.Config): The configuration object.
        model (transformers.T5ForConditionalGeneration): The T5 model to train.
        dataset (data.Dataset): The dataset to train on.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        data_collator (transformers.DataCollatorForSeq2Seq): The data collator to use.
        
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The simple trainer and optional additional info.
    """
    # Create DataLoader for batching
    loader = data.DataLoader(
        dataset,
        batch_size=conf.batch_size,
        collate_fn=data_collator
    )
    
    # Move model to GPU
    model = model.cuda()
    
    # Initialize optimizer
    optimizer = init_t5_optim(conf, model)
    
    # Linear learning rate scheduler with warmup
    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader),
    )
    
    # Initialize trainer statistics
    stats = trainer_stats.init_from_conf(
        conf=conf,
        device=model.device,
        num_train_steps=len(loader)
    )
    
    # Return the SimpleTrainer with initialized components
    return trainer.SimpleTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=model.device,
        stats=stats
    ), None


################################################################################
##################################    Init    ##################################
################################################################################

def t5_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """
    Initializes the T5 model and returns the appropriate trainer based on the configuration.
    
    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
        
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The initialized trainer and optional additional info.
    """
    model, dataset, tokenizer, data_collator = pre_init_t5(conf, dataset)
    
    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, tokenizer, data_collator)
    else:
        raise Exception(f"Unknown trainer type: {conf.trainer}")
