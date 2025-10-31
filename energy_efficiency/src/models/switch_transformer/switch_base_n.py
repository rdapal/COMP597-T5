from typing import Dict, Optional, Tuple
import src.config as config
import src.hardware_management as hardware_management
import src.trainer as trainer
import src.trainer.stats as trainer_stats
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import transformers

def init_switch_base_8_tokenizer():
    # Based on https://huggingface.co/docs/transformers/en/training
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/switch-base-8")

    # This is necessary to perform MLM using the data collator.
    tokenizer.mask_token = "<mask>"

    return tokenizer

def process_dataset(conf : config.Config, tokenizer : transformers.PreTrainedTokenizer, dataset : data.Dataset) -> data.Dataset:
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

    dataset = dataset.map(tokenize, batched=True, num_proc=conf.tokenize_num_process)
    dataset = dataset.remove_columns(column_names=["text", "url","timestamp"])

    return dataset

def init_switch_base_8_optim(conf : config.Config, model : nn.Module) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=conf.learning_rate) 

def pre_init_switch_base_n(conf : config.Config, dataset : data.Dataset) -> Tuple[transformers.SwitchTransformersForConditionalGeneration, data.Dataset, transformers.PreTrainedTokenizer, transformers.DataCollatorForLanguageModeling]:
    tokenizer = init_switch_base_8_tokenizer()
    dataset = process_dataset(conf, tokenizer, dataset)

    # Based on https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # The defaults of SwitchTransformerConfig roughly correspond to switch-base-8 but with configurable number of experts.
    model_config = transformers.SwitchTransformersConfig(num_experts=conf.switch_transformers_num_experts)
    model = transformers.SwitchTransformersForConditionalGeneration(model_config)

    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, dataset, tokenizer, data_collator

################################################################################
#################################    Simple    #################################
################################################################################

def simple_trainer(conf : config.Config, model : transformers.SwitchTransformersForConditionalGeneration, dataset : data.Dataset, tokenizer : transformers.PreTrainedTokenizer, data_collator : transformers.DataCollatorForLanguageModeling) -> Tuple[trainer.Trainer, Optional[Dict]]:
    loader = data.DataLoader(dataset, batch_size=conf.batch_size, collate_fn=data_collator)
    model = model.cuda()
    optimizer = init_switch_base_8_optim(conf, model)
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

def switch_base_n_init(conf : config.Config, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    model, dataset, tokenizer, data_collator = pre_init_switch_base_n(conf, dataset)

    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, tokenizer, data_collator)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")
