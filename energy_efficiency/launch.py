from typing import Any, Dict, Optional, Tuple
from datasets import load_dataset
import argparse
import gc
import src.models.switch_transformer as switch_transformer
import src.models.qwen as qwen
import src.models.unet3d_mlcommons.pytorch as unet3d_ml_commons
import src.models.unet3d.unet3d as unet3d_simple
import src.models.gpt2.gpt2 as gpt2
import src.trainer as trainer
import src.config as config

def load_data(conf : config.Config):
    train_files = None
    if conf.dataset_train_files is not None and conf.dataset_train_files != "":
        train_files = {"train": conf.dataset_train_files}
    return load_dataset(conf.dataset, data_files=train_files, split=conf.dataset_split, num_proc=conf.dataset_load_num_proc)

def process_conf(conf : config.Config) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    dataset = load_data(conf)
    print(f"Dataset loaded with {len(dataset)} samples.")

    if conf.model == "switch-base-8":
        return switch_transformer.switch_base_8_init(conf, dataset)
    elif conf.model == "switch-base-n":
        return switch_transformer.switch_base_n_init(conf, dataset)
    elif conf.model == "qwen-moe":
        return qwen.qwen_init(conf, dataset)
    elif conf.model == "unet3d_mlcommons":
        # call the MLCommons unet3d training script
        data_dir = "/raid/data/imseg/raw-data/kits19/preproc-data" # unet3d dataset on the server
        return unet3d_ml_commons.unet3d_mlcommons_init(conf, data_dir)
    elif conf.model == "unet3d":
        data_dir = "/raid/data/imseg/raw-data/kits19/preproc-data" # unet3d dataset on the server
        return unet3d_simple.unet3d_mlcommons_init(conf, data_dir)
    elif conf.model == "gpt2":
        return gpt2.gpt2_init(conf, dataset)
    else:
        raise Exception(f"Unknown model {conf.model}")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(config.ConfigArgs.MODEL.to_arg(), type=str, help="Which model to train.", choices=["switch-base-8", "switch-base-n", "qwen-moe", "unet3d", "unet3d_mlcommons", "gpt2"])
    parser.add_argument(config.ConfigArgs.TRAINER.to_arg(), type=str, help="How to train the model", default="simple", choices=["simple"])
    parser.add_argument(config.ConfigArgs.DATASET.to_arg(), type=str, help="Which dataset to use.", default="allenai/c4",  choices=["allenai/c4"])
    # TODO Remove default, it makes no sense.
    parser.add_argument(config.ConfigArgs.DATASET_TRAIN_FILES.to_arg(), type=str, help="Which files to use for the dataset.", default="/mnt/nobackup/omicha1/c4/downloaded/multilingual/c4-en.tfrecord-00000-of-*.json.gz")
    # TODO Again, default makes no sense. Should be removed.
    parser.add_argument(config.ConfigArgs.DATASET_SPLIT.to_arg(), type=str, help="How to split the dataset (ex: train[:100])", default="train[:100]")
    parser.add_argument(config.ConfigArgs.DATASET_LOAD_NUM_PROC.to_arg(), type=int, help="Number of threads used to load the dataset.", default=20)
    parser.add_argument(config.ConfigArgs.TOKENIZE_NUM_PROCESS.to_arg(), type=int, help="Number of threads used to tokenize the dataset.", default=20)
    parser.add_argument(config.ConfigArgs.BATCH_SIZE.to_arg(), type=int, help="Size of batches", default=4)
    parser.add_argument(config.ConfigArgs.TRAIN_STATS.to_arg(), type=str, help="Type of statistics to gather. By default it is set to no-op, which ignores everything.", default="no-op", choices=["no-op", "no-op-sync", "simple", "torch-profiler", "codecarbon", "averaged-energy"])
    parser.add_argument(config.ConfigArgs.SWITCH_TRANSFORMER_NUM_EXPERTS.to_arg(), type=int, help="When the selected model a switch-base-n, sets the number of experts per sparse layer. It is recommended to only use powers of two.", default=8)
    parser.add_argument(config.ConfigArgs.QWEN_NUM_EXPERTS.to_arg(), type=int, help="When the selected model is qwen, sets the number of experts per sparse layer. It is recommended to only use powers of two.", default=8)
    # the following arguments are used for codecarbon tracking
    parser.add_argument(config.ConfigArgs.RUN_NUM.to_arg(), type=int, help="The run number used for codecarbon file tracking.", default=0)
    parser.add_argument(config.ConfigArgs.PROJECT_NAME.to_arg(), type=str, help="The name of the project used for codecarbon file tracking.", default="energy-efficiency")
    
    # The default is set to 1e-6 which is a good default rate for training Qwen models. 1e-7 is a good default rate for Switch-Transformers. Make sure to adjust it for different models
    parser.add_argument(config.ConfigArgs.LEARNING_RATE.to_arg(), type=float, help="The learning rate for training. It is used by the optimizer for both Switch Transformers and Qwen models.", default=1e-6)

    # For GPU throttling
    parser.add_argument(config.ConfigArgs.ENABLE_THROTTLING.to_arg(), action="store_true", help="Whether to use GPU throttling to reduce energy consumption at pre-selected points during training.")
    parser.add_argument(config.ConfigArgs.EXPERT_THROTTLING_PERFORMANCE_THRESHOLD.to_arg(), type=float, help="How much performance can be reduced by thottling. For example, a value of 0.05 means can throttling can increase computation time by at most 5%.", default=0.02)
    # (greta) testing unet3d and throttling
    parser.add_argument(config.ConfigArgs.THROTTLE_TYPE.to_arg(), type=str, help="Which pass to throttle.", choices=["forward", "backward", "optimizer", "all_fixed", "dym_best"], default="optimizer")
    parser.add_argument(config.ConfigArgs.THROTTLE_FREQUENCY.to_arg(), type=int, help="The frequency to set the GPU to during the fixed pass when throttling is enabled.", default=1305)

    # return parser.parse_args()
    # Using parse_known_args since deepspeed adds the flag --local-rank, which is currently unhandled
    args, _ = parser.parse_known_args()
    return args

def main():
    args = get_args()

    conf = config.Config(args)

    model_trainer, model_kwargs = process_conf(conf)

    if model_trainer is None:
        print("No model trainer, unet3d testing, exiting.")
    else:
        model_trainer.train(model_kwargs)
    del conf
    del model_kwargs
    del model_trainer

if __name__ == "__main__":
    main()
    gc.collect()

