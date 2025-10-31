import os
import subprocess
import src.config as config

def unet3d_mlcommons_init(conf: config.Config, dataset=None):
    """
    Launches the MLCommons UNet3D training script with specified parameters.
    """
    # docker?
    in_docker = os.path.exists("/.dockerenv")

    # path adjust
    if in_docker:
        data_dir = "/data"  # mounted volume inside container
        results_dir = "/results"
    else:
        data_dir = "/raid/data/imseg/raw-data/kits19/preproc-data"
        results_dir = "/mnt/nobackup/gzu/tmp/data/imseg/raw-data/kits19/results"

    os.makedirs(results_dir, exist_ok=True)

    # MLCommons UNet3D training script
    unet3d_main_path = os.path.join(os.getcwd(), "src/models/unet3d_mlcommons/pytorch/main.py")

    # MLCommons training
    # TODO: add to conf
    cmd = [
        "python", unet3d_main_path,
        "--data_dir", data_dir,
        "--epochs", "2",  
        "--evaluate_every", "20",
        "--start_eval_at", "1000",
        "--quality_threshold", "0.908",
        "--batch_size", str(conf.batch_size),
        "--optimizer", "sgd",
        "--ga_steps", "1",
        "--learning_rate", str(conf.learning_rate),
        "--seed", str(conf.run_num if conf.run_num != 0 else -1),
        "--lr_warmup_epochs", "200"
    ]

    print(f"\nRunning MLCommons UNet3D:\n{' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    return None, None
