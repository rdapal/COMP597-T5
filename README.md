# COMP597-starter-code
This repository contains starter code for COMP597: Responsible AI - Energy Efficiency analysis using CodeCarbon. 
TODO: add more description on for course description, project description and instructions on the project.
### Course Description
TODO: course description.

### Project Description
TODO: project description.

### Instructions
TODO: instructions for the project. eg:
1. Set up your environment using the provided instructions below under [Environment Setup](#environment-setup).
2. Familiarize yourself with the CodeCarbon library and its usage. Resources can be found in the [CodeCarbon Resources](#codecarbon-resources) section.
3. Implement your language/vision/other model and run experiements to collect data.
4. Document your process and findings in a report.

---

TODO: add a section for repository structure. with a tree and description of important folders and files.

Structure: #TODO
```
COMP597-starter-code
.
├── README.md                               # instructions and description of the project
├── requirements.txt                        # list of required packages to install
├── env.sh                                  # script to setup the conda environment variables
├── .gitignore                              # gitignore file to exclude unnecessary files
├── ...
├── energy_effiency
│   ├── codecarbonlogs                      # folder containing all codecarbon logs from running codecarbon scripts
│   ├── codecarbon_scripts
│   │   ├── codecarbon_datacollect.sh       # script that runs and collects codecarbon information for switch-transformers (simple, custom, fmoe, 
│   │   │                                     deepspeed) + qwen (simple, deepspeed, fmoe) with different parameters
│   │   ├── tmp_fmoe_datacollection.sh      # script that runs qwen/switch with fmoe and creates graphs 
│   │   ├── codecarbon_cleanup.sh           # script that cleans up the log files and merges them after collecting codecarbon data
│   │   ├── codecarbon_pipeline.sh          # script that pipelines the process of data collection, cleanup and plotting
│   │   ├── plot_losses.sh                  # script that runs the loss plotting
│   │   ├── plot_losses.py                  # python code for loss plotting
│   │   └── plots.sh                        # script that creates graphs for the codecarbon collected data (LAURA)
│   ├── src
│   │   ├── config
│   │   │   └── config.py                   # configurations file with user given arguments
│   │   ├── models
│   │   │   ├── qwen
│   │   │   │   ├── __init__.py
│   │   │   │   ├── qwen_small.py           # qwen3moe implementation with deepspeed, fmoe, simple
│   │   │   │   └── structures.txt          # qwen-moe and switch transformers configuration structures + collected data for memory capacity qwen3moe
│   │   ├── trainer
│   │   │   ├── stats
│   │   │   │   ├── codecarbon.py           # trainer stats to collect codecarbon information with losses
│   │   │   │   └── ...
│   │   │   ├── base.py                     # abstract methods + added loss tracking function for more data (log_loss())
│   │   │   ├── distributed_base.py
│   │   │   ├── deepspeed.py
│   │   │   └── ...
│   ├── launch.py
│   ├── start-qwen.sh                       # script to easily start qwen using deepspeed
│   └── ...
└── ...
```

TODO: add section for setup and installations
#### environment setup
- setting up the environment on the server:
    - `cd mnt/nobackup`
    - `mkdir <dirname>` 
    - `cd /mnt/nobackup/omicha1`
    - `conda init bash` and check by `cd` and `cat .bashrc` should have
    ```
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/usr/local/pkgs/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/usr/local/pkgs/anaconda/etc/profile.d/conda.sh" ]; then
            . "/usr/local/pkgs/anaconda/etc/profile.d/conda.sh"
        else
            export PATH="/usr/local/pkgs/anaconda/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    ```
    - `conda create --prefix=/mnt/nobackup/<dirname>/conda/<envname> python=3.12`
    - `source .bashrc` before trying to activate the conda env
    - `cp /mnt/nobackup/omicha1/msc-research-exploration/energy_effiency/env.sh /mnt/nobackup/<dirname>/` and modify necessary names
    ```
    #!/bin/bash

    . ~/.bashrc

    conda activate /mnt/nobackup/<dirname>/conda/<envname>/

    # This prevents HuggingFace from using ~/.cache which would fill up my disk quota
    export HF_HOME="/mnt/nobackup/<dirname>/cache/huggingface/datasets"

    # This prevents pip from using ~/.cache which would fill up my disk quota
    export PIP_CACHE_DIR="/mnt/nobackup/<dirname>/cache/pip"
    ```
    - `mkdir cache` (from /mnt/nobackup/<dirname>/)
    - from now on, on login to the server, simply `cd /mnt/nobackup/<dirname>` and `. env.sh` to activate the conda environment
    - once you have cloned the project `msc-research-exploration`, `cd msc-research-exploration` and install the requirements by `pip install -r requirements.txt`
    - to disconnect from the environment: `conda deactivate` or simply disconnect from the server
- to run the deepspeed model and Olivier code:
    - `cd /mnt/nobackup/omicha1/msc-research-exploration/energy_effiency` and run `python3 launch.py --model switch-base-8` for switch-base-8 code and `./start-deepspeed.sh --train_stats simple` to start the deepspeed model
    - double check before logging out: `nvidia-smi`

TODO: add section for resources
### CodeCarbon Resources
- [olivier-tutorial-code-carbon](https://colab.research.google.com/drive/1eBLk-Fne8YCzuwVLiyLU8w0wNsrfh3xq?usp=sharing#scrollTo=rUCBJMaymHw0)
- [laura-documentation](https://docs.google.com/document/d/1GSxPYXRVjkb1eSwnZZBsSQMVICRLdM_pM-iEVeivAIE/edit?tab=t.0#heading=h.br99yzi307vk)
- [laura documentation](https://docs.google.com/document/d/1Ihfniv1CaWz79tO4IcXx3JG7pAZDIGWigAMDKiVTNDc/edit?tab=t.xsnyx8yl1r62)

TODO: add section for how to run experiments, how to edit files to add a new model etc.

---