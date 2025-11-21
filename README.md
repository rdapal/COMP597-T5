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
├── energy_efficiency
│   ├── src
│   │   ├── config
│   │   │   └── config.py                   # configurations file with user given arguments
│   │   ├── models
│   │   │   ├── gpt2
│   │   │   │   ├── __init__.py
│   │   │   │   └── gpt2.py                 # gpt2 model simple trainer example
│   │   ├── trainer
│   │   │   ├── stats
│   │   │   │   ├── codecarbon.py           # trainer stats to collect codecarbon information with losses
│   │   │   │   └── ...
│   │   │   ├── base.py                     # abstract methods 
│   │   │   ├── simple.py                   # simple trainer 
│   │   │   └── ...
│   ├── launch.py
│   ├── start-gpt2.sh                       # script to easily start gpt2 
│   └── ...
└── ...
```

TODO: add section for setup and installations
#### environment setup

To ensure dependencies are installed, we will use a Conda environment. 

1. **Setting up storage** <br> Your home directory on the McGill server is part of a network file system where users get limited amounts of storage. You can check your storage usage and how much you are allowed to use using the command `quota`. Python packages, pip's cache, Conda's cache and datasets can use quite a bit of storage, so we need to ensure they are stored outside your directory to avoid any issues with disk quotas. Say you have your own directory, stored in `SOME_PATH`, on a server that is not part of the network file system (hence not affected by disk quotas). Of course, you should replace `SOME_PATH` with the appropriate path for any command below that uses it. The steps to go around the disk quota are as follows:
    1. We can make a cache directory using `mkdir SOME_PATH/cache`. 
    2. For pip's cache, we can redirect it to that directory using `export PIP_CACHE_DIR=SOME_PATH/cache/pip`. 
    3. For Hugging Face datasets, we can use `export HF_HOME=SOME_PATH/cache/huggingface`. 
    4. For Conda, we will give it its own directory using `mkdir SOME_PATH/conda`. Then we can configure downloads to happens there using `conda config --add pkgs_dirs SOME_PATH/conda/pkgs`, which will update the `~/.condarc` configuration file.
2. **Initializing Conda** <br> If you have never used Conda with this user, you need to initialize Conda with `conda init bash`. This modifies the `~/.bashrc` file. Unfortunately, the `~/.bashrc` file is not always executed at login, depending on the server configurations. For that reason, it is recommended to run `. ~/.bashrc` before running any Conda commands. 
3. **Verifying the installation environment** <br> Run the command `conda --version`. If the version is at least 23.0, skip to the next step. Now we need to verify that mamba is available. To do so, run `conda list conda-libmamba-solver`. If the output contains a version for mamba, skip to the next step. If you haven't skipped to the next step, follow the instructions below.
    1. Create a base environment using `conda env create --prefix SOME_PATH/conda/envs/COMP597-base --file energy_efficiency/base-environment.yaml`. 
    2. Activate the newly created environment using `conda activate SOME_PATH/conda/envs/COMP597-base`.
    3. Create the following environment variable: `export CONDA_BASE_ENV_PATH=SOME_PATH/conda/envs/COMP597-base`.
    4. You will need to make sure you use the Conda version installed in the environment you just created. For any `conda` command in step 4, use `${CONDA_BASE_ENV_PATH}/bin/conda` instead of `conda`.
4. **Creating the project environment** <br> You can now simply run `conda env create --prefix SOME_PATH/conda/envs/COMP597-project --file energy_efficiency/environment.yaml` to create the environment. You can use your environment by activating it with `conda activate SOME_PATH/conda/envs/COMP597-project`. 
5. **Using the environment** <br> For any future use of the environment, you can create a script, let's name it `env.sh`, which will contain the configuration to set up the environment. You can then execute the script with `. env.sh` to set up activate your environment. The script would look like this:
    ```
    #!/bin/bash
    
    . ~/.bashrc
    conda activate SOME_PATH/conda/envs/COMP597-project
    export PIP_CACHE_DIR=SOME_PATH/cache/pip
    export HF_HOME=SOME_PATH/cache/huggingface
    ```
6. **Quitting** <br> If you want to quit the environment, or reset your sheel to before you activate the environment, simply run `conda deactivate`.

#TODO:change the paths from /mnt to absolute paths
- setting up the environment on the server:
    - `cd /mnt/nobackup`
    - `mkdir <dirname>` 
    - `cd /mnt/nobackup/<dirname>`
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
    - `conda create --prefix=/mnt/nobackup/<dirname>/conda/<envname> python=3.14`
    - `source .bashrc` before trying to activate the conda env
    - `cp /mnt/nobackup/omicha1/COMP597-starter-code/energy_effiency/env.sh /mnt/nobackup/<dirname>/` and modify necessary names
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
    - once you have cloned the project `COMP597-starter-code`, `cd COMP597-starter-code` and install the requirements by `pip install -r requirements.txt`
    - to disconnect from the environment: `conda deactivate` or simply disconnect from the server

TODO: add section for resources
### CodeCarbon Resources
- [olivier-tutorial-code-carbon](https://colab.research.google.com/drive/1eBLk-Fne8YCzuwVLiyLU8w0wNsrfh3xq)
- [laura-documentation](https://docs.google.com/document/d/1GSxPYXRVjkb1eSwnZZBsSQMVICRLdM_pM-iEVeivAIE/edit)
- [laura documentation](https://docs.google.com/document/d/1Ihfniv1CaWz79tO4IcXx3JG7pAZDIGWigAMDKiVTNDc/edit)

TODO: add section for how to run experiments, how to edit files to add a new model etc.

---
