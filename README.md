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

### Repository Structure: 
```
COMP597-starter-code
.
├── energy_efficiency
│   ├── src
│   │   ├── config                          # Configuration related files                     
│   │   ├── models
│   │   │   ├── gpt2
│   │   │   └── ...                 
│   │   ├── trainer                         
│   │   │   ├── stats                       # Stats collection for trainers
│   │   │   │  ├── base.py 
│   │   │   │  └── ...
│   │   │   ├── base.py                     # Trainer base class
│   │   └── ...   
│   ├── launch.py                           # Main script to launch training experiments                           
│   ├── requirements.txt                        
│   └── start-gpt2.sh
├── .gitignore
├── env_setup.sh                            # Script to setup the conda environment                               
└── README.md
```

#### Environment setup

We will use a Conda envrionment to install the required dependencies. The steps below will walk you through the steps. A setup script `env_setup.sh` is also provided and will execute all the steps below given as input the path `SOME_PATH` as described in step one below.

1. **Setting up storage** <br> Your home directory on the McGill server is part of a network file system where users get limited amounts of storage. You can check your storage usage and how much you are allowed to use using the command `quota`. Python packages, pip's cache, Conda's cache and datasets can use quite a bit of storage, so we need to ensure they are stored outside your directory to avoid any issues with disk quotas. Say you have your own directory, stored in `SOME_PATH`, on a server that is not part of the network file system (hence not affected by disk quotas). Use `export BASE_STORAGE_PATH=SOME_PATH` where you replace `SOME_PATH` with the actual path. The steps to go around the disk quota are as follows:
    1. We can make a cache directory using `mkdir ${BASE_STORAGE_PATH}/cache`. 
    2. For pip's cache, we can redirect it to that directory using `export PIP_CACHE_DIR=${BASE_STORAGE_PATH}/cache/pip`. 
    3. For Hugging Face datasets, we can use `export HF_HOME=${BASE_STORAGE_PATH}/cache/huggingface`. While this variable is not strictly needed for the environment set up, it is needed when using the Hugging Face datasets module.
2. **Initializing Conda** <br> If you have never used Conda with this user, you need to initialize Conda with `conda init bash`. This modifies the `~/.bashrc` file. Unfortunately, the `~/.bashrc` file is not always executed at login, depending on the server configurations. For that reason, it is recommended to run `. ~/.bashrc` before running any Conda commands. 
3. **Creating the project environment** <br> First, let's make sure to create the directory to store the environment using `mkdir -p ${BASE_STORAGE_PATH}/conda/envs`. You can now simply run `conda create --prefix ${BASE_STORAGE_PATH}/conda/envs/COMP597-project python=3.14` to create the environment. 
4. **Activating the environment** <br> You can use your environment by activating it with `conda activate ${BASE_STORAGE_PATH}/conda/envs/COMP597-project`. 
5. **Installing dependencies** <br> The dependencies are provided as a requirements file. You can install them using `pip install -r energy_efficiency/requirements.txt`.
6. **Using the environment** <br> For any future use of the environment, you can create a script, let's name it `local_env.sh`, which will contain the configuration to set up the environment. You can then execute the script with `. local_env.sh` to set up activate your environment. The script would look like this (where you need to replace `SOME_PATH`):
    ```
    #!/bin/bash
    
    . ~/.bashrc
    conda activate SOME_PATH/conda/envs/COMP597-project
    export PIP_CACHE_DIR=SOME_PATH/cache/pip
    export HF_HOME=SOME_PATH/cache/huggingface
    ```
7. **Quitting** <br> If you want to quit the environment, or reset your sheel to before you activate the environment, simply run `conda deactivate`.

TODO: add section for resources
### CodeCarbon Resources
- [olivier-tutorial-code-carbon](https://colab.research.google.com/drive/1eBLk-Fne8YCzuwVLiyLU8w0wNsrfh3xq)
- [laura-documentation](https://docs.google.com/document/d/1GSxPYXRVjkb1eSwnZZBsSQMVICRLdM_pM-iEVeivAIE/edit)
- [laura documentation](https://docs.google.com/document/d/1Ihfniv1CaWz79tO4IcXx3JG7pAZDIGWigAMDKiVTNDc/edit)

TODO: add section for how to run experiments, how to edit files to add a new model etc.


### GPT2 example
#### How to setup a new model (GPT2)
Files to edit/add:
- Add a new model under the models directory, `energy_efficiency/src/models/gpt2/gpt2.py` : contains the GPT2 model definition, optimizer initialization, and trainer setup.
- Create a bash script to run the experiments (optional), `energy_efficiency/start-gpt2.sh` : script to launch experiments with GPT2 model.
- Edit the main launch file to add the new model, `energy_efficiency/launch.py` : add the model choice in the argument parser.
- Edit the configuration file to add any model-specific configurations, `energy_efficiency/src/config/config.py`.
- Edit the requirements file if new dependencies are needed, `energy_efficiency/requirements.txt`.
- Add any additional files as needed for data processing, evaluation, etc.
- Add trainer objects and/or trainer stats if needed under `energy_efficiency/src/trainer/`.

Setting up a model - GPT2 example:
1. Find and setup a tokenizer from Hugging Face transformers. Make adjustements as needed to make it compatible with your dataset.
2. Find and setup an optimizer. Make sure to set the learning rate from the configuration object.
3. Setup data processing using the tokenizer and dataset.
4. Setup the model using data collator, a config from the model and a model (do not take the pretrained one).
5. implement the trainer setup function. You can start with a simple trainer as shown in the example. You can implement more complex trainers as needed.
6. Initialize the model and add it to the [init file](energy_efficiency/src/models/__init__.py) in the model factory. Make sure to add the model choice in the launch file argument parser and add any needed arguments to the configuration file.

#### How to run experiments with GPT2
Running experiments using launch.py - TODO: see laura documentation !!!

Example commands to run experiments with GPT2 can be found in the `energy_efficiency/start-gpt2.sh` script.

To run the model with codecarbon tracking, make necessary modifications to the codecarbon trainer stats and run the experiments as shown in the script.
Add any other trainer stats objects as needed and run experiments accordingly.

---
