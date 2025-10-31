import matplotlib.pyplot as plt 
import experiment
import pandas as pd 
import re
import os

def prepare_data_as_dict(filepath, granularity, label):
    # Load original csv file as a dataframe
    # Add new column with iteration number; extract iteration number from task name, e.g. '3' from 'Forward pass #3' or 'Step #3'
    df = pd.read_csv(filepath, usecols = ['task_name', 'duration', 'emissions', 'energy_consumed', 'gpu_rank'])
    df[['iteration']] = df['task_name'].str.extract(r'#(\d+)').astype(int) #convert to int
    
    # Define key parameters
    rank_list = sorted(df['gpu_rank'].unique()) #use sorted() to return a new list
    num_iterations = len(sorted(df['iteration'].unique()))
    metrics = ['duration', 'emissions', 'energy_consumed']

    # Populate dictionary
    if granularity == "substep":
        '''
        ASSUMPTIONS:
        * CSV file is ordered by rank, then by iteration, then in the order Forward pass > Backward pass > Optimiser step
        * There are no missing values; each iteration has three substeps (forward, backward, and optimiser); iteration numbers are consecutive and start at 1
        * CSV's task_name column is in the format 'Forward pass #3', 'Backward pass #3', 'Optimiser step #3', etc
        '''

        # Extract data from df and save it in a dictionary:
        # data[metric][rank] = [[], [], ...] (i.e., a list of [Forward, Backward, Optimiser] values, one per iteration)
        data = {
                metric: {rank: [[] for _ in range(num_iterations)] for rank in rank_list}
                for metric in metrics
                }

        for row in df.itertuples(index=False):
            data['duration'][row.gpu_rank][row.iteration-1].append(row.duration)
            data['emissions'][row.gpu_rank][row.iteration-1].append(row.emissions)
            data['energy_consumed'][row.gpu_rank][row.iteration-1].append(row.energy_consumed)

    else: # granularity == step
        '''
        ASSUMPTIONS:
        * CSV file is ordered by rank, then by iteration
        * There are no missing values; iteration numbers are consecutive and start at 1
        * CSV's task_name column is in the format 'Step #1', 'Step #2', etc
        '''

        # Extract data from df and save it in a dictionary:
        # data[metric][rank] = [] (i.e., a list of the value recorded for each iteration)
        data = {
                metric: {rank: [] for rank in rank_list}
                for metric in metrics
                }

        for row in df.itertuples(index=False):
            data['duration'][row.gpu_rank].append(row.duration)
            data['emissions'][row.gpu_rank].append(row.emissions)
            data['energy_consumed'][row.gpu_rank].append(row.energy_consumed)

    # Wrap data in a dictionary with metadata
    exp = add_experiment_metadata(data, label, num_iterations, rank_list)
    return exp

def add_experiment_metadata(data_dict, label, num_iterations, rank_list):
    '''
    ASSUMPTIONS:
    * label has format: '<model_name>_<trainer_type>_<num_experts>_<batch_size>_<num_gpus>' e.g. 'qwen_moe_deepspeed_60_2_4'
    '''
    pattern = r'^(.+)_([^_]+)_(\d+)_(\d+)_(\d+)$'
    match = re.match(pattern, label)
    if not match:
        raise ValueError(
            f"Label '{label}' does not match expected format: <model_name>_<trainer_type>_<num_experts>_<batch_size>_<num_gpus>")

    dict_with_metadata = {
        'model_name': match.group(1),
        'trainer_type': match.group(2),
        'num_experts': int(match.group(3)),
        'batch_size': int(match.group(4)),
        'num_gpus': int(match.group(5)),
        'num_iterations': num_iterations,
        'rank_list': rank_list,
        'data': data_dict,
    }
    return dict_with_metadata

def get_unit(metric): 
    unit_map = {
            "duration": "s",
            "energy_consumed": "kWh",
            "emissions": "kg"
            }
    
    return unit_map.get(metric)

def get_abbreviated_trainer_name(full_name): 
    abbrev_map = {
            "custom": "c",
            "simple": "s",
            "fmoe": "f",
            "deepspeed": "d"
            }

    return abbrev_map.get(full_name)
