import os 
import re 
import pandas as pd

'''
Attributes:
    model : String
    trainer : String
    granularity: String
    num_experts : int
    batch_size : int
    num_gpus : int
    num_iterations : int
    ranks : int list
    df: pandas dataframe contaiing the the training data
'''

class BaseExperiment: 
    def __init__(self, filepath, label):
        self._init_metadata(filepath, label)
        self._init_data(filepath)

    def _init_metadata(self, filepath, label):
        # Extract metadata from label
        label_pattern = r'^(.+)_([^_]+)_([^_]+)_(\d+)_(\d+)_(\d+)$'
        label_match = re.match(label_pattern, label)
        if not label_match:
            raise ValueError(
                f"Label '{label}' does not match expected format: <model_name>_<trainer_type>_<granularity>_<num_experts>_<batch_size>_<num_gpus>")
        self.model = label_match.group(1)
        self.trainer = label_match.group(2)
        self.granularity = label_match.group(3)
        self.num_experts = int(label_match.group(4))
        self.batch_size = int(label_match.group(5))
        self.num_gpus = int(label_match.group(6))

        # Extract metadata from file
        df = pd.read_csv(filepath, usecols = ['task_name', 'gpu_rank'])
        df[['iteration']] = df['task_name'].str.extract(r'#(\d+)').astype(int)
        self.ranks = sorted(df['gpu_rank'].unique()) #use sorted() to return a new list
        self.num_iterations = len(sorted(df['iteration'].unique()))
    
    # TODO: instead of always using these columns, adapt to the metrics passed in by the user
    def _init_data(self, filepath):
        cols = ['task_name', 'duration', 'emissions', 'energy_consumed', 'gpu_rank', 'loss']
        df = pd.read_csv(filepath, usecols = lambda x: x in cols) 
        df[['iteration']] = df['task_name'].str.extract(r'#(\d+)').astype(int) #convert to int
        #TODO: this does not work with "STEP #", modify so that it does
        df['task_name'] = df['task_name'].str.extract(r'(Forward|Backward|Optimiser|Step)')
        self.df = df
   

class StepExperiment(BaseExperiment): 
    def __init__(self, filepath, label):
        super().__init__(filepath, label)

    def get_cumulative_series_df(self, rank, metric): 
        attribute = f"cumulative_{metric}_df"
        
        if not hasattr(self, attribute): 
            df = self.df.sort_values(['gpu_rank', 'iteration'])
            df[f"cumulative_{metric.get_col_name()}"] = df.groupby('gpu_rank')[metric.value].cumsum() # Sum cumulated over iterations, independent for each rank
            df = df[[f"cumulative_{metric.get_col_name()}", "gpu_rank", "iteration"]]
            setattr(self, attribute, df)
        
        df = getattr(self, attribute)
        return df[[df['gpu_rank'] == rank]]

    def get_smoothed_loss(self, ): 
        

class SubstepExperiment(BaseExperiment): 
    def __init__(self, filepath, label):
        super().__init__(filepath, label)

    def get_stacked_components(self, rank, iteration, metric):
        # Returns [Forward, Backward, Optimiser] numbers in a stable order
        filtered_df = self.df[(self.df["gpu_rank"] == rank) & (self.df["iteration"] == iteration)]
        components = []
        for task_name in ['Forward','Backward','Optimiser']:
            components.append(float(filtered_df.loc[filtered_df['task_name']==task_name, metric.value]))
        return components

