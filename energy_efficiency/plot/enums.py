from enum import Enum 

class Trainer(Enum): 
    CUSTOM = ("custom", "c")
    SIMPLE = ("simple", "s")
    DEEPSPEED = ("deepspeed", "d")
    FMOE = ("fmoe", "f")

    def __init__(self, label, abbrev): 
        self.label = label
        self.abbrev = abbrev

    def get_abbrev(self, throttle=False):
        abbrev = self.abbrev
        if throttle: 
            return abbrev + "-th"
        return abbrev
    
    def get_label(self): 
        return self.label


class Model(Enum): 
    SWITCH = ("switch", "sw")
    QWEN_MOE = ("qwen_moe", "qw")

    def __init__(self, label, abbrev): 
        self.label = label 
        self.abbrev = abbrev

    def get_abbrev(self, throttle=False):
        abbrev = self.abbrev
        if throttle: 
            return abbrev + "-th"
        return abbrev

    def get_label(self):
        return self.label


class PerformanceMetric(Enum): 
    DURATION = ("duration", "s", "duration")
    ENERGY_CONSUMED = ("energy consumed", "kWh", "energy_consumed")
    EMISSIONS = ("emissions", "kg", "emissions")


    def __init__(self, label, unit, col_name): 
        self.label = label
        self.unit = unit
        self.col_name = col_name

    def get_label(self):
        return self.label

    def get_unit(self): 
        return self.unit

    def get_col_name(self): 
        return self.col_name


'''

TODO LATER
Allow different x-axis progress variables beyond training iterations. For example: num tokens processed, time elapsed
Skeleton enum:

class ProgressVariable(Enum):
    ITERATIONS = ("", "training iterations")
    NUM_TOKENS_PROCESSED = ("", "number of tokens processed")

''' 
   




