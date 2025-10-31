import enum

class _TrainingComponentsAutoName(enum.Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name

@enum.unique
class TrainingComponents(_TrainingComponentsAutoName):
    STEP = enum.auto()
    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    OPTIMIZER_STEP = enum.auto()
    LOSS_ACCUMULATION = enum.auto()
    GRADIENTS_ALL_REDUCE = enum.auto()
    SAVE_CHECKPOINT = enum.auto()

    ALL_FIXED = enum.auto()

