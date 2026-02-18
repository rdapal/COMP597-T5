from src.config.util.base_config import _Arg, _BaseConfig

config_name = "hardware"

class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(type=str, help="Directory to save hardware metrics CSV and JSON files.", default="./hardware_stats")
        self._arg_run_id = _Arg(type=str, help="Unique run identifier. Auto-generated timestamp if not provided.", default=None)
