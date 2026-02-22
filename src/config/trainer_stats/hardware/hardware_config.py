from src.config.util.base_config import _Arg, _BaseConfig

config_name = "hardware"


class TrainerStatsConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(
            type=str,
            help="Directory to save hardware metrics output CSV and JSON files",
            default="./hardware_stats",
        )
        self._arg_run_id = _Arg(
            type=str,
            help="Unique run identifier. Auto-generated timestamp if not provided.",
            default=None,
        )
        self._arg_carbon_intensity = _Arg(
            type=float,
            help=(
                "Carbon intensity of the electricity grid in gCO2eq/kWh. "
                "Quebec (Hydro-Quebec) ≈ 30 gCO2eq/kWh. "
                "Canada average ≈ 110. US average ≈ 390. "
                "Used to estimate CO2 emissions from energy measurements."
            ),
            default=30.0,
        )
