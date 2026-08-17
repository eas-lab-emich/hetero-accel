from src.evaluation_result import EvaluationResult
from src.logging.metric_logger import MetricLogger


class SubacceleratorParamsLogger(MetricLogger):
    def __init__(self, base_dir):
        super().__init__(base_dir, "subaccelerator_params_metrics.csv")
        self._write_row(
            ["step",
             "is_improved",
             "is_accepted",
             "precision",
             "pe_array_x",
             "pe_array_y",
             "sram_size",
             "ifmap-spad_size",
             "weights_spad_size",
             "psum_spad_size",
             "evaluation_result"])

    def log(self, *, iteration, is_improved, is_accepted, precision, pe_array_x, pe_array_y,
            sram_size, ifmap_spad_size, weights_spad_size, psum_spad_size, evaluation_result: EvaluationResult):
        self._check_closed()
        self._write_row([
            iteration,
            self._format_bool(is_improved),
            self._format_bool(is_accepted),
            precision,
            pe_array_x,
            pe_array_y,
            sram_size,
            ifmap_spad_size,
            weights_spad_size,
            psum_spad_size,
            evaluation_result.value
        ])
