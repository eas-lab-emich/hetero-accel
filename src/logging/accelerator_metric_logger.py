from collections import defaultdict

from src.evaluation_result import EvaluationResult
from src.logging.metric_logger import MetricLogger
from src.optimization.scheduling import Schedule


class AcceleratorMetricLogger(MetricLogger):
    def __init__(self, base_dir):
        super().__init__(base_dir, "accelerator_metrics.csv")
        self._write_row(
            ["step",
             "is_improved",
             "sim_temperature",
             "energy",
             "latency",
             "edp",
             "area",
             "scheduled_dnns",
             "evaluation_result"])

    @staticmethod
    def _parse_scheduled(schedule: Schedule):
        if not schedule:
            return None
        scheduled_dnns = defaultdict(int)
        for entry in schedule.entries:
            scheduled_dnns[entry.bin.precision] += 1
        return ";".join(f"{k}:{v}" for k, v in sorted(scheduled_dnns.items()))

    def log(self, *, iteration, is_improved, sim_temperature, energy, latency, edp, area, scheduled: Schedule, evaluation_result: EvaluationResult):
        self._check_closed()
        self._write_row([
            iteration,
            self._format_bool(is_improved),
            self._format_float(sim_temperature),
            self._format_float(energy),
            latency,
            self._format_sci_notation(edp),
            self._format_float(area),
            self._parse_scheduled(scheduled),
            evaluation_result.value
        ])
