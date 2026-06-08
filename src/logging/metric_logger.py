import csv
from abc import ABC
from pathlib import Path

from src.exception.invalid_state import InvalidStateException


class MetricLogger(ABC):
    def __init__(self, base_dir, file_name):
        path = Path(base_dir, file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8")  # disable translation of newlines
        self.writer = csv.writer(self.file)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _check_closed(self):
        if self.closed:
            raise InvalidStateException("Logger already closed")

    def close(self):
        if not self.closed:
            self.file.close()
            self.closed = True

    def _write_row(self, row):
        self.writer.writerow(row)
        self.file.flush()

    @staticmethod
    def _format_float(to_format):
        return None if to_format is None else f"{to_format:.4f}"

    @staticmethod
    def _format_sci_notation(to_format):
        return None if to_format is None else f"{to_format:.6g}"

    @staticmethod
    def _format_bool(to_format):
        if to_format is None:
            return None
        return "True" if to_format else "False"
