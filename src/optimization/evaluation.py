from src.optimization.scheduling import Schedule


class SchedulePenalizer:

    def __init__(self):
        self.schedule_history = []
        self.lambda_p3 = 1.1

    def penalize(self, schedule: Schedule):
        self.schedule_history.append(schedule)
        return self.lambda_p3 * self.compute_p3(self.schedule_history)

    @staticmethod
    def compute_p3(schedule_history):
        c = 1e17  # huge c to signal annealing there's a huge problem
        lookback_window = 4
        enforced_precision = 8
        if len(schedule_history) < lookback_window:
            return 0
        lookback = schedule_history[-lookback_window:]
        for s in lookback:
            if not s or not s.assigned:
                continue
            for a in s.assigned.values():
                if a.precision == enforced_precision:
                    return 0
        return c
