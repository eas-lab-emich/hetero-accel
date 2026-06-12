import numpy as np

from src.optimization.scheduling import Schedule


class SchedulePenalizer:

    def __init__(self, accuracy_lut):
        self.lambda_p1 = 1.2
        self.lambda_p2 = 1.04
        self.lambda_p3 = 1.1
        self.window_size = 4
        self.accuracy_lut = accuracy_lut
        self.schedule_history = []
        self.baseline_precision = 8
        self.risk_threshold = 12  # 12% accumulated accuracy loss vs baseline

    def penalize(self, schedule: Schedule):
        self.schedule_history.append(schedule)
        risk = self.__aggregate_loss(schedule)
        print(f"step={len(self.schedule_history) - 1}, risk={risk}")
        return (self.lambda_p1 * self.__compute_p1(self.schedule_history)
                + self.lambda_p2 * self.__compute_p2(self.schedule_history)
                + self.lambda_p3 * self.__compute_p3(self.schedule_history))

    def __compute_p1(self, schedule_history):
        schedules = schedule_history[:-self.window_size + 1]
        accumulated_risk = 0
        for start_idx in range(len(schedules)):
            window_sum = 0
            windowed = schedule_history[start_idx: start_idx + self.window_size]
            for schedule in windowed:
                window_sum += self.__aggregate_loss(schedule)
            if window_sum > self.risk_threshold:
                accumulated_risk += window_sum - self.risk_threshold
        return accumulated_risk

    def __compute_p2(self, schedule_history):
        streak = 0
        max_streak = 0

        for schedule in schedule_history:
            if self.__bad(schedule):
                streak += 1
            else:
                streak = 0

            if streak > max_streak:
                max_streak = streak
        return max(0, max_streak - self.window_size)

    def __bad(self, schedule):
        return self.__aggregate_loss(schedule) > self.risk_threshold

    def __compute_p3(self, schedule_history):
        c = 1e17  # huge c to signal annealing there's a huge problem
        enforced_precision = 8
        window = self.window_size
        if len(schedule_history) < window:
            return 0
        lookback = schedule_history[-window:]
        for s in lookback:
            if not s or not s.assigned:
                continue
            for a in s.assigned.values():
                if a.precision == enforced_precision:
                    return 0
        return c

    def __aggregate_loss(self, schedule):
        losses = []
        for (network, assignment) in schedule.assigned.items():
            assignment_precision = assignment.precision
            assignment_accuracy = self.__network_accuracy(network, assignment_precision)
            baseline_accuracy = self.__network_accuracy(network, self.baseline_precision)
            loss = baseline_accuracy - assignment_accuracy
            losses.append(loss)
        if not losses: # if somehow we get here and have no assignments
            return 0
        return np.mean(losses)

    def __network_accuracy(self, network, precision):
        return self.accuracy_lut.loc[
            (self.accuracy_lut["Network"] == network) &
            (self.accuracy_lut["QuantBits"] == precision),
            "Accuracy"
        ].iloc[0]
