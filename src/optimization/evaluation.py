from collections import namedtuple
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.evaluation_result import EvaluationResult
from src.optimization.scheduling import Schedule

Penalty = namedtuple('Penalty',
                     ['total_penalty', 'aggregate_accuracy_loss',
                      'p1', 'p2', 'p3', 'lambda_1', 'lambda_2', 'lambda_3', 'window', 'risk_threshold'])


class StepResult(NamedTuple):
    edp: float
    penalty: float
    accepted: bool
    improved: bool
    schedule: Schedule
    evaluation_result: EvaluationResult


class SchedulePenalizer:

    def __init__(self, accuracy_lut):
        self.lambda_p1 = 0.015
        self.lambda_p2 = 0
        self.lambda_p3 = 0
        self.window_size = 3
        self.accuracy_lut = accuracy_lut
        self.baseline_precision = 8
        self.risk_threshold = 8

    def penalize(self, e, latest_schedule: Schedule, history, enabled) -> Penalty:
        p1 = self.__compute_p1(latest_schedule, history) if latest_schedule and latest_schedule.assigned else 0 # FIXME across components, really should be setting penalty to inf rather than edp
        # p2 = self.__compute_p2(self.schedule_history)
        p2 = 0
        # p3 = self.__compute_p3(latest_schedule)
        p3 = 0
        total_penalty = (self.lambda_p1 * e * p1 + self.lambda_p2 * e * p2 + self.lambda_p3 * e * p3) if enabled else 0
        return Penalty(total_penalty, self.__aggregate_loss(latest_schedule),
                       p1, p2, p3,
                       self.lambda_p1, self.lambda_p2, self.lambda_p3,
                       self.window_size, self.risk_threshold)

    def __compute_p1(self, latest_schedule, history):
        eligible_schedules = [result.schedule for result in history if result.accepted]
        eligible_schedules.append(latest_schedule)
        penalty = 0
        for index, schedule in enumerate(eligible_schedules):
            budget_remainder = self.__budget_remainder(schedule)
            base_penalty = max(0, budget_remainder - 10)
            penalty += np.exp(1.2 * (index - len(eligible_schedules) + 1)) * base_penalty
        return penalty

    def __budget_remainder(self, schedule):
        min_accuracies = self.accuracy_lut.loc[
            self.accuracy_lut[self.accuracy_lut["Valid"] == 1].groupby("Network")["Accuracy"].idxmin()
        ].set_index("Network")["Accuracy"]

        max_accuracies = self.accuracy_lut.loc[
            self.accuracy_lut[self.accuracy_lut["Valid"] == 1].groupby("Network")["Accuracy"].idxmax()
        ].set_index("Network")["Accuracy"]

        total_budget = (max_accuracies - min_accuracies).sum()

        accel_accuracies = pd.DataFrame.from_dict(schedule.assigned, orient="index") \
            .reset_index(names="Network").rename(columns={"precision": "QuantBits"}) \
            .merge(self.accuracy_lut, on=["Network", "QuantBits"]).set_index("Network")["Accuracy"]

        budget_used = (max_accuracies - accel_accuracies).sum()
        return total_budget - budget_used

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

    def __compute_p3(self, latest_schedule):
        c = 1
        enforced_precisions = 6, 7, 8
        assigned_precisions = [a.precision for a in latest_schedule.assigned.values()]
        for p in enforced_precisions:
            if not p in assigned_precisions:
                return c
        return 0

        # schedule_history = [result.schedule for result in self.step_results if result.accepted]
        # schedule_history.append(latest_schedule)
        # c = 1
        # window = self.window_size
        # if len(schedule_history) < window:
        #     return 0
        # lookback = schedule_history[-window:]
        # for s in lookback:
        #     if not s or not s.assigned:
        #         continue
        #     for a in s.assigned.values():
        #         if a.precision == enforced_precision:
        #             return 0
        return c

    def __aggregate_loss(self, schedule):
        if not schedule or not schedule.assigned:
            return 0
        losses = []
        for (network, assignment) in schedule.assigned.items():
            assignment_precision = assignment.precision
            assignment_accuracy = self.__network_accuracy(network, assignment_precision)
            baseline_accuracy = self.__network_accuracy(network, self.baseline_precision)
            loss = baseline_accuracy - assignment_accuracy
            losses.append(loss)
        return np.mean(losses)

    def __network_accuracy(self, network, precision):
        return self.accuracy_lut.loc[
            (self.accuracy_lut["Network"] == network) &
            (self.accuracy_lut["QuantBits"] == precision),
            "Accuracy"
        ].iloc[0]
