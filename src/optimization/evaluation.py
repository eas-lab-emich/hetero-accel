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
    gamma = 0.4
    lambda_p1 = 0.0045
    baseline_precision = 8

    def __init__(self, accuracy_lut):
        self.accuracy_lut = accuracy_lut

    def penalize(self, edp, latest_schedule: Schedule, history, enabled) -> Penalty:
        penalty = self.__compute_p1(latest_schedule, history) if latest_schedule and latest_schedule.assigned else 0
        total_penalty = (self.lambda_p1 * edp * penalty) if enabled else 0
        return Penalty(total_penalty, self.__aggregate_loss(latest_schedule),
                       penalty, 0, 0, self.lambda_p1, 0, 0, 0, 0)

    def __compute_p1(self, latest_schedule, history):
        eligible_schedules = [result.schedule for result in history if result.accepted]
        eligible_schedules.append(latest_schedule)
        penalty = 0
        for index, schedule in enumerate(eligible_schedules):
            budget_remainder = self.__budget_remainder(schedule)
            base_penalty = budget_remainder - 10 # no longer clipping
            penalty += np.exp(self.gamma * (index - len(eligible_schedules) + 1)) * base_penalty
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
