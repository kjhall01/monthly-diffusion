"""Stop between batches, then let the script export its best validated model."""

from time import perf_counter


class TrainingTimeLimit(TimeoutError):
    """Only the training deadline expired; ordinary training failures still raise."""


def check_time_limit(deadline):
    if perf_counter() >= deadline:
        raise TrainingTimeLimit("Training time limit reached")
