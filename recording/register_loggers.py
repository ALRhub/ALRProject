from typing import List

from recording.loggers.AbstractLogger import AbstractLogger
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.tasks.AbstractTask import AbstractTask
from util.Types import ConfigDict


def register_loggers(config: ConfigDict,
                     algorithm: AbstractIterativeAlgorithm,
                     task: AbstractTask) -> List[AbstractLogger]:
    """
    Create a list of all loggers used for the current run. The order of the loggers may matter, since loggers can pass
    computed values to subsequent ones.
    Args:
        config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
            used by cw2 for the current run.
        algorithm: An instance of the algorithm to run.
        task: The task instance to perform the algorithm on. Can e.g., contain training data or a gym environment

    Returns: A list of loggers to use.

    """
    recording_dict = config.get("recording", {})

    from recording.loggers.ConfigLogger import ConfigLogger
    from recording.loggers.ScalarsLogger import ScalarsLogger
    loggers = [ConfigLogger(config=config, algorithm=algorithm, task=task),
               ScalarsLogger(config=config, algorithm=algorithm, task=task)]
    if recording_dict.get("visualization", False):
        from recording.loggers.VisualizationLogger import VisualizationLogger
        loggers.append(VisualizationLogger(config=config, algorithm=algorithm, task=task))
    if recording_dict.get("wandb", False):
        from recording.loggers.CustomWAndBLogger import CustomWAndBLogger
        loggers.append(CustomWAndBLogger(config=config, algorithm=algorithm, task=task))

    return loggers