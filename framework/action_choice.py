from enum import Enum


class ActionChoice(Enum):
    """
    Represents if the choice of the action should be decided by largest Q-Value or random
    """
    RANDOM = "random"
    LARGEST_Q = "largestQ"
