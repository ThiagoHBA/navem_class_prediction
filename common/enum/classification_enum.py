from enum import Enum

class LinearClassificationClass(Enum):
    URGENT_STOP = 0
    REDUCE_A_LOT = 1
    SLIGHTLY_REDUCE = 2
    KEEP_SPEED = 3
    SPEED_UP = 4

class SidesClassificationClass(Enum):
    STRONG_SIDESTEP_LEFT = 0
    SLIGHTLY_SIDESTEP_LEFT = 1
    NO_SIDESTEP = 2
    SLIGHTLY_SIDESTEP_RIGHT = 3
    STRONG_SIDESTEP_RIGHT = 4