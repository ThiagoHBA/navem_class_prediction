from enum import Enum

class LinearClassificationClass(Enum):
    URGENT_STOP = 4
    REDUCE_A_LOT = 3
    SLIGHTLY_REDUCE = 2
    KEEP_SPEED = 1
    SPEED_UP = 0

class SidesClassificationClass(Enum):
    STRONG_SIDESTEP_LEFT = 4
    SLIGHTLY_SIDESTEP_LEFT = 3
    NO_SIDESTEP = 2
    SLIGHT_SIDESTEP_RIGHT = 1
    STRONG_SIDESTEP_RIGHT = 0