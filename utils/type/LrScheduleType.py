from enum import Enum


class LrScheduleType(Enum):
    NONE = 1,
    CYCLIC = 2,
    COSINE_ANNEALING = 3,
    REDUCE_ON_PLATEAU = 4
