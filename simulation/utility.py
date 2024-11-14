import numpy as np
from typing import Any
from numpy.typing import NDArray
from numpy import float_

DEFAULT_SEED = 42


def is_number(x: Any) -> bool:
    """
    Checks whether x is int or float.
    """
    return isinstance(x, int) or isinstance(x, float) or isinstance(x, complex)


def to_numpy(x: Any) -> NDArray[float_]:
    """
    Converts x to numpy array.
    """
    if is_number(x):
        return np.array([x])
    else:
        return np.array(x)
