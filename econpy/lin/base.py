from collections.abc import Callable
from typing import Optional, Any

import random
import numpy as np
import torch


def set_deterministic_and_all_seed(seed: Optional[int]) -> None:
    """Sets seed for all pytorch related random generators"""
    deterministic = (seed is not None)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = (not deterministic)
    if deterministic:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def timeit(
        fn: Callable[[], Any],
        repeat: int = 2**8, number: int = 1, groups: int = 2**2) -> float:
    """
    timeit for torch cuda functions.
    Aim for more than 10ms, when choosing repeat
    """
    def timed_fn(fn, number):
        """Capture for timing"""
        start.record()
        for _ in range(number):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = tuple(timed_fn(fn, number) for _ in range(repeat))
    groups = min(groups, repeat)
    min_times = tuple(
        min(times[repeat*i//groups:repeat*(i+1)//groups])\
            for i in range(groups))
    t0, t1 = min(min_times), max(min_times) # shows variance in estimates
    return t0, t1, (t1 - t0) / t0


CUDA = torch.device("cuda")