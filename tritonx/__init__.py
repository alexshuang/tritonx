from .replay import dump_inputs, replay_inputs, get_dump_output_dir
from .utils import get_peak_performance

__all__ = [
    "dump_inputs",
    "replay_inputs",
    "get_dump_output_dir",
    "get_peak_performance",
]


def _launch_metadata(grid, kernel, args):
    return {
        'grid': grid,
        'args': args,
    }

import functools
import triton
triton.jit = functools.partial(triton.jit, launch_metadata=_launch_metadata)
