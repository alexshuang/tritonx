from .replay import dump_inputs, replay_inputs, get_dump_output_dir

__all__ = [
    "dump_inputs",
    "replay_inputs",
    "get_dump_output_dir",
]


def _launch_metadata(grid, kernel, args):
    return {
        'grid': grid,
        'args': args,
    }

import functools
import triton
triton.jit = functools.partial(triton.jit, launch_metadata=_launch_metadata)
