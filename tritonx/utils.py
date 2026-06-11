import triton
import torch
import functools

try:
    from triton.backends.amd.testing import MPMark as _Mark
except ImportError:
    from triton.testing import Mark as _Mark


TORCH_DTYPE_TO_DTYPE = {
    torch.float32: "f32",
    torch.float: "f32",
    torch.float16: "f16",
    torch.half: "f16",
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.double: "f64",
    torch.float8_e4m3fn: "f8_e4m3fn",
    torch.float8_e5m2: "f8_e5m2",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.long: "i64",
    torch.uint8: "u8",
    torch.bool: "bool",
}


DTYPE_TO_TORCH_DTYPE = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f64": torch.float64,
    "f8_e4m3fn": torch.float8_e4m3fn,
    "f8_e5m2": torch.float8_e5m2,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "u8": torch.uint8,
    "bool": torch.bool,
}


PEAK_PERF = {
    "gfx938_cu72": {
        'bf16': {
            'flops': 383.3856e12,
            'bw': 2.4e12,
            'hz': 1.3e9,
        },
    },
    "gfx938_cu64": {
        'bf16': {
            'flops': 340.7872e12,
            'bw': 2.4e12, # same as cu72 ?
            'hz': 1.3e9,
        },
    }
}


@functools.lru_cache
def get_gpu_label():
    target = triton.runtime.driver.active.get_current_target()
    device = torch.cuda.current_device()
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    return f"{target.arch}_cu{num_cu}"


def get_peak_performance(torch_dtype):
    gpu_label = get_gpu_label()
    dtype = TORCH_DTYPE_TO_DTYPE[torch_dtype]
    try:
        return PEAK_PERF[gpu_label][dtype]
    except Exception as e:
        print(f"Undefined peak performance for {gpu_label}_{dtype}")
        raise NotImplementedError


def tensor_shape_dtype_str(t: torch.Tensor) -> str:
    shape_part = "x".join(str(d) for d in t.shape)
    dtype_abbr = TORCH_DTYPE_TO_DTYPE.get(t.dtype, str(t.dtype))
    return f"{shape_part}x{dtype_abbr}"


class Mark(_Mark):
    def __init__(self, fn, benchmarks):
        super().__init__(fn, benchmarks)
        if 'flops_fn' in benchmarks.args:
            self.flops_fn = benchmarks.args['flops_fn']
        if 'bytes_fn' in benchmarks.args:
            self.bytes_fn = benchmarks.args['bytes_fn']

    def _run(self, bench: triton.testing.Benchmark, save_path: str, show_plots: bool, print_data: bool, diff_col=False,
             save_precision=6, **kwrags):
        import os

        # import matplotlib.pyplot as plt
        import pandas as pd
        y_mean = bench.line_names
        y_min = [f'{x}-min' for x in bench.line_names]
        y_max = [f'{x}-max' for x in bench.line_names]
        x_names = list(bench.x_names)
        prefix_names = ['*.pt']
        perf_names = []
        columns=prefix_names + x_names + y_mean + y_min + y_max
        if self.flops_fn:
            _names = ['TFLOPS', 'Peak TFLOPS', 'TFLOPS Utilization']
            perf_names.extend(_names)
            columns.extend(_names)
        if self.bytes_fn:
            _names = ['BW', 'Peak BW', 'BW Utilization']
            perf_names.extend(_names)
            columns.extend(_names)
        if self.flops_fn and self.bytes_fn:
            _names = ['AI', 'Compute Bound Ratio']
            perf_names.extend(_names)
            columns.extend(_names)
        df = pd.DataFrame(columns=columns)
        pt_files = bench.args['pt']
        del bench.args['pt']
        for x, pt in zip(bench.x_vals, pt_files):
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]

            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_mean, row_min, row_max = [], [], []
            tflops, peak_tflops, flops_utilization = [], [], []
            bw, peak_bw, bw_utilization = [], [], []
            ai, compute_bound_ratio = [], []
            for y in bench.line_vals:
                ret = self.fn(**{'inputs': x_args}, **{bench.line_arg: y}, **bench.args, **kwrags)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]

                if self.flops_fn or self.bytes_fn:
                    from .replay import _move_to_device
                    _x_args = _move_to_device(x_args, 'cpu')

                if self.flops_fn:
                    flops = self.flops_fn(_x_args)
                    flops_per_sec = flops / (y_mean / 1000)
                    peak_flops = get_peak_performance(_x_args['q_extend'].dtype)['flops']
                    tflops += [flops_per_sec / 1e12]
                    peak_tflops += [peak_flops / 1e12]
                    flops_utilization += [flops_per_sec / peak_flops]

                if self.bytes_fn:
                    bytes = self.bytes_fn(_x_args)
                    bytes_per_sec = bytes / (y_mean / 1000)
                    peak_bytes = get_peak_performance(_x_args['q_extend'].dtype)['bw']
                    bw += [bytes_per_sec / 1e12]
                    peak_bw += [peak_bytes / 1e12]
                    bw_utilization += [bytes_per_sec / peak_bytes]

                if self.flops_fn and self.bytes_fn:
                    _ai = flops / bytes
                    ai += [_ai]
                    compute_bound_ratio += [_ai / (peak_flops / peak_bytes)]

            from .replay import FloatTensor
            _x = [tensor_shape_dtype_str(o) if hasattr(o, 'dtype') else
                  o.get('type_str') if isinstance(o, FloatTensor) else o
                  for o in x]
            df.loc[len(df)] = [pt.stem] + _x + row_mean + row_min + row_max + \
                                tflops + peak_tflops + flops_utilization + \
                                bw + peak_bw + bw_utilization + ai + compute_bound_ratio

        # NYI:
        # if bench.plot_name:
        #     plt.figure()
        #     ax = plt.subplot()
        #     Plot first x value on x axis if there are multiple.
        #     first_x = x_names[0]
        #     for i, y in enumerate(bench.line_names):
        #         y_min, y_max = df[y + '-min'], df[y + '-max']
        #         col = bench.styles[i][0] if bench.styles else None
        #         sty = bench.styles[i][1] if bench.styles else None
        #         ax.plot(df[first_x], df[y], label=y, color=col, ls=sty)
        #         if not y_min.isnull().all() and not y_max.isnull().all():
        #             y_min = y_min.astype(float)
        #             y_max = y_max.astype(float)
        #             ax.fill_between(df[first_x], y_min, y_max, alpha=0.15, color=col)
        #     ax.legend()
        #     ax.set_xlabel(bench.xlabel or first_x)
        #     ax.set_ylabel(bench.ylabel)
        #     # ax.set_title(bench.plot_name)
        #     ax.set_xscale("log" if bench.x_log else "linear")
        #     ax.set_yscale("log" if bench.y_log else "linear")
        #     if show_plots:
        #         plt.show()
        #     if save_path:
        #         plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
        df = df[prefix_names + x_names + bench.line_names + perf_names]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df['Diff'] = df[col1] - df[col0]

        if print_data:
            print(bench.plot_name + ':')
            print(df.to_string())
        if save_path:
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), float_format=f"%.{save_precision}f",
                      index=False)
        return df


def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper
