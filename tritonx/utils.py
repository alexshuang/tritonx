import triton
import torch

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


def tensor_shape_dtype_str(t: torch.Tensor) -> str:
    shape_part = "x".join(str(d) for d in t.shape)
    dtype_abbr = TORCH_DTYPE_TO_DTYPE.get(t.dtype, str(t.dtype))
    return f"{shape_part}x{dtype_abbr}"


class Mark(_Mark):

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
        df = pd.DataFrame(columns=prefix_names + x_names + y_mean + y_min + y_max)
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
            for y in bench.line_vals:
                ret = self.fn(**{'inputs': x_args}, **{bench.line_arg: y}, **bench.args, **kwrags)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = ret, None, None
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            _x = [tensor_shape_dtype_str(o) if hasattr(o, 'dtype') else o for o in x]
            df.loc[len(df)] = [pt.stem] + _x + row_mean + row_min + row_max

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
        df = df[prefix_names + x_names + bench.line_names]
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
