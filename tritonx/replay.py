import os
import json
import torch
import hashlib
import inspect
import functools
from pathlib import Path
from typing import Any, Callable, List, Optional
import triton.testing as tt
from .utils import perf_report, TORCH_DTYPE_TO_DTYPE, DTYPE_TO_TORCH_DTYPE

from triton import __version__ as triton_version
triton_major_version = int(triton_version.split(".")[0])
triton_minor_version = int(triton_version.split(".")[1])
triton_version_float = triton_major_version + float(triton_minor_version / 10)


def _get_dump_dir():
    if triton_version_float >= 3.3:
        from triton.knobs import cache as cache_knob
        dump_dir = os.getenv("TRITON_DUMP_DIR", "").strip() or cache_knob.dump_dir
    else:
        from triton.runtime.cache import default_dump_dir
        dump_dir = os.getenv("TRITON_DUMP_DIR", "").strip() or default_dump_dir()
    return os.path.join(dump_dir, "inputs")


def _to_hashable_obj(x: Any):
    if isinstance(x, torch.Tensor):
        t = x.detach()
        return {
            "__type__": "tensor",
            "dtype": str(t.dtype),
            "shape": list(t.shape),
        }
    elif isinstance(x, (str, int, float, bool)) or x is None:
        return x
    elif isinstance(x, (list, tuple)):
        return {"__type__": type(x).__name__, "items": [_to_hashable_obj(v) for v in x]}
    elif isinstance(x, dict):
        return {
            "__type__": "dict",
            "items": {str(k): _to_hashable_obj(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))},
        }
    else:
        return {"__type__": type(x).__name__, "repr": repr(x)}


def _compute_inputs_sha256(bound_args: dict[str, Any]) -> str:
    obj = {k: _to_hashable_obj(v) for k, v in sorted(bound_args.items(), key=lambda kv: kv[0])}
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def tensor_type_str(x: torch.Tensor) -> str:
    shape = "x".join(str(s) for s in x.shape)
    dtype = TORCH_DTYPE_TO_DTYPE.get(x.dtype, str(x.dtype).replace("torch.", ""))
    return f"{shape}x{dtype}"


def _to_savable_obj(x: Any, move_tensor_to_cpu: bool = True):
    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x):
            return tensor_type_str(x)

        t = x.detach()
        if move_tensor_to_cpu:
            t = t.cpu()
        return t
    elif isinstance(x, list):
        return [_to_savable_obj(v, move_tensor_to_cpu) for v in x]
    elif isinstance(x, tuple):
        return tuple(_to_savable_obj(v, move_tensor_to_cpu) for v in x)
    elif isinstance(x, dict):
        return {k: _to_savable_obj(v, move_tensor_to_cpu) for k, v in x.items()}
    else:
        return x


def get_func_name(func) -> str:
    if func.__name__ == 'forward' and hasattr(func, '__qualname__'):
        return func.__qualname__
    return func.__name__


def dump_inputs(
    fn: Optional[Callable] = None,
    *,
    move_tensor_to_cpu: bool = True,
    overwrite: bool = False,
):
    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If a function is decorated with both @dump_inputs and @replay_inputs,
            # enable TRITONX_DISABLE_DUMP=1 during replay to skip dumping and avoid
            # affecting benchmark result.
            if os.getenv("TRITONX_DISABLE_DUMP", "0") == "1":
                return func(*args, **kwargs)

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            bound_args = dict(bound.arguments)

            sha = _compute_inputs_sha256(bound_args)

            dump_dir = _get_dump_dir()
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            func_name = get_func_name(func)
            file_path = os.path.join(dump_dir, f"{func_name}-{sha}.pt")

            if overwrite or not os.path.exists(file_path):
                torch.save(
                    {
                        "func_name": func_name,
                        "inputs": {k: _to_savable_obj(v, move_tensor_to_cpu) for k, v in bound_args.items()},
                    },
                    file_path,
                )
                print(f"Dump inputs of {func_name} to {file_path}")

            return func(*args, **kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    else:
        return decorator


def tensor_from_type_str(s: str, device="cpu"):
    shape_part, dtype_part = s.rsplit("x", 1)
    shape = tuple(int(x) for x in shape_part.split("x"))
    dtype = DTYPE_TO_TORCH_DTYPE[dtype_part]

    if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return torch.rand(shape, dtype=dtype, device=device)

    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        x = torch.rand(shape, dtype=torch.float32, device=device)
        return x.to(dtype)

    raise ValueError(f"Unsupported dtype: {dtype_part}")


def replay_inputs(
    fn: Optional[Callable] = None,
    *,
    device: str ='cuda',
    strict_func_name: bool = True,
    benchmark: bool = True,
):
    def decorator(func):
        func_name = get_func_name(func)

        def _iter_case_paths(hash_prefix: str = None) -> List[Path]:
            cache_dir = _get_dump_dir()
            root = Path(cache_dir)
            if not root.exists():
                raise FileNotFoundError(f"cache_dir does not exist: {root}")

            glob_expr = f"{func_name}-{hash_prefix}.pt" if hash_prefix else f"{func_name}-*.pt"
            paths = sorted(root.glob(glob_expr))
            if not paths:
                raise RuntimeError(f"No found {glob_expr} under {root}")
            return paths

        def _move_to_device(obj: Any, target_device: str) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.to(target_device)
            if isinstance(obj, str):
                return tensor_from_type_str(obj, target_device)
            if isinstance(obj, dict):
                return {k: _move_to_device(v, target_device) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_move_to_device(v, target_device) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_move_to_device(v, target_device) for v in obj)
            return obj

        def _build_benchmark_from_pt() -> tt.Benchmark:
            file_paths = _iter_case_paths()

            def is_same_argument_names(lists: list[list[str]]) -> bool:
                if not lists:
                    return True
                first = lists[0]
                return all(lst == first for lst in lists[1:])

            keys, vals = [], []
            for f in file_paths:
                obj = torch.load(f, map_location="cpu")
                keys.append(obj['inputs'].keys())
                vals.append([o for o in obj['inputs'].values()])

            assert is_same_argument_names(keys) == True

            x_names = keys[0]
            x_vals = vals
            plot_name = f"Benchmark {func_name}"
            return tt.Benchmark(
                x_names=x_names,
                x_vals=x_vals,
                line_arg="provider",
                line_vals=["triton"],
                line_names=["triton"],
                styles=[("blue", "-")],
                ylabel="ms",
                plot_name=plot_name,
                args={
                    "device": device,
                    "pt": file_paths,
                },
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        def _run(file: str, device, ref_fn: Callable = None):
            obj = torch.load(file, map_location="cpu")
            # if strict_func_name and obj.get("func_name") != func_name:
            #     continue
            inputs = _move_to_device(obj["inputs"], device)
            print(f"Replaying {os.path.basename(file)} ...")
            result = func(**inputs)

            if ref_fn:
                ref_fn(inputs)

            return result

        def replay_all(device_override=None, ref_fn: Callable = None):
            files = _iter_case_paths()
            results = []
            for f in files:
                results.append(
                    {
                        "file": str(f),
                        "result": _run(f,
                                       device_override if device_override is not None else device,
                                       ref_fn),
                    }
                )
            return results

        def replay(hash_prefix: str, device_override=None, ref_fn: Callable = None):
            files = _iter_case_paths(hash_prefix)
            if len(files) > 1:
                raise RuntimeError(f"multiple matches for prefix {hash_prefix}: {[str(x) for x in files]}")
            return _run(files[0], device_override if device_override is not None else device, ref_fn),

        def benchmark(
            *,
            show_plots: bool = False,
            print_data: bool = True,
            save_path: str = "",
            return_df: bool = False,
            warmup: int = 25,
            rep: int = 100,
            quantiles: Optional[List[float]] = None,
        ):
            bench_cfgs = _build_benchmark_from_pt()
            os.makedirs(save_path, exist_ok=True)

            @perf_report(bench_cfgs)
            def _bench(**bench_kwargs):
                inputs = _move_to_device(bench_kwargs["inputs"], device)
                provider = bench_kwargs.get("provider", "triton")
                runtime_device = bench_kwargs.get("device", device)

                if provider != "triton":
                    raise ValueError(f"Only provider='triton' is supported, got {provider!r}")
                if runtime_device != device:
                    raise ValueError(
                        f"Benchmark device mismatch: expected {device!r}, got {runtime_device!r}"
                    )

                run = lambda: func(**inputs)

                if quantiles is None:
                    q = [0.5, 0.2, 0.8]
                else:
                    q = quantiles

                # return tt.do_bench(run, warmup=warmup, rep=rep, quantiles=q)
                return tt.do_bench(run, quantiles=q)

            return _bench.run(
                show_plots=show_plots,
                print_data=print_data,
                save_path=save_path,
                return_df=return_df,
            )

        wrapper.replay_all = replay_all
        wrapper.replay = replay
        wrapper.benchmark = benchmark
        return wrapper

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
