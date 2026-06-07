import os
import json
import torch
import triton
import hashlib
import inspect
import functools
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
from collections import defaultdict
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
    return dump_dir


def get_dump_input_dir():
    return os.path.join(_get_dump_dir(), "inputs")


def get_dump_output_dir():
    return os.path.join(_get_dump_dir(), "outputs")


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


def _make_deterministic_seed(func_name: str, arg_name: str, type_str: str) -> int:
    key = f"{func_name}::{arg_name}::{type_str}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)


class FloatTensor(dict):
    def __repr__(self):
        return self.get("type_str")


def all_scalars_equal(x: torch.Tensor):
    u = torch.unique(x)
    if u.numel() == 1:
        return True, u.item()
    else:
        return False, None


@dataclass
class CSRMask:
    length: int
    indptr: torch.Tensor  # shape: (num_segments * 2,), [start, end)

    @classmethod
    def from_mask(cls, mask: torch.Tensor) -> "CSRMask":
        assert mask.dtype == torch.bool and mask.dim() == 1

        padded = torch.cat([
            torch.tensor([False]),
            mask.detach().cpu(),
            torch.tensor([False])
        ])
        diff = padded[1:].int() - padded[:-1].int()
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends   = (diff == -1).nonzero(as_tuple=True)[0]

        indptr = torch.stack([starts, ends], dim=1).flatten()
        return cls(length=mask.shape[0], indptr=indptr)

    def to_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.length, dtype=torch.bool)
        for start, end in self.indptr.reshape(-1, 2):
            mask[start:end] = True
        return mask

    @property
    def num_segments(self) -> int:
        return self.indptr.shape[0] // 2

    @property
    def num_true(self) -> int:
        pairs = self.indptr.reshape(-1, 2)
        return int((pairs[:, 1] - pairs[:, 0]).sum())

    def __repr__(self) -> str:
        return f"{self.length}xi1"
        # return (f"CSRMask(length={self.length}, "
        #         f"num_segments={self.num_segments}, "
        #         f"num_true={self.num_true}, "
        #         f"segments=[{segs}])")


def _to_savable_obj(x: Any, move_tensor_to_cpu: bool = True,
                    func_name: str = "", arg_name: str = "",
                    compress_all: bool = True) -> Any:
    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x):
            type_str = tensor_type_str(x)
            seed = _make_deterministic_seed(func_name, arg_name, type_str)
            return FloatTensor({"type_str": type_str, "seed": seed})
        else:
            if compress_all:
                # all same scalar
                u = torch.unique(x)
                if u.numel() == 1:
                    return f"{tensor_type_str(x)}={u.item()}"

                # compress by CSR format
                if x.dtype == torch.bool and x.dim() == 1:
                    return CSRMask.from_mask(x)

        t = x.detach()
        if move_tensor_to_cpu:
            t = t.cpu()
        return t
    elif isinstance(x, list):
        return [_to_savable_obj(v, move_tensor_to_cpu, func_name, arg_name, compress_all) for v in x]
    elif isinstance(x, tuple):
        return tuple(_to_savable_obj(v, move_tensor_to_cpu, func_name, arg_name, compress_all) for v in x)
    elif isinstance(x, dict):
        return {k: _to_savable_obj(v, move_tensor_to_cpu, func_name, arg_name, compress_all) for k, v in x.items()}
    else:
        return x


def get_func_name(func) -> str:
    if func.__name__ == 'forward' and hasattr(func, '__qualname__'):
        return func.__qualname__
    return func.__name__


def is_dump_inputs(func) -> bool:
    return hasattr(func, '__decorator_name__') and func.__decorator_name__ == 'dump_inputs'


def dump_inputs(
    fn: Optional[Callable] = None,
    *,
    move_tensor_to_cpu: bool = True,
    overwrite: bool = False,
    compress_all: bool = True,
):
    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, dump=True, **kwargs):
            # @replay_inputs don't dump inputs by dump_inputs=False to avoid affecting benchmark
            if not dump:
                return func(*args, **kwargs)

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            bound_args = dict(bound.arguments)

            sha = _compute_inputs_sha256(bound_args)

            dump_dir = get_dump_input_dir()
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            func_name = get_func_name(func)
            file_path = os.path.join(dump_dir, f"{func_name}-{sha}.pt")

            inputs = {}
            for k, v in bound_args.items():
                try:
                    inputs[k] = _to_savable_obj(v, move_tensor_to_cpu, func_name, k, compress_all)
                except Exception as e:
                    print(f'Warning occurred {e}, trying "not move_tensor_to_cpu" again')
                    inputs[k] = _to_savable_obj(v, not move_tensor_to_cpu, func_name, k, compress_all)

            torch.save(
                {
                    "func_name": func_name,
                    "inputs": inputs,
                    # "inputs": {
                    #     k: _to_savable_obj(v, move_tensor_to_cpu, func_name, k)
                    #     for k, v in bound_args.items()
                    # },
                },
                file_path,
            )
            print(f"Dump inputs of {func_name} to {file_path}")

            return func(*args, **kwargs)

        wrapper.__decorator_name__ = "dump_inputs"
        wrapper.__decorator_params__ = {
            "move_tensor_to_cpu": move_tensor_to_cpu,
            "overwrite": overwrite,
        }
        return wrapper

    if fn is not None:
        return decorator(fn)
    else:
        return decorator


def parse_type(s: str):
    shape_part, dtype_part = s.rsplit("x", 1)
    shape = tuple(int(x) for x in shape_part.split("x"))
    dtype = DTYPE_TO_TORCH_DTYPE[dtype_part]
    return shape, dtype


def float_tensor_from_type_str(s: str, device="cpu", seed: int = None):
    shape, dtype = parse_type(s)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return torch.rand(shape, dtype=dtype, device=device, generator=gen)

    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        x = torch.rand(shape, dtype=torch.float32, device=device, generator=gen)
        return x.to(dtype)

    raise ValueError(f"Unsupported dtype: {s.split('x')[-1]}")


def tensor_from_type_str(s: str, device="cpu", seed: int = None):
    dtype = s.split('x')[-1]
    if dtype.startswith("f") or dtype.startswith("bf"):
        return float_tensor_from_type_str(s, device, seed)
    elif '=' in dtype:
        type_str, value_str = s.split('=')
        shape, dtype = parse_type(type_str)
        return torch.full(shape, eval(value_str), dtype=dtype).to(device)

    raise ValueError(f"Unsupported dtype: {dtype}")


def next_power_of_2(x):
    if x < 1:
        return 1
    return 1 << (x - 1).bit_length()


def prev_power_of_2(x):
    if x < 1:
        return 1
    return 1 << (x.bit_length() - 1)


def p2_reduce_dim_vals(vals: list[int]) -> set[int]:
    """ power-of-2 """
    vals = sorted(vals)
    min_val, max_val = vals[0], vals[-1]
    kept = {min_val, max_val}

    p = prev_power_of_2(min_val)
    while p <= max_val:
        if p in vals:
            kept.add(p)
        else:
            kept.add(min(vals, key=lambda v: abs(v - p)))
        p *= 2

    return kept


def prune_grids(grids: list[tuple], threshold: int = 10, mode='next_power_of_2') -> list[tuple]:
    # Step 1: unique
    unique_grids = list(dict.fromkeys(grids))
    ndim = len(unique_grids[0])

    # Step 2: find the dimensions that vary significantly (number of distinct values > threshold)
    vary_dims = set(
        d for d in range(ndim)
        if len(set(g[d] for g in unique_grids)) > threshold
    )

    # Step 3: For each varying dimension, group by the prefix key and reduce the values
     # Filter dimension by dimension, reducing the grid set in each round
    current_grids = set(unique_grids)

    for vary_dim in sorted(vary_dims):
        groups = defaultdict(set)
        for g in current_grids:
            key = tuple(g[d] for d in range(ndim) if d != vary_dim)
            groups[key].add(g[vary_dim])

        next_grids = set()
        for key, vary_vals in groups.items():
            kept_vals = p2_reduce_dim_vals(sorted(vary_vals))
            for v in kept_vals:
                g = list(key)
                g.insert(vary_dim, v)
                next_grids.add(tuple(g))

        current_grids = next_grids

    return sorted(current_grids)


def replay_inputs(
    fn: Optional[Callable] = None,
    *,
    device: str ='cuda',
    strict_func_name: bool = True,
    benchmark: bool = True,
    move_tensor_to_cpu: bool = True,
):
    def decorator(func):
        func_name = get_func_name(func)
        _user_hooks = []

        def add_user_hook(hook):
            if hook not in _user_hooks:
                _user_hooks.append(hook)

        def remove_user_hook(hook):
            if hook in _user_hooks:
                _user_hooks.remove(hook)

        def _iter_case_paths(hash_prefix: str = None) -> List[Path]:
            cache_dir = get_dump_input_dir()
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
            if isinstance(obj, FloatTensor):
                # format: {"type_str": "2x3xf32", "seed": 123}
                return float_tensor_from_type_str(obj["type_str"], target_device, seed=obj["seed"])
            if isinstance(obj, CSRMask):
                return obj.to_mask().to(target_device)
            if isinstance(obj, dict):
                return {k: _move_to_device(v, target_device) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_move_to_device(v, target_device) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_move_to_device(v, target_device) for v in obj)
            return obj

        def _build_benchmark_from_pt(pt_manifest: str = '') -> tt.Benchmark:
            file_paths = [Path(o) for o in open(pt_manifest).read().splitlines()] \
                if pt_manifest else _iter_case_paths()

            def is_same_argument_names(lists: list[list[str]]) -> bool:
                if not lists:
                    return True
                first = lists[0]
                return all(lst == first for lst in lists[1:])

            keys, vals = [], []
            for f in file_paths:
                obj = torch.load(f, map_location="cpu", weights_only=False)
                keys.append(obj['inputs'].keys())
                vals.append([o for o in obj['inputs'].values()])

            assert is_same_argument_names(keys) == True

            x_names = keys[0]
            x_vals = vals
            plot_name = f"{func_name}"
            print(f"Benchmarking {func_name} with {len(file_paths)} cases")
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
            fn = lambda: func(*args, dump=False, **kwargs) if is_dump_inputs(func) else func(*args, **kwargs)
            return fn()

        def _run(file: str, device, cache_results: list = None, ref_fn: Callable = None):
            def _launch_enter_hook(launch_metadata):
                metadata = launch_metadata.get() | {'input_file': file}
                for h in _user_hooks:
                    h(metadata)

            add_hook = True if _user_hooks else False

            if add_hook:
                if triton_version_float >= 3.5:
                    triton.knobs.runtime.launch_enter_hook.add(_launch_enter_hook)
                elif triton_version_float >= 3.4:
                    triton.knobs.runtime.launch_enter_hook = _launch_enter_hook
                else:
                    triton.compiler.CompiledKernel.launch_enter_hook = _launch_enter_hook

            obj = torch.load(file, map_location="cpu", weights_only=False)
            # if strict_func_name and obj.get("func_name") != func_name:
            #     continue
            inputs = _move_to_device(obj["inputs"], device)
            print(f"Replaying {os.path.basename(file)} ...")
            if is_dump_inputs(func):
                ret = func(dump=func.__decorator_params__.get("overwrite"),
                           **inputs)
            else:
                ret = func(**inputs)

            if add_hook:
                if triton_version_float >= 3.5:
                    triton.knobs.runtime.launch_enter_hook.remove(_launch_enter_hook)
                elif triton_version_float >= 3.4:
                    triton.knobs.runtime.launch_enter_hook = None
                else:
                    triton.compiler.CompiledKernel.launch_enter_hook = None

            if cache_results:
                if isinstance(cache_results, str):
                    cache_results = [cache_results]
                dump_dir = get_dump_output_dir()
                Path(dump_dir).mkdir(parents=True, exist_ok=True)
                file_path = os.path.join(dump_dir, os.path.basename(file))
                torch.save(
                    {
                        "func_name": func_name,
                        "outputs": {k: inputs[k].detach().cpu() for k in cache_results},
                    },
                    file_path,
                )
                print(f"Cache results {cache_results} to {file_path}")

            if ref_fn:
                ref_fn({'input_file': file, **inputs})

            return ret

        def replay_all(device_override=None, cache_results: str = None, ref_fn: Callable = None,
                       replay_enter_hook: Callable = None):
            files = _iter_case_paths()
            if replay_enter_hook:
                add_user_hook(replay_enter_hook)
            results = [
                {
                    "file": str(f),
                    "result": _run(f,
                                    device_override if device_override is not None else device,
                                    cache_results=cache_results,
                                    ref_fn=ref_fn),
                } for f in files
            ]
            return results

        def replay(hash_prefix: str, device_override=None, ref_fn: Callable = None):
            files = _iter_case_paths(hash_prefix)
            if len(files) > 1:
                raise RuntimeError(f"multiple matches for prefix {hash_prefix}: {[str(x) for x in files]}")
            return _run(files[0], device_override if device_override is not None else device, ref_fn=ref_fn)

        def get_grids_by_key(keys: list):
            grids = defaultdict(set)
            def _hook(metadata):
                args = metadata['args']
                key = tuple([str(args[o].dtype) if hasattr(args[o], 'dtype') else args[o]
                            for o in keys if o in args])
                grids[key].add((metadata['input_file'], metadata['grid']))

            replay_all(replay_enter_hook=_hook)
            return grids

        def prune_files_by_grids(key: list, mode='next_power_of_2'):
            res = {}
            for k, v in get_grids_by_key(key).items():
                files, grids = zip(*v)

                indices = defaultdict(list)
                for idx, g in enumerate(grids):
                    indices[g].append(idx)

                pruned = prune_grids(grids, mode=mode)
                pruned_indices = [indices.get(grid, [])[0] for grid in pruned]
                res[k] = [(files[i], grids[i]) for i in pruned_indices]
            return res

        def benchmark(
            *,
            show_plots: bool = False,
            print_data: bool = True,
            save_path: str = "",
            return_df: bool = False,
            warmup: int = 25,
            rep: int = 100,
            quantiles: Optional[List[float]] = None,
            pt_manifest: str = '',
        ):
            bench_cfgs = _build_benchmark_from_pt(pt_manifest=pt_manifest)
            os.makedirs(save_path, exist_ok=True)

            compile_only = True if os.getenv("TRITON_HCUTUNE_COMPILE_ONLY", "0") == "1" else False
            if compile_only:
                print(f"[tritonx] HCU auto-tuning with cpu tensors on compile-only mode")
                _device = "cpu"
            else:
                _device = device

            @perf_report(bench_cfgs)
            def _bench(**bench_kwargs):
                inputs = _move_to_device(bench_kwargs["inputs"], _device)
                provider = bench_kwargs.get("provider", "triton")
                runtime_device = bench_kwargs.get("device", _device)

                if provider != "triton":
                    raise ValueError(f"Only provider='triton' is supported, got {provider!r}")
                if not compile_only and runtime_device != _device:
                    raise ValueError(
                        f"Benchmark device mismatch: expected {_device!r}, got {runtime_device!r}"
                    )

                run = lambda: func(dump=False, **inputs) if is_dump_inputs(func) else func(**inputs)

                if quantiles is None:
                    q = [0.5, 0.2, 0.8]
                else:
                    q = quantiles

                # return tt.do_bench(run, warmup=warmup, rep=rep, quantiles=q)
                return tt.do_bench(run, quantiles=q)

            ret = _bench.run(
                show_plots=show_plots,
                print_data=print_data,
                save_path=save_path,
                return_df=return_df,
            )

            if save_path:
                print('\n' + '*' * 72)
                print('*')
                print(f'*  Benchmark result save path: {os.path.join(save_path, f"{bench_cfgs.plot_name}.csv")}')
                print('*')
                print('*' * 72 + '\n')

            return ret

        wrapper.replay_all = replay_all
        wrapper.replay = replay
        wrapper.benchmark = benchmark
        wrapper.get_grids_by_key = get_grids_by_key
        wrapper.prune_files_by_grids = prune_files_by_grids
        wrapper.__decorator_name__ = "replay_inputs"
        return wrapper

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
