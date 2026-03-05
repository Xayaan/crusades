import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Dummy:
    """Small object that safely absorbs arbitrary attribute/call usage."""

    def __call__(self, *args, **kwargs):
        return _Dummy()

    def __getattr__(self, _name):
        return _Dummy()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _FakeFastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def post(self, *args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def get(self, *args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator


@contextmanager
def _patched_modules(overrides: dict[str, object]):
    sentinel = object()
    previous: dict[str, object] = {}
    for name, module in overrides.items():
        previous[name] = sys.modules.get(name, sentinel)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_security_defs():
    return _load_module("crusades.core.security_defs", ROOT / "src/crusades/core/security_defs.py")


def _make_torch_stub() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__getattr__ = lambda _name: _Dummy()
    torch.Tensor = object
    torch.cuda = _Dummy()
    torch.cuda.is_available = lambda: False
    torch.cuda.Event = types.SimpleNamespace(elapsed_time=lambda *_a, **_k: 0.0)
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.__getattr__ = lambda _name: _Dummy()
    torch.nn.Module = object
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    torch.nn.functional.__getattr__ = lambda _name: _Dummy()
    torch.nn.functional.cross_entropy = _Dummy()
    torch.optim = types.ModuleType("torch.optim")
    torch.optim.__getattr__ = lambda _name: _Dummy()
    torch.optim.Optimizer = object
    torch.backends = _Dummy()
    torch.set_grad_enabled = _Dummy()
    torch.set_default_dtype = _Dummy()
    torch.set_float32_matmul_precision = _Dummy()
    torch.get_float32_matmul_precision = lambda: "high"
    torch.manual_seed = _Dummy()
    return torch


def _make_crusades_overrides(security_defs) -> dict[str, object]:
    crusades_pkg = types.ModuleType("crusades")
    core_pkg = types.ModuleType("crusades.core")
    crusades_pkg.core = core_pkg
    core_pkg.security_defs = security_defs
    return {
        "crusades": crusades_pkg,
        "crusades.core": core_pkg,
        "crusades.core.security_defs": security_defs,
    }


def _load_scanner():
    """Load env.py's _validate_code_structure (the production scanner)."""
    security_defs = _load_security_defs()
    torch_stub = _make_torch_stub()
    overrides = {
        **_make_crusades_overrides(security_defs),
        "torch": torch_stub,
        "torch.nn": torch_stub.nn,
        "torch.nn.functional": torch_stub.nn.functional,
        "torch.optim": torch_stub.optim,
        "fastapi": types.SimpleNamespace(FastAPI=_FakeFastAPI),
        "pydantic": types.SimpleNamespace(BaseModel=object),
    }
    with _patched_modules(overrides):
        env_module = _load_module("env_under_test", ROOT / "environments/templar/env.py")
    return env_module._validate_code_structure


_STRUCTURAL_FAILURES = {
    "Missing required function: inner_steps",
}


def _security_blocked(validate_code_structure, code: str) -> tuple[bool, str]:
    safe, message = validate_code_structure(code)
    reason = message or ("No validation message" if safe else "Validation failed without message")
    if not safe and message in _STRUCTURAL_FAILURES:
        return False, f"(structural) {message}"
    blocked = not safe
    return blocked, reason


MULTI_GPU_RELAXED: set[str] = {
    "functools",
    "torch.distributed",
    "torch.multiprocessing",
    "type",
    "__class__",
}


def _bypass_cases() -> list[tuple[str, str, bool]]:
    """Return (name, code, expect_blocked) triples.

    Cases with expect_blocked=False are intentionally allowed for multi-GPU
    support and should NOT trigger the scanner.
    """
    cases: list[tuple[str, str, bool]] = []
    n = 0

    def add(name, code, *, expect_blocked=True):
        nonlocal n
        n += 1
        cases.append((f"{n}. {name}", code, expect_blocked))

    # === __name__ BINDING VECTORS ===
    add("MatchAs binding", "match '__main__':\n    case __name__:\n        pass\n")
    add("MatchStar binding", "match ['__main__']:\n    case [*__name__]:\n        pass\n")
    add(
        "MatchMapping binding", "match {'key': '__main__'}:\n    case {**__name__}:\n        pass\n"
    )
    add(
        "Exception handler binding",
        "try:\n    raise Exception()\nexcept Exception as __name__:\n    pass\n",
    )
    add("For loop binding", "for __name__ in ['__main__']:\n    pass\n")
    add("List comprehension binding", "_ = [__name__ for __name__ in ['__main__']]\n")
    add("Context manager binding", "with open('/dev/null') as __name__:\n    pass\n")
    add("Type parameters (generics)", "def hack[__name__]():\n    pass\n")
    add("Walrus operator binding", "(__name__ := '__main__')\n")
    add("Class namespace hijack", "class __name__:\n    pass\n")
    add("Global declaration", "def hack():\n    global __name__\n    __name__ = '__main__'\n")
    add(
        "Nonlocal declaration",
        "def outer():\n    x = 1\n    def inner():\n        nonlocal __name__\n",
    )
    add("ImportFrom __name__", "from math import __name__\n")
    add("Import alias to __name__", "import math as __name__\n")
    add("AnnAssign __name__", "__name__: str = '__main__'\n")
    add("AugAssign __name__", "__name__ += '_main_'\n")
    add("Del __name__", "del __name__\n")
    add("__name__ as function arg", "def hack(__name__='__main__'):\n    pass\n")
    add("Torch attr mutation __name__", "import torch\ntorch.nn.functional.__name__ = '__main__'\n")
    add("Tuple unpacking __name__", "__name__, _ = '__main__', None\n")
    add("List unpacking __name__", "[__name__] = ['__main__']\n")
    add("Star unpacking __name__", "*__name__, = ['__main__']\n")
    add("Multi-target assignment __name__", "_ = __name__ = '__main__'\n")
    add("Dict comprehension __name__", "{__name__: 1 for __name__ in ['__main__']}\n")
    add("Set comprehension __name__", "{__name__ for __name__ in ['__main__']}\n")
    add("Generator expression __name__", "next(__name__ for __name__ in ['__main__'])\n")
    add("Async function def __name__", "async def __name__(): pass\n")
    add("Lambda __name__ param", "f = lambda __name__: __name__\n")

    # === STRING OBFUSCATION ===
    add("str(bytearray) obfuscation", "x = str(bytearray([102, 95, 98, 97, 99, 107]), 'ascii')\n")
    add("str(bytes) obfuscation", "x = str(bytes([102, 95, 98, 97, 99, 107]), 'ascii')\n")
    add("bytes().decode() obfuscation", "x = bytes([102, 95, 98, 97, 99, 107]).decode()\n")
    add("b-literal .decode()", "x = b'f_back'.decode()\n")
    add("str.join() obfuscation", 'x = "".join(["f", "_", "b", "a", "c", "k"])\n')
    add("String concatenation", 'x = "f_" + "back"\n')
    add("%-format obfuscation (single)", 'x = "%s_back" % "f"\n')
    add("%-format obfuscation (tuple)", 'x = "%s%s" % ("f_", "back")\n')
    add("f-string constant parts", "x = f\"{'f'}_back\"\n")
    add("Hex escape string literal", 'x = "\\x66\\x5f\\x62\\x61\\x63\\x6b"\n')
    add("Unicode escape string literal", 'x = "\\u0066\\u005f\\u0062\\u0061\\u0063\\u006b"\n')
    add("bytes.fromhex().decode()", 'x = bytes.fromhex("665f6261636b").decode()\n')
    add("bytearray.fromhex().decode()", 'x = bytearray.fromhex("665f6261636b").decode()\n')
    add("Reversed string join", 'x = "".join(reversed("kcab_f"))\n')
    add("Multi-line string concat", 'x = ("f_"\n"back")\n')
    add("Byte addition decode", "x = (b'f_' + b'back').decode()\n")

    # Non-default encoding bypass attempts
    add(
        "bytes([...]).decode('utf-16le')",
        'x = bytes([102, 0, 95, 0, 98, 0, 97, 0, 99, 0, 107, 0]).decode("utf-16le")\n',
    )
    add(
        "bytes.fromhex().decode(encoding='utf-16le')",
        'x = bytes.fromhex("66005f006200610063006b00").decode(encoding="utf-16le")\n',
    )
    add(
        "str(bytes([...]), 'utf-16le')",
        'x = str(bytes([102, 0, 95, 0, 98, 0, 97, 0, 99, 0, 107, 0]), "utf-16le")\n',
    )
    add(
        "str(bytearray([...]), encoding='utf-16le')",
        'x = str(bytearray([102, 0, 95, 0, 98, 0, 97, 0, 99, 0, 107, 0]), encoding="utf-16le")\n',
    )
    add(
        "bytes([...]).decode('latin-1')",
        'x = bytes([102, 95, 98, 97, 99, 107]).decode("latin-1")\n',
    )
    add(
        "b'literal'.decode('ascii')",
        "x = b'f_back'.decode('ascii')\n",
    )

    # Dynamic encoding (fail-closed)
    add(
        "str(bytes([...]), dynamic_encoding)",
        'enc = "utf-8"\nx = str(bytes([102, 95, 98, 97, 99, 107]), enc)\n',
    )
    add(
        "bytes([...]).decode(dynamic_encoding)",
        'enc = "utf-8"\nx = bytes([102, 95, 98, 97, 99, 107]).decode(enc)\n',
    )
    add(
        "str(bytearray([...]), encoding=var)",
        'enc = "ascii"\nx = str(bytearray([102, 95, 98, 97, 99, 107]), encoding=enc)\n',
    )

    # Oversized bytes literal (DoS guard)
    add(
        "bytes([0]*5000).decode() DoS",
        "x = bytes([" + ",".join(["0"] * 5000) + "]).decode()\n",
    )

    # === FORBIDDEN MODULE IMPORTS ===
    add("import os", "import os\n")
    add("import sys", "import sys\n")
    add("import subprocess", "import subprocess\n")
    add("import inspect", "import inspect\n")
    add("import ctypes", "import ctypes\n")
    add("import gc", "import gc\n")
    add("import time", "import time\n")
    add("import socket", "import socket\n")
    add("import pickle", "import pickle\n")
    add("import threading", "import threading\n")
    add("import ast", "import ast\n")
    add("import builtins", "import builtins\n")
    add("import operator", "import operator\n")
    add("import types", "import types\n")
    add("import functools", "import functools\n", expect_blocked=False)
    add("import weakref", "import weakref\n")
    add("import logging", "import logging\n")
    add("from os import path", "from os import path\n")
    add("import unittest.mock", "import unittest.mock\n")
    add("Dotted torch.distributed", "import torch.distributed\n", expect_blocked=False)
    add("import _thread", "import _thread\n")
    add("import multiprocessing", "import multiprocessing\n")
    add("import signal", "import signal\n")
    add("import shutil", "import shutil\n")
    add("import tempfile", "import tempfile\n")
    add("import marshal", "import marshal\n")
    add("import dis", "import dis\n")
    add("import code", "import code\n")
    add("import codecs", "import codecs\n")
    add("import base64", "import base64\n")
    add("import pdb", "import pdb\n")
    add("import runpy", "import runpy\n")
    add("import atexit", "import atexit\n")
    add("import traceback", "import traceback\n")
    add("import io", "import io\n")
    add("import http", "import http\n")
    add("import urllib", "import urllib\n")
    add(
        "from torch.multiprocessing import spawn",
        "from torch.multiprocessing import spawn\n",
        expect_blocked=False,
    )
    add("import numpy.ctypeslib", "import numpy.ctypeslib\n")

    # === FORBIDDEN BUILTINS / NAMES ===
    add("exec() call", "exec('print(1)')\n")
    add("eval() call", "eval('1+1')\n")
    add("compile() call", "compile('x=1', '<s>', 'exec')\n")
    add("__import__() call", "__import__('os')\n")
    add("setattr() reference", "f = setattr\n")
    add("getattr() reference", "f = getattr\n")
    add("delattr() reference", "f = delattr\n")
    add("globals() reference", "f = globals\n")
    add("locals() reference", "f = locals\n")
    add("open() reference", "f = open\n")
    add("vars() reference", "f = vars\n")
    add("dir() reference", "f = dir\n")
    add("type() reference", "f = type\n", expect_blocked=False)
    add("memoryview() reference", "f = memoryview\n")
    add("chr() reference", "f = chr\n")
    add("ord() reference", "f = ord\n")
    add("breakpoint() call", "breakpoint()\n")
    add("input() reference", "f = input\n")
    add("__build_class__ reference", "f = __build_class__\n")
    add("exec via method call", "import torch\ntorch.exec('x=1')\n")
    add("eval via method call", "import torch\ntorch.eval('1+1')\n")

    # === FRAME / INTROSPECTION ACCESS ===
    add("f_globals attribute", "import torch\nx = torch.f_globals\n")
    add("f_back attribute", "import torch\nx = torch.f_back\n")
    add("f_locals attribute", "import torch\nx = torch.f_locals\n")
    add("__globals__ attribute", "import torch\nx = torch.__globals__\n")
    add("__code__ attribute", "import torch\nx = torch.__code__\n")
    add("__subclasses__ attribute", "import torch\nx = torch.__subclasses__\n")
    add("__traceback__ attribute", "import torch\nx = torch.__traceback__\n")
    add("co_consts attribute", "import torch\nx = torch.co_consts\n")
    add("gi_frame attribute", "import torch\nx = torch.gi_frame\n")
    add("cr_frame attribute", "import torch\nx = torch.cr_frame\n")
    add("ag_frame attribute", "import torch\nx = torch.ag_frame\n")
    add("tb_frame attribute", "import torch\nx = torch.tb_frame\n")
    add("__getattribute__ attribute", "import torch\nx = torch.__getattribute__\n")
    add("f_code attribute", "import torch\nx = torch.f_code\n")
    add("f_builtins attribute", "import torch\nx = torch.f_builtins\n")
    add("gi_code attribute", "import torch\nx = torch.gi_code\n")
    add("cr_code attribute", "import torch\nx = torch.cr_code\n")
    add("ag_code attribute", "import torch\nx = torch.ag_code\n")
    add("tb_next attribute", "import torch\nx = torch.tb_next\n")
    add("__func__ attribute", "import torch\nx = torch.__func__\n")
    add("__self__ attribute", "import torch\nx = torch.__self__\n")
    add("__bases__ attribute", "import torch\nx = torch.__bases__\n")
    add("__mro__ attribute", "import torch\nx = torch.__mro__\n")
    add("__init_subclass__ attribute", "import torch\nx = torch.__init_subclass__\n")
    add("__class__ string literal", 'x = "__class__"\n', expect_blocked=False)
    add("__setattr__ string literal", 'x = "__setattr__"\n')
    add("__delattr__ string literal", 'x = "__delattr__"\n')
    add("__import__ string literal", 'x = "__import__"\n')
    add("__globals__ string literal", 'x = "__globals__"\n')
    add("__code__ string literal", 'x = "__code__"\n')

    # === CLASS HIERARCHY ESCAPE ===
    add("str class bases access", "x = ''.__class__.__bases__\n")
    add("int class subclasses", "x = (1).__class__.__subclasses__\n")

    # === TORCH MONKEY-PATCHING ===
    add("torch._C access", "import torch\nx = torch._C\n")
    add("torch._dynamo access", "import torch\nx = torch._dynamo\n")
    add("torch.compile alias", "import torch\nc = torch.compile\n")
    add("torch.load alias", "import torch\nld = torch.load\n")
    add("from torch import _C", "from torch import _C\n")
    add("from torch import compile", "from torch import compile\n")
    add("torch._dynamo.config write", "import torch\ntorch._dynamo.config.suppress_errors = True\n")
    add(
        "torch.nn.functional.cross_entropy overwrite",
        "import torch\nimport torch.nn.functional as F\nF.cross_entropy = lambda *a, **k: None\n",
    )
    add("from torch import load", "from torch import load\n")
    add("torch._inductor access", "import torch\nx = torch._inductor\n")
    add(
        "torch._inductor.config write",
        "import torch\ntorch._inductor.config.fallback_random = True\n",
    )
    add(
        "torch.autograd.backward overwrite",
        "import torch\ntorch.autograd.backward = lambda *a: None\n",
    )
    add(
        "torch.backends.cudnn.deterministic write",
        "import torch\ntorch.backends.cudnn.deterministic = True\n",
    )
    add("Star import from torch", "from torch import *\n")
    add("Star import from torch.nn", "from torch.nn import *\n")
    add("Star import from torch.nn.functional", "from torch.nn.functional import *\n")

    # === TIMER TAMPERING ===
    add("perf_counter attr access", "import torch\nx = torch.perf_counter\n")
    add("monotonic attr access", "import torch\nx = torch.monotonic\n")
    add("elapsed_time assignment", "import torch\ntorch.cuda.Event.elapsed_time = lambda *a: 0.0\n")
    add("synchronize assignment", "import torch\ntorch.cuda.synchronize = lambda: None\n")
    add("perf_counter_ns access", "import torch\nx = torch.perf_counter_ns\n")
    add("monotonic_ns access", "import torch\nx = torch.monotonic_ns\n")
    add("_perf_counter string", 'x = "_perf_counter"\n')
    add("_monotonic string", 'x = "_monotonic"\n')
    add("_REAL_PC_ID string", 'x = "_REAL_PC_ID"\n')
    add("_REAL_MONO_ID string", 'x = "_REAL_MONO_ID"\n')
    add("_REAL_ET_ID string", 'x = "_REAL_ET_ID"\n')
    add("_cuda_elapsed_time string", 'x = "_cuda_elapsed_time"\n')
    add("_cuda_synchronize string", 'x = "_cuda_synchronize"\n')

    # === GRADIENT TOGGLE ===
    add("set_grad_enabled call", "import torch\ntorch.set_grad_enabled(False)\n")
    add("inference_mode call", "import torch\ntorch.inference_mode()\n")

    # === __builtins__ / __dict__ ACCESS ===
    add("__builtins__ name access", "x = __builtins__\n")
    add("__dict__ attribute", "import torch\nx = torch.__dict__\n")
    add("object.__setattr__", "import torch\nobject.__setattr__(torch, 'x', 1)\n")
    add("object.__delattr__", "import torch\nobject.__delattr__(torch, 'x')\n")

    # === MOCK / PATCHING ===
    add("mock.patch string", 'x = "mock.patch"\n')
    add("MagicMock string", 'x = "MagicMock"\n')
    add("unittest.mock string", 'x = "unittest.mock"\n')

    # === OPERATOR HELPERS ===
    add("attrgetter string", 'x = "attrgetter"\n')
    add("methodcaller string", 'x = "methodcaller"\n')
    add("operator.attrgetter string", 'x = "operator.attrgetter"\n')

    # === HIDDEN MODULES / SENSITIVE KEYS ===
    add("_hidden_modules string", 'x = "_hidden_modules"\n')
    add("_sensitive_keys string", 'x = "_sensitive_keys"\n')

    # === OPTIMIZER INTERNALS ===
    add("_opt_impl string", 'x = "_opt_impl"\n')
    add("_grad_snapshot_gpu string", 'x = "_grad_snapshot_gpu"\n')
    add("step_count string", 'x = "step_count"\n')

    # === sys.modules ACCESS ===
    add("sys.modules string", 'x = "sys.modules"\n')

    # === VALIDATOR INTERNAL STRINGS ===
    add("_CACHE string", 'x = "_CACHE"\n')
    add("initial_state string", 'x = "initial_state"\n')
    add("GradientCapturingOptimizer string", 'x = "GradientCapturingOptimizer"\n')
    add("captured_gradients string", 'x = "captured_gradients"\n')

    # === FORBIDDEN DECORATORS ===
    add("@exec decorator", "@exec\ndef f(): pass\n")
    add("@eval decorator", "@eval\ndef f(): pass\n")

    # === FORBIDDEN IMPORT SUBSTRINGS ===
    add("cpp_extension import", "from torch.utils import cpp_extension\n")
    add("cpp_extension in dotted import", "import torch.utils.cpp_extension\n")

    # === DYNAMIC IMPORT / MODULE ACCESS ===
    add("importlib string", 'x = "importlib"\n')
    add("import_module string", 'x = "import_module"\n')
    add("__import__ via name", "f = __import__\n")

    # === GC / INTROSPECTION STRINGS ===
    add("get_objects string", 'x = "get_objects"\n')
    add("get_referrers string", 'x = "get_referrers"\n')
    add("get_referents string", 'x = "get_referents"\n')

    # === NESTED / INDIRECT ATTACKS ===
    add(
        "Nested class with __name__ attr",
        "class X:\n    __name__ = '__main__'\n",
    )
    add(
        "Lambda body with forbidden name",
        "f = lambda: exec\n",
    )
    add(
        "Comprehension hiding setattr",
        "_ = [setattr for _ in range(1)]\n",
    )
    add(
        "Default arg hiding forbidden func",
        "def f(x=exec): pass\n",
    )
    add(
        "Annotation hiding forbidden string",
        "def f(x: '__globals__' = None): pass\n",
    )

    # === DOUBLE / TRIPLE ENCODING ===
    add(
        "Nested bytes decode",
        "x = bytes([95, 95, 103, 108, 111, 98, 97, 108, 115, 95, 95]).decode('utf-8')\n",
    )
    add(
        "Bytearray list + decode",
        "x = bytearray([95, 95, 103, 108, 111, 98, 97, 108, 115, 95, 95]).decode()\n",
    )

    # === MULTI-GPU RELAXATION ABUSE ===
    # These try to weaponize the modules/builtins that were unblocked for multi-GPU.
    # The imports themselves are allowed, but using them for forbidden operations must
    # still be caught by other layers of the scanner.
    add(
        "torch.distributed + exec via string",
        'import torch.distributed\nexec("import os")\n',
    )
    add(
        "functools + globals() access",
        "import functools\ng = globals()\n",
    )
    add(
        "functools.partial wrapping eval",
        "import functools\nf = functools.partial(eval, '1+1')\n",
    )
    add(
        "type() + __bases__ escape",
        "t = type.__bases__\n",
    )
    add(
        "type() + __subclasses__ escape",
        "t = type.__subclasses__\n",
    )
    add(
        "__class__.__bases__ escape",
        "class X:\n    b = X.__class__.__bases__\n",
    )
    add(
        "__class__.__mro__ escape",
        "class X:\n    m = X.__class__.__mro__\n",
    )
    add(
        "torch.multiprocessing + os import",
        "from torch.multiprocessing import spawn\nimport os\n",
    )
    add(
        "torch.distributed + inspect import",
        "import torch.distributed\nimport inspect\n",
    )
    add(
        "functools + __import__",
        "import functools\n__import__('os')\n",
    )

    return cases


def main() -> int:
    scanner = _load_scanner()

    print("=" * 60)
    print("  Templar Crusades — Security Scanner Regression Suite")
    print("  (tests env.py production scanner used by simulate_validator)")
    print("=" * 60)

    failures: list[str] = []
    total_checks = 0
    relaxed_count = 0

    for case_name, case_code, expect_blocked in _bypass_cases():
        blocked, reason = _security_blocked(scanner, case_code)
        total_checks += 1

        if expect_blocked:
            ok = blocked
            status = "\033[32mBLOCKED\033[0m" if ok else "\033[31mBYPASSED\033[0m"
            print(f"\n  [{case_name}]")
            print(f"    {status} | {reason}")
            if not ok:
                failures.append(case_name)
        else:
            relaxed_count += 1
            ok = not blocked
            status = "\033[36mALLOWED\033[0m" if ok else "\033[33mBLOCKED (unexpected)\033[0m"
            print(f"\n  [{case_name}] (multi-GPU relaxation)")
            print(f"    {status} | {reason}")
            if not ok:
                failures.append(f"{case_name} — should be allowed")

    passed = total_checks - len(failures)
    print("\n" + "=" * 60)
    print(f"  Result: {passed}/{total_checks} checks passed")
    print(f"  ({relaxed_count} cases intentionally allowed for multi-GPU)")
    if failures:
        print(f"\n  \033[31mFAILURES ({len(failures)}):\033[0m")
        for f in failures:
            print(f"    - {f}")
    else:
        print("\n  \033[32mAll attack vectors pass.\033[0m")
    print("=" * 60)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
