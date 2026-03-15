"""
Shared security scanner for miner code validation.

Pure AST-based scanning — no heavy dependencies (torch, fastapi, etc.).
Used by both the production evaluator (environments/templar/env.py) and
the local Arbos agent (arbos/agent.py).

All rules come from security_defs.py; this module only provides the
scanning logic.
"""

from __future__ import annotations

import ast

from crusades.core.security_defs import (
    _MAX_BYTES_LITERAL_ELTS,
    ALLOWED_TORCH_ASSIGNMENT_PREFIXES,
    ALLOWED_TORCH_SUBMODULE_IMPORTS,
    FORBIDDEN_ASSIGNMENT_ATTRS,
    FORBIDDEN_ATTR_CALLS,
    FORBIDDEN_BACKEND_TOGGLE_ATTRS,
    FORBIDDEN_BUILTINS,
    FORBIDDEN_CUDNN_ATTRS,
    FORBIDDEN_DIRECT_CALLS,
    FORBIDDEN_DOTTED_MODULES,
    FORBIDDEN_GC_ATTRS,
    FORBIDDEN_GRAD_TOGGLE_CALLS,
    FORBIDDEN_IMPORT_SUBSTRINGS,
    FORBIDDEN_INTROSPECTION_ATTRS,
    FORBIDDEN_MODULES,
    FORBIDDEN_NAMES,
    FORBIDDEN_OBJECT_DUNDER_ATTRS,
    FORBIDDEN_STRINGS,
    FORBIDDEN_SYS_MODULE_NAMES,
    FORBIDDEN_TIMER_ATTRS,
    FORBIDDEN_TORCH_ATTRIBUTE_ALIASES,
    FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS,
    FORBIDDEN_TORCH_CONFIG_MODULES,
    FORBIDDEN_TORCH_SYMBOL_IMPORTS,
    SuspiciousConstructionError,
    forbidden_name_binding_reason,
    try_decode_bytes_node,
    try_decode_str_bytes_constructor,
    try_resolve_concat,
    try_resolve_format,
    try_resolve_fstring,
    try_resolve_join,
)


def scan_for_dangerous_patterns(tree: ast.AST) -> tuple[bool, str | None]:
    """AST scan to reject forbidden code patterns.

    Returns ``(True, None)`` when the tree is clean, or
    ``(False, "Line N: reason")`` on the first violation found.
    """
    _forbidden_names = FORBIDDEN_NAMES
    _forbidden_modules = FORBIDDEN_MODULES
    _forbidden_dotted_modules = FORBIDDEN_DOTTED_MODULES
    _forbidden_torch_symbol_imports = FORBIDDEN_TORCH_SYMBOL_IMPORTS
    _forbidden_torch_backend_symbol_imports = FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS
    _forbidden_torch_attribute_aliases = FORBIDDEN_TORCH_ATTRIBUTE_ALIASES
    _forbidden_builtins = FORBIDDEN_BUILTINS

    torch_aliases = {"torch"}
    torch_submodule_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    local_name = alias.asname or alias.name
                    if local_name != "torch":
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: aliasing torch is forbidden"
                    torch_aliases.add(local_name)
                elif alias.name.startswith("torch.") and alias.asname:
                    if alias.name in ALLOWED_TORCH_SUBMODULE_IMPORTS:
                        torch_submodule_aliases.add(alias.asname)

        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Name) and node.value.id in torch_aliases:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id != "torch":
                            line = getattr(node, "lineno", "?")
                            return False, f"Line {line}: aliasing torch is forbidden"
                        torch_aliases.add(target.id)

        if isinstance(node, ast.NamedExpr):
            if isinstance(node.value, ast.Name) and node.value.id in torch_aliases:
                if isinstance(node.target, ast.Name) and node.target.id != "torch":
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: aliasing torch via walrus operator is forbidden"
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in torch_aliases
                and node.value.attr in _forbidden_torch_attribute_aliases
            ):
                line = getattr(node, "lineno", "?")
                return False, (
                    f"Line {line}: binding torch.{node.value.attr} via walrus operator is forbidden"
                )

        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Tuple, ast.List)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Name) and elt.id in torch_aliases:
                    for target in node.targets:
                        names = []
                        if isinstance(target, (ast.Tuple, ast.List)):
                            names = [n.id for n in target.elts if isinstance(n, ast.Name)]
                        elif isinstance(target, ast.Name):
                            names = [target.id]
                        for n in names:
                            if n != "torch":
                                line = getattr(node, "lineno", "?")
                                return False, (
                                    f"Line {line}: aliasing torch via unpacking is forbidden"
                                )
                if (
                    isinstance(elt, ast.Attribute)
                    and isinstance(elt.value, ast.Name)
                    and elt.value.id in torch_aliases
                    and elt.attr in _forbidden_torch_attribute_aliases
                ):
                    line = getattr(node, "lineno", "?")
                    return False, (
                        f"Line {line}: binding torch.{elt.attr} via unpacking is forbidden"
                    )

        _attr_targets: list[ast.Attribute] = []
        if isinstance(node, ast.Assign):
            _attr_targets = [t for t in node.targets if isinstance(t, ast.Attribute)]
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute):
            _attr_targets = [node.target]
        elif isinstance(node, ast.Delete):
            _attr_targets = [t for t in node.targets if isinstance(t, ast.Attribute)]

        for target in _attr_targets:
            root_node = target.value
            while isinstance(root_node, ast.Attribute):
                root_node = root_node.value
            if not isinstance(root_node, ast.Name):
                continue
            root_name = root_node.id

            if root_name in torch_submodule_aliases:
                line = getattr(node, "lineno", "?")
                return (
                    False,
                    f"Line {line}: mutating {root_name}.{target.attr} is forbidden"
                    " (monkey-patching torch modules is not allowed)",
                )

            if root_name in torch_aliases:
                if isinstance(target.value, ast.Name):
                    line = getattr(node, "lineno", "?")
                    return (
                        False,
                        f"Line {line}: mutating {root_name}.{target.attr} is forbidden"
                        " (monkey-patching torch modules is not allowed)",
                    )
                parts: list[str] = []
                walk = target.value
                while isinstance(walk, ast.Attribute):
                    parts.append(walk.attr)
                    walk = walk.value
                if isinstance(walk, ast.Name):
                    parts.append(walk.id)
                    parent_path = ".".join(reversed(parts))
                    if any(
                        parent_path == pfx or parent_path.startswith(pfx + ".")
                        for pfx in ALLOWED_TORCH_ASSIGNMENT_PREFIXES
                    ):
                        continue
                    line = getattr(node, "lineno", "?")
                    return (
                        False,
                        f"Line {line}: mutating {parent_path}.{target.attr}"
                        " is forbidden (monkey-patching torch modules is not allowed)",
                    )

        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in torch_aliases
            and node.value.attr in _forbidden_torch_attribute_aliases
        ):
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: aliasing torch.{node.value.attr} is forbidden"

        if isinstance(node, ast.Name) and node.id in _forbidden_names:
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: reference to '{node.id}' is forbidden"

        name_binding_violation = forbidden_name_binding_reason(node)
        if name_binding_violation:
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: {name_binding_violation}"

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_OBJECT_DUNDER_ATTRS:
            if isinstance(node.value, ast.Name) and node.value.id == "object":
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "_C":
            if isinstance(node.value, ast.Name) and node.value.id in torch_aliases:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: torch._C access is forbidden"

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute):
                    if target.value.attr == "config" and isinstance(
                        target.value.value, ast.Attribute
                    ):
                        if target.value.value.attr in FORBIDDEN_TORCH_CONFIG_MODULES:
                            line = getattr(node, "lineno", "?")
                            return (
                                False,
                                f"Line {line}: modifying torch.{target.value.value.attr}.config is forbidden",
                            )

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "__class__":
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_TIMER_ATTRS:
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in FORBIDDEN_ASSIGNMENT_ATTRS:
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "__slots__":
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_GC_ATTRS:
            if isinstance(node.value, ast.Name) and node.value.id == "gc":
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_CUDNN_ATTRS:
            if (
                isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "cudnn"
            ):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_BACKEND_TOGGLE_ATTRS:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_GRAD_TOGGLE_CALLS:
                line = getattr(node, "lineno", "?")
                return (
                    False,
                    f"Line {line}: {func.attr}() is forbidden"
                    " (disabling gradients would bypass verification)",
                )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_DIRECT_CALLS:
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: {node.func.id}() is forbidden"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_ATTR_CALLS:
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: .{node.func.attr}() is forbidden"
                if node.func.attr == "compile":
                    if not (
                        isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
                    ):
                        line = getattr(node, "lineno", "?")
                        return (
                            False,
                            f"Line {line}: .compile() is forbidden (only torch.compile is allowed)",
                        )

        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module in _forbidden_modules or alias.name.startswith("importlib"):
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden import"
                for substr in FORBIDDEN_IMPORT_SUBSTRINGS:
                    if substr in alias.name:
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: forbidden import"
                for forbidden_path in _forbidden_dotted_modules:
                    if alias.name == forbidden_path or alias.name.startswith(forbidden_path + "."):
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: forbidden import"

        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: star imports (from ... import *) are forbidden"
            if not node.module:
                continue
            base_module = node.module.split(".")[0]
            if base_module in _forbidden_modules or node.module.startswith("importlib"):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden import"
            if node.module == "torch":
                for alias in node.names:
                    if alias.name in _forbidden_torch_symbol_imports:
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: importing torch.{alias.name} is forbidden"
            if node.module.startswith("torch.backends"):
                for alias in node.names:
                    if alias.name in _forbidden_torch_backend_symbol_imports:
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: importing torch backend toggle is forbidden"
            for substr in FORBIDDEN_IMPORT_SUBSTRINGS:
                if substr in node.module:
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden import"
            for forbidden_path in _forbidden_dotted_modules:
                if node.module == forbidden_path or node.module.startswith(forbidden_path + "."):
                    line = getattr(node, "lineno", "?")
                    return False, f"Line {line}: forbidden import"
            for alias in node.names:
                for substr in FORBIDDEN_IMPORT_SUBSTRINGS:
                    if substr in alias.name:
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: forbidden import"
                full_path = f"{node.module}.{alias.name}"
                for forbidden_path in _forbidden_dotted_modules:
                    if full_path == forbidden_path or full_path.startswith(forbidden_path + "."):
                        line = getattr(node, "lineno", "?")
                        return False, f"Line {line}: forbidden import"
                if full_path in ALLOWED_TORCH_SUBMODULE_IMPORTS:
                    local_name = alias.asname or alias.name
                    torch_submodule_aliases.add(local_name)

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _forbidden_builtins:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: {node.func.id}() is forbidden"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _forbidden_builtins:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: .{node.func.attr}() is forbidden"

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in torch_aliases
        ):
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: torch.load() is forbidden (uses pickle internally)"

        if isinstance(node, ast.Attribute) and node.attr == "ctypeslib":
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: ctypeslib access is forbidden"

        if isinstance(node, ast.Name) and node.id == "__builtins__":
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: __builtins__ access is forbidden"
        if isinstance(node, ast.Attribute) and node.attr == "__builtins__":
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: __builtins__ access is forbidden"

        if isinstance(node, ast.Attribute) and node.attr == "modules":
            if isinstance(node.value, ast.Name) and node.value.id in FORBIDDEN_SYS_MODULE_NAMES:
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "__dict__":
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: __dict__ access is forbidden"

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_INTROSPECTION_ATTRS:
            line = getattr(node, "lineno", "?")
            return False, f"Line {line}: forbidden pattern detected"

        if isinstance(node, ast.Attribute) and node.attr == "optimizer":
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                line = getattr(node, "lineno", "?")
                return False, f"Line {line}: accessing .optimizer attribute is forbidden"

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for deco in node.decorator_list:
                if isinstance(deco, ast.Name) and deco.id in _forbidden_names:
                    line = getattr(deco, "lineno", "?")
                    return False, f"Line {line}: decorator @{deco.id} is forbidden"

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("bytes", "bytearray")
            and node.args
            and len(node.args) == 1
            and isinstance(node.args[0], (ast.List, ast.Tuple))
            and len(node.args[0].elts) > _MAX_BYTES_LITERAL_ELTS
        ):
            line = getattr(node, "lineno", "?")
            return False, (
                f"Line {line}: bytes/bytearray literal with {len(node.args[0].elts)} "
                f"elements exceeds limit of {_MAX_BYTES_LITERAL_ELTS}"
            )

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
        ):
            for kw in node.keywords:
                if (
                    kw.arg == "weights_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                ):
                    line = getattr(node, "lineno", "?")
                    return False, (
                        f"Line {line}: weights_only=False is forbidden"
                        " (enables arbitrary code execution via pickle)"
                    )

    return True, None


def validate_code_security(code: str) -> tuple[bool, str | None]:
    """Full security validation of miner code.

    Runs AST pattern scanning, forbidden string checking, and obfuscation
    detection.  Returns ``(True, None)`` if clean, ``(False, reason)`` on
    the first violation.

    This is the single entry-point both the production evaluator and the
    local Arbos agent should use.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error at line {exc.lineno}: {exc.msg}"

    safe, danger_error = scan_for_dangerous_patterns(tree)
    if not safe:
        return False, f"Security violation: {danger_error}"

    _forbidden_strings = FORBIDDEN_STRINGS

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pattern in _forbidden_strings:
                if pattern in node.value:
                    line = getattr(node, "lineno", "?")
                    return False, (
                        f"Security violation: Line {line}: forbidden string pattern detected"
                    )

    _obfuscation_resolvers = [
        ("bytes decode", try_decode_bytes_node),
        ("str(bytes(...))", try_decode_str_bytes_constructor),
        ("str.join()", try_resolve_join),
        ("concatenation", try_resolve_concat),
        ("%-format", try_resolve_format),
        ("f-string", try_resolve_fstring),
    ]
    for node in ast.walk(tree):
        for label, resolver in _obfuscation_resolvers:
            try:
                resolved = resolver(node)
            except SuspiciousConstructionError as exc:
                line = getattr(node, "lineno", "?")
                return False, f"Security violation: Line {line}: {exc}"
            if resolved is not None:
                for pattern in _forbidden_strings:
                    if pattern in resolved:
                        line = getattr(node, "lineno", "?")
                        return False, (
                            f"Security violation: Line {line}: "
                            f"forbidden string constructed via {label}"
                        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            for pattern in _forbidden_strings:
                if node.attr == pattern:
                    line = getattr(node, "lineno", "?")
                    return False, (
                        f"Security violation: Line {line}: forbidden attribute name '{node.attr}'"
                    )

    return True, None
