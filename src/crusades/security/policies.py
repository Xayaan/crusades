"""Security policy re-exports with type annotations.

Single source of truth remains in crusades.core.security_defs.
This module provides a cleaner import path for consumers that
want ``from crusades.security.policies import FORBIDDEN_MODULES``
instead of reaching into ``crusades.core.security_defs``.
"""

from crusades.core.security_defs import (
    ALLOWED_TORCH_ASSIGNMENT_PREFIXES as ALLOWED_TORCH_ASSIGNMENT_PREFIXES,
)
from crusades.core.security_defs import (
    ALLOWED_TORCH_SUBMODULE_IMPORTS as ALLOWED_TORCH_SUBMODULE_IMPORTS,
)
from crusades.core.security_defs import (
    FORBIDDEN_ASSIGNMENT_ATTRS as FORBIDDEN_ASSIGNMENT_ATTRS,
)
from crusades.core.security_defs import (
    FORBIDDEN_ATTR_CALLS as FORBIDDEN_ATTR_CALLS,
)
from crusades.core.security_defs import (
    FORBIDDEN_BACKEND_TOGGLE_ATTRS as FORBIDDEN_BACKEND_TOGGLE_ATTRS,
)
from crusades.core.security_defs import (
    FORBIDDEN_BUILTINS as FORBIDDEN_BUILTINS,
)
from crusades.core.security_defs import (
    FORBIDDEN_CUDNN_ATTRS as FORBIDDEN_CUDNN_ATTRS,
)
from crusades.core.security_defs import (
    FORBIDDEN_DIRECT_CALLS as FORBIDDEN_DIRECT_CALLS,
)
from crusades.core.security_defs import (
    FORBIDDEN_DOTTED_MODULES as FORBIDDEN_DOTTED_MODULES,
)
from crusades.core.security_defs import (
    FORBIDDEN_GC_ATTRS as FORBIDDEN_GC_ATTRS,
)
from crusades.core.security_defs import (
    FORBIDDEN_GRAD_TOGGLE_CALLS as FORBIDDEN_GRAD_TOGGLE_CALLS,
)
from crusades.core.security_defs import (
    FORBIDDEN_IMPORT_SUBSTRINGS as FORBIDDEN_IMPORT_SUBSTRINGS,
)
from crusades.core.security_defs import (
    FORBIDDEN_INTROSPECTION_ATTRS as FORBIDDEN_INTROSPECTION_ATTRS,
)
from crusades.core.security_defs import (
    FORBIDDEN_MODULES as FORBIDDEN_MODULES,
)
from crusades.core.security_defs import (
    FORBIDDEN_NAMES as FORBIDDEN_NAMES,
)
from crusades.core.security_defs import (
    FORBIDDEN_OBJECT_DUNDER_ATTRS as FORBIDDEN_OBJECT_DUNDER_ATTRS,
)
from crusades.core.security_defs import (
    FORBIDDEN_STRINGS as FORBIDDEN_STRINGS,
)
from crusades.core.security_defs import (
    FORBIDDEN_SYS_MODULE_NAMES as FORBIDDEN_SYS_MODULE_NAMES,
)
from crusades.core.security_defs import (
    FORBIDDEN_TIMER_ATTRS as FORBIDDEN_TIMER_ATTRS,
)
from crusades.core.security_defs import (
    FORBIDDEN_TORCH_ATTRIBUTE_ALIASES as FORBIDDEN_TORCH_ATTRIBUTE_ALIASES,
)
from crusades.core.security_defs import (
    FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS as FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS,
)
from crusades.core.security_defs import (
    FORBIDDEN_TORCH_CONFIG_MODULES as FORBIDDEN_TORCH_CONFIG_MODULES,
)
from crusades.core.security_defs import (
    FORBIDDEN_TORCH_SYMBOL_IMPORTS as FORBIDDEN_TORCH_SYMBOL_IMPORTS,
)
from crusades.core.security_defs import (
    SuspiciousConstructionError as SuspiciousConstructionError,
)
from crusades.core.security_defs import (
    forbidden_name_binding_reason as forbidden_name_binding_reason,
)
from crusades.core.security_defs import (
    try_decode_bytes_node as try_decode_bytes_node,
)
from crusades.core.security_defs import (
    try_decode_str_bytes_constructor as try_decode_str_bytes_constructor,
)
from crusades.core.security_defs import (
    try_resolve_concat as try_resolve_concat,
)
from crusades.core.security_defs import (
    try_resolve_format as try_resolve_format,
)
from crusades.core.security_defs import (
    try_resolve_fstring as try_resolve_fstring,
)
from crusades.core.security_defs import (
    try_resolve_join as try_resolve_join,
)

__all__: list[str] = [
    # Policy constant sets
    "ALLOWED_TORCH_ASSIGNMENT_PREFIXES",
    "ALLOWED_TORCH_SUBMODULE_IMPORTS",
    "FORBIDDEN_ASSIGNMENT_ATTRS",
    "FORBIDDEN_ATTR_CALLS",
    "FORBIDDEN_BACKEND_TOGGLE_ATTRS",
    "FORBIDDEN_BUILTINS",
    "FORBIDDEN_CUDNN_ATTRS",
    "FORBIDDEN_DIRECT_CALLS",
    "FORBIDDEN_DOTTED_MODULES",
    "FORBIDDEN_GC_ATTRS",
    "FORBIDDEN_GRAD_TOGGLE_CALLS",
    "FORBIDDEN_IMPORT_SUBSTRINGS",
    "FORBIDDEN_INTROSPECTION_ATTRS",
    "FORBIDDEN_MODULES",
    "FORBIDDEN_NAMES",
    "FORBIDDEN_OBJECT_DUNDER_ATTRS",
    "FORBIDDEN_STRINGS",
    "FORBIDDEN_SYS_MODULE_NAMES",
    "FORBIDDEN_TIMER_ATTRS",
    "FORBIDDEN_TORCH_ATTRIBUTE_ALIASES",
    "FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS",
    "FORBIDDEN_TORCH_CONFIG_MODULES",
    "FORBIDDEN_TORCH_SYMBOL_IMPORTS",
    # Exceptions
    "SuspiciousConstructionError",
    # Detection helpers
    "forbidden_name_binding_reason",
    "try_decode_bytes_node",
    "try_decode_str_bytes_constructor",
    "try_resolve_concat",
    "try_resolve_format",
    "try_resolve_fstring",
    "try_resolve_join",
]
