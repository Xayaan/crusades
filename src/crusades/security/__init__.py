"""Security scanning and policy enforcement for miner code validation.

This package provides AST-based code validation that can be used outside
the Docker container environment.  Inside the container, env.py retains
its own copy of the scanning functions since it cannot import from
``crusades.security``.
"""

from crusades.security.scanner import CodeScanner

__all__ = ["CodeScanner"]
