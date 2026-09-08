"""benchmarks/grading/mbpp.py - the MBPP (sanitized) harness.

Each item ships `test_imports` and `test_list` (three asserts). This module
only arranges for the dataset's own verdict to be reached; it never states an
answer.
"""

from __future__ import annotations

NAME = 'mbpp'
WANT_DEF = True


def build_program(item: dict, code: str) -> str | None:
    """The full program to execute, or None when the answer has no usable code."""
    if not code.strip():
        return None
    parts = list(item.get('test_imports') or [])
    parts.append(code)
    parts.extend(item.get('test_list') or [])
    return '\n\n'.join(parts) + '\n'


def reference_answer(item: dict) -> str | None:
    """The dataset's own solution, dressed as if a model had answered it.

    Returns None when the item ships no reference `code`.
    """
    code = item.get('code')
    return f'```python\n{code}\n```\n' if code else None
