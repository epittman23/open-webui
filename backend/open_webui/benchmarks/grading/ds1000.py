"""benchmarks/grading/ds1000.py - the DS-1000 harness.

Each item carries `code_context`, which defines test_execution(solution). This
module only arranges for the dataset's own verdict to be reached; it never
states an answer.
"""

from __future__ import annotations

import json
import re

NAME = 'ds1000'

# DS-1000 solutions are graded whole (not required to be a def), so the last
# fenced block wins outright rather than the last one containing a def.
WANT_DEF = False


def build_program(item: dict, code: str) -> str | None:
    """The full program to execute, or None when the answer has no usable code."""
    if not code.strip():
        return None
    # json.dumps emits a valid Python string literal, which keeps the model's
    # quotes, backslashes and newlines out of the program's syntax.
    solution = json.dumps(code)
    parts = [item['code_context'], '',
             f'__solution = {solution}',
             'test_execution(__solution)']
    # Some items additionally constrain the surface form (a required API, a
    # banned keyword). The upstream runner calls it when present, so so do we.
    if re.search(r'^def\s+test_string\s*\(', item['code_context'], re.M):
        parts.append('test_string(__solution)')
    return '\n'.join(parts) + '\n'


def reference_answer(item: dict) -> str | None:
    """The dataset's own solution, dressed as if a model had answered it.

    Returns None when the item ships no reference_code.
    """
    code = item.get('reference_code')
    return f'```python\n{code}\n```\n' if code else None
