"""benchmarks/grading/humaneval.py - the HumanEval harness.

Each item ships `test` (a check(candidate) function) and `entry_point`. This
module only arranges for the dataset's own verdict to be reached; it never
states an answer.
"""

from __future__ import annotations

import re

NAME = 'humaneval'

# Solutions are graded as function definitions: a model that reconsiders and
# follows a complete solution with a short usage snippet must not have that
# snippet mistaken for its answer.
WANT_DEF = True


def build_program(item: dict, code: str) -> str | None:
    """The full program to execute, or None when the answer has no usable code."""
    if not code.strip():
        return None
    entry = item['entry_point']
    # A model asked for the complete function normally returns it. When it
    # returned only a body -- the completion-style answer the stub invites --
    # the stub is prepended so that answer is graded rather than discarded.
    if not re.search(rf'^\s*(async\s+)?def\s+{re.escape(entry)}\b', code, re.M):
        code = item['prompt'] + '\n' + code
    return '\n\n'.join([code, item['test'], f'check({entry})', ''])


def reference_answer(item: dict) -> str | None:
    """The dataset's own solution, dressed as if a model had answered it.

    Returns None when the item ships no canonical_solution.
    """
    body = item.get('canonical_solution')
    code = item['prompt'] + body if body else None
    return f'```python\n{code}\n```\n' if code else None
