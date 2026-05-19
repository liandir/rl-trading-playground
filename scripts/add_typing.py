"""Apply a lighter typing pass to a tree of Python sources.

For each ``FunctionDef``/``AsyncFunctionDef`` the script will:

1. Add type annotations to bare parameters when the type can be safely inferred
   from a literal default value (``bool``, ``int``, ``float``, ``str``,
   ``list``/``dict``/``tuple`` containers) or from a small allow-list of
   recognised PyTorch defaults (``torch.tanh`` → ``Callable[[torch.Tensor],
   torch.Tensor]``).
2. Convert annotations whose default is the literal ``None`` to use ``| None``
   (e.g. ``device: torch.device = None`` → ``device: torch.device | None =
   None``).
3. Add ``-> None`` to functions whose body never returns a value.
4. Update Google-style ``(Any)`` Args entries in the function's docstring to
   match the resolved annotation.
5. Ensure ``from collections.abc import Callable`` is present when the file
   newly references ``Callable``.

Anything that is already annotated, or where the script cannot confidently
infer a type, is left untouched.

Usage::

    python3 scripts/add_typing.py src/ [--dry-run] [--write]

The script edits files in place when ``--write`` is given; otherwise it prints
a unified diff to stdout.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


_TORCH_ACTIVATIONS = {
    "torch.tanh",
    "torch.relu",
    "torch.sigmoid",
    "torch.nn.functional.relu",
    "torch.nn.functional.tanh",
    "torch.nn.functional.sigmoid",
    "F.relu",
    "F.tanh",
    "F.sigmoid",
}

_CALLABLE_TENSOR = "Callable[[torch.Tensor], torch.Tensor]"
_TORCH_MODULE = "torch.nn.Module"


def _annotation_text(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _infer_from_default(default: ast.expr) -> str | None:
    if isinstance(default, ast.Constant):
        value = default.value
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if value is None:
            return None  # Need accompanying base annotation to wrap with | None.
    if isinstance(default, ast.List):
        return "list"
    if isinstance(default, ast.Dict):
        return "dict"
    if isinstance(default, ast.Tuple):
        return "tuple"
    if isinstance(default, ast.Set):
        return "set"
    if isinstance(default, ast.Attribute):
        text = ast.unparse(default)
        if text in _TORCH_ACTIVATIONS:
            return _CALLABLE_TENSOR
    if isinstance(default, ast.Call):
        text = ast.unparse(default)
        if text == "torch.nn.Identity()":
            return _TORCH_MODULE
    return None


def _default_is_none(default: ast.expr | None) -> bool:
    return isinstance(default, ast.Constant) and default.value is None


def _has_value_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if any reachable Return carries a non-None value."""

    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and child is not node:
            continue
        if isinstance(child, ast.Return):
            if child.value is None:
                continue
            if isinstance(child.value, ast.Constant) and child.value.value is None:
                continue
            return True
        if isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
            return True
    return False


# ---------------------------------------------------------------------------
# Parameter inventory
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParamPlan:
    name: str
    arg_node: ast.arg
    existing: str | None
    has_default: bool
    default_node: ast.expr | None
    default_is_none: bool
    inferred: str | None  # what we'd like to annotate it with (only when adding new)
    final_annotation: str | None  # the final annotation we want in source (existing or new)


def _gather_params(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ParamPlan]:
    args = func.args
    plans: list[ParamPlan] = []

    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    default_offset = len(positional) - len(defaults)

    for index, arg in enumerate(positional):
        has_default = index >= default_offset
        default = defaults[index - default_offset] if has_default else None
        plans.append(_plan_for_arg(arg, has_default, default))

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        plans.append(_plan_for_arg(arg, default is not None, default))

    return plans


def _plan_for_arg(arg: ast.arg, has_default: bool, default: ast.expr | None) -> ParamPlan:
    existing = _annotation_text(arg.annotation)
    is_none_default = has_default and _default_is_none(default)
    inferred: str | None = None
    final = existing

    if existing is None:
        if has_default and default is not None and not is_none_default:
            inferred = _infer_from_default(default)
            final = inferred
        # Otherwise: cannot infer (no default, or default is plain None).
    else:
        if is_none_default and "None" not in existing:
            final = f"{existing} | None"

    return ParamPlan(
        name=arg.arg,
        arg_node=arg,
        existing=existing,
        has_default=has_default,
        default_node=default,
        default_is_none=is_none_default,
        inferred=inferred,
        final_annotation=final,
    )


# ---------------------------------------------------------------------------
# Source editing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TextEdit:
    """A single targeted text edit, applied right-to-left to preserve offsets."""

    start: int  # absolute character offset in the source
    end: int  # exclusive
    new_text: str


def _line_starts(source: str) -> list[int]:
    starts = [0]
    for idx, ch in enumerate(source):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def _abs_offset(line_starts: list[int], line: int, col: int) -> int:
    return line_starts[line - 1] + col


def _plan_arg_annotation_edit(
    source: str,
    line_starts: list[int],
    plan: ParamPlan,
) -> TextEdit | None:
    """If the arg has no existing annotation but we now want one, emit an edit."""

    if plan.existing is not None:
        return None
    if plan.final_annotation is None:
        return None

    arg = plan.arg_node
    name_start = _abs_offset(line_starts, arg.lineno, arg.col_offset)
    name_end = name_start + len(plan.name)

    # Emit `: TYPE` immediately after the name. If a default follows directly
    # (e.g. `act=torch.tanh`) normalise spacing to `: TYPE = default`.
    rest_idx = name_end
    follows_equals = rest_idx < len(source) and source[rest_idx] == "="
    if follows_equals:
        return TextEdit(name_end, name_end + 1, f": {plan.final_annotation} = ")
    return TextEdit(name_end, name_end, f": {plan.final_annotation}")


def _plan_optional_widen_edit(
    source: str,
    line_starts: list[int],
    plan: ParamPlan,
) -> TextEdit | None:
    """Widen `X = None` to `X | None = None` when the existing annotation lacks None."""

    if plan.existing is None or not plan.default_is_none:
        return None
    if "None" in plan.existing:
        return None
    arg = plan.arg_node
    assert arg.annotation is not None
    ann_start = _abs_offset(line_starts, arg.annotation.lineno, arg.annotation.col_offset)
    ann_end = _abs_offset(line_starts, arg.annotation.end_lineno, arg.annotation.end_col_offset)
    return TextEdit(ann_start, ann_end, plan.final_annotation or plan.existing)


def _plan_return_annotation_edit(
    source: str,
    line_starts: list[int],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> TextEdit | None:
    """Add `-> None` when missing and the function never returns a value."""

    if func.returns is not None:
        return None
    if _has_value_return(func):
        return None

    # Find the colon that ends the signature line. We must skip colons inside
    # default expressions or annotations by tracking parenthesis depth and
    # respecting string/comment boundaries.
    sig_start = _abs_offset(line_starts, func.lineno, func.col_offset)
    # body[0] is the first statement; its position bounds the signature end.
    if not func.body:
        return None
    body_first = func.body[0]
    body_start = _abs_offset(line_starts, body_first.lineno, body_first.col_offset)

    sig_text = source[sig_start:body_start]
    # Find the rightmost colon at paren-depth 0, ignoring colons inside strings.
    paren = 0
    bracket = 0
    brace = 0
    in_string: str | None = None
    string_skip = 0
    last_colon = -1
    i = 0
    while i < len(sig_text):
        ch = sig_text[i]
        if string_skip:
            string_skip -= 1
            i += 1
            continue
        if in_string is not None:
            if ch == "\\":
                string_skip = 1
            elif sig_text[i : i + len(in_string)] == in_string:
                i += len(in_string)
                in_string = None
                continue
            i += 1
            continue
        if ch in ("'", '"'):
            # Triple-quoted?
            triple = sig_text[i : i + 3]
            if triple in ("'''", '"""'):
                in_string = triple
                i += 3
                continue
            in_string = ch
            i += 1
            continue
        if ch == "#":
            nl = sig_text.find("\n", i)
            i = len(sig_text) if nl == -1 else nl
            continue
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == ":" and paren == 0 and bracket == 0 and brace == 0:
            last_colon = i
        i += 1

    if last_colon < 0:
        return None
    abs_colon = sig_start + last_colon
    return TextEdit(abs_colon, abs_colon, " -> None")


# ---------------------------------------------------------------------------
# Docstring (Any) replacement
# ---------------------------------------------------------------------------


_ANY_LINE_RE = re.compile(
    r"^(?P<indent>\s+)(?P<name>\*{0,2}[A-Za-z_][A-Za-z0-9_]*) \(Any\)(?P<rest>:.*)$",
    re.MULTILINE,
)


def _docstring_node(func: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Constant | None:
    if not func.body:
        return None
    first = func.body[0]
    if not isinstance(first, ast.Expr):
        return None
    if not isinstance(first.value, ast.Constant) or not isinstance(first.value.value, str):
        return None
    return first.value


def _plan_docstring_edit(
    source: str,
    line_starts: list[int],
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name_to_type: dict[str, str],
) -> TextEdit | None:
    node = _docstring_node(func)
    if node is None:
        return None
    doc_start = _abs_offset(line_starts, node.lineno, node.col_offset)
    doc_end = _abs_offset(line_starts, node.end_lineno, node.end_col_offset)
    doc_src = source[doc_start:doc_end]

    def repl(match: re.Match[str]) -> str:
        name = match.group("name").lstrip("*")
        typ = name_to_type.get(name)
        if typ is None:
            return match.group(0)
        return f'{match.group("indent")}{match.group("name")} ({typ}){match.group("rest")}'

    new_doc = _ANY_LINE_RE.sub(repl, doc_src)
    if new_doc == doc_src:
        return None
    return TextEdit(doc_start, doc_end, new_doc)


# ---------------------------------------------------------------------------
# File orchestration
# ---------------------------------------------------------------------------


def _ensure_callable_import(source: str) -> str:
    """Add `from collections.abc import Callable` if Callable is now used."""

    if "Callable" not in source:
        return source
    # Already imported?
    if re.search(r"^from collections\.abc import [^\n]*\bCallable\b", source, re.MULTILINE):
        return source
    if re.search(r"^from typing import [^\n]*\bCallable\b", source, re.MULTILINE):
        return source
    # Insert after the module docstring (if any) and any leading future imports.
    tree = ast.parse(source)
    insert_line = 1  # 1-based
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            insert_line = node.end_lineno + 1
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_line = node.end_lineno + 1
            continue
        break
    lines = source.splitlines(keepends=True)
    # Try to merge with an existing `from collections.abc import ...` line.
    merged = False
    for idx, line in enumerate(lines):
        m = re.match(r"^from collections\.abc import (.+)$", line.rstrip("\n"))
        if m:
            existing = [s.strip() for s in m.group(1).split(",")]
            if "Callable" not in existing:
                existing.append("Callable")
                existing.sort()
                lines[idx] = f"from collections.abc import {', '.join(existing)}\n"
            merged = True
            break
    if merged:
        return "".join(lines)
    new_line = "from collections.abc import Callable\n"
    lines.insert(insert_line - 1, new_line)
    return "".join(lines)


def _apply_edits(source: str, edits: list[TextEdit]) -> str:
    # Sort right-to-left to keep earlier offsets valid.
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        source = source[: edit.start] + edit.new_text + source[edit.end :]
    return source


def process_file(path: Path) -> tuple[str, str]:
    """Return ``(original, modified)`` source for ``path``.

    Args:
        path (Path): Python source file to analyse.

    Returns:
        tuple[str, str]: The original and the transformed source. They are
            identical when no edits applied.
    """

    original = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(original, filename=str(path))
    except SyntaxError:
        return original, original

    line_starts = _line_starts(original)
    edits: list[TextEdit] = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = _gather_params(func)
        # Build edits per param.
        name_to_type: dict[str, str] = {}
        for plan in params:
            if plan.name in {"self", "cls"}:
                continue
            if plan.final_annotation is not None:
                name_to_type[plan.name] = plan.final_annotation
            edit = _plan_arg_annotation_edit(original, line_starts, plan)
            if edit:
                edits.append(edit)
            edit = _plan_optional_widen_edit(original, line_starts, plan)
            if edit:
                edits.append(edit)

        ret_edit = _plan_return_annotation_edit(original, line_starts, func)
        if ret_edit:
            edits.append(ret_edit)

        doc_edit = _plan_docstring_edit(original, line_starts, func, name_to_type)
        if doc_edit:
            edits.append(doc_edit)

    modified = _apply_edits(original, edits)
    modified = _ensure_callable_import(modified)
    return original, modified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory to walk for .py files.")
    parser.add_argument("--write", action="store_true", help="Apply edits in place.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files.")
    args = parser.parse_args()

    paths = [args.root] if args.root.is_file() else sorted(args.root.rglob("*.py"))
    if args.limit:
        paths = paths[: args.limit]
    changed = 0
    for path in paths:
        original, modified = process_file(path)
        if original == modified:
            continue
        changed += 1
        if args.write:
            path.write_text(modified, encoding="utf-8")
            print(f"updated {path}")
        else:
            for line in difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=str(path),
                tofile=str(path) + " (modified)",
            ):
                sys.stdout.write(line)
    print(f"\n{changed} of {len(paths)} files would change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
