"""Build static HTML documentation from repository docstrings.

The builder intentionally avoids importing project modules. It parses Python
source files with ``ast`` so docs can be generated without optional runtime
dependencies, API credentials, network access, or GPU libraries.
"""

from __future__ import annotations

import ast
import html
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
SOURCE_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts")


@dataclass(slots=True)
class ApiObject:
    """Documented Python object extracted from the syntax tree."""

    kind: str
    name: str
    qualname: str
    anchor: str
    bases: list[str]
    params: list[str]
    signature: str
    signature_html: str
    docstring: str
    lineno: int
    children: list["ApiObject"] = field(default_factory=list)


@dataclass(slots=True)
class ModuleDoc:
    """Rendered documentation payload for one Python module or script."""

    source_path: Path
    source_rel: Path
    module_name: str
    page_path: Path
    docstring: str
    objects: list[ApiObject]


def main() -> None:
    """Generate the documentation site under ``docs/``."""

    modules = [_parse_module(path) for root in SOURCE_ROOTS for path in sorted(root.rglob("*.py"))]
    _assert_documented_arguments(modules)
    _prepare_docs_dir()
    for module in modules:
        _write_module_page(module, modules)
    _write_directory_pages(modules)
    _write_api_index(modules)
    _write_markdown_pages(modules)
    _write_site_index(modules)
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote {len(modules)} API pages to {DOCS_DIR}")


def _prepare_docs_dir() -> None:
    """Create a clean API output directory while preserving hand-written docs."""

    api_dir = DOCS_DIR / "api"
    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir(parents=True, exist_ok=True)


def _parse_module(path: Path) -> ModuleDoc:
    """Parse one source file and collect its public docstrings."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_rel = path.relative_to(PROJECT_ROOT)
    if path.is_relative_to(PROJECT_ROOT / "src"):
        rel = path.relative_to(PROJECT_ROOT / "src")
        module_name = ".".join(("src", *rel.with_suffix("").parts))
    else:
        rel = source_rel
        module_name = ".".join(rel.with_suffix("").parts)

    page_path = DOCS_DIR / "api" / source_rel.with_suffix(".html")
    return ModuleDoc(
        source_path=path,
        source_rel=source_rel,
        module_name=module_name,
        page_path=page_path,
        docstring=ast.get_docstring(tree) or "",
        objects=[_parse_object(node) for node in tree.body if _is_documentable(node)],
    )


def _parse_object(node: ast.AST, parent: str = "") -> ApiObject:
    """Parse a class or function node into a serializable documentation object."""

    assert isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    children: list[ApiObject] = []
    qualname = f"{parent}.{node.name}" if parent else node.name
    if isinstance(node, ast.ClassDef):
        kind = "class"
        bases = [ast.unparse(base) for base in [*node.bases, *[kw.value for kw in node.keywords]]]
        params = []
        signature = _class_signature(node)
        signature_html = _class_signature_html(node)
        children = [_parse_object(child, qualname) for child in node.body if _is_documentable(child)]
    elif isinstance(node, ast.AsyncFunctionDef):
        kind = "async method" if parent else "async function"
        bases = []
        params = _documentable_arg_names(node.args)
        signature = _function_signature(node)
        signature_html = _function_signature_html(node)
    else:
        kind = "method" if parent else "function"
        bases = []
        params = _documentable_arg_names(node.args)
        signature = _function_signature(node)
        signature_html = _function_signature_html(node)

    return ApiObject(
        kind=kind,
        name=node.name,
        qualname=qualname,
        anchor=_anchor(qualname),
        bases=bases,
        params=params,
        signature=signature,
        signature_html=signature_html,
        docstring=ast.get_docstring(node) or "",
        lineno=node.lineno,
        children=children,
    )


def _is_documentable(node: ast.AST) -> bool:
    """Return whether a node should appear in API documentation."""

    return isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)


def _class_signature(node: ast.ClassDef) -> str:
    """Return a static class signature including base classes."""

    bases = [ast.unparse(base) for base in [*node.bases, *[kw.value for kw in node.keywords]]]
    return f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a static function signature, including return annotation."""

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({args}){returns}"


def _class_signature_html(node: ast.ClassDef) -> str:
    """Return syntax-highlighted HTML for a class signature."""

    bases = [ast.unparse(base) for base in [*node.bases, *[kw.value for kw in node.keywords]]]
    base_html = ""
    if bases:
        rendered = ", ".join(f'<span class="sig-type">{_escape(base)}</span>' for base in bases)
        base_html = f'<span class="sig-punct">(</span>{rendered}<span class="sig-punct">)</span>'
    return (
        '<span class="sig-keyword">class</span> '
        f'<span class="sig-name">{_escape(node.name)}</span>{base_html}'
    )


def _function_signature_html(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return syntax-highlighted HTML for a function signature."""

    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    params = _signature_params_html(node.args)
    returns = ""
    if node.returns is not None:
        returns = (
            ' <span class="sig-punct">-&gt;</span> '
            f'<span class="sig-return">{_escape(ast.unparse(node.returns))}</span>'
        )
    return (
        f'<span class="sig-keyword">{keyword}</span> '
        f'<span class="sig-name">{_escape(node.name)}</span>'
        f'<span class="sig-punct">(</span>{params}<span class="sig-punct">)</span>{returns}'
    )


def _signature_params_html(args: ast.arguments) -> str:
    """Return syntax-highlighted HTML for function parameters."""

    parts: list[str] = []
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        parts.append(_signature_arg_html(arg, default))
        if index == len(args.posonlyargs) - 1:
            parts.append('<span class="sig-punct">/</span>')
    if args.vararg is not None:
        parts.append(_signature_arg_html(args.vararg, None, prefix="*"))
    elif args.kwonlyargs:
        parts.append('<span class="sig-punct">*</span>')
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(_signature_arg_html(arg, default))
    if args.kwarg is not None:
        parts.append(_signature_arg_html(args.kwarg, None, prefix="**"))
    return '<span class="sig-punct">,</span> '.join(parts)


def _signature_arg_html(arg: ast.arg, default: ast.expr | None, *, prefix: str = "") -> str:
    """Return syntax-highlighted HTML for a single function parameter."""

    annotation = ""
    if arg.annotation is not None:
        annotation = (
            '<span class="sig-punct">: </span>'
            f'<span class="sig-type">{_escape(ast.unparse(arg.annotation))}</span>'
        )
    default_html = ""
    if default is not None:
        default_html = (
            '<span class="sig-punct"> = </span>'
            f'<span class="sig-default">{_escape(ast.unparse(default))}</span>'
        )
    prefix_html = f'<span class="sig-punct">{prefix}</span>' if prefix else ""
    return f'{prefix_html}<span class="sig-param">{_escape(arg.arg)}</span>{annotation}{default_html}'


def _documentable_arg_names(args: ast.arguments) -> list[str]:
    """Return parameter names that must be documented."""

    names = [arg.arg for arg in [*args.posonlyargs, *args.args] if arg.arg not in {"self", "cls"}]
    if args.vararg is not None:
        names.append(f"*{args.vararg.arg}")
    names.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(f"**{args.kwarg.arg}")
    return names


def _assert_documented_arguments(modules: list[ModuleDoc]) -> None:
    """Assert every public callable documents all of its arguments.

    Underscore-prefixed callables (including dunders) are skipped — they are
    internal helpers and not held to the same Args-section coverage rule.
    """

    failures: list[str] = []
    for module in modules:
        for obj in _iter_objects(module.objects):
            if obj.name.startswith("_"):
                continue
            if not obj.params:
                continue
            documented = _documented_arg_names(obj.docstring)
            missing = [param for param in obj.params if param not in documented]
            if missing:
                failures.append(
                    f"{module.source_rel}:{obj.lineno} {obj.qualname} missing Args entries: "
                    f"{', '.join(missing)}"
                )
    if failures:
        raise AssertionError("Undocumented callable arguments:\n" + "\n".join(failures))


def _documented_arg_names(docstring: str) -> set[str]:
    """Return argument names documented in a structured docstring."""

    sections = _split_docstring_sections(docstring)
    return {field["name"] for field in _parse_doc_fields(sections["args"]) if field["name"]}


def _iter_objects(objects: list[ApiObject]) -> list[ApiObject]:
    """Return a flattened list of documented objects."""

    flattened: list[ApiObject] = []
    for obj in objects:
        flattened.append(obj)
        flattened.extend(_iter_objects(obj.children))
    return flattened


def _write_site_index(modules: list[ModuleDoc]) -> None:
    """Write the GitHub Pages entry point."""

    existing_markdown = _markdown_docs()
    api_entries = _api_entry_directories(modules)
    api_links = "\n".join(
        f"<li><a href='{_rel(DOCS_DIR / 'index.html', _directory_index_path(root))}'>"
        f"{_escape(root.as_posix())}/</a><span>{_module_count(modules, root)} modules</span></li>"
        for root in api_entries
    )
    markdown_links = "\n".join(
        f"<li><a href='{_rel(DOCS_DIR / 'index.html', _markdown_page_path(path))}'>"
        f"{_escape(path.stem.replace('_', ' ').title())}</a></li>"
        for path in existing_markdown
    )
    body = f"""
    {_global_sidebar(DOCS_DIR / "index.html", modules)}
    <main class="content">
      <section class="hero">
        <p class="eyebrow">Multi-Currency Trading</p>
        <h1>Project Documentation</h1>
        <p>Static documentation generated from Python docstrings and the existing markdown notes.</p>
        <p><code>python3 scripts/build_docs.py</code></p>
      </section>
      <section>
        <h2>API Reference</h2>
        <ul class="module-list detailed">{api_links}</ul>
      </section>
      <section>
        <h2>Design Notes</h2>
        <ul class="module-list">{markdown_links or '<li>No markdown notes found.</li>'}</ul>
      </section>
    </main>
    """
    (DOCS_DIR / "index.html").write_text(_page("Project Documentation", body, layout="split"), encoding="utf-8")


def _write_api_index(modules: list[ModuleDoc]) -> None:
    """Write an API-only index page."""

    api_entries = _api_entry_directories(modules)
    rows = "\n".join(
        f"<li><a href='{_rel(DOCS_DIR / 'api' / 'index.html', _directory_index_path(root))}'>"
        f"{_escape(root.as_posix())}/</a><span>{_module_count(modules, root)} modules</span></li>"
        for root in api_entries
    )
    body = f"""
    {_global_sidebar(DOCS_DIR / "api" / "index.html", modules)}
    <main class="content">
      <p class="eyebrow">Repository API</p>
      <h1>API Reference</h1>
      <p class="muted">Browse by source subdirectory. Module pages are linked from their containing directory.</p>
      <ul class="module-list detailed">{rows}</ul>
    </main>
    """
    (DOCS_DIR / "api" / "index.html").write_text(_page("API Reference", body, layout="split"), encoding="utf-8")


def _write_markdown_pages(modules: list[ModuleDoc]) -> None:
    """Write sidebar-wrapped HTML pages for hand-written markdown notes."""

    for path in _markdown_docs():
        page_path = _markdown_page_path(path)
        body = f"""
        {_global_sidebar(page_path, modules)}
        <main class="content">
          <p class="eyebrow">Design Note</p>
          <h1>{_escape(path.stem.replace('_', ' ').title())}</h1>
          <article class="markdown-doc">{_markdown_to_html(path.read_text(encoding="utf-8"))}</article>
        </main>
        """
        page_path.write_text(_page(path.stem.replace("_", " ").title(), body, layout="split"), encoding="utf-8")


def _write_directory_pages(modules: list[ModuleDoc]) -> None:
    """Write recursive directory index pages for the API reference."""

    dirs = _all_directories(modules)
    for directory in dirs:
        _write_directory_page(directory, modules, dirs)


def _write_directory_page(directory: Path, modules: list[ModuleDoc], dirs: set[Path]) -> None:
    """Write a directory index with immediate child directories and modules."""

    child_dirs = sorted(
        child for child in dirs if child.parent == directory and child != directory
    )
    child_modules = sorted(
        (module for module in modules if module.source_rel.parent == directory),
        key=lambda module: module.source_rel.name,
    )

    child_dir_links = "\n".join(
        f"<li><a href='{_rel(_directory_index_path(directory), _directory_index_path(child))}'>"
        f"{_escape(child.name)}/</a><span>{_module_count(modules, child)} modules</span></li>"
        for child in child_dirs
    )
    module_links = "\n".join(
        f"<li><a href='{_rel(_directory_index_path(directory), module.page_path)}'>"
        f"{_escape(module.source_rel.name)}</a><span>{_escape(_summary(module.docstring))}</span></li>"
        for module in child_modules
    )
    parent_link = ""
    if directory.parent != Path("."):
        parent_link = (
            f"<a class='back' href='{_rel(_directory_index_path(directory), _directory_index_path(directory.parent))}'>"
            f"Parent Directory</a>"
        )
    else:
        parent_link = f"<a class='back' href='{_rel(_directory_index_path(directory), DOCS_DIR / 'api' / 'index.html')}'>API Index</a>"

    body = f"""
    {_global_sidebar(_directory_index_path(directory), modules, current_directory=directory)}
    <main class="content">
      <p class="eyebrow">{_escape(directory.as_posix())}/</p>
      <h1>{_escape(directory.as_posix())}/</h1>
      <p class="muted">{_module_count(modules, directory)} modules in this branch.</p>
      <div class="crumbs">
        <a href="{_rel(_directory_index_path(directory), DOCS_DIR / 'index.html')}">Project Docs</a>
        <span>/</span>
        <a href="{_rel(_directory_index_path(directory), DOCS_DIR / 'api' / 'index.html')}">API Reference</a>
      </div>
      {parent_link}
      <section class="directory-section">
        <h2>Directories</h2>
        <ul class="module-list detailed">{child_dir_links or '<li><span>No child directories.</span></li>'}</ul>
      </section>
      <section class="directory-section">
        <h2>Modules</h2>
        <ul class="module-list detailed">{module_links or '<li><span>No modules in this directory.</span></li>'}</ul>
      </section>
    </main>
    """
    path = _directory_index_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_page(f"{directory.as_posix()}/", body, layout="split"), encoding="utf-8")


def _write_module_page(module: ModuleDoc, modules: list[ModuleDoc]) -> None:
    """Write one module API page."""

    module.page_path.parent.mkdir(parents=True, exist_ok=True)
    class_index = _class_index(modules)
    objects = "\n".join(_render_object(obj, module, class_index) for obj in module.objects)
    body = f"""
    {_module_sidebar(module, modules)}
    <main class="content">
      <p class="eyebrow">{_escape(module.source_path.relative_to(PROJECT_ROOT).as_posix())}</p>
      <h1>{_escape(module.module_name)}</h1>
      {_doc_block(module.docstring)}
      {objects or '<p class="muted">No public classes or functions found.</p>'}
    </main>
    """
    module.page_path.write_text(_page(module.module_name, body, layout="split"), encoding="utf-8")


def _global_sidebar(
    page_path: Path,
    modules: list[ModuleDoc],
    *,
    current_directory: Path | None = None,
) -> str:
    """Render the navigation sidebar used by non-module pages."""

    api_links = "".join(
        f'<li><a href="{_rel(page_path, _directory_index_path(directory))}">'
        f'{_escape(directory.as_posix())}/</a></li>'
        for directory in _api_entry_directories(modules)
    )
    design_links = "".join(
        f'<li><a href="{_rel(page_path, _markdown_page_path(path))}">'
        f'{_escape(path.stem.replace("_", " ").title())}</a></li>'
        for path in _markdown_docs()
    )
    current = ""
    if current_directory is not None:
        current = f"""
      <div class="side-section">
        <h2>Current Branch</h2>
        {_directory_sidebar_tree(page_path, current_directory)}
      </div>
        """
    return f"""
    <nav class="side">
      <div class="side-section">
        <a class="back" href="{_rel(page_path, DOCS_DIR / 'index.html')}">Project Docs</a>
        <a class="back" href="{_rel(page_path, DOCS_DIR / 'api' / 'index.html')}">API Index</a>
      </div>
      <div class="side-section">
        <h2>API</h2>
        <ul class="toc-list">{api_links}</ul>
      </div>
      {current}
      <div class="side-section">
        <h2>Design Notes</h2>
        <ul class="toc-list">{design_links or '<li><span class="muted">No notes.</span></li>'}</ul>
      </div>
    </nav>
    """


def _directory_sidebar_tree(page_path: Path, directory: Path) -> str:
    """Render parent-directory links for a directory page."""

    items: list[str] = []
    current = Path(directory.parts[0])
    while current != Path("."):
        label = f"{current.as_posix()}/"
        if current == directory:
            items.append(f'<li><span class="current-source">{_escape(label)}</span></li>')
            break
        items.append(
            f'<li><a href="{_rel(page_path, _directory_index_path(current))}">{_escape(label)}</a></li>'
        )
        child_index = len(current.parts)
        if child_index >= len(directory.parts):
            break
        current = current / directory.parts[child_index]
    return f'<ul class="source-tree">{"".join(items)}</ul>'


def _module_sidebar(module: ModuleDoc, modules: list[ModuleDoc]) -> str:
    """Render navigation and table of contents for a module page."""

    directory_page = _directory_index_path(module.source_rel.parent)
    return f"""
    <nav class="side">
      <div class="side-section">
        <a class="back" href="{_rel(module.page_path, DOCS_DIR / 'index.html')}">Project Docs</a>
        <a class="back" href="{_rel(module.page_path, DOCS_DIR / 'api' / 'index.html')}">API Index</a>
        <a class="back" href="{_rel(module.page_path, directory_page)}">Containing Directory</a>
      </div>
      <div class="side-section">
        <h2>Source</h2>
        {_source_tree(module, modules)}
      </div>
      <div class="side-section">
        <h2>On This Page</h2>
        {_module_toc(module.objects)}
      </div>
    </nav>
    """


def _source_tree(module: ModuleDoc, modules: list[ModuleDoc]) -> str:
    """Render source path navigation for the current module."""

    items: list[str] = []
    current = Path(module.source_rel.parts[0])
    while current != module.source_rel.parent and current != Path("."):
        items.append(
            f'<li><a href="{_rel(module.page_path, _directory_index_path(current))}">'
            f'{_escape(current.as_posix())}/</a></li>'
        )
        child_index = len(current.parts)
        if child_index >= len(module.source_rel.parts) - 1:
            break
        current = current / module.source_rel.parts[child_index]
    items.append(
        f'<li><a href="{_rel(module.page_path, _directory_index_path(module.source_rel.parent))}">'
        f'{_escape(module.source_rel.parent.as_posix())}/</a></li>'
    )
    siblings = sorted(
        (other for other in modules if other.source_rel.parent == module.source_rel.parent),
        key=lambda other: other.source_rel.name,
    )
    sibling_items = []
    for sibling in siblings:
        if sibling.source_rel == module.source_rel:
            sibling_items.append(f'<li><span class="current-source">{_escape(sibling.source_rel.name)}</span></li>')
        else:
            sibling_items.append(
                f'<li><a href="{_rel(module.page_path, sibling.page_path)}">'
                f'{_escape(sibling.source_rel.name)}</a></li>'
            )
    items.append(f'<li><ul class="source-siblings">{"".join(sibling_items)}</ul></li>')
    return f'<ul class="source-tree">{"".join(items)}</ul>'


def _module_toc(objects: list[ApiObject]) -> str:
    """Render a module table of contents with classes, functions, and methods."""

    if not objects:
        return '<p class="muted">No public objects.</p>'
    return f'<ul class="toc-list">{"".join(_toc_item(obj) for obj in objects)}</ul>'


def _toc_item(obj: ApiObject) -> str:
    """Render one table-of-contents item."""

    children = f'<ul>{"".join(_toc_item(child) for child in obj.children)}</ul>' if obj.children else ""
    return (
        '<li>'
        f'<a href="#{_escape(obj.anchor)}"><span class="toc-kind">{_escape(obj.kind)}</span>'
        f'<span class="toc-name">{_escape(obj.name)}</span></a>'
        f'{children}'
        '</li>'
    )


def _render_object(
    obj: ApiObject,
    module: ModuleDoc,
    class_index: dict[str, tuple[ModuleDoc, ApiObject]],
) -> str:
    """Render an extracted class or function."""

    children = "\n".join(_render_object(child, module, class_index) for child in obj.children)
    return f"""
    <article class="api-object" id="{_escape(obj.anchor)}">
      <div class="object-meta">{_escape(obj.kind)} · line {obj.lineno}</div>
      {_object_heading(obj, module, class_index)}
      <div class="signature"><code>{obj.signature_html}</code></div>
      {_doc_block(obj.docstring)}
      {children}
    </article>
    """


def _object_heading(
    obj: ApiObject,
    module: ModuleDoc,
    class_index: dict[str, tuple[ModuleDoc, ApiObject]],
) -> str:
    """Render an API object heading, including class inheritance."""

    if obj.kind != "class" or not obj.bases:
        return f"<h2>{_escape(obj.name)}</h2>"
    bases = ", ".join(_base_link(base, module, class_index) for base in obj.bases)
    return f'<h2>{_escape(obj.name)} <span class="inherits">inherits {bases}</span></h2>'


def _base_link(
    base: str,
    module: ModuleDoc,
    class_index: dict[str, tuple[ModuleDoc, ApiObject]],
) -> str:
    """Render a superclass name, linking to internal docs when available."""

    base_name = base.rsplit(".", 1)[-1]
    target = class_index.get(base_name)
    if target is None:
        return f'<span class="superclass">{_escape(base)}</span>'
    target_module, target_obj = target
    href = f"{_rel(module.page_path, target_module.page_path)}#{target_obj.anchor}"
    return f'<a class="superclass" href="{_escape(href)}">{_escape(base)}</a>'


def _class_index(modules: list[ModuleDoc]) -> dict[str, tuple[ModuleDoc, ApiObject]]:
    """Return a unique-name index of documented classes."""

    candidates: dict[str, list[tuple[ModuleDoc, ApiObject]]] = {}
    for module in modules:
        for obj in _iter_objects(module.objects):
            if obj.kind == "class":
                candidates.setdefault(obj.name, []).append((module, obj))
    return {name: values[0] for name, values in candidates.items() if len(values) == 1}


def _api_entry_directories(modules: list[ModuleDoc]) -> list[Path]:
    """Return the first directory links shown on the API reference pages."""

    dirs = _all_directories(modules)
    entries = sorted(path for path in dirs if path.parent == Path("src"))
    entries.extend(path for path in (Path("scripts"),) if path in dirs)
    return entries


def _all_directories(modules: list[ModuleDoc]) -> set[Path]:
    """Return every source directory that needs a directory index page."""

    dirs: set[Path] = set()
    for module in modules:
        current = module.source_rel.parent
        while current != Path("."):
            dirs.add(current)
            current = current.parent
    return dirs


def _directory_index_path(directory: Path) -> Path:
    """Return the generated API index path for a source directory."""

    return DOCS_DIR / "api" / directory / "index.html"


def _module_count(modules: list[ModuleDoc], directory: Path) -> int:
    """Count modules contained recursively under a source directory."""

    directory_parts = directory.parts
    return sum(module.source_rel.parts[: len(directory_parts)] == directory_parts for module in modules)


def _anchor(value: str) -> str:
    """Return a stable HTML anchor id for an API object."""

    return "api-" + "".join(ch if ch.isalnum() else "-" for ch in value).strip("-").lower()


def _markdown_docs() -> list[Path]:
    """Return hand-written markdown documents that should get HTML wrappers."""

    return sorted(p for p in DOCS_DIR.glob("*.md") if p.name.lower() != "readme.md")


def _markdown_page_path(path: Path) -> Path:
    """Return the generated HTML path for a markdown note."""

    return path.with_suffix(".html")


def _markdown_to_html(markdown: str) -> str:
    """Render a conservative subset of Markdown as HTML."""

    html_parts: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            html_parts.append(f"<p>{_inline_code(_escape(' '.join(paragraph).strip()))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            html_parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>")
            list_items.clear()

    def flush_code() -> None:
        if code_lines:
            html_parts.append(f'<pre class="doc-extra">{_escape("".join(code_lines).rstrip())}</pre>')
            code_lines.clear()

    for raw_line in markdown.splitlines(keepends=True):
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                flush_list()
                in_code = True
            continue
        if in_code:
            code_lines.append(raw_line)
            continue
        if not stripped:
            flush_paragraph()
            flush_list()
            continue
        if stripped.startswith("#"):
            flush_paragraph()
            flush_list()
            level = min(len(stripped) - len(stripped.lstrip("#")), 4)
            text = stripped[level:].strip()
            html_parts.append(f"<h{level}>{_inline_code(_escape(text))}</h{level}>")
            continue
        if stripped.startswith(("- ", "* ")):
            flush_paragraph()
            list_items.append(_inline_code(_escape(stripped[2:].strip())))
            continue
        paragraph.append(stripped)
    flush_paragraph()
    flush_list()
    flush_code()
    return "\n".join(html_parts)


def _doc_block(docstring: str) -> str:
    """Render a docstring with structured sections when available."""

    if not docstring:
        return '<p class="muted">No docstring.</p>'
    if _has_structured_sections(docstring):
        return _structured_doc_block(docstring)
    return _plain_doc_block(docstring)


def _plain_doc_block(docstring: str) -> str:
    """Render a free-form docstring as prose with optional literal blocks."""

    parts = _render_prose_lines(docstring.splitlines())
    return f'<div class="docstring plain-docstring">{"".join(parts)}</div>'


def _has_structured_sections(docstring: str) -> bool:
    """Return whether a docstring contains renderable Google-style sections."""

    headings = {"args:", "arguments:", "parameters:", "returns:", "yields:", "raises:"}
    return any(line.strip().lower() in headings | {"example", "examples"} for line in docstring.splitlines())


def _structured_doc_block(docstring: str) -> str:
    """Render Google-style docstring sections as semantic HTML."""

    sections = _split_docstring_sections(docstring)
    parts: list[str] = ['<div class="docstring structured-docstring">']
    if sections["summary"]:
        parts.append(_render_summary(sections["summary"]))
    if sections["examples"]:
        parts.append(_render_examples_section(sections["examples"]))
    if sections["args"]:
        parts.append(_render_field_section("Arguments", sections["args"]))
    if sections["returns"]:
        parts.append(_render_field_section("Returns", sections["returns"]))
    if sections["yields"]:
        parts.append(_render_field_section("Yields", sections["yields"]))
    if sections["raises"]:
        parts.append(_render_field_section("Raises", sections["raises"]))
    if sections["other"]:
        extra = "\n".join(sections["other"]).strip()
        parts.append(f'<pre class="doc-extra">{_escape(extra)}</pre>')
    parts.append("</div>")
    return "\n".join(part for part in parts if part)


def _split_docstring_sections(docstring: str) -> dict[str, list[str]]:
    """Split a docstring into summary and named structured sections."""

    aliases = {
        "args:": "args",
        "arguments:": "args",
        "parameters:": "args",
        "returns:": "returns",
        "yields:": "yields",
        "raises:": "raises",
    }
    sections: dict[str, list[str]] = {
        "summary": [],
        "args": [],
        "examples": [],
        "returns": [],
        "yields": [],
        "raises": [],
        "other": [],
    }
    current = "summary"
    lines = docstring.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped_lower = line.strip().lower()
        if stripped_lower in {"example", "examples"}:
            current = "examples"
            if index + 1 < len(lines) and set(lines[index + 1].strip()) <= {"-"}:
                index += 2
                continue
            index += 1
            continue
        key = aliases.get(line.strip().lower())
        if key is not None:
            current = key
            index += 1
            continue
        sections[current if current in sections else "other"].append(line.rstrip())
        index += 1
    return sections


def _render_summary(lines: list[str]) -> str:
    """Render summary lines before the first structured section."""

    return "".join(_render_prose_lines(lines))


def _render_prose_lines(lines: list[str]) -> list[str]:
    """Render prose lines while preserving indented literal blocks."""

    parts: list[str] = []
    paragraph: list[str] = []
    literal: list[str] = []
    math_block: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(line.strip() for line in paragraph).strip()
            if text:
                parts.append(f"<p>{_inline_code(_escape(text))}</p>")
            paragraph.clear()

    def flush_literal() -> None:
        if literal:
            text = "\n".join(line.rstrip() for line in literal).strip("\n")
            if text.strip():
                parts.append(f'<pre class="doc-literal">{_escape(text)}</pre>')
            literal.clear()

    def flush_math() -> None:
        if math_block:
            text = "\n".join(line.strip() for line in math_block).strip()
            if text:
                parts.append(f'<div class="math-block">{_escape(text)}</div>')
            math_block.clear()

    for line in lines:
        stripped = line.strip()
        if math_block:
            math_block.append(line)
            if stripped in {"\\]", "$$"}:
                flush_math()
            continue
        if not line.strip():
            flush_paragraph()
            flush_literal()
            continue
        if stripped in {"\\[", "$$"}:
            flush_paragraph()
            flush_literal()
            math_block.append(line)
            continue
        if line.startswith((" ", "\t")):
            flush_paragraph()
            literal.append(line)
            continue
        flush_literal()
        paragraph.append(line)
    flush_paragraph()
    flush_literal()
    flush_math()
    return parts


def _render_field_section(title: str, lines: list[str]) -> str:
    """Render a structured docstring field section."""

    fields = _parse_return_fields(lines) if title in {"Returns", "Yields"} else _parse_doc_fields(lines)
    if not fields:
        return ""
    rows = "\n".join(_render_doc_field(field) for field in fields)
    return f"""
    <section class="doc-section">
      <h3>{_escape(title)}</h3>
      <dl class="doc-fields">{rows}</dl>
    </section>
    """


def _render_examples_section(lines: list[str]) -> str:
    """Render example docstring lines with doctest code blocks."""

    chunks: list[str] = []
    prose: list[str] = []
    code: list[str] = []

    def flush_prose() -> None:
        if prose:
            chunks.append(_render_summary(prose))
            prose.clear()

    def flush_code() -> None:
        if code:
            code_text = "\n".join(code).strip()
            chunks.append(f'<pre class="example-code"><code>{_escape(code_text)}</code></pre>')
            code.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_prose()
            flush_code()
            continue
        if stripped.startswith((">>>", "...")):
            flush_prose()
            code.append(stripped)
        elif code and (raw_line.startswith(" ") or raw_line.startswith("\t")):
            code.append(line)
        else:
            flush_code()
            prose.append(stripped)
    flush_prose()
    flush_code()
    if not chunks:
        return ""
    return f"""
    <section class="doc-section">
      <h3>Examples</h3>
      {"".join(chunks)}
    </section>
    """


def _parse_return_fields(lines: list[str]) -> list[dict[str, str]]:
    """Parse return or yield section lines as typed values."""

    fields = _parse_doc_fields(lines)
    for field in fields:
        if field["name"] and not field["type"]:
            field["type"] = field["name"]
            field["name"] = ""
    return fields


def _parse_doc_fields(lines: list[str]) -> list[dict[str, str]]:
    """Parse indented Google-style field lines."""

    fields: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _looks_like_field(line):
            if current is not None:
                fields.append(current)
            current = _parse_doc_field(line)
        elif current is not None:
            current["description"] = f'{current["description"]} {line}'.strip()
        else:
            current = {"name": "", "type": "", "description": line}
    if current is not None:
        fields.append(current)
    return fields


def _looks_like_field(line: str) -> bool:
    """Return whether a line looks like a docstring field."""

    return ":" in line


def _parse_doc_field(line: str) -> dict[str, str]:
    """Parse one ``name (type): description`` docstring field."""

    head, _, description = line.partition(":")
    head = head.strip()
    type_name = ""
    name = head
    if head.endswith(")") and "(" in head:
        name, _, type_part = head.rpartition("(")
        name = name.strip()
        type_name = type_part[:-1].strip()
    elif " " not in head and head:
        name = head
    else:
        name = ""
        type_name = head
    return {"name": name, "type": type_name, "description": description.strip()}


def _render_doc_field(field: dict[str, str]) -> str:
    """Render one parsed docstring field."""

    name = field["name"]
    type_name = field["type"]
    description = field["description"]
    name_html = f'<span class="doc-field-name">{_escape(name)}</span>' if name else ""
    type_html = f'<span class="doc-field-type">{_escape(type_name)}</span>' if type_name else ""
    description_html = _inline_code(_escape(description))
    return f"""
      <div class="doc-field">
        <dt>{name_html}{type_html}</dt>
        <dd>{description_html}</dd>
      </div>
    """


def _inline_code(text: str) -> str:
    """Render double-backtick inline code spans in already-escaped text."""

    parts = text.split("``")
    if len(parts) == 1:
        return text
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if index % 2:
            rendered.append(f"<code>{part}</code>")
        else:
            rendered.append(part)
    return "".join(rendered)


def _page(title: str, body: str, *, layout: str = "default") -> str:
    """Wrap page content in shared HTML, CSS, and metadata."""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1c2024;
      --muted: #687076;
      --line: #dfe3e6;
      --panel: #f7f9fa;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --code: #263238;
      --sig-name: #075985;
      --sig-param: #9a3412;
      --sig-type: #6d28d9;
      --sig-default: #64748b;
      --sig-return: #047857;
      --bg: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    a {{ color: var(--accent-dark); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    h1, h2, h3 {{ line-height: 1.18; margin: 0 0 0.7rem; }}
    h1 {{ font-size: clamp(2rem, 4vw, 3.8rem); max-width: 980px; }}
    h2 {{ font-size: 1.35rem; margin-top: 2rem; }}
    .inherits {{
      display: block;
      margin-top: 0.25rem;
      color: var(--muted);
      font-size: 0.92rem;
      font-weight: 500;
    }}
    .superclass {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
    .hero {{
      min-height: 46vh;
      padding: 9vh min(7vw, 5rem) 5vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff 0%, #eef8f6 100%);
    }}
    .hero p {{ max-width: 760px; font-size: 1.1rem; color: var(--muted); }}
    .eyebrow {{
      color: var(--accent-dark);
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
      font-size: 0.78rem;
    }}
    section {{ padding: 2.5rem min(7vw, 5rem); }}
    .module-list {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 0.65rem;
      padding: 0;
      list-style: none;
    }}
    .module-list li {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.8rem 0.95rem;
      background: #fff;
      overflow-wrap: anywhere;
    }}
    .module-list.detailed li {{ display: flex; flex-direction: column; gap: 0.25rem; }}
    .module-list span, .muted {{ color: var(--muted); }}
    .directory-section {{ padding: 0; margin-top: 2rem; }}
    .crumbs {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      margin: 1rem 0;
      color: var(--muted);
    }}
    .split {{
      display: grid;
      grid-template-columns: minmax(240px, 300px) minmax(0, 1fr);
      min-height: 100vh;
    }}
    .side {{
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 1.25rem;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }}
    .side h2 {{
      margin: 0 0 0.55rem;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
    }}
    .side ul {{ list-style: none; padding: 0; margin: 0; }}
    .side li {{ margin: 0.35rem 0; overflow-wrap: anywhere; }}
    .back {{ display: block; font-weight: 700; margin-bottom: 0.6rem; }}
    .side-section {{
      border-bottom: 1px solid var(--line);
      padding-bottom: 1rem;
      margin-bottom: 1rem;
    }}
    .side-section:last-child {{
      border-bottom: 0;
      margin-bottom: 0;
      padding-bottom: 0;
    }}
    .source-tree li {{
      margin: 0.25rem 0;
      padding-left: 0.65rem;
      border-left: 2px solid var(--line);
    }}
    .source-tree .source-siblings {{
      margin-top: 0.35rem;
    }}
    .source-tree .source-siblings li {{
      border-left: 0;
      padding-left: 0;
    }}
    .current-source {{
      color: var(--ink);
      font-weight: 700;
    }}
    .toc-list ul {{
      margin: 0.25rem 0 0.55rem 0.8rem;
      padding-left: 0.7rem;
      border-left: 2px solid var(--line);
    }}
    .toc-list a {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 0.45rem;
      align-items: baseline;
      padding: 0.2rem 0;
    }}
    .toc-kind {{
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 0.68rem;
      line-height: 1;
      padding: 0.18rem 0.35rem;
      text-transform: uppercase;
    }}
    .toc-name {{
      overflow-wrap: anywhere;
    }}
    .content {{ padding: 2.5rem min(6vw, 4rem); max-width: 1100px; }}
    .api-object {{
      border-top: 1px solid var(--line);
      padding-top: 1.3rem;
      margin-top: 1.6rem;
    }}
    .api-object .api-object {{
      margin-left: 1rem;
      padding-left: 1rem;
      border-left: 3px solid var(--line);
    }}
    .object-meta {{
      color: var(--muted);
      font-size: 0.85rem;
      margin-bottom: 0.25rem;
    }}
    .signature {{
      white-space: pre-wrap;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.9rem;
      background: var(--panel);
    }}
    .docstring {{
      margin: 0.7rem 0 1.1rem;
    }}
    .structured-docstring {{
      white-space: normal;
      overflow-x: visible;
    }}
    .structured-docstring p {{
      margin: 0 0 0.9rem;
    }}
    .doc-section {{
      padding: 0;
      margin-top: 1rem;
    }}
    .doc-section h3 {{
      margin: 0 0 0.55rem;
      color: var(--accent-dark);
      font-size: 1rem;
    }}
    .doc-fields {{
      display: grid;
      gap: 0.55rem;
      margin: 0;
    }}
    .doc-field {{
      display: grid;
      grid-template-columns: minmax(140px, 260px) minmax(0, 1fr);
      gap: 0.85rem;
      border-top: 1px solid var(--line);
      padding-top: 0.55rem;
    }}
    .doc-field dt {{
      display: flex;
      flex-wrap: wrap;
      align-content: flex-start;
      gap: 0.35rem;
      min-width: 0;
    }}
    .doc-field dd {{
      margin: 0;
      min-width: 0;
    }}
    .doc-field-name {{
      color: var(--sig-param);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-weight: 700;
    }}
    .doc-field-type {{
      color: var(--sig-type);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }}
    .doc-field-type::before {{ content: "("; color: #4b5563; }}
    .doc-field-type::after {{ content: ")"; color: #4b5563; }}
    .doc-extra {{
      white-space: pre-wrap;
      overflow-x: auto;
      margin: 1rem 0 0;
    }}
    .plain-docstring {{
      white-space: normal;
      overflow-x: visible;
    }}
    .plain-docstring p {{
      margin: 0 0 0.9rem;
    }}
    .doc-literal {{
      white-space: pre-wrap;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.9rem;
      background: #ffffff;
      margin: 0.7rem 0 0.9rem;
    }}
    .math-block {{
      margin: 0.9rem 0;
      overflow-x: auto;
    }}
    .example-code {{
      white-space: pre-wrap;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.9rem;
      background: #ffffff;
      margin: 0.6rem 0 0;
    }}
    .markdown-doc {{
      max-width: 980px;
    }}
    .markdown-doc h1, .markdown-doc h2, .markdown-doc h3, .markdown-doc h4 {{
      margin-top: 1.6rem;
    }}
    .markdown-doc p {{
      margin: 0.9rem 0;
    }}
    .markdown-doc ul {{
      padding-left: 1.3rem;
    }}
    .signature {{ color: var(--code); }}
    .signature code {{ white-space: pre-wrap; }}
    .sig-keyword {{ color: var(--accent-dark); font-weight: 700; }}
    .sig-name {{ color: var(--sig-name); font-weight: 700; }}
    .sig-param {{ color: var(--sig-param); }}
    .sig-type {{ color: var(--sig-type); font-weight: 600; }}
    .sig-default {{ color: var(--sig-default); }}
    .sig-return {{ color: var(--sig-return); font-weight: 700; }}
    .sig-punct {{ color: #4b5563; }}
    @media (max-width: 860px) {{
      .split {{ display: block; }}
      .side {{ position: static; height: auto; max-height: 45vh; border-right: 0; border-bottom: 1px solid var(--line); }}
      .content {{ padding: 1.5rem; }}
      .doc-field {{ grid-template-columns: 1fr; gap: 0.2rem; }}
      section, .hero {{ padding-left: 1.25rem; padding-right: 1.25rem; }}
    }}
  </style>
</head>
<body class="{layout}">
{body}
</body>
</html>
"""


def _summary(docstring: str) -> str:
    """Return the first sentence or line from a docstring."""

    stripped = " ".join(docstring.split())
    if not stripped:
        return "No module docstring."
    sentence, _, _ = stripped.partition(". ")
    return sentence + ("." if not sentence.endswith(".") else "")


def _rel(from_path: Path, to_path: Path) -> str:
    """Return a POSIX relative link between two documentation files."""

    start = from_path.parent if from_path.suffix else from_path
    return Path(os.path.relpath(to_path, start=start)).as_posix()


def _escape(value: object) -> str:
    """HTML-escape a value for text and attribute contexts."""

    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
