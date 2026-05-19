# Documentation Site

Build the static HTML documentation with:

```bash
python3 scripts/build_docs.py
```

The generated `docs/index.html` file is the GitHub Pages entry point. The build
uses the Python AST instead of importing modules, so optional runtime
dependencies and API credentials are not required to compile the docs.
