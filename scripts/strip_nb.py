import sys
import nbformat

def strip_nb(nb):
    # remove cell outputs and execution counts, clear metadata
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        # optionally clear cell metadata:
        cell["metadata"] = {}
    # clear top-level metadata that stores execution info
    nb["metadata"].pop("kernelspec", None)
    nb["metadata"].pop("language_info", None)
    return nb

def main():
    data = sys.stdin.read()
    try:
        nb = nbformat.reads(data, as_version=nbformat.NO_CONVERT)
    except Exception:
        # if parsing fails, just pass content through
        sys.stdout.write(data)
        return
    nb = strip_nb(nb)
    sys.stdout.write(nbformat.writes(nb))

if __name__ == "__main__":
    main()