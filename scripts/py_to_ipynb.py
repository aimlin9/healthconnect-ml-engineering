"""Convert a `# %%` / `# %% [markdown]` cell-marked .py script into a real
.ipynb and execute it in-place with a live kernel, so the notebook deliverable
contains real outputs (printed metrics, tables, charts) rather than static code.

Usage: python scripts/py_to_ipynb.py notebooks/healthconnect_ml_track.py
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SRC = Path(sys.argv[1])
DEST = SRC.with_suffix(".ipynb")

text = SRC.read_text()
chunks = text.split("\n# %%")
cells = []
for i, chunk in enumerate(chunks):
    chunk = chunk if i == 0 else "# %%" + chunk
    is_markdown = chunk.lstrip().startswith("# %% [markdown]")
    body = chunk.split("\n", 1)[1] if "\n" in chunk else ""
    body = body.strip("\n")
    if not body.strip():
        continue
    if is_markdown:
        md_lines = [l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l
                    for l in body.split("\n")]
        cells.append(nbformat.v4.new_markdown_cell("\n".join(md_lines)))
    else:
        cells.append(nbformat.v4.new_code_cell(body))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "HealthConnect Week6 (.venv)",
    "language": "python",
    "name": "healthconnect-week6",
}

print(f"Built {len(cells)} cells. Executing...")
client = NotebookClient(nb, timeout=900, kernel_name="healthconnect-week6",
                         resources={"metadata": {"path": str(SRC.parent)}})
client.execute()

nbformat.write(nb, DEST)
print(f"Executed and wrote {DEST}")
