# Raw data

Third-party datasets used by this project, committed so that every clone trains and evaluates on identical
files. Nothing in this folder was created by this project, and nothing here is ever edited.

| Folder | Dataset | Source | License | Details |
|---|---|---|---|---|
| `ledgar/` | LEDGAR (via LexGLUE) | https://huggingface.co/datasets/coastalcph/lex_glue | CC BY 4.0 | [`ledgar/README.md`](ledgar/README.md): per-file download URLs, dates, SHA-256 checksums, citations |

Rules for adding data here (see `AGENTS.md`, Data rules):

1. Every file must be listed in its folder's `README.md` with the exact URL it was downloaded from, the
   download date, its SHA-256 checksum, the license, and the citation the source asks for.
2. Files are copied unchanged. Derived or generated data goes to `../cache/` (git-ignored).
3. Check the license allows redistribution before committing; keep the attribution the license requires.
