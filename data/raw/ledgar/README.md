# LEDGAR (LexGLUE packaging)

These files are the training, validation and test data for the clause classifier, committed so that every
clone trains and evaluates on exactly the same rows.

| File | Rows | Content |
|---|---|---|
| `train.parquet` | 60,000 | contract provisions filed 2016-2017 |
| `validation.parquet` | 10,000 | filed 2018 |
| `test.parquet` | 10,000 | filed 2019 |
| `labels.json` | 100 | label id -> LEDGAR heading name |

Columns: `text` (the provision) and `label` (integer id into `labels.json`). The classifier merges synonym
labels into clause types at load time (`configs/ledgar_label_groups.yaml`); these files are never edited.

## Source and attribution

- Dataset: **LEDGAR**, Tuggener, D., von Däniken, P., Peetz, T., Cieliebak, M. (2020). *LEDGAR: A Large-Scale
  Multi-label Corpus for Text Classification of Legal Provisions in Contracts.* LREC 2020.
  https://aclanthology.org/2020.lrec-1.155
- Packaging: **LexGLUE**, Chalkidis, I. et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language
  Understanding in English.* ACL 2022. https://arxiv.org/abs/2110.00976
- Downloaded from https://huggingface.co/datasets/coastalcph/lex_glue (config `ledgar`, parquet export).
- Underlying documents: contracts filed with the US SEC and published on EDGAR (public records).

## License

LexGLUE is released under **Creative Commons Attribution 4.0 (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/
You may share and adapt the data, including commercially, provided you keep this attribution. Models trained
on it are derivatives; keep the attribution when redistributing them.

To re-download from the source instead of using these files, delete them and run `make data`.
