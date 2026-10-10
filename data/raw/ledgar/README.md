# LEDGAR data files: sources and credits

Every file in this folder was downloaded from, or derived from, the **LexGLUE** dataset published by the
CoAStaL NLP group (University of Copenhagen) on the Hugging Face Hub, which packages the **LEDGAR** corpus
created at ZHAW (Zurich University of Applied Sciences). The underlying texts are contract provisions from
filings on the US SEC's EDGAR system. Full credit to these sources; this project did not create any of this
data and does not edit these files.

## Per-file provenance

| File | Rows | Downloaded from (exact URL) | Date | SHA-256 |
|---|---|---|---|---|
| `train.parquet` | 60,000 | https://huggingface.co/datasets/coastalcph/lex_glue/resolve/refs%2Fconvert%2Fparquet/ledgar/train/0000.parquet | 2026-10-11 | `e2901d9e019f5652dadda839d488155c681e50b90bd33f00574949418827867d` |
| `validation.parquet` | 10,000 | https://huggingface.co/datasets/coastalcph/lex_glue/resolve/refs%2Fconvert%2Fparquet/ledgar/validation/0000.parquet | 2026-10-11 | `55ef0deada1ed75bbf13c68fdd1c1a1ebfa5b09176d5e68b92499234f51d6c9f` |
| `test.parquet` | 10,000 | https://huggingface.co/datasets/coastalcph/lex_glue/resolve/refs%2Fconvert%2Fparquet/ledgar/test/0000.parquet | 2026-10-11 | `af9ed0ec09545d39449ffddca3996a1a7774e2bd0a9d6fc09d4e9dd067892a7a` |
| `labels.json` | 100 names | Not downloaded as a file: extracted by `scripts/01_prepare_data.py` from the `ClassLabel` names stored in the schema metadata of `train.parquet` above (the same list shown on the dataset page). | 2026-10-11 | `57c16c446815c517e769e07cb38177afaac58273535bfbabf6ea72b26e4b298b` |

The parquet files are the Hugging Face Hub's automatic parquet conversion (`refs/convert/parquet` branch) of
the `ledgar` configuration of `coastalcph/lex_glue`, downloaded with `huggingface_hub.hf_hub_download` and
copied here unchanged. Verify a file with `shasum -a 256 <file>`.

Columns: `text` (the provision) and `label` (integer id into `labels.json`). The classifier merges synonym
labels into clause types at load time (`configs/ledgar_label_groups.yaml`); these files are never modified.
To re-download from the source instead of using these copies, delete them and run `make data`.

## Credits

**Dataset page (download source)**
LexGLUE on the Hugging Face Hub, maintained by the CoAStaL NLP group, University of Copenhagen:
https://huggingface.co/datasets/coastalcph/lex_glue (code and benchmark: https://github.com/coastalcph/lex-glue)

**LexGLUE (the benchmark that packages LEDGAR with this split and these 100 labels)**
Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., and Aletras, N. (2022).
*LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics (ACL 2022). https://arxiv.org/abs/2110.00976

**LEDGAR (the original corpus)**
Tuggener, D., von Däniken, P., Peetz, T., and Cieliebak, M. (2020). *LEDGAR: A Large-Scale Multi-label
Corpus for Text Classification of Legal Provisions in Contracts.* Proceedings of the 12th Language Resources
and Evaluation Conference (LREC 2020), pages 1235–1241. https://aclanthology.org/2020.lrec-1.155
ZHAW Zurich University of Applied Sciences, record: https://digitalcollection.zhaw.ch/handle/11475/20087

**Underlying documents**
Contracts filed with the U.S. Securities and Exchange Commission and published on EDGAR, which are public
records: https://www.sec.gov/edgar

## License

LexGLUE is released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/. You may share and adapt the data, including commercially,
provided you keep this attribution. Models trained on it are derivatives; keep the attribution when
redistributing them. The project's own code is Apache-2.0 (see `../../../LICENSE`).

BibTeX:

```bibtex
@inproceedings{chalkidis-etal-2022-lexglue,
  title     = {{L}ex{GLUE}: A Benchmark Dataset for Legal Language Understanding in {E}nglish},
  author    = {Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and Bommarito, Michael and
               Androutsopoulos, Ion and Katz, Daniel Martin and Aletras, Nikolaos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year      = {2022},
  url       = {https://arxiv.org/abs/2110.00976}
}

@inproceedings{tuggener-etal-2020-ledgar,
  title     = {{LEDGAR}: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts},
  author    = {Tuggener, Don and von D{\"a}niken, Pius and Peetz, Thomas and Cieliebak, Mark},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  year      = {2020},
  pages     = {1235--1241},
  url       = {https://aclanthology.org/2020.lrec-1.155}
}
```
