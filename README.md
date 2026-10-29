# biobankQA-RU: code and reported results

This reduced release accompanies the biobankQA-RU study. It contains the research pipeline, prompt templates, analysis code, reported numerical tables, and figures.

The 47 multiple-choice questions and 14 open situational tasks are withheld for future evaluation. Task files, answer keys, saved model responses, item-level expert and judge records, retrieval caches, parsed corpus, prebuilt index, and manuscript documents are excluded. These materials must not be added to this repository.

## Contents

- `benchmark/`: generation, retrieval, parsing, judging, export, and analysis code.
- `prompts/` and `documentation/`: prompt templates and corpus metadata; no evaluation questions.
- `code/`: offline analysis code and release checks.
- `expected/`: frozen reference result tables.
- `results/reproduced/`: previously computed tables, plots, and historical validation report.
- `figures/article/`: figure assets supplied with the manuscript.

## Check this release

Use Python 3.11 or later. The integrity and release tests use only the standard library:

```sh
python code/verify_release.py
python code/test_release.py
```

`MANIFEST.json` records the size and SHA-256 digest of every released file. The checks also reject withheld data directories and intermediate artifact formats.

## Reproduction and model runs

Full numerical reproduction and generation require the withheld inputs and are unavailable from this reduced release alone. `code/reproduce.py` preserves the original analysis and exits with an explanation when those inputs are absent. The saved `results/reproduced/validation.json` describes the original full-data run; it is not a validation of a new run from this release.

The analysis dependencies are in `requirements.txt`; pipeline dependencies are in `requirements-pipeline.txt`. Model scripts additionally require provider credentials supplied locally through environment variables. Run any experiments with private inputs in a separate private checkout. Hosted model aliases and provider behavior may change, so new generation can differ from the study outputs.

## License

The MIT license applies to software code in `benchmark/` and `code/`. Research materials have no reuse license in this release; see `DATA-LICENSE.md` and `RIGHTS-STATUS.md`.
