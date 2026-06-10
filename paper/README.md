# Paper build

`main.tex` is the submission draft. **Every number is auto-generated** — never hand-typed:

```bash
.venv-train/bin/python paper/make_numbers.py   # reports/*.json -> paper/numbers.tex (+ 12 consistency checks; fails build on contradiction)
.venv-train/bin/python paper/make_figures.py   # reports/*.json -> paper/figures/*.pdf
cd paper && latexmk -pdf main.tex
```

The ONE RULE: a number that `make_numbers.py` cannot derive from a harness-emitted report
does not appear in the paper. The script cross-checks counts across reports and exits non-zero
on any inconsistency, so the build itself enforces the rule.
