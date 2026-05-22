# RCK architecture paper

This directory holds the LaTeX source for the headline RCK paper.

## Build

You have three options.

### Option 1: Overleaf (easiest, no install)

1. Zip this whole `papers/rck-architecture/` directory.
2. Go to [overleaf.com](https://overleaf.com), create a new project,
   "Upload Project", drop the zip.
3. Overleaf compiles automatically. Edit, recompile, download PDF.

You'll want to update one thing: the `\author{...}` block in
`paper.tex` if you want a different affiliation or contact.

### Option 2: Local LaTeX

Requires `pdflatex`, `bibtex`, Python with `matplotlib`.

```bash
# from this directory
python figures/generate_figures.py    # produces 3 PDF figures
make                                   # builds paper.pdf
```

On Windows, install [MiKTeX](https://miktex.org/) and the GNU Make
port (e.g. via Chocolatey: `choco install make`). MiKTeX will
auto-install missing packages on first compile.

### Option 3: Tectonic (single binary, no fuss)

```bash
python figures/generate_figures.py
tectonic paper.tex
```

[Tectonic](https://tectonic-typesetting.github.io/) is a self-contained
LaTeX engine that downloads only the packages it needs.

## Files

```
paper.tex                  # the manuscript
references.bib             # bibliography
Makefile                   # build script
figures/
  generate_figures.py      # reads data/*.json, writes PDFs
  stack-diagram.pdf        # figure 1 (generated)
  chain-depth.pdf          # figure 2 (generated)
  sparse-vs-dense.pdf      # figure 3 (generated)
```

## Where to submit

The paper is structured for an 8-12 page workshop / journal track.
Realistic targets:

### Preprint servers (do these first, no review, fast)

| Venue | URL | Notes |
|---|---|---|
| **Zenodo** | https://zenodo.org | DOI in hours, no affiliation required, fully citable. Best first stop. |
| **TechRxiv** | https://techrxiv.org | IEEE-sponsored. No affiliation needed. |
| **OSF Preprints** | https://osf.io/preprints/ | Academic, accepts unaffiliated. |
| **HAL** | https://hal.archives-ouvertes.fr | French national archive, international submissions OK. |
| **arXiv** (cs.AI / cs.LG) | https://arxiv.org | Needs an endorser for first submission. See below. |

### Peer-reviewed (no affiliation barrier)

| Venue | URL | Notes |
|---|---|---|
| **TMLR** | https://openreview.net/group?id=TMLR | Transactions on Machine Learning Research. Open peer review, no fees, ~1-3 month turnaround, accepts unaffiliated authors. Probably the best target. |
| **NeSy** | https://neurosymbolic-ai.org | Neuro-Symbolic AI workshop / conference. Direct fit. |
| **KR** | https://kr.org | Knowledge Representation conference. Also a good fit. |
| **NeurIPS workshops** | https://neurips.cc | The NeSy workshop in particular. |
| **ICLR workshops** | https://iclr.cc | Same idea. |

### arXiv endorser

If you've never posted to arXiv in `cs.AI` you need one endorsement
from an existing author. Three ways to get one:

1. Cold-email an HRR/HDC/NeSy researcher with the PDF attached.
   Pentti Kanerva (Redwood Center, UC Berkeley), Tony Plate, Pei
   Wang (Temple University), Alexander Gray (IBM Research), or
   the maintainers of `torchhd` are good candidates.
2. Post to the NeSy mailing list with the draft and ask politely.
3. Submit to Zenodo and TMLR first; once you have one published
   preprint, future arXiv endorsements become easier.

## Distribution checklist (after the paper is up)

- [ ] Zenodo upload (canonical DOI)
- [ ] Mirror to TechRxiv + OSF Preprints
- [ ] Submit to TMLR via OpenReview
- [ ] arXiv (after endorser secured)
- [ ] GitHub README updated with paper link
- [ ] Hacker News post ("Show HN: RCK \[paper + code\]")
- [ ] r/MachineLearning [R] post
- [ ] r/LocalLLaMA post
- [ ] Twitter / X thread tagging @ylecun, @karpathy, @SchmidhuberAI,
      relevant HDC/NeSy researchers
- [ ] Mastodon (sigmoid.social)
- [ ] LinkedIn post
- [ ] Direct emails to authors of cited papers ("we built on your
      work, here's what we did, would love your feedback")
- [ ] Newsletter pitches: Import AI (Jack Clark), The Batch (Andrew
      Ng), AlphaSignal, TLDR AI

## Drafts

This is paper 1 of an intended series:

1. **Architecture (this paper)**: the v15 stack + headline empirical
   results.
2. **Chain induction with filters**: deep-dive on the four-gate
   filter stack, 100% precision result.
3. **Confidence-propagation bottleneck**: depth study + the
   geometric-mean finding.
4. **Sparse HRR negative result**: capacity cliff measurement.

Papers 2-4 are planned but not yet drafted; the architecture paper
is the priority for first publication.
