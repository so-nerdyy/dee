# arXiv submission checklist (Nivin submits, from his own account)

Nothing here has been submitted. arXiv requires the submitting author to be an author and to
take responsibility for the paper, so only you can do this.

## 1. Before you submit (fix these in `dee.tex`)

- [x] **Your name.** Set to "Nivin Karthick" (2026-11-20). Affiliation is "Independent researcher"; change it to your school if you prefer.
- [ ] **Code link.** The Availability paragraph points at the repo. After the repo cleanup,
      replace it with a tagged release (for example `v0.1-paper`) so the paper points at the
      exact code. Also add a `LICENSE` file to the repo first: right now the repo has none, which
      means nobody may legally reuse the code. MIT or Apache-2.0 are the usual choices.
- [ ] **DeepSeek-V4-Flash citation.** `refs.bib` cites the model card. If DeepSeek published a
      technical report for V4-Flash, cite that instead.
- [ ] **Read the whole paper once yourself.** Every number in it comes from committed scorecards
      and addenda, but you are the one signing it. Things to check especially: the cross-GPU
      token pattern in section 7.2 (it comes from the run threads' comparisons, not a committed
      table), and the competitor descriptions in section 2.
- [ ] **AI disclosure.** The Acknowledgments say AI assistants helped with engineering,
      analysis and writing, and that you take responsibility. arXiv's policy is that the
      authors are responsible for everything, and that AI tools can't be authors. Keep that
      paragraph.
- [ ] Rebuild: `cd paper && ./make_arxiv.sh` (needs TeX Live and matplotlib). It writes
      `dee.pdf` and `arxiv.tar.gz`.

## 2. What to fill in on the submission form

| field | value |
|---|---|
| Primary category | **cs.DC** (Distributed, Parallel, and Cluster Computing) |
| Cross-lists | **cs.LG** (Machine Learning), **cs.PF** (Performance) |
| License | **CC BY 4.0** (anyone may share and reuse with credit; best for visibility). arXiv's default "non-exclusive license to distribute" also works if you'd rather keep more rights. |
| Title | Exact by Construction, Graded in Advance: A Pre-Registered Hardware Evaluation of SSD-to-GPU Expert Offloading for DeepSeek-V4-Flash |
| Authors | Nivin Karthick |
| Comments | `15 pages, 10 figures, 3 tables. Code, frozen predictions and scorecards: https://github.com/so-nerdyy/dee` |
| Abstract | paste from below (plain text, no LaTeX commands; arXiv limit is 1,920 characters) |
| Upload | `paper/arxiv.tar.gz` (contains `dee.tex`, `dee.bbl`, two figure PDFs). arXiv does not run BibTeX, which is why the `.bbl` is included. |

### Abstract (plain text for the form)

Mixture-of-experts (MoE) models can run on GPUs far smaller than the model by keeping most expert weights on SSD or in host RAM and loading them on demand. Recent offloading systems report large speedups, but rarely check that the offloaded model produces the same tokens as the unoffloaded one, and their performance numbers are not tested against predictions made before the experiment. We describe DEE (Dynamic Expert Eviction), an engine that serves DeepSeek-V4-Flash (43 MoE layers, 256 routed experts per layer, a 146.6 GiB FP4 expert store) from an NVMe-backed two-tier cache, and evaluate it so that it can fail. DEE enforces an exactness contract: placement may change, but generated token ids and the full routing journal must be byte-identical to a reference run. Before any cloud GPU run existed, we froze a prediction matrix with kill bands and graded every later run against it. Across seven runs on six Modal GPU configurations (under $14 of compute), every run met the exactness contract on its own hardware, and a 2xT4 container reproduced a reference answer sealed three months earlier on other infrastructure token for token. Most throughput predictions were wrong: 9 of 12 testable decode-speed rows were killed, 2-6.5x below nominal, which located the bottleneck in the host-side copy path. Prefetch hints did not pay for their extra reads, offline or on hardware. Next-use eviction removed a 0% host-hit cliff at batch 8 and raised throughput 1.30x, as registered (1.27x). The fastest configuration, an RTX PRO 6000, decoded 6.27 tokens/s aggregate at batch 8. We release the engine, frozen predictions, scorecards and scorer.

(About 1,600 characters. Check the count in the form.)

## 3. Endorsement (you will probably need one)

arXiv asks first-time submitters to a category for an **endorsement** from someone who has
already published in that category, unless your account is tied to a recognized institution
email. A personal Gmail almost certainly needs one.

1. Create your account at arxiv.org (use your real name; you can use a personal email).
2. Start the submission and choose cs.DC. If an endorsement is needed, arXiv shows a message
   with a **6-character endorsement code** and a link.
3. Send that code to one person who has published in cs.DC or cs.LG on arXiv. Good options:
   a teacher, professor or researcher you already know; an author of a paper you cite who works
   on MoE offloading (keep the email short: who you are, the paper's title, the PDF attached,
   the code, and "would you be willing to endorse this submission?"). An endorser is vouching
   that it is a real research paper, not reviewing it, so one short polite email is normal.
4. Once they endorse (they click a link and enter the code), finish the submission.
5. arXiv moderators review new submissions. Papers usually appear the next weekday after the
   daily cutoff (14:00 US Eastern). Moderators can reclassify or put a paper on hold; if that
   happens, the email tells you why.

## 4. After it is public

- [ ] Add the arXiv link and a BibTeX entry to the repo README.
- [ ] Post on Hacker News (draft in `HN_POST.md`), ideally a weekday morning US time, and stay
      around for the first two hours to answer comments.
- [ ] Any fix after posting goes up as a new version (v2) on arXiv; v1 stays visible.
