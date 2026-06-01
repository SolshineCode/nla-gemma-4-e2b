# Content-specificity eval: does the v0.1 AV read source content?

**Date:** 2026-05-31
**Question:** Does the v0.1 AV text output carry *any* recoverable information about
which source document its activation came from? This was raised as a direct challenge
to the "content-blind / template-clustered" verdict: the intuition that the outputs
*do* contain some content-specific signal, even if not the surface topic (e.g. an
activation on a Hillary Clinton article might encode her personality or the event
rather than "politics", which would still be useful for interpretability).

**Answer:** No. Across six independent probes (lexical retrieval, semantic retrieval,
local-window retrieval, non-template-subset retrieval, and two LLM-judge forced-choice
panels), the AV output does not recover its source document above chance. The verdict
is robust and convergent. The one thing that *did* change: the v0.1 AV is genuinely
**diverse** (45/50 unique exact strings), so the blanket "template-clustered" label is
too strong. But that diversity is decoupled from source content. Diverse is not the
same as content-specific.

## Why this is a new test

The published "content-blind" verdict rested on AV->AR *activation-reconstruction*
discriminability (cross-row argmax = 2/50, ~chance; see `ar_scoreboard.json`). That
measures whether the **AR** can use the AV output to pick the right activation. It does
not test whether the AV's **text** carries source-content signal, because the AR is a
known structural-projector bottleneck (paraphrasing the AV moves round-trip cosine only
~3%). This eval tests the AV text directly, with the AR removed from the loop.

## Data

The 50 v0.1 AV outputs already generated for the Neuronpedia head-to-head
(`av_outputs_v0_1_dd_n50_from_neuronpedia.json`), row-aligned to `rl.parquet` rows
0..49 (alignment verified: `rl[i][:120] == basic_func[i]` for all 50). Those 50 rows
come from **13 unique documents** (~4 token-positions each, positions 53-940). Source
text per row = `detokenized_text_truncated` (the document truncated at the activation
token), which is the correct retrieval target, not the document's first 200 chars.

Because positions within one document are legitimately confusable, the fair, well-posed
task is **doc-level retrieval** (chance = 1/13 = 7.7%); row-level (chance = 1/50) is the
harder secondary metric. Null hypotheses are tested with a 5,000-iteration permutation
test (shuffle which AV output is assigned to which row, recompute the metric).

## Methods and results

| Probe | Metric | Observed | Chance | p (perm/binom) | Signal? |
|---|---|---|---|---|---|
| TF-IDF word (1-2 gram) retrieval | doc top-1 | 0.100 | 0.077 | 0.34 (gap) | no |
| TF-IDF char (3-5 gram) retrieval | doc top-1 | 0.080 | 0.077 | 0.58 (gap) | no |
| Semantic (all-MiniLM-L6-v2) retrieval | doc top-1 | 0.100 | 0.077 | 0.46 (gap) | no |
| Local-window (last 300 chars) TF-IDF | doc top-1 | 0.020 | 0.077 | 0.24 (gap) | no |
| Local-window semantic | doc top-1 | 0.060 | 0.077 | 0.59 (gap) | no |
| Non-template subset, tail, TF-IDF | doc top-1 | 0.094 | 0.077 | 0.43 (gap) | no |
| Non-template subset, tail, semantic | doc top-1 | 0.062 | 0.077 | 0.84 (gap) | no |
| LLM judge, Claude Haiku 4.5 (5-way) | accuracy | 0.240 | 0.200 | 0.29 (binom) | no |
| LLM judge, Claude Sonnet 4.6 (5-way, n=30) | accuracy | 0.267 | 0.200 | 0.24 (binom) | no |
| LLM judge, Gemini 3.5 Flash (5-way) | accuracy | n/a | 0.200 | n/a | unavailable (auth) |

The Sonnet panel timed out on a 300s `claude -p` call after 3 of 5 batches; the 30
completed trials are scored above (a stronger judge does not rescue signal either). The
Gemini panel was unavailable because Antigravity's Windows OAuth subprocess returns
empty output until an interactive `agy` login is done; the conclusion does not depend on
it, given the non-LLM semantic embedding already provides a cross-family independent
probe.

Supporting metrics already on file:
- **Output diversity** (`av_diversity_v0_1_dd_n50.json`): 45/50 unique exact strings,
  34/50 unique 100-char prefixes. Real surface variation.
- **Template collapse**: 18/50 outputs share the "The model tracks a list of
  country-specific..." prefix. So the collapse is real but partial.
- **AR cross-row discriminability** (`ar_scoreboard.json`): v0.1 argmax 2/50 (~chance),
  identity margin 0.002. Confirms the AR side too.

## Addressing the interpretability caveat directly

The challenge included a fair point: the AV might map an activation to a real but
*non-topical* feature (a person's personality, the event, the tone), which retrieval
against the surface topic would miss. Three things rule this out as a rescue:

1. **Semantic retrieval is feature-agnostic.** all-MiniLM embeddings place
   "Clinton's assertive personality" closer to the Clinton article than to a football
   article. A real personality/event/tone feature would still produce a same-doc vs
   different-doc similarity gap. There is none (p = 0.46).
2. **The LLM judge was told to use any connection.** The judge prompt explicitly
   instructs: match on "topic, named entities, the event itself, a person's personality
   or role, tone, register, or any other content link (it need not be the surface
   topic)." It still cannot beat chance.
3. **Both probes converge with the non-semantic lexical probe.** All three method
   families agree.

## The one honest scope-limit

This eval tests **discriminability among these 13 news documents**. A feature that is
genuinely *constant* across all 13 (e.g. "formal news register", "third-person
reporting") would be real, would be content, and would be **invisible** to this eval
because it does not vary across the set. So the precise claim is: *the AV output carries
no per-document-discriminative content signal on this set*, not the stronger *the AV
encodes nothing*. Distinguishing those requires a more diverse source set (multiple
genres) and ideally per-claim feature probes. Flagged for the next eval round.

## Bottom line for the public release framing

The current public docs describe the output as "theme-correct, detail-confabulated."
The "detail-confabulated" half is right. The "theme-correct" half is a mild overclaim:
the output is **format/genre-plausible but not per-row content- or theme-discriminative**.
The correction to the published docs is to (a) replace "theme-correct" with that more
precise phrasing, (b) cite this direct retrieval eval as the strongest evidence, and
(c) keep the genuine nuance that v0.1 output is diverse, just not content-tracking.

## Reproduce

```
python experiments/v8_nla_local/eval_content_specificity.py          # lexical + semantic retrieval
python experiments/v8_nla_local/eval_content_specificity_refine.py   # tail-window + non-template
python experiments/v8_nla_local/eval_content_specificity_judge.py                       # Haiku judge
python experiments/v8_nla_local/eval_content_specificity_judge.py --model claude-sonnet-4-6
```
All outputs land in `results/content_aware_eval/content_specificity_*`.
