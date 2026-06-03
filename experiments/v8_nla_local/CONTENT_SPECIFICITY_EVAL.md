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

**The most important result is the follow-up.** A ceiling test shows the raw L23
activation IS strongly content-discriminative (doc-retrieval 0.24 vs 0.077 chance,
p=0.0006; linear probe 60% on 13-way doc id). So the content is present in the
activation and the AV throws it away: **the verbalizer is the bottleneck, not the
activation, and the gap is therefore fixable rather than intrinsic.** A layer sweep
adds that L23 is not even the best site to read (L17 is more than 2x more
discriminative). Caveats up front: n=50 (13 docs) for the core eval and n=200 (~50
docs) for the sweep; the sweep uses last-token retrieval, so L17's lead may partly
reflect surface structure; all probes are on politically-themed news text.

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

## The decisive follow-up: the content IS in the activation (the AV is the bottleneck)

A null on the AV output has two very different explanations: either (A) the L23
activation encodes doc-discriminative content but the AV fails to surface it, or
(B) the activation does not encode it and no AV could. We tested this directly with
a **ceiling experiment**: run the same doc-level retrieval on the RAW L23
activations, plus a cross-validated linear probe (`eval_activation_ceiling.py`).

| Readout on RAW L23 activation (n=50, 13 docs) | Result | Chance | Signal? |
|---|---|---|---|
| Doc-level retrieval top-1 | **0.240** | 0.077 | yes (p=0.0006, z=4.6) |
| Linear-probe (logistic) CV accuracy | **0.604** | 0.083 | yes |

So the answer is **(A)**. The raw activation is strongly document-discriminative:
a simple linear probe reads doc identity at 60% (13-way), and nearest-activation
retrieval runs at 3x chance. The content is right there. The v0.1 AV throws it
away, landing its TEXT output at chance (0.10, p=0.34). A single forward-injection
single-token probe behaves the same (shift-vs-baseline 0.28, p=0.36, non-degenerate).

Three consequences:
1. **The gap is the verbalizer's reading of the activation, not an intrinsic ceiling.**
   The information needed for per-row content fidelity is present in the 2B-L23
   activation. The open problem is the verbalizer's conditioning on the injected
   activation — it largely ignores the injection and emits a learned prior (the
   template collapse on file). Surfacing that signal through the AV is an active
   research direction, not a settled matter of training budget. This is the hopeful
   reading, and it is the measured one.
2. **It weighs against the polysemanticity-at-2B-scale hypothesis for doc-level content.**
   The prior calibration doc speculated the 2B activation might intrinsically encode
   less per-instance specificity. For document identity that reading is hard to
   sustain: 60% linear decodability says the coarse specificity is there. How much
   finer per-entity specificity the activation carries — and the role of
   polysemanticity in it — remains under active investigation.
3. **It validates the eval method.** The same retrieval that finds nothing in the AV
   output finds a strong signal in the activation, so the AV-output null is a real
   property of the AV, not a blind spot of the metric.

The likely mechanism is the template collapse already on file (18/50 outputs share
one prefix): the AV largely ignores the injected activation and emits a learned
prior, so the rich activation signal never reaches the text.

### Layer sweep: L23 is not even the best layer to read

`eval_layer_ceiling_sweep.py` re-extracts last-token activations at eight layers
for 200 held-out texts (~50 docs, chance = 0.020) and runs the same doc-level
retrieval ceiling at each.

| Layer | Doc-retrieval top-1 | z | note |
|---|---|---|---|
| L5 | 0.300 | 27 | |
| L9 | 0.350 | 32 | |
| L13 | 0.285 | 25 | |
| **L17** | **0.805** | **73** | most discriminative by far |
| L21 | 0.405 | 38 | |
| L23 | 0.350 | 31 | the layer the NLA reads |
| L27 | 0.530 | 48 | |
| L31 | 0.515 | 48 | |

Every layer is strongly above chance (all p ~ 0.0005), so doc-discriminative content
is present throughout the stack. But **L23, the NLA's target, is among the weaker
layers, and L17 carries more than twice its doc-discriminability** (0.805 vs 0.350).
Two clean levers for a better NLA follow: (a) fix the AV so it actually reads the L23
signal that is already there, and (b) consider retargeting to L17, where there is far
more to read. (The retrieval is over last-token activations, so L17's edge may partly
reflect richer lexical structure at the middle of the stack; the point that all
layers are content-rich and L23 is not optimal holds either way.)

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

But the headline is not just a downgrade. The ceiling test reframes the whole gap as
**hopeful and fixable**: the content the AV misses is demonstrably present in the L23
activation (60% linear-probe doc id), so this is a verbalizer-training problem, not a
model-scale ceiling. The public framing should carry both halves: the v0.1 AV does not
yet read per-row content, AND the information is there to be read, so a better-trained
AV (and/or a more discriminative layer like L17) is the clear path to a content-faithful
NLA at this scale. Floor is real, headroom is real.

## Reproduce

```
python experiments/v8_nla_local/eval_content_specificity.py          # lexical + semantic retrieval
python experiments/v8_nla_local/eval_content_specificity_refine.py   # tail-window + non-template
python experiments/v8_nla_local/eval_content_specificity_judge.py                       # Haiku judge
python experiments/v8_nla_local/eval_content_specificity_judge.py --model claude-sonnet-4-6
```
All outputs land in `results/content_aware_eval/content_specificity_*`.
