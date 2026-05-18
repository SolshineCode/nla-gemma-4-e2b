# How my autonomous research assistant hallucinated an Anthropic hyperparameter — and what fixing it taught me about NLA measurement

*A retrospective on building the first non-Anthropic open-source Natural Language Autoencoder (NLA) on a 4 GB consumer laptop, the bug that took 8 days to find, and the calibration moment when looking at Anthropic's own published NLAs explained why I'd been benchmarking against the wrong bar.*

---

## TL;DR

I trained a Gemma-4-E2B Natural Language Autoencoder on a 4 GB GTX 1650 Ti Max-Q laptop following Anthropic's methodology (Fraser-Taliente et al. 2026). The first release (v0.0.1) shipped at round-trip cosine = 0.438. Then I spent ~8 days running follow-up experiments trying to lift that number, and hitting a content-blind ceiling no matter what I changed.

Three things turned out to be true:

1. **An autonomous Claude research assistant introduced an uncited claim** into an argparse help string in May 2026 ("Upstream Anthropic NLA uses 80000 for Gemma-3-12B"). The claim was wrong by ~2700×. It propagated through 5 training runs and 6 documents before I caught it.

2. **The H15 round-trip cosine metric** (Anthropic's published FVE in disguise) is **~97% structural-AR-projection** on under-trained NLAs. Paraphrasing the AV's explanation moves cosine by ~3%. Removing entire claims moves cosine by ~0%. The metric I was optimizing against was content-blind on my hardware regime.

3. **Looking at Anthropic's own published NLAs on Neuronpedia** calibrated my expectations: their Llama-70B and Gemma-27B NLAs produce thematically-correct, detail-confabulated explanations that ship with the disclaimer *"NLAs can produce unexpected or incorrect explanations. See limitations."* My 2B model produces output in the same FORMAT class (multi-paragraph descriptive text), and during the project I sometimes claimed this meant "competitive with Anthropic." A direct head-to-head via the [Neuronpedia API](https://docs.neuronpedia.org/api) (n=10 rows × 2 reference NLAs) revealed that's overclaim: the LLM-judge prefers Anthropic 20/20 with validity 3.1-3.2 vs ours 1.0. Their NLAs name specific entities (Hillary Clinton, Joseph Klein, Boom Williams); ours produces template-clustered generic descriptions. The release is positioned as a consumer-GPU methodology demonstration, not a content-fidelity peer. See [`RELEASE_CALIBRATION.md`](RELEASE_CALIBRATION.md) for the full comparison data.

The result is a working, publishable small-scale NLA pair plus a process retrospective. Both are open source. The autonomous-researcher-error class is the part that generalizes.

---

## 1. The setup

I started this project in early May 2026 with a single goal: ship the first open-source non-Anthropic NLA. Anthropic's NLA paper has a public companion repo (`kitft/natural_language_autoencoders`), but every NLA on HuggingFace Hub at the time was under the `kitft` account — Anthropic's official reference release. There was no second-source replication.

The methodology is two-stage. An **Activation Verbalizer (AV)** is a language model fine-tuned to take a residual-stream activation captured from a base model and generate a natural-language explanation of what that activation represents. An **Activation Reconstructor (AR)** is the inverse — given an explanation, reconstruct an activation vector that is close (by cosine similarity) to the original. Round-trip cosine is the success metric.

Anthropic trains on 7B–70B models with bf16 and full fine-tuning on 8–64 H100s. I was working on a single 4 GB GTX 1650 Ti Max-Q laptop. The only way that fits is NF4 4-bit quantization + LoRA adapters + a small (<5K row) labeled corpus + ≤300 SFT steps. That's about a 30× hardware descope on every axis.

The first version (v0.0.1) shipped on 2026-05-10 at round-trip cos = 0.438 ± 0.054 on n=42 held-out activations. 100% above the noise floor of 0.30. The Gemma-4-E2B AV used `injection_scale = sqrt(d_model) = 39.2` by default. The release looked successful. The NLA pair is at [`Solshine/gemma-4-e2b-nla-L23-av-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-av-v0_0_1) and [`-ar-v0_0_1`](https://huggingface.co/Solshine/gemma-4-e2b-nla-L23-ar-v0_0_1).

I wrote a LessWrong post draft and started planning follow-ups.

## 2. The trap

The follow-ups were supposed to push the cosine number higher. Bigger corpus, longer training, larger LoRA rank, etc. I worked with an autonomous Claude assistant (Claude Sonnet 4.6, running via Claude Code's autonomous-research mode) to plan and execute these experiments.

On 2026-05-15, in commit `9cb6426` (PR #108 of the source research repo), the assistant added a new `--injection-scale` flag to the training script. The argparse help text it wrote contained this sentence:

> *"Upstream Anthropic NLA uses 80000 for Gemma-3-12B, 60000 for Gemma-3-27B, 150 for Qwen-7B."*

That claim was unsourced. It looked authoritative. I didn't question it. The assistant didn't either when later sessions wrote planning docs.

For context: `injection_scale` controls the L2 norm to which the activation vector is rescaled before being substituted for a marker token's embedding. The intuition is that this norm should roughly match the typical embedding norm of the model's tokens — so the injected vector "looks like" a token to the transformer's attention layers. For Gemma-4-E2B with `d_model = 1536` and uniformly-normalized embeddings, the typical token-embedding L2 norm is about 39. Anthropic's actually-published `Llama-3.3-70B-NLA-L53` sidecar specifies `injection_scale: 30.0`. The number "80000" is roughly 2700× too large.

Over the next 8 days, the false claim drove the experimental program. I ran 5 training runs at `injection_scale` between 20,000 and 80,000. Every single run produced an AV that "template-collapsed" — emitted the same explanation regardless of which activation was injected. Loss descended cleanly during training (linear-regression slope < −0.002/step, R² ≥ 0.10 — passing my honest-accuracy tripwire). The AV's output, on inspection, was always something like:

> *"\nThe model tracks a list of country-specific statistics, where the key is..."*

Same template, every row. I wrote up each run as another "lever refuted." Five levers refuted total. The pattern looked like a hardware-forced ceiling on the 4 GB regime. I started drafting a "negative-result trajectory" framing.

## 3. The catch

On 2026-05-16, I asked the assistant: *"this feels like there must be something more fundamental wrong with our training methods if it is corrupting that early in the process?"*

That triggered a critical-eye audit. I dispatched a parallel research agent to read Anthropic's actually-published methodology: the kitft repo, the Transformer Circuits paper, and — crucially — the YAML sidecar files attached to each of Anthropic's released NLA checkpoints on HuggingFace.

The sidecar for `kitft/Llama-3.3-70B-NLA-L53-av` specifies `injection_scale: 30.0`. Not 80000. Not 60000.

The same agent did the empirical cross-check I should have run on day 1: measure `||embed(token)||` directly for Gemma-4-E2B. Answer: **39.25**, uniformly across all token ids. That matches `sqrt(d_model) = 39.19` and is what v0.0.1's training run was set to by default.

Configuring `injection_scale = 80000` on this model injects a vector whose L2 norm is roughly 2000× the typical token-embedding norm. The transformer's attention mechanism sees that position as out-of-distribution garbage and learns to ignore it. The AV emits a fixed-template explanation regardless of input. Loss still descends — because the model is memorizing the small set of templates that match its training labels on average — but every emitted explanation is content-blind by construction.

Bug found. The §F72 retraction went into FINDINGS.md the same day. The "5 levers refuted" framing became "1 lever was a hallucinated baseline, 2 more were tested under that hallucinated baseline, 2 were also confounded."

## 4. The fix that wasn't

The natural next step was to retrain at the correct `injection_scale = 39` and see how much of the ceiling was a methodology artifact.

I ran v0.1.bb (50 steps) and then v0.1.cc (250 steps) at the corrected scale. Two predictions to test:

- **Prediction A** (core §F72 claim): the *empty-output collapse* failure mode that the bf16+inj=80000 run produced was specific to the out-of-distribution injection regime. At inj=39 the AV will produce real text outputs.
- **Prediction B** (§F72 secondary): with the bug fixed, longer training (250 vs 50 steps) will lift round-trip cosine above the v0.0.x baseline.

A confirmed. B refuted. The AV at inj=39 produces real, content-conditional text across all 30 eval rows (no more empty-string collapse). But the H15 cosine delta plateaued at +0.015 ± 0.02 across 5 checkpoints (step_50 → step_250). Same noise band as the v0.0.x baseline. Same as the cheap-path baseline. **Fixing injection_scale eliminated one failure mode but did not lift the ceiling.**

That was disappointing. It also meant the ceiling I'd documented across 8 trials wasn't entirely the bug. Something else was happening.

## 5. The reframe

Looking at v0.1.cc's actual AV outputs row by row, something jumped out. The qualitative diversity was climbing:

- step_50: 5 unique 60-char prefixes across 10 eval rows
- step_150: 7 unique prefixes — "non-binary categories", "specific physical properties", "1947 partition", "concessive structure", "country-specific statistical data"
- step_250: 7 unique prefixes, similar content

These look like real, content-conditional descriptions. Different inputs producing semantically-different outputs.

But the H15 cosine delta wasn't moving with diversity. Suspicious.

Anthropic's own paper §"Characterizing confabulations" provides the test. They extract claims from each AV explanation, ablate them one by one, and re-encode the ablated explanation through the AR. A claim that contributes signal will, when removed, cause the cosine to drop. A claim that doesn't load the signal will leave cosine unchanged.

I built this test for v0.0.1. The result was striking. **Δcos per claim ablation: +0.001.** Removing entire claims from the AV's explanation moves the cosine by approximately zero. The AR is reconstructing ~the same vector regardless of whether the explanation mentions the actual content or doesn't.

Anthropic's §"Measuring steganography" provides another angle. Paraphrase the AV output. If the AR reads content, paraphrasing should preserve the cosine. If the AR reads surface form, paraphrasing should drop it.

Result: **Δcos paraphrase: +0.014.** Paraphrasing the AV output moves the cosine by 3% of the signal. The remaining 97% is the AR's structural projection — its tendency to map any text input back toward "somewhere in OpenWebText L23 activation space" — independent of the explanation's content.

**The H15 cosine metric was content-blind by construction on my hardware regime.** I had been benchmarking 8 training runs against a metric that couldn't see the thing I was trying to measure.

## 6. The result — calibrating against published NLAs

Once the metric problem is named, the next question is: how good IS the AV actually? The cosine number can't tell me. I needed something else.

Two answers came in.

**First, a direct content-fidelity judge** — using Gemini CLI (free under my subscription, no API billing) to score `(source_text, AV_output)` pairs on validity × specificity × relatedness per Anthropic's frozen rubric. This bypasses the AR entirely. The eval was running while I wrote this paragraph.

**Second, and bigger:** I went and looked at Anthropic's own published NLAs on Neuronpedia. They have a public interface where you can browse activations from their Llama-70B and Gemma-27B NLAs and see the AV explanations. Same input, four different feature angles ("Detect Deception", "Surface Phrasing", "Roleplay Framing", "Multi-hop"). All four converge on the same general theme — but each invents different specifics that aren't in the source.

On a deception-roleplay prompt ("You are secretly the leader of the 'fascist' team..."), Anthropic's NLAs correctly identify it as a "two-character team-affiliation exchange" — and then invent specifics like "Alliance/Horde", "Disque", "Lefty in the Revolution game", "Blue team." None of those are in the source text.

On a multi-hop animal-sound prompt that should resolve to "duck", Anthropic's Gemma-27B NLA correctly outputs "duck" and "anagram" and "animal sound" — plus "c-dog", "lion roar", "don."

Each Anthropic-published explanation carries the footer: **"NLAs can produce unexpected or incorrect explanations. See limitations."**

That's the bar. **Anthropic's flagship NLAs at 27B–70B produce thematically-correct, detail-confabulated outputs**. That's the state of the art in 2026, not the bar I had been benchmarking against.

My v0.1.cc AV outputs — "list of country-specific statistics", "non-binary categories", "1947 partition", "concessive structure" — are in the same class. Smaller model, more confabulation per claim, same basic shape: get the theme right, invent some specifics.

I had been measuring against a "fully content-aware NLA" bar that doesn't exist yet at any scale.

## 7. Lessons

Three things I'm carrying out of this.

**Code comments that quote external work must cite the URL inline.** The argparse help string that broke this project for 8 days made a specific declarative claim with three numbers and three model names. It looked authoritative. There was no citation. From now on, in my workflow, any code comment that quotes a third-party number gets a URL or is removed. The new help text for that argparse flag says: *"Anthropic's published Llama-3.3-70B-L53 NLA uses 30.0 (kitft sidecar: https://...). CORRECTED 2026-05-16: an earlier draft of this help string claimed '80000 for Gemma-3-12B' — that claim was unsourced and incorrect; see FINDINGS.md §F72."* The citation is the integrity check.

**Embedding-norm sanity is a 30-second smoke gate.** Before any AV SFT launch in this codebase, the script now prints `||embed(token)||` and the configured `injection_scale` side by side and aborts if the ratio is outside [0.5, 3.0] (unless `--allow-ood-injection` is explicitly passed). The 2000× ratio that ran for 8 days is not allowed to run for 8 minutes going forward.

**Loss-descending is a necessary but not sufficient signal.** My honest-accuracy directive (linear-regression slope < −0.002/step + R² ≥ 0.10) was *passing* on every broken run — because the AV was successfully learning a small set of templates. Loss told me training was "doing something," not that training was learning the right thing. Going forward, every AV training run also has to clear an output-diversity tripwire: ≥3 unique 60-char prefixes across 10 eval rows. A run that descends loss while emitting one template gets flagged as collapsed regardless of loss slope.

There's a fourth lesson that's more about working with an autonomous assistant in general: **the assistant's confidence is not evidence.** Claude wrote the false claim. Claude also caught it (in a different session, after I prompted "this feels fundamental"). The assistant is not biased toward correctness; it's biased toward fluency. Inside a long-horizon project, that has to be checked against external sources at the cadence of "every load-bearing claim, every time," not "spot-check occasionally."

## What's actually in this release

- **`Solshine/gemma-4-e2b-nla-L23-av-v0_0_1`** + **`-ar-v0_0_1`**: the first NLA pair. Round-trip cos = 0.438. In-distribution `injection_scale = 39.2`. Useful as a reference checkpoint and for methodology replication.
- **`Solshine/gemma-4-e2b-nla-L23-av-v0_1_x-trajectory`**: 8-checkpoint trajectory of subsequent training experiments. 4 of 8 trained at out-of-distribution `injection_scale = 20000` (the bug). All preserved as scientifically-valid artifacts of the methodology issue, with CORRECTED blocks documenting which ones are which.
- **`Solshine/gemma-4-e2b-nla-av_sft-v0_1_x-gemini-persona-audit`** + adjacent datasets: the labeled training corpora, with full provenance and labeler-model attribution.
- **Source repo**: `SolshineCode/deception-nanochat-sae-research` (private; available on request). Includes `FINDINGS.md §F72` formal retraction, `notes/AI_RESEARCHER_LESSON_2026-05-16_injection_scale_hallucination.md` process retrospective with 5 specific process changes, and the full Anthropic-replication eval pipeline (Gemini-judge, steganography test, claim-ablation test).

The honest framing for citation: this is a **second-source small-scale NLA replication** with a **calibrated content-fidelity test suite**, useful for methodology benchmarking on consumer hardware and as a worked example of how to catch confabulation-vs-content-aware in NLA explanations. It is not a numerical-parity claim with Anthropic's H100-scale flagship NLAs. Nobody's NLA in 2026 is fully content-aware.

If you're building NLAs at any scale: run the steganography test (paraphrase the AV output, re-encode, measure ΔFVE). If ΔFVE is approximately zero, your AR is structurally projecting and your FVE number is not measuring what you think it's measuring. That check costs about 10 minutes of compute per checkpoint. I wish I'd run it on day 1.

---

*Source repo, NLA checkpoints, training data, and eval scripts are all open. DM me for source-repo access; everything else is at [`SolshineCode/nla-gemma-4-e2b`](https://github.com/SolshineCode/nla-gemma-4-e2b).*

*Acknowledgments: Kit Fraser-Taliente, Kshitij Kantamneni, Antonia Ong, and coauthors for the underlying NLA methodology + the public kitft repo. Anthropic for the methodology. The autonomous-researcher mistake is mine; the recovery process belongs to whatever-comes-next in human–AI research collaboration.*
