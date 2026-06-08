Ryota Nakamura
Faculty of Data Science, Musashino University
Tokyo 135-8181, Japan
ryota.nakamura@ds.musashino-u.ac.jp

June 8, 2026

To the Editor-in-Chief and Editorial Board,
*IEEE Access*

**Re: Submission of "Cache the Known, Ask Only the Novel: A Confidence-Routed Harness for Sub-1B Language Models"**

Dear Editors,

Please consider the attached manuscript for publication in *IEEE Access*. The work studies how to deploy sub-1-billion-parameter ("tiny") language models reliably and cheaply on device, and reaches a conclusion that we believe is both practical and counter to common practice.

**Problem and contribution.** The usual route to making a tiny model useful is to strengthen the model (fine-tuning, larger checkpoints). We argue and demonstrate the opposite for tiny-model *systems*: the path to better performance, cost, and reliability is not a stronger model but a better *division of labour* — separate inputs into "known" and "novel," answer the known with deterministic, non-parametric components, and spend the language model only on the genuinely novel. Concretely, we pair a frozen sub-1B model with an experience cache of verified cases that exposes two complementary paths (a no-LLM embedding classifier and the LLM conditioned on retrieved exemplars and distilled rules), and we add a *confidence gate* that decides between them.

**Why it is a contribution, not a heuristic.** The central engineering result is that the routing decision is principled and validated: the non-parametric classifier's own confidence predicts whether its no-LLM answer is correct at mean AUROC 0.83 across seven datasets (nearest-neighbor similarity alone is near chance). Routing by that signal matches or beats the best single path (85.6% vs 83.9% no-LLM, 57.0% LLM) while invoking the language model on only ~4% of inputs, and approaches an oracle ceiling. A comprehensive sweep — 11 datasets × 8 sub-1B models (88 model–dataset cells) — shows the no-LLM path dominates every LLM-path variant (71.9% vs 26–29% mean), is nearly invariant to class count (≈80% from 2 to 77 classes), and rivals LoRA fine-tuning at zero training cost; structuring the model's own failures into discriminative rules helps the LLM path only marginally. We also delimit the claim honestly: a generative-QA pilot marks where the no-LLM path stops transferring, so all claims are scoped to closed-set classification, with open-ended generation and reasoning left to future work.

**Suitability and significance for IEEE Access.** The paper offers a clear, reproducible engineering message for the large practitioner audience deploying small models at the edge — *when not to call the model* — supported by a large empirical study, validity controls (leave-one-theme-out splits, label-corruption and 1-NN baselines, Wilson confidence intervals, paired McNemar tests), an external re-test on six real-world datasets, and a full open-source release of code, scripts, and result data.

**Originality and ethics.** This manuscript is original, has not been published previously, and is not under consideration elsewhere. There are no conflicts of interest. All datasets used are public benchmarks or author-created; no human-subjects or private data are involved. The author has approved the submission.

Thank you for your consideration. I look forward to the reviewers' feedback.

Sincerely,

Ryota Nakamura
Faculty of Data Science, Musashino University
