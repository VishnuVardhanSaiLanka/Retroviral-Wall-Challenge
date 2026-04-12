# Literature-to-Feature Map

## Goal

Use a small number of high-value external papers to identify the next feature and experiment blocks most likely to improve the frozen `0.7217` benchmark.

This is the 80/20 pass:

- read only the most relevant primary sources
- extract only the mechanisms that can plausibly change the current model
- map them into a small number of candidate feature blocks

## Key Sources

### 1. Structural basis for pegRNA-guided reverse transcription by a prime editor

Source:

- Nature 2024
- https://www.nature.com/articles/s41586-024-07497-8

Why it matters:

- directly describes how M-MLV RT sits relative to Cas9 and how reverse transcription proceeds in prime editing
- shows that the RT stays in a relatively fixed position while the pegRNA-synthesized DNA heteroduplex accumulates along the Cas9 surface
- implies that productive PE function is strongly shaped by geometry of tethered reverse transcription, not generic RT competence alone

High-value implications:

- add features tied to fixed RT positioning and accessibility in the fused complex
- add features tied to heteroduplex build-up path rather than only local pocket chemistry
- model the difference between “can polymerize” and “can polymerize in the PE geometry”

### 2. Engineered CRISPR prime editors with compact, untethered reverse transcriptases

Source:

- Nature Biotechnology 2023
- https://www.nature.com/articles/s41587-022-01473-1

Why it matters:

- shows that intact fusion architecture is not the only viable organization
- broadens the RT design space used for prime editing
- suggests that properties supporting PE may not reduce to similarity to MMLV itself

High-value implications:

- de-emphasize direct MMLV similarity as a scientific endpoint
- focus on transferable RT properties compatible with PE function
- use literature on compact/untethered RTs to motivate broader mechanistic candidates

### 3. Phage-assisted evolution and protein engineering yield compact, efficient prime editors

Source:

- Cell 2023
- https://www.sciencedirect.com/science/article/pii/S0092867423008541

Why it matters:

- shows that different RTs specialize in different edit types
- demonstrates that pegRNA length and secondary structure interact with optimal RT choice
- implies PE efficiency depends on interaction between RT properties and substrate/edit context, not a single generic activity score

High-value implications:

- treat RT competence as conditional on substrate geometry
- add feature blocks describing tolerance to longer/more structured extension paths
- consider whether some current benchmark failures reflect unmodeled specialization dimensions

### 4. Reverse transcriptases prime DNA synthesis

Source:

- Nucleic Acids Research 2023
- https://academic.oup.com/nar/article/51/14/7125/7188755

Why it matters:

- argues that direct priming is a conserved RT property across major RT classes
- suggests initiation/priming capacity may be a more transferable RT axis than family identity

High-value implications:

- add features around catalytic-neighborhood accessibility and local initiation geometry
- model whether the active site is arranged to support initiating extension, not just elongating after engagement

### 5. Prime editing: therapeutic advances and mechanistic insights

Source:

- Gene Therapy 2024/2025
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11946880/

Why it matters:

- synthesizes the mechanistic and engineering bottlenecks in prime editing
- useful for prioritizing which PE-specific constraints deserve modeling effort

High-value implications:

- focus on mechanistic rules that repeatedly affect PE efficiency
- use as a filter to decide which feature ideas are worth implementing

## Highest-Leverage New Feature Blocks

### 1. PE Extension-Path Features

Hypothesis:

- PE success depends on whether the RT can reverse transcribe while the pegRNA-synthesized DNA heteroduplex builds along the Cas9 surface.

Candidate features:

- active-site exposure under a tethered orientation prior
- extension-path openness from catalytic center outward
- steric burden around the likely heteroduplex accumulation path
- corridor continuity for ordered basic residues rather than generic positive charge

Why high value:

- directly motivated by the 2024 Nature structure
- closest match to what the current benchmark seems to undermeasure

### 2. Priming / Initiation Features

Hypothesis:

- transferable RT function may depend on the ability to initiate productive synthesis on the PE substrate, not only sustain polymerization.

Candidate features:

- local catalytic-neighborhood accessibility
- arrangement of nearby Lys/Arg around the acidic cluster
- cavity openness in the first shell around the catalytic center
- motif + exposure features that describe initiation readiness

Why high value:

- supported by the NAR 2023 RT priming paper
- may generalize across RT families better than family-specific elongation heuristics

### 3. Edit-Path / Substrate-Tolerance Features

Hypothesis:

- different RTs may be specialized for different template/extension regimes, as seen in engineered prime editors.

Candidate features:

- tolerance proxies for longer extension paths
- structured-path burden proxies
- geometric room for extended heteroduplex growth

Why high value:

- supported by the Cell 2023 PE evolution paper
- could explain why some RTs fail despite plausible catalytic competence

## What Not to Overinvest In First

- raw ESM embeddings
- broad family-similarity structural features
- global composition summaries without localization
- replacing the frozen benchmark wholesale with a brand-new biology-only branch

## Recommended Next Research Program

### Phase 1

- keep the frozen `0.7217` benchmark intact
- add one PE extension-path block informed by the Nature 2024 structure
- test it only as an augmentation to the frozen benchmark

### Phase 2

- add one priming/initiation block informed by the NAR 2023 RT priming paper
- compare whether it helps the same Retroviral false negatives

### Phase 3

- if neither Phase 1 nor Phase 2 improves LOFO, treat the current internal feature regime as near-exhausted and prioritize new data or more precise structure-derived features

## Current Conclusion

The fastest nontrivial path forward is not more generic model tuning.

It is:

1. literature-backed PE geometry features
2. literature-backed priming/initiation features
3. adding them narrowly on top of the frozen benchmark

If those do not beat `0.7217`, then the bottleneck is likely the information content of the current data/feature regime rather than the optimizer.
