# HARBOR-QA: Retrieval-Augmented Few-Shot Prompting for Bengali Factoid QA
<p align="center">
  <img src="https://img.shields.io/badge/Task-Bengali%20QA-blue" />
  <img src="https://img.shields.io/badge/Private%20F1-0.6550-brightgreen" />
  <img src="https://img.shields.io/badge/Public%20F1-0.6475-green" />
  <img src="https://img.shields.io/badge/Training-Free-orange" />
  <img src="https://img.shields.io/badge/GPU-T4%2015GB-lightgrey" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

> **Competition:** "Are You Sure LLM Is Enough?" — Intra-CUET ML Contest 2.0  
> **Venue:** Published as a system description paper at **ACL 2026 Workshops**  
> **Authors:** Shuva Dey · Priyangshu Barua — *Chittagong University of Engineering & Technology (CUET)*
<p align="center">
  <img src="docs/banner.png" width="900"/>
</p>

<h1 align="center">
HARBOR-QA
</h1>

<p align="center">
Training-Free Retrieval-Augmented Few-Shot Question Answering for Bengali Wikipedia
</p>

<p align="center">

[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue)]()
[![Private F1](https://img.shields.io/badge/Private%20F1-0.6550-success)]()
[![Training](https://img.shields.io/badge/Training-Free-orange)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

</p>
---

## Overview

HARBOR-QA is a **training-free, four-phase Retrieval-Augmented Generation (RAG) pipeline** for short-answer factoid question answering over a Bengali Wikipedia knowledge base. The system requires **zero model fine-tuning** and runs entirely on a single free-tier Kaggle T4 GPU, yet outperforms both a pure-extractive baseline and a QLoRA fine-tuned model.

The core insight: *for closed-domain QA over a fixed knowledge base, a well-designed retrieval front-end contributes more to final F1 than model scale or fine-tuning.*

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HARBOR-QA Pipeline                       │
│                                                                 │
│  Phase 1 — Offline Indexing                                     │
│  ┌──────────┐   ┌───────────┐   ┌─────────────┐                │
│  │  6,720   │──▶│  55,841   │──▶│ BGE-M3 FAISS│                │
│  │ KB Arts  │   │  Chunks   │   │  Dense Idx  │                │
│  └──────────┘   └───────────┘   ├─────────────┤                │
│                                 │ BM25 Sparse │                │
│                                 │    Index    │                │
│                                 └─────────────┘                │
│                                                                 │
│  Phase 2 — Online Hybrid Retrieval (per question)               │
│  ┌──────────────────────────────────────────────┐               │
│  │  Top-20 dense hits + Top-20 sparse hits      │               │
│  │         ──── RRF Fusion ────                 │               │
│  │         Top-4 chunks → context (≤ 1,800 ch)  │               │
│  └──────────────────────────────────────────────┘               │
│                                                                 │
│  Phase 3 — Few-Shot Generation                                  │
│  ┌──────────────────────────────────────────────┐               │
│  │  Gemma-2-9B-IT (4-bit NF4) + 5 exemplars    │               │
│  │  Greedy decode · max 32 tokens · batch = 8   │               │
│  └──────────────────────────────────────────────┘               │
│                                                                 │
│  Phase 4 — Answer Refinement                                    │
│  ┌──────────────────────────────────────────────┐               │
│  │  Unknown? → Pass 1: Force re-prompt          │               │
│  │  Still unknown? → Pass 2: Extractive fallback│               │
│  └──────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1 — Knowledge Base Indexing

- Custom parser strips 6,720 Bengali Wikipedia articles of navigation chrome, reference sections, and inline markup; **disambiguation pages are intentionally retained** (gold answer = `দ্ব্যর্থতা নিরসন।`)
- `RecursiveCharacterTextSplitter` with Bengali-aware separators (`।`, `.`, `?`, `\n`) produces **55,841 chunks** (512-char, 128-char overlap)
- **Dense index:** BGE-M3 embeddings (dim=1024, L2-normalised) → FAISS `IndexFlatIP`
- **Sparse index:** BM25Okapi over whitespace-tokenised chunk text

### Phase 2 — Hybrid Retrieval with RRF

For each question, top-20 candidates are retrieved from each index and fused via Reciprocal Rank Fusion:

$$\text{RRF}(d) = \frac{1}{r_d(d)+1} + \frac{1}{r_s(d)+1}$$

The top-4 RRF chunks are concatenated into the LLM context, hard-capped at 1,800 characters. **All contexts are pre-computed before the LLM is loaded**, so the embedder and the generator never occupy GPU memory simultaneously — the key memory management decision enabling this pipeline on a 15 GB T4.

### Phase 3 — Few-Shot Prompted Generation

- `google/gemma-2-9b-it` loaded in **4-bit NF4 quantisation** (~5–6 GB VRAM)
- **5 curated Bengali Q→A exemplars** teach the terse, daari-terminated answer style without any training
- Greedy decoding (`do_sample=False`, `num_beams=1`), `max_new_tokens=32`, batch size 8

### Phase 4 — Two-Pass Answer Refinement

| Pass | Trigger | Action |
|------|---------|--------|
| Post-processor | Every answer | Strip prefix, truncate to 16 tokens, append daari |
| Pass 1 | Answer = `অজানা।` | Force LLM re-prompt prohibiting the unknown token |
| Pass 2 | Still unknown | Keyword-matched extractive span from context |

---

## Results

| System | Public F1 | Private F1 | Training Required? |
|--------|-----------|------------|--------------------|
| Extractive baseline (XLM-RoBERTa-Large-SQuAD2) | 0.538 | 0.539 | No |
| QLoRA Gemma-2-2B (fine-tuned) | 0.635 | 0.628 | Yes |
| **HARBOR-QA (ours)** | **0.6475** | **0.6550** | **No** |

HARBOR-QA surpasses the fine-tuned baseline by **+2.70 F1 points** (private) with zero training.

### Answer Quality Breakdown (1,500 test predictions)

| F1 Range | Count | Percentage |
|----------|-------|------------|
| Perfect (F1 = 1.0) | 709 | 47.3% |
| Partial (0.5 ≤ F1 < 1.0) | 194 | 12.9% |
| Low (0 < F1 < 0.5) | 123 | 8.2% |
| Zero (F1 = 0.0) | 474 | 31.6% |

**Retrieval ceiling:** 75.3% of training questions have the gold answer verbatim in the retrieved context — the theoretical upper bound for any purely extractive approach on this configuration.

---

## Setup & Reproduction

### Requirements

```bash
pip install sentence-transformers rank_bm25 langchain-text-splitters faiss-cpu
pip install -U "transformers>=4.45,<5" "accelerate>=1.0" "bitsandbytes>=0.43"
```

### Data

The competition dataset is derived from [Indic-RAG-Suite](https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite) by AI4Bharat. Place the following files under `Dataset/`:

```
Dataset/
├── train.csv          # 3,000 Q&A pairs with gold answers
├── test.csv           # 1,500 test questions
└── Knowledge_Base.txt # ~6,720 Bengali Wikipedia articles
```

### HuggingFace Token (for Gemma)

`google/gemma-2-9b-it` is a gated model. You must:
1. Accept the licence at [huggingface.co/google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
2. Create a read-access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add it as a Kaggle Secret named `HF_TOKEN`

> **Alternative (ungated):** Replace `MODELS` in Cell 3 with `["Qwen/Qwen2.5-7B-Instruct"]` to skip the HF login step.

### Running on Kaggle

1. Upload the notebook to Kaggle and attach the competition dataset
2. Set accelerator to **GPU T4 x2** (the notebook pins `CUDA_VISIBLE_DEVICES=0` to avoid DataParallel conflicts with 4-bit quantisation)
3. Select **Run → Restart & Run All**

> **Expected runtime:** ~10–15 min for BGE-M3 embedding + ~2–3 hours for batched LLM inference over 1,500 questions on a single T4.

---

## Hyperparameter Configuration

All parameters are centralised in **Cell 3** for full reproducibility:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `CHUNK_SIZE` | 512 chars | Fits short factual answers |
| `CHUNK_OVERLAP` | 128 chars | Avoids splitting answers across boundaries |
| `DENSE_K` / `BM25_K` | 20 | Top-k from each retriever |
| `FINAL_K` | 5 | Chunks after RRF fusion |
| `CTX_CHUNKS` | 4 | Chunks fed to the LLM |
| `CTX_CHAR_CAP` | 1,800 chars | Hard context character cap |
| `EMBED_MODEL` | `BAAI/bge-m3` | Multilingual dense embedding |
| `N_FEWSHOT` | 5 | In-context exemplars |
| `MAX_NEW_TOKENS` | 32 | Sufficient for short factual answers |
| `GEN_BATCH` | 8 | Throughput vs. VRAM balance |
| `MAX_ANS_TOKENS` | 16 | Post-processing safety cap (99th %ile ≈ 12 tokens) |
| `SEED` | 42 | Global random seed |

---

## Repository Structure

```
├── bengali-rag-qa-final-submission.ipynb   # Full reproducible pipeline (Kaggle notebook)
├── README.md
└── paper/
    └── HARBOR-QA_ACL2026.pdf               # System description paper (ACL 2026 Workshops)
```

---

## Key Design Decisions

**Sequential memory management.** By pre-computing all 4,500 retrieval contexts (train + test) before loading the LLM, the pipeline avoids ever having both the BGE-M3 encoder and the 9B-parameter generator in GPU memory at the same time. This single decision makes the pipeline feasible on a 15 GB T4.

**Disambiguation page retention.** The custom KB parser intentionally keeps disambiguation pages, whose gold answers are the fixed string `দ্ব্যর্থতা নিরসন।`. Stripping them would cause guaranteed zero-F1 on that question subset.

**RRF with k=1.** The standard RRF formulation uses k=60; we use k=1 (i.e., `1/(rank+1)` directly) to give higher weight to top-ranked documents, which empirically produces better fusion on this domain.

---

## Limitations

- **Retrieval ceiling (75.3%):** Questions whose gold answers do not appear verbatim in the retrieved context cannot be answered correctly by any purely extractive component.
- **Morphological tokenisation:** BM25 uses whitespace tokenisation and does not account for Bengali morphological inflection. A morpheme-aware tokeniser would improve sparse recall.
- **Long gold answers:** The 16-token post-processing cap is rarely active (99th percentile ≈ 12 tokens), but answers of 6+ words show only 3.8% perfect match — the primary area for future improvement.

---

## Acknowledgements

The competition dataset is derived from [Indic-RAG-Suite](https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite) assembled by AI4Bharat. Knowledge base construction is credited to [Ratnajit Dhar](https://www.kaggle.com/code/ratnajitdhar08/creating-dataset-for-ml-contest). We thank the competition organisers at CUET for providing evaluation infrastructure.
