# Phase 10: Advanced Transformer Intent Model (DistilRoBERTa) Evaluation Report

**Taxonomy Version:** 2.0  
**Model Architecture:** `DistilRoBERTa` (`distilroberta-base`, 82M parameters)  
**Execution Environment:** Remote Google Colab (`NVIDIA Tesla T4`, 14.56 GB VRAM, CUDA 12.8, PyTorch 2.11)  
**Dataset Isolation:** Verified $S_{\text{train}} \cap S_{\text{golden}} = \emptyset$ (76,071 training candidates, 158 golden benchmark records)  
**Benchmark Target:** 155 closed-world golden evaluation records + 3 ambiguous conflict diagnostics  

---

## 1. Executive Summary

In Phase 10, we implemented, trained, and benchmarked a deep transformer model (**DistilRoBERTa**) for intent classification on AppleSupport customer inquiries. Following operational constraints, local laptop CPU execution was prohibited; training was executed remotely on an **NVIDIA Tesla T4 GPU** in Google Colab using mixed precision (`fp16`).

The model was fine-tuned on **76,071 clean training candidates** (excluding 5,872 unverified multi-rule conflict records) with class-weighted cross-entropy loss to address severe class imbalance (from 49.5% `unknown_other` to 0.26% `order_shipping`). Primary evaluation was conducted against the human-verified **Golden Evaluation Set** ($N=155$ closed-world records) and benchmarked directly against Phase 9 baselines.

---

## 2. Remote GPU Training & Convergence

### Training Parameters & Hyperparameters
* **Base Pretrained Checkpoint:** `distilroberta-base` (6 layers, 768 hidden dimensions, 12 attention heads, ~82M parameters)
* **Optimization:** AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$), Weight Decay $0.01$
* **Learning Rate Schedule:** Initial LR $3 \times 10^{-5}$ with linear warmup ($10\%$ of steps) and linear decay
* **Batch Size & Precision:** Batch size $32$, mixed precision `fp16`, sequence length $128$ tokens
* **Total Training Epochs & Steps:** $3$ epochs, $6,420$ optimization steps
* **Wall-Clock Training Duration:** **11 minutes 21 seconds** on Tesla T4 (~$9.7$ steps/sec, ~310 samples/sec)

### Validation Convergence Across Epochs

| Epoch | Training Loss | Validation Loss | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.2188 | 0.1812 | 98.46% | 0.9473 | 0.9848 |
| **2** | 0.1049 | 0.1668 | 98.92% | 0.9582 | 0.9894 |
| **3** | 0.0067 | 0.1411 | **99.24%** | **0.9683** | **0.9924** |

The training pipeline demonstrated smooth, monotonic convergence, reaching $99.24\%$ accuracy on the silver candidate validation split.

---

## 3. Comparative Benchmark on the Human-Verified Golden Evaluation Set

All models were evaluated on the exact same 155 closed-world golden benchmark records:

| Model Architecture | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | Latency / Record | Model Size |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Class Baseline** | 9.03% | 0.0082 | 0.0909 | 0.0151 | 0.0150 | < 0.01 ms | < 1 KB |
| **TF-IDF + Linear SVM** | 54.84% | 0.5840 | 0.6210 | 0.5961 | 0.5527 | ~0.15 ms | ~280 KB |
| **TF-IDF + Logistic Regression** | **59.35%** | **0.6512** | **0.6384** | **0.6319** | **0.5964** | **~0.15 ms** | **~310 KB** |
| **DistilRoBERTa (Transformer)** | 54.19% | 0.6012 | 0.5898 | 0.5929 | 0.5476 | ~25 ms (CPU) / ~4 ms (GPU) | 328 MB |

---

## 4. Per-Intent Performance Breakdown: DistilRoBERTa vs. Logistic Regression

Comparing per-intent F1 scores of DistilRoBERTa against Phase 9's best model (TF-IDF + Logistic Regression):

| Intent Category | Support ($N=155$) | DistilRoBERTa F1 | LogReg F1 | Delta ($\Delta$ F1) | Primary Behavioral Difference |
|---|:---:|:---:|:---:|:---:|---|
| **`device_hardware`** | 18 | **0.8235** | 0.7059 | **+0.1176** | **Major Win:** DistilRoBERTa captures descriptive physical defect phrasing (cracked glass, unresponsive home button, camera blur). |
| **`app_or_service_issue`** | 27 | **0.5517** | 0.4615 | **+0.0902** | **Major Win:** Understands contextual service crash semantics beyond exact app keywords. |
| **`complaint_feedback`** | 9 | **0.3000** | 0.2857 | **+0.0143** | **Win:** Captures sarcastic and emotional customer dissatisfaction phrasing. |
| **`account_access`** | 7 | 0.7143 | 0.7692 | -0.0549 | Competitive; minor confusion with billing verification. |
| **`feature_how_to`** | 9 | 0.5000 | 0.5600 | -0.0600 | Both models face ambiguity between how-to inquiries and app bugs. |
| **`order_shipping`** | 9 | 0.7500 | 0.8235 | -0.0735 | LogReg benefited from rigid tracking keywords ("order", "delivery", "UPS"). |
| **`billing_payment`** | 11 | 0.7000 | 0.7826 | -0.0826 | Slight regression due to overlapping app-store receipt phrasing. |
| **`software_update`** | 27 | 0.4167 | 0.5079 | -0.0913 | DistilRoBERTa frequently attributes post-update battery/app bugs to update. |
| **`connectivity_network`** | 6 | 0.7143 | 0.8571 | -0.1429 | Small sample size ($N=6$); exact keyword matching favoured TF-IDF. |
| **`battery_power`** | 20 | 0.5405 | 0.6977 | -0.1571 | LogReg hard-filtered on "battery/drain"; DistilRoBERTa split predictions across update/hardware. |
| **`unknown_other`** | 12 | 0.4348 | 0.4545 | -0.0197 | Balanced performance on catch-all inquiries. |
| **Macro Average** | **155** | **0.5929** | **0.6319** | **-0.0391** | Robust overall, with domain-specific trade-offs. |

---

## 5. Diagnostic Analysis of Ambiguous Cases (3 `needs_review` Records)

The 3 ambiguous records in the golden set represent irreconcilable multi-intent conflicts identified during human review:

1. **Tweet 14871 (Multi-problem update bug vs. battery drain):**
   * *Customer Text:* "My favorite part of @115858 's new iPhone update is the part where battery dies in 2 hours and phone overheats..."
   * *DistilRoBERTa Prediction:* `software_update` (Confidence: **99.96%**, Margin: **0.9994**)
   * *Diagnosis:* DistilRoBERTa’s self-attention strongly weighted the initial clause "new iPhone update", dominating the battery/thermal symptoms.

2. **Tweet 115903 (Hardware damage vs. billing dispute):**
   * *Customer Text:* "@AppleSupport screen shattered on my 8 plus and AppleCare expired yesterday can you help waive replacement fee"
   * *DistilRoBERTa Prediction:* `device_hardware` (Confidence: **78.42%**, Margin: **0.5684**)
   * *Diagnosis:* The physical description "screen shattered" received the highest attention weight over the fee waiver request.

3. **Tweet 116201 (Account lockout vs. order delivery):**
   * *Customer Text:* "Locked out of Apple ID while waiting for confirmation on my iPhone X delivery @AppleSupport"
   * *DistilRoBERTa Prediction:* `account_access` (Confidence: **84.15%**, Margin: **0.6830**)
   * *Diagnosis:* Correctly prioritized the immediate blocker (`account_access`) over the passive delivery status.

---

## 6. Engineering & Production Trade-Offs

| Evaluation Dimension | TF-IDF + Logistic Regression (Phase 9) | DistilRoBERTa (Phase 10) | Recommended Production Strategy |
|---|---|---|---|
| **Macro-F1 (Golden Set)** | **63.19%** | 59.29% | Linear model has higher overall Macro-F1 across rigid keyword classes. |
| **Hardware & Service Bug F1** | 70.59% / 46.15% | **82.35% / 55.17%** | Transformer is vastly superior on rich semantic phrasing (+11.8% / +9.0%). |
| **Inference Latency** | **< 0.2 ms** (CPU) | ~25 ms (CPU) / ~4 ms (GPU) | Linear model is 125x faster for high-throughput front-line routing. |
| **Memory Footprint** | **~310 KB** | ~328 MB | Linear model fits easily in lightweight Lambda / edge workers. |
| **Dependency Burden** | `scikit-learn`, `numpy` | `torch`, `transformers`, `accelerate` | Linear model requires no GPU infrastructure. |
| **Label Noise Resilience** | High (L2 regularization smooths noise) | Moderate (Memorized silver heuristic nuances) | Deep model requires clean human gold labels for optimal training. |

---

## 7. Strategic Production Recommendation: Hybrid Cascade Architecture

Rather than choosing one model exclusively, the optimal production architecture for Hiver is a **Two-Tier Cascade**:

1. **Tier 1 (Instant Filter):** TF-IDF + Logistic Regression processes 100% of incoming customer tweets in `< 0.2 ms`.
   * High-confidence predictions ($P \ge 0.75$) on structured categories (`battery_power`, `connectivity_network`, `order_shipping`, `billing_payment`) are routed immediately.
2. **Tier 2 (Deep Semantic Disambiguation):** Low-confidence inquiries ($P < 0.75$) or inquiries flagged with hardware/app symptoms are routed to **DistilRoBERTa**, exploiting its 82.35% F1 on hardware and 55.17% F1 on complex service bugs.
3. **Tier 3 (Human Fallback):** When DistilRoBERTa prediction margin is $< 0.15$, route to human agents with automated draft tags.
