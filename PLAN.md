# THE ULTIMATE 100-DAY AI/ML CHALLENGE

**Goal**: Build nanochat from scratch with continual learning capability

**Timeline**: 14+ weeks (Weeks 0-2 Foundations, then 12 weeks core)

---

## TABLE OF CONTENTS

1. [Design Principles](#design-principles)
2. [Hierarchical Dependency Graph](#hierarchical-dependency-graph)
3. [The Ultimate 14-Week Plan](#the-ultimate-14-week-plan)
4. [Math Refresher](#math-refresher)
5. [Paper Reading Strategy](#paper-reading-strategy)
6. [Tools & Resources](#tools--resources)
7. [Evaluation Framework](#evaluation-framework)
8. [Weekend Reflection Protocol](#weekend-reflection-protocol)
9. [Flexibility Mechanisms](#flexibility-mechanisms)
10. [Final Checklist](#final-checklist)

---

## DESIGN PRINCIPLES

1. **Top-down with safety**: Start with nanochat goal, recurse to fundamentals, but build foundations first (Weeks 0-2)
2. **Deep Dive First**: Complete exercises and read papers (Tier 1) immediately, don't rush.
3. **Modular weeks**: Each week = **Core** + **Stretch** + **Mega-stretch**
4. **Measure everything**: Metrics dashboard from Week 4 (loss, perplexity, throughput, memory)
5. **Flexible timeline**: Compress or extend based on deep dive needs.
6. **Weekend protocol**: Formalized reflection + pivot decision tree every Sunday.
7. **SOTA-first after foundations**: 2025 techniques (MLA, GRPO, FlashAttention-2) by Week 11.
8. **Continual learning throughout**: Integrated from Week 9.
9. **Tool use early**: Week 7, not Week 11 (early integration crucial).
10. **Open-ended ceiling**: Week 16+ research tracks (HTM, JEPA, MoE).

---

## HIERARCHICAL DEPENDENCY GRAPH

*(Same complexity as before, just stretched timeline)*

### Critical Path (Dependencies)

- **Weeks 0-2**: Micrograd + Makemore 1-2 + Papers → Deep Foundation
- **Week 3**: Makemore 3-5 (Optimization/Architectures) → DNN Mastery
- **Week 4**: Transformer (A1) ← depends on Week 3
- **Week 5**: Tokenization (B3) → Unblocks data experiments
- **Week 6**: Training loop (B2) + A1 → First real training
- **Week 7**: Modern arch (A2) + Tool use basics (C3) → 2025-ready
- **Week 8**: Evaluation (B3) → Measurement infrastructure
- **Week 9**: LoRA (C1) → PEFT foundation for continual learning
- **Week 10**: Continual learning core (C2) → Your obsession begins
- **Week 11**: SOTA 2025 (A3) → MLA, GRPO, cutting-edge
- **Week 12**: Alignment (C3) → SFT, GRPO, DPO
- **Week 13**: Integration (C2 + C3) → Hydra model with tools
- **Week 14**: Scale (B3) → Multi-GPU, 1B params
- **Week 15**: Nanochat speedrun → Final assembly
- **Week 16+**: Research tracks

---

### WEEKS 0-2: FOUNDATIONS & MAKEMORE DEEP DIVE

**Goal**: Deep understanding of Autograd, MLPs, and Model foundations. Complete all exercises and read core papers.

#### Core - 14 Days (Completed)
- [x] **Math Refresher**: Derivatives, Chain Rule, Matrix Ops.
- [x] **Micrograd**: Build autograd engine from scratch.
  - [x] Implemented `Value` class, topological sort, backward pass.
  - [x] Exercises: Gradient checking, manual backprop.
- [x] **Makemore Part 1 (Bigram)**:
  - [x] Probability tables, sampling, loss.
  - [x] Exercises: Tensor broadcasting, effective batching.
- [x] **Makemore Part 2 (MLP)**:
  - [x] Embedding layer, hidden layer, interactions.
  - [x] Train/Dev/Test splits, hyperparameter tuning.
  - [x] Exercises: Learning rate finding, model capacity experiments.
- [x] **Paper Reading**:
  - [x] *A Neural Probabilistic Language Model* (Bengio et al., 2003) - Read & Highlighted.
  - [x] Exercises: Replicate key figures/tables from the paper.

#### Projects
- [x] **Vivid Descent**: 3D Neural Net Visualizer.

---

### WEEK 3: MAKEMORE MASTERY (OPTIMIZATION & ARCHITECTURES)

**Goal**: Look under the hood. Understand why deep networks are hard to train and how modern architectures (CNNs, WaveNet) fix it.

#### Core - 5-7 Days
- [ ] **Part 3: Activations & Gradients (BatchNorm)**:
  - Diagnose vanishing/exploding gradients.
  - Implement BatchNorm from scratch.
  - **Exercise**: Visualize activation histograms before/after BatchNorm.
- [ ] **Part 4: Backprop Ninja**:
  - Manual backpropagation through a standard MLP (cross-entropy, linear, tanh).
  - **Exercise**: Match PyTorch gradients exactly.
- [ ] **Part 5: WaveNet (CNNs)**:
  - Hierarchical structure, dilated convolutions.
  - **Exercise**: Implement causal convolutions.
  
#### Papers (Critical)
- *Batch Normalization* (Ioffe & Szegedy, 2015)
- *WaveNet* (DeepMind, 2016)

---

### WEEK 4: MINIMAL TRANSFORMER FROM SCRATCH

**Goal**: Build GPT architecture, understand attention deeply.

#### Core - 5 Days
- [ ] **Implement from scratch** (no HuggingFace, pure PyTorch):
  - Scaled dot-product attention
  - Multi-head attention
  - MLP block + Residuals + LayerNorm
  - Causal mask
- [ ] **Train tiny GPT**: Character-level on Shakespeare.
- [ ] **Exercises**:
  - Visualize attention maps.
  - Implement "Attention is All You Need" Section 3 faithfully.

#### Papers (Critical)
- *Attention Is All You Need* (Vaswani et al., 2017)

---

### WEEK 5: TOKENIZATION & DATA PIPELINE

**Goal**: Build BPE tokenizer, create data shards.

#### Core - 4 Days
- [ ] **Build BPE tokenizer**:
  - Byte-level BPE, merge rules.
  - Special tokens handling.
- [ ] **Data Pipeline**:
  - Download TinyStories / OpenWebText.
  - Create memory-mapped shards (numpy.memmap).
- [ ] **Exercises**:
  - Compare BPE vocab size effects on compression ratio.

#### Papers
- *Neural Machine Translation of Rare Words with Subword Units* (BPE Paper)

---

### WEEK 6: TRAINING LOOP & EFFICIENCY

**Goal**: Full training pipeline with optimization, logging.

#### Core - 5 Days
- [ ] **Training Engine**:
  - AdamW optimizer, Cosine LR scheduler.
  - Gradient Clipping, AMP (Mixed Precision).
  - Gradient Accumulation.
- [ ] **Logging**: TensorBoard / WandB integration.
- [ ] **Exercises**:
  - Sweep learning rates, visualize loss landscapes.

#### Papers
- *Adam: A Method for Stochastic Optimization* (Kingma & Ba)

---

### WEEK 7: MODERN ARCHITECTURE + TOOL USE BASICS

**Goal**: Upgrade to 2025 architecture (Llama-style) and add tool use.

#### Core - 5 Days
- [ ] **Modernize GPT**:
  - RoPE (Rotary Embeddings).
  - RMSNorm.
  - SwiGLU activations.
- [ ] **Tool Use**:
  - Special tokens `<CALC>`.
  - Simple interpreter for math/logic.

#### Papers
- *RoFormer* (RoPE)
- *Llama 2 Technical Report*
- *GLU Variants Improve Transformer* (SwiGLU)

---

### WEEK 8: EVALUATION FRAMEWORK

**Goal**: Build the yardstick.

#### Core - 4 Days
- [ ] **Eval Harness**:
  - Perplexity (WikiText-2, C4).
  - Accuracy (ARC-Easy, GSM8K-Tiny).
- [ ] **Baseline Training**:
  - Train 124M param model (GPT-2 Small equivalent).
  - Establish baseline metrics.

#### Papers
- *Scaling Laws for Neural Language Models* (Kaplan et al., 2020)

---

### WEEK 9: PEFT MASTERY (LoRA)

**Goal**: Parameter-Efficient Fine-Tuning.

#### Core - 5 Days
- [ ] **LoRA from Scratch**:
  - Low-rank decomposition matrices.
  - Integration into Linear layers.
- [ ] **Experiments**:
  - Finetune on specific domain (Medical/Code).
  - Compare LoRA vs Full Fine-Tuning.

#### Papers
- *LoRA: Low-Rank Adaptation of LLMs* (Hu et al.)

---

### WEEK 10: CONTINUAL LEARNING CORE

**Goal**: Solve Catastrophic Forgetting.

#### Core - 6 Days
- [ ] **Strategies**:
  - Replay Buffers.
  - EWC (Elastic Weight Consolidation).
  - LoRA-per-Task.
- [ ] **Experiments**:
  - Sequential Tasks: Fiction -> Code -> Science.
  - Measure Forward/Backward Transfer.

#### Papers
- *Overcoming Catastrophic Forgetting* (EWC)
- *Continual Learning Review*

---

### WEEK 11: SOTA 2025 TECHNIQUES

**Goal**: MLA, MoE, FlashAttention.

#### Core - 6 Days
- [ ] **DeepSeek Architecture**:
  - MLA (Multi-Head Latent Attention).
  - MoE (Mixture of Experts) basics.
- [ ] **Efficiency**:
  - FlashAttention-2 integration.

#### Papers
- *DeepSeek-V2 / V3 Technical Reports*
- *FlashAttention-2*

---

### WEEK 12: ALIGNMENT (SFT, DPO, GRPO)

**Goal**: Instruct tuning & preference optimization.

#### Core - 6 Days
- [ ] **SFT**: Chat templates, instruction masking.
- [ ] **GRPO / DPO**:
  - Group Relative Policy Optimization (DeepSeek style).
  - Direct Preference Optimization.

#### Papers
- *DeepSeek-R1*
- *DPO*

---

### WEEK 13: INTEGRATION (HYDRA MODEL)

**Goal**: Tools + Adapters + RAG.

#### Core - 6 Days
- [ ] **Hydra Architecture**:
  - Dynamic adapter routing.
  - RAG integration (FAISS).
  - Tool orchestration loop.

---

### WEEK 14: SCALE & DISTRIBUTED

**Goal**: Multi-GPU, 1B params.

#### Core - 5 Days
- [ ] **Distributed Training**:
  - DDP / FSDP.
  - Train 1B param model on FineWeb-EDU.

---

### WEEK 15: NANOCHAT SPEEDRUN

**Goal**: Final assembly and UI.

#### Core - 6 Days
- [ ] **Full Pipeline**: Tokenizer -> Pretrain -> SFT -> UI.
- [ ] **Deployment**: Docker, FastAPI.

---

### WEEK 16+: RESEARCH TRACKS

*(See previous PLAN.md for details on HTM, JEPA, etc.)*

---

## PROGRESS TRACKING

- **Weeks 0-2**: [x] Foundations (Micrograd, Makemore 1-2, Bengio Paper)
- **Week 3**: [ ] Makemore 3-5
- **Week 4**: [ ] Transformer
- ...
