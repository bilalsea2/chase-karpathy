# THE ULTIMATE 100-DAY AI/ML CHALLENGE

**Goal**: Build nanochat from scratch with continual learning capability

**Timeline**: 12 weeks base (compressible to 8, extendable to 16+)

---

## TABLE OF CONTENTS

1. [Design Principles](#design-principles)
2. [Hierarchical Dependency Graph](#hierarchical-dependency-graph)
3. [The Ultimate 12-Week Plan](#the-ultimate-12-week-plan)
4. [Math Refresher](#math-refresher)
5. [Paper Reading Strategy](#paper-reading-strategy)
6. [Tools & Resources](#tools--resources)
7. [Evaluation Framework](#evaluation-framework)
8. [Weekend Reflection Protocol](#weekend-reflection-protocol)
9. [Flexibility Mechanisms](#flexibility-mechanisms)
10. [Final Checklist](#final-checklist)

---

## DESIGN PRINCIPLES

1. **Top-down with safety**: Start with nanochat goal, recurse to fundamentals, but build foundations first (Week 0-1)
2. **Modular weeks**: Each week = **Core** + **Stretch** + **Mega-stretch**
3. **Measure everything**: Metrics dashboard from Week 1 (loss, perplexity, throughput, memory)
4. **Flexible timeline**: Compress to 8 weeks or extend to 16 weeks based on progress
5. **Weekend protocol**: Formalized reflection + pivot decision tree every Sunday
6. **SOTA-first after foundations**: 2025 techniques (MLA, GRPO, FlashAttention-2) by Week 8
7. **Continual learning throughout**: Not just Weeks 9-10, integrated from Week 6
8. **Tool use early**: Week 4, not Week 8 (early integration crucial)
9. **Paper reading tiers**: Critical path (implement) + Context (notes)
10. **Open-ended ceiling**: Week 13+ research tracks (HTM, JEPA, MoE, 50-paper analysis)

---

## HIERARCHICAL DEPENDENCY GRAPH

### Visual Map: Top-Down Decomposition

```
┌──────────────────────────────────────────────────────────────────────┐
│  FINAL BOSS: Continual-Learning NanoChat with Tool Use              │
│  • Never stops learning, never forgets, uses tools for facts        │
│  • Small (560M-1B params), fast, production-ready                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐       ┌────▼────┐      ┌────▼────┐
    │  BRAIN  │       │ TRAINING│      │ADAPTATION│
    │  (Arch) │       │ (Engine)│      │(Learning)│
    └────┬────┘       └────┬────┘      └────┬────┘
         │                 │                 │
┌────────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
│ A1: Transformer │ │ B1: Autograd│ │ C1: PEFT        │
│ • Attention     │ │ • Backprop  │ │ • LoRA          │
│ • MLP           │ │ • Loss fns  │ │ • QLoRA         │
│ • Residual      │ │ • Optimizers│ │ • Adapters      │
│ • LayerNorm     │ │ • AdamW     │ │ • Merging       │
└─────────────────┘ └─────────────┘ └─────────────────┘
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│ A2: Modern Arch │ │B2: Efficiency│ │C2: Continual    │
│ • RoPE          │ │ • AMP        │ │ • Replay        │
│ • GQA/MQA       │ │ • DDP/FSDP   │ │ • EWC           │
│ • SwiGLU        │ │ • FlashAttn  │ │ • LoRA-per-task │
│ • RMSNorm       │ │ • Grad accum │ │ • Nested LoRA   │
│ • KV cache      │ │ • Mixed prec │ │ • Merging       │
└─────────────────┘ └─────────────┘ └─────────────────┘
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│ A3: SOTA 2025   │ │ B3: Scale   │ │ C3: Alignment   │
│ • MLA (DeepSeek)│ │ • Tokenize  │ │ • SFT           │
│ • MoE           │ │ • Data pipe │ │ • DPO/GRPO      │
│ • FlashAttn-2   │ │ • Multi-GPU │ │ • Tool use      │
│ • Hybrid attn   │ │ • ZeRO/FSDP │ │ • RAG           │
└─────────────────┘ └─────────────┘ └─────────────────┘
```

### Critical Path (Dependencies)

- **Week 0**: Math refresher + Autograd (B1) → Foundation
- **Week 1**: Transformer (A1) ← depends on B1
- **Week 2**: Tokenization (B3) → Unblocks data experiments
- **Week 3**: Training loop (B2) + A1 → First real training
- **Week 4**: Modern arch (A2) + Tool use basics (C3) → 2025-ready
- **Week 5**: Evaluation (B3) → Measurement infrastructure
- **Week 6**: LoRA (C1) → PEFT foundation for continual learning
- **Week 7**: Continual learning core (C2) → Your obsession begins
- **Week 8**: SOTA 2025 (A3) → MLA, GRPO, cutting-edge
- **Week 9**: Alignment (C3) → SFT, GRPO, DPO
- **Week 10**: Integration (C2 + C3) → Hydra model with tools
- **Week 11**: Scale (B3) → Multi-GPU, 1B params
- **Week 12**: Nanochat speedrun → Final assembly
- **Week 13+**: Research tracks (HTM, JEPA, MoE, papers)

---

### WEEK 0: (5-7 days)

**Goal**: Foundation - Math + Autograd engine from scratch

#### Core - 4-5 days
- [ ] **Math refresher** (see [Math Refresher](#math-refresher) section):
  - Linear algebra: shapes, matmul, broadcasting, SVD intuition
  - Calculus: chain rule, gradients, backprop
  - Optimization: SGD, Adam, AdamW update rules
  - Probability: softmax, cross-entropy, KL divergence
- [x] **Micrograd**: Build autograd engine (Karpathy video)
  - Value class with \_\_add\_\_, \_\_mul\_\_, etc.
  - Topological sort for backward pass
  - Train tiny MLP on toy dataset
- [x] **Deliverable**: `autograd.py` (100 lines), gradient checker, math cheat sheet

#### Stretch - 2 days
- [ ] Makemore part 1: Bigram character model
- [ ] Makemore part 2: MLP character predictor
- [ ] Read: *Attention Is All You Need* (skim)

#### Mega-stretch - 2 days
- [ ] Implement BatchNorm from scratch
- [ ] Numerical stability debugging toolkit
- [x] Gradient visualization tool

#### Resources
- Karpathy NN:0→Hero Lecture 1 (micrograd)
- 3Blue1Brown: Linear Algebra + Calculus essentials
- Matrix Cookbook (reference)

---

### WEEK 1: MINIMAL TRANSFORMER FROM SCRATCH

**Goal**: Build GPT architecture, understand attention deeply

#### Core - 4 days
- [ ] **Implement from scratch** (no HuggingFace, pure PyTorch):
  - Scaled dot-product attention
  - Multi-head attention (parallel heads)
  - MLP block (Linear → GELU → Linear)
  - Residual connections + LayerNorm
  - Causal mask (triangular)
  - Token + positional embeddings
- [ ] **Train tiny GPT on Shakespeare** (character-level, ~3M params)
- [ ] **Deliverable**: `transformer.py` (250 lines), loss curve plot, samples

#### Stretch - 2 days
- [ ] Add sinusoidal positional encoding
- [ ] Implement weight tying (embedding = lm_head)
- [ ] Gradient clipping + warmup schedule
- [ ] Read: Karpathy "Let's Build GPT" video + nanoGPT repo

#### Mega-stretch - 2 days
- [ ] FlashAttention-1 (use library, understand algorithm)
- [ ] Attention visualization notebook (`attention_viz.ipynb`)
- [ ] Compare multi-head vs single-head attention

#### Metrics
- **Training loss** (target: <1.5 on Shakespeare)
- **Sample quality** (generate 10 samples, check coherence)
- **Throughput**: tokens/sec (CPU vs GPU)
- **Memory**: peak GPU usage

#### Papers (Critical)
- *Attention Is All You Need* (Vaswani et al.) - Section 3 (architecture)

#### Tools Introduced
- PyTorch, einsum, torch.nn.functional, matplotlib

---

### WEEK 2: TOKENIZATION & DATA PIPELINE

**Goal**: Build BPE tokenizer, create data shards, unblock experiments

#### Core - 3 days
- [ ] **Build BPE tokenizer** (Karpathy minbpe or HF tokenizers):
  - Byte-level BPE training
  - Encode/decode roundtrip test
  - Special tokens: <|endoftext|>, <|pad|>
- [ ] **Download & process data**:
  - TinyStories (1GB) or OpenWebText-10BT
  - Create memory-mapped shards
  - Test data loading speed
- [ ] **Deliverable**: `tokenizer.py`, `data.py`, 10 shards, loading benchmark

#### Stretch - 2 days
- [ ] Sequence packing (constant-length batches)
- [ ] Data quality checks (dedup, language filter)
- [ ] Compression ratio analysis (bytes/token)
- [ ] Read: *SentencePiece* paper

#### Mega-stretch - 2 days
- [ ] Compare BPE vs WordPiece vs Unigram
- [ ] Implement tiktoken-style regex splitting
- [ ] Multi-language tokenizer experiment

#### Metrics
- **Vocab size** (target: 8k-32k)
- **Compression ratio** (bytes/token on English text)
- **Loading speed** (samples/sec)
- **Tokenization speed** (tokens/sec)

#### Tools Introduced
- HuggingFace datasets, tokenizers, regex, numpy.memmap

---

### WEEK 3: TRAINING LOOP & EFFICIENCY

**Goal**: Full training pipeline with optimization, logging, checkpointing

#### Core - 4 days
- [ ] **Implement training loop**:
  - DataLoader with shuffle + batching
  - AdamW optimizer (torch.optim.AdamW)
  - LR schedule: linear warmup (500 steps) + cosine decay
  - Mixed precision (torch.amp)
  - Gradient accumulation (simulate large batch)
  - Checkpointing (save/load model + optimizer state)
  - TensorBoard logging
- [ ] **Train 50M param GPT on TinyStories** (5-10k steps)
- [ ] **Deliverable**: `train.py`, checkpoint, TensorBoard logs

#### Stretch - 2 days
- [ ] Profile GPU utilization (nvidia-smi, torch.profiler)
- [ ] Optimize batch size + sequence length
- [ ] Gradient clipping (norm-based, not value-based)
- [ ] Read: *HuggingFace Smol Training Playbook*

#### Mega-stretch - 2 days
- [ ] DDP (DistributedDataParallel) on single node, 2+ GPUs
- [ ] Benchmark FP16 vs BF16 vs FP32 (if Ampere+ GPU)
- [ ] Gradient checkpointing (activation recomputation)

#### Metrics
- **Training loss curve** (smooth descent, no spikes)
- **Throughput**: tokens/sec (target: >10k on single GPU)
- **GPU memory usage** (target: <80% to avoid OOM)
- **Validation perplexity** (target: <30 on TinyStories)
- **Wall-clock time** per 1000 steps

#### Tools Introduced
- torch.amp, torch.distributed, TensorBoard, nvidia-smi, torch.profiler

---

### WEEK 4: MODERN ARCHITECTURE + TOOL USE BASICS

**Goal**: Upgrade to 2025 architecture (RoPE, GQA, SwiGLU), add simple tool use

#### Core - 5 days
- [ ] **Upgrade transformer** (convert GPT-2 → Llama-3 style):
  - RoPE positional embeddings (replace learned/sinusoidal)
  - RMSNorm (replace LayerNorm)
  - SwiGLU activation (replace GELU in MLP)
  - Remove biases (Llama-style: no bias in Linear layers)
- [ ] **Simple tool use**:
  - Train model to output `<CALC>expression</CALC>` special tokens
  - Build calculator interpreter (safe eval sandbox)
  - Test on 10 math problems
- [ ] **Deliverable**: `modern_gpt.py`, tool-calling demo script

#### Stretch - 2 days
- [ ] Implement GQA (Grouped Query Attention)
  - Group K,V heads (e.g., 8 Q heads → 2 KV heads)
- [ ] Add KV cache for inference (autoregressive caching)
- [ ] Read: *RoFormer* (RoPE), *Llama 2* paper

#### Mega-stretch - 2 days
- [ ] FlashAttention-2 integration (if not done in Week 1)
- [ ] Benchmark attention memory (standard vs GQA vs FlashAttn)
- [ ] Hybrid attention (local + global, Longformer-style)

#### Metrics
- **Model quality**: Generate 100 samples, check coherence (1-5 scale)
- **Tool use accuracy**: 10 math problems, % correct final answers
- **Inference speed**: tokens/sec with KV cache vs without
- **Memory**: KV cache size (bytes) for 2048 context

#### Papers (Critical)
- *RoFormer* (Su et al.) - RoPE equations
- *Llama 2* (Touvron et al.) - modern architecture choices
- *GQA* (Ainslie et al.) - grouped query attention

#### Tools Introduced
- ReAct pattern (Reason + Act), safe eval sandbox, function calling basics

**CRITICAL**: Tool use starts here (Week 4), not Week 8. Early integration.

---

### WEEK 5: EVALUATION FRAMEWORK + BASELINE MODEL

**Goal**: Build comprehensive eval suite, establish baseline performance

#### Core - 4 days
- [ ] **Build evaluation suite** (`eval.py`):
  - Perplexity on multiple datasets (C4, WikiText-2, domain-specific)
  - Simple reasoning: ARC-easy subset (25 questions)
  - Simple math: GSM8K-tiny (10 problems)
  - Toxicity check (basic keyword filter)
- [ ] **Train baseline model**:
  - 124M params (GPT-2 small size) on 10B tokens
  - Full training run (not just 5k steps)
- [ ] **Deliverable**: `eval.py`, evaluation dashboard (notebook), baseline checkpoint

#### Stretch - 2 days
- [ ] Add MMLU subset (5 categories: STEM, humanities, etc.)
- [ ] HumanEval subset (code generation, 10 problems)
- [ ] Read: *Scaling Laws* (Kaplan et al.)

#### Mega-stretch - 2 days
- [ ] Contamination detection (train/test data overlap check)
- [ ] Build leaderboard comparison (your model vs GPT-2, TinyLlama)
- [ ] Automated eval pipeline (GitHub Actions or cron job)

#### Metrics
- **Perplexity**:
  - C4: target <20
  - WikiText-2: target <25
- **Reasoning**:
  - ARC-easy: target >30% (for 124M model)
  - GSM8K: target >5%
- **Code**:
  - HumanEval: target >5%
- **Toxicity**: <2% toxic samples (on 1000 random generations)

#### Papers (Critical)
- *Scaling Laws for Neural Language Models* (Kaplan et al.)

#### Tools Introduced
- lm-evaluation-harness (EleutherAI), HumanEval dataset, toxicity classifiers

---

### WEEK 6: PEFT MASTERY (LoRA, QLoRA, Foundation for Continual Learning)

**Goal**: Master parameter-efficient fine-tuning, foundation for continual learning

#### Core - 5 days
- [ ] **Implement LoRA from scratch** (no PEFT library yet):
  - Low-rank matrices A (d×r), B (r×d) where r<<d
  - Apply to ALL linear layers (Attention Q,K,V,O + MLP up,down)
  - Merging: W' = W + s·A·B (scaling factor s)
  - Unmerging (restore original weights)
- [ ] **LoRA finetune experiment**:
  - Pick domain (medical, legal, or coding)
  - Finetune with LoRA (r=8) vs full fine-tuning
  - Compare loss curves, memory, speed
- [ ] **Deliverable**: `lora.py`, domain adapter checkpoint, LoRA vs FT comparison report

#### Stretch - 2 days
- [ ] QLoRA (4-bit quantization + LoRA):
  - Use bitsandbytes for 4-bit quantization
  - Train QLoRA on larger model (560M)
- [ ] Experiment with ranks (r=4, 8, 16, 32, 64)
- [ ] Read: *LoRA* paper + *Thinking Machines "LoRA Without Regret"* blog

#### Mega-stretch - 2 days
- [ ] Implement DoRA (Directional LoRA variant)
- [ ] LoRA on Attention only vs All layers (ablation study)
- [ ] LoRA merging strategies (TIES, DARE)
- [ ] Read: *QLoRA* paper (Dettmers et al.)

#### Metrics
- **LoRA vs Full FT**:
  - Final loss (should be comparable)
  - Memory usage (LoRA should be ~10-20% of full FT)
  - Training speed (LoRA should be faster)
- **Forgetting**: Perplexity on original domain (measure domain shift)
- **Rank analysis**: How does r affect performance?

#### Papers (Critical)
- *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al.)
- *Thinking Machines "LoRA Without Regret"* blog (practical hyperparameters)
- *QLoRA* (Dettmers et al.)

#### Tools Introduced
- bitsandbytes (quantization), HuggingFace PEFT library (compare to yours)

**CRITICAL**: LoRA is the foundation for continual learning. Master it deeply this week.

---

### WEEK 7: CONTINUAL LEARNING CORE (Your Obsession Begins)

**Goal**: Implement & compare 3 continual learning strategies, measure forgetting

#### Core - 6 days
- [ ] **Implement 3 strategies**:
  1. **Rehearsal buffer**: Store 1k samples from each task, replay during new task training
  2. **EWC (Elastic Weight Consolidation)**: Compute Fisher diagonal, add quadratic penalty
  3. **LoRA-per-task**: Freeze base model, train separate LoRA adapter per domain
- [ ] **Sequential learning experiment**:
  - Task 1: TinyStories (fiction)
  - Task 2: Python code (GitHub Code dataset)
  - Task 3: Scientific text (arXiv abstracts)
- [ ] **Measure forgetting**:
  - Perplexity on Task 1 after training Task 2, 3
  - Forward transfer: Does Task 1 help Task 2?
  - Backward transfer: Does Task 2 hurt Task 1?
- [ ] **Deliverable**: `continual.py`, forgetting curves (plot), 3-task experiment report

#### Stretch - 2 days
- [ ] Nested LoRA (stack adapters hierarchically)
- [ ] LoRA merging strategies (weighted average, TIES, DARE)
- [ ] Read: *EWC* paper, *Continual Learning review* (Parisi et al.)

#### Mega-stretch - 2 days
- [ ] Generative replay (train small model to generate Task 1 data)
- [ ] PackNet (prune & pack weights per task)
- [ ] Progressive Neural Networks (lateral connections)
- [ ] Read: *Thinking Machines nested learning* blog

#### Metrics (Critical for continual learning)
- **Forgetting**: Δ perplexity on Task 1 after Task 2, 3
  - Target: <20% increase (good continual learning)
- **Forward transfer**: Task 2 performance with vs without Task 1 pretraining
- **Backward transfer**: Task 1 performance after Task 3 vs baseline
- **Memory overhead**: bytes per task (adapters, replay buffer)
- **Training time**: wall-clock per task

#### Papers (Critical)
- *Overcoming catastrophic forgetting in neural networks* (Kirkpatrick et al.) - EWC
- *Beyond Supervised Continual Learning: a Review* (Parisi et al.)
- *Thinking Machines nested learning* blog

**CRITICAL**: This is your obsession. Spend extra time here. Understand forgetting deeply.

---

### WEEK 8: SOTA 2025 TECHNIQUES (MLA, FlashAttention-2, MoE)

**Goal**: Implement cutting-edge 2025 techniques, push efficiency frontier

#### Core - 5 days
- [ ] **Implement MLA (Multi-Head Latent Attention)** - DeepSeek style:
  - Compress KV heads to latent vectors (low-rank projection)
  - Up-project in attention computation
  - Benchmark memory savings vs standard attention
- [ ] **Integrate FlashAttention-2** properly (if not done earlier):
  - Understand algorithm (tiled computation)
  - Use library (flash-attn or torch.nn.functional.scaled_dot_product_attention)
  - Benchmark speed
- [ ] **Deliverable**: `deepseek_attn.py`, memory benchmark report (MLA vs standard vs GQA)

#### Stretch - 2 days
- [ ] Implement MoE (Mixture of Experts) layer:
  - 4-8 experts (MLP variants)
  - Top-k routing (k=2)
  - Load balancing loss
- [ ] Read: *DeepSeek-V2* paper (MLA), *DeepSeek-V3* (MoE)

#### Mega-stretch - 2 days
- [ ] Train small MoE model (4 experts, 124M params total)
- [ ] Compare dense vs MoE at same compute budget
- [ ] Expert specialization analysis (which expert handles what?)
- [ ] Read: *Switch Transformers* (Google MoE)

#### Metrics
- **Memory**:
  - KV cache size (bytes) for 2048 context:
    - Standard attention: baseline
    - GQA: ~50% reduction
    - MLA: ~70% reduction
- **Speed**:
  - Attention computation time (ms):
    - Standard: baseline
    - FlashAttention-2: ~2-3x faster
- **MoE metrics** (if implemented):
  - Expert utilization (should be balanced, not 90% to one expert)
  - Total params vs active params per forward pass

#### Papers (Critical)
- *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* - MLA architecture
- *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (Dao)
- *Switch Transformers* (Fedus et al.) - MoE at scale

#### Tools Introduced
- DeepSeek codebase (reference), flash-attn library, MoE routing implementations

---

### WEEK 9: ALIGNMENT & GRPO (SFT, DPO, GRPO)

**Goal**: Turn base model into aligned chat model using SFT + GRPO

#### Core - 5 days
- [ ] **Supervised Fine-Tuning (SFT)**:
  - Chat template format: `<|user|>...<|assistant|>...`
  - Loss masking (don't train on user prompts, only assistant responses)
  - Finetune on SmolTalk or Alpaca dataset (10k samples)
- [ ] **Implement GRPO (Group Relative Policy Optimization)**:
  - Group sampling: generate K=4 responses per prompt
  - Relative ranking: compare within group (no reward model)
  - Loss: maximize log-prob of best response, minimize worst
- [ ] **Deliverable**: `sft.py`, `grpo.py`, chat model checkpoint

#### Stretch - 2 days
- [ ] DPO (Direct Preference Optimization) implementation
- [ ] Synthetic preference generation (use GPT-4 to rank responses)
- [ ] Read: *DeepSeek-R1* (GRPO), *DPO* paper

#### Mega-stretch - 2 days
- [ ] Online DPO (continual preference learning)
- [ ] Multi-objective GRPO (helpfulness + harmlessness + factuality)
- [ ] Constitutional AI principles (Anthropic-style)

#### Metrics
- **Chat quality** (human eval on 10 prompts, 1-5 scale):
  - Helpfulness
  - Harmlessness
  - Coherence
- **Win rate**: Pairwise comparison vs baseline (target: >60%)
- **Alignment tax**: Perplexity shift after GRPO (should be small, <10%)
- **Throughput**: Still fast inference? (should not degrade)

#### Papers (Critical)
- *InstructGPT: Training language models to follow instructions* (Ouyang et al.) - SFT/RLHF
- *Direct Preference Optimization* (Rafailov et al.) - DPO
- *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs* - GRPO details

---

### WEEK 10: CONTINUAL + TOOLS INTEGRATION (Hydra Model)

**Goal**: Build multi-adapter manager + RAG + tool orchestration

#### Core - 6 days
- [ ] **Build "Hydra" model**:
  - Frozen base model (your 124M-560M from Week 5/8)
  - Library of LoRA adapters: math, code, chat, search, medical
  - Dynamic adapter loading based on prompt classification
  - Adapter router (simple classifier or embedding similarity)
- [ ] **Integrate RAG (Retrieval-Augmented Generation)**:
  - FAISS vector store (dense embeddings)
  - Embed documents (Wikipedia, custom knowledge base)
  - Retrieve top-k relevant docs, prepend to context
- [ ] **Tool orchestration**:
  - Calculator (Python eval in sandbox)
  - Search (Wikipedia API or Google search stub)
  - Python REPL (code execution in Docker)
- [ ] **Deliverable**: `hydra.py`, multi-tool demo notebook

#### Stretch - 2 days
- [ ] Continual tool learning (add new tools without retraining base)
- [ ] Adapter conflict detection & resolution (when 2+ adapters activated)
- [ ] Read: *ReAct* paper, *Toolformer*

#### Mega-stretch - 2 days
- [ ] Hierarchical tool planning (multi-step tool use, chain-of-thought)
- [ ] Memory-augmented adapter (episodic buffer per adapter)
- [ ] Tool use reflection (model critiques its own tool calls)

#### Metrics
- **Tool use accuracy**: 20 tasks requiring tools (math, search, code)
  - Target: >80% correct final answer
- **Adapter switching latency**: ms to load/unload adapter
  - Target: <50ms
- **Forgetting after adding new adapter**: perplexity shift on old tasks
  - Target: <5%
- **RAG quality**: Does retrieval help? (perplexity with vs without RAG)

#### Papers (Critical)
- *ReAct: Synergizing Reasoning and Acting in Language Models* (Yao et al.)
- *Toolformer: Language Models Can Teach Themselves to Use Tools* (Schick et al.)

#### Tools Introduced
- FAISS (vector search), LangChain (compare to yours), function calling APIs, Docker (sandboxing)

---

### WEEK 11: SCALING & DISTRIBUTED TRAINING

**Goal**: Scale to 560M-1B params, multi-GPU training, 50-100B tokens

#### Core - 5 days
- [ ] **Scale model architecture**:
  - 560M-1B parameters (Llama-like config)
  - Example: n_layers=24, n_heads=16, d_model=1024
- [ ] **Multi-GPU training**:
  - DDP (DistributedDataParallel) on 2-4 GPUs
  - FSDP (Fully Sharded Data Parallel) for memory efficiency
  - ZeRO-1 or ZeRO-2 (DeepSpeed)
- [ ] **Pretrain on large dataset**:
  - FineWeb-EDU (50-100B tokens, ~50-100GB)
  - Train for 1-2 days on rented GPU cluster (4-8x A100)
- [ ] **Deliverable**: 1B model checkpoint, training logs, distributed training guide

#### Stretch - 2 days
- [ ] Pipeline parallelism (simple 2-stage: layers 0-11 on GPU0, 12-23 on GPU1)
- [ ] Gradient checkpointing (activation recomputation to save memory)
- [ ] Read: *Megatron-LM* paper, FSDP documentation

#### Mega-stretch - 2 days
- [ ] 3D parallelism (data + pipeline + tensor parallelism)
- [ ] Mixed expert-data parallelism (for MoE models)
- [ ] Benchmark scaling efficiency (strong scaling vs weak scaling)

#### Metrics
- **Scaling efficiency**: tokens/sec/GPU as you add GPUs
  - Ideal: linear scaling (2 GPUs = 2x speed)
  - Reality: ~80-90% efficiency is good
- **Memory breakdown** (GB per GPU):
  - Model weights
  - Optimizer state (AdamW = 2x model size)
  - Activations
  - KV cache (during inference)
- **Final perplexity** on FineWeb-EDU:
  - Target: <12 for 1B model trained on 100B tokens

#### Papers (Critical)
- *Megatron-LM: Training Multi-Billion Parameter Language Models* (Shoeybi et al.)
- *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (Rajbhandari et al.)
- *PyTorch FSDP* documentation

#### Tools Introduced
- DeepSpeed, FSDP (torch.distributed.fsdp), Megatron-LM, GPU cluster (RunPod, Lambda, Vast.ai)

**CRITICAL**: Budget $300-500 for GPU rental (4-8x A100 for 1-2 days). Plan compute carefully.

---

### WEEK 12: NANOCHAT INTEGRATION (Final Assembly)

**Goal**: Full nanochat speedrun - tokenizer → pretrain → SFT → eval → UI

#### Core - 6 days
- [ ] **Full nanochat speedrun**:
  - Use your 124M-560M-1B model from Week 11
  - SFT on chat dataset (if not done in Week 9)
  - Build web interface (FastAPI backend + simple UI)
  - Streaming inference with KV cache
  - Deploy locally (Docker container)
- [ ] **Continual learning API**:
  - POST /learn endpoint (accepts new data, trains LoRA adapter)
  - GET /adapters endpoint (list available adapters)
  - POST /switch_adapter (change active adapter)
- [ ] **Deliverable**: Working chat UI, Docker image, deployment guide, API docs

#### Stretch - 2 days
- [ ] Add continual learning dashboard (visualize forgetting curves, adapter library)
- [ ] Multi-user adapter isolation (user-specific adapters)
- [ ] Read: *Karpathy nanochat* repo thoroughly (study speedrun.sh)

#### Mega-stretch - 2 days
- [ ] Production monitoring (latency, throughput, error rates)
- [ ] A/B testing framework (compare adapter versions)
- [ ] Automated adapter merging pipeline (nightly job)

#### Metrics (End-to-end)
- **Latency**:
  - Time to first token: target <500ms
  - Full response (100 tokens): target <5 seconds
- **Throughput**: concurrent users (target: 5-10 users on single GPU)
- **Chat quality**: vs GPT-3.5 on 50 prompts (human eval, pairwise comparison)
  - Target: >40% win rate against GPT-3.5 (realistic for 1B model)
- **Continual learning**: Add 5 new facts via /learn API, verify retention next day

#### Tools Introduced
- FastAPI, Gradio (or Streamlit), Docker, nginx (reverse proxy), monitoring (Prometheus, Grafana)

---

### WEEK 13+: RESEARCH TRACKS (Open-Ended)

Choose one or more tracks based on interest:

#### Track A: Hierarchical Temporal Memory (HTM)
- [ ] Study Jeff Hawkins' *Thousand Brains Theory*
- [ ] Implement simple HTM layer (spatial pooler + temporal memory)
- [ ] Compare to transformer on sequence prediction task
- [ ] Hybrid architecture: HTM + Transformer
- [ ] **Deliverable**: HTM experiment report, code

#### Track B: JEPA (Joint Embedding Predictive Architecture)
- [ ] Study Yann LeCun's JEPA vision (I-JEPA, V-JEPA)
- [ ] Implement I-JEPA on simple task (masked image modeling)
- [ ] Hybrid: Transformer + JEPA objective (predict embeddings, not tokens)
- [ ] **Deliverable**: JEPA prototype, comparison report

#### Track C: Advanced Continual Learning
- [ ] Progressive Neural Networks (lateral connections between columns)
- [ ] Meta-learning for fast adaptation (MAML, Reptile)
- [ ] Lifelong learning benchmarks (CLOC, CTrL)
- [ ] **Deliverable**: Benchmark results, potential leaderboard submission

#### Track D: MoE Mastery
- [ ] Scale to 16-32 experts (billions of params, but sparse)
- [ ] Expert specialization analysis (what does each expert learn?)
- [ ] Soft vs hard routing (compare top-k, softmax routing, learned routing)
- [ ] **Deliverable**: MoE training guide, scaling analysis

#### Track E: 50 Papers Deep Dive
- [ ] Structured reading protocol:
  - **Tier 1 (Critical)**: 15 papers, full implementation of core idea
  - **Tier 2 (Important)**: 20 papers, key ideas coded + notes
  - **Tier 3 (Context)**: 15 papers, notes only
- [ ] **Deliverable**: Annotated bibliography, code snippets repo, blog post series

---

## MATH REFRESHER

**Time Budget: 5-7 days (Week 0)**

### Linear Algebra (The Shape Game)

**Must-know operations:**
- **Vectors & matrices**: add, multiply, transpose, reshape
- **Dot product**: a·b = Σᵢ aᵢbᵢ = |a||b|cos(θ) (similarity measure, core of attention)
- **Matrix multiplication**: (M,K) @ (K,N) → (M,N)
  - **Rule**: Inner dimensions must match
  - **Practice**: Multiply (3,5) @ (5,7) by hand to verify (3,7)
- **Broadcasting**: PyTorch magic for different shapes
  - (B,T,C) + (C) → broadcasts (C) across batch & time
  - **Critical**: 90% of bugs are shape errors
- **Norms**: L1 (sum |x|), L2 (√Σx²), Frobenius (matrix L2)
- **SVD (intuition)**: A = UΣVᵀ (decompose into rotations + scaling)
  - **Why**: LoRA uses low-rank approximation (keep top-r singular values)
- **PyTorch shapes**: (B, T, C) notation
  - B = batch size
  - T = sequence length (time)
  - C = channels (embedding dimension)
  - **Einsum**: `torch.einsum('btc,cd->btd', x, W)` (powerful for custom ops)

**Practice:**
- [ ] Implement batched matmul, check all shapes
- [ ] Reproduce attention scores: Q @ K.T → (B, T_q, T_k)
- [ ] Debug 5 intentional shape mismatches (add breakpoints)
- [ ] Visualize SVD (2D matrix → rotate + scale + rotate)

**Why:** 90% of bugs are shape errors. Master this, save 100 hours debugging.

---

### Calculus (The Gradient Flow)

**Must-know concepts:**
- **Partial derivatives**: ∂f/∂x (rate of change wrt one variable, others fixed)
- **Gradient**: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ] (direction of steepest ascent)
- **Chain rule** (scalar): df/dx = (df/dy)(dy/dx)
- **Chain rule** (vector): Jacobian matrices multiply
- **Taylor expansion** (intuition): f(x+ε) ≈ f(x) + ε∇f(x) (local linear approximation)

**Practice:**
- [ ] Derive softmax gradient by hand (once, painful but enlightening)
- [ ] Derive cross-entropy gradient by hand
- [ ] Implement numerical gradient check (finite differences)
- [ ] Build micrograd Value class (40 lines)

**Why:** Backprop is just chain rule. If you can't derive a gradient, you can't debug training.

---

### Optimization (The Update Rules)

**Must-know algorithms:**

**SGD**:
```
θ ← θ - lr * ∇L
```

**SGD + Momentum**:
```
v ← β*v + ∇L
θ ← θ - lr * v
```
- Smooths gradients, accelerates in consistent directions

**AdamW** (default for transformers):
```
m ← β1*m + (1-β1)*∇L          # first moment (mean)
v ← β2*v + (1-β2)*(∇L)²        # second moment (variance)
m_hat ← m / (1 - β1^t)          # bias correction
v_hat ← v / (1 - β2^t)
θ ← θ - lr * (m_hat / (√v_hat + ε) + wd*θ)  # update + weight decay
```
- **Hyperparams**: lr=3e-4, β1=0.9, β2=0.95, ε=1e-8, wd=0.1 (typical)
- **Why AdamW not Adam**: Decoupled weight decay (better generalization)

**Learning rate schedules**:
- **Warmup**: Linear increase from 0 to max_lr (first 500-2000 steps)
  - Prevents early instability
- **Cosine decay**: lr = min_lr + 0.5*(max_lr - min_lr)*(1 + cos(π*t/T))
  - Smooth decrease to ~10% of max_lr

**Gradient clipping**:
```
if ||∇L|| > max_norm:
    ∇L ← ∇L * (max_norm / ||∇L||)
```
- Typical max_norm: 1.0 (prevents gradient explosion)

**Practice:**
- [ ] Implement AdamW from scratch (30 lines)
- [ ] Plot LR schedule: 500-step warmup + cosine decay over 10k steps
- [ ] Compare SGD vs Adam on tiny MLP (MNIST)

**Why:** You'll tune hyperparameters 1000 times. Know what each param does.

---

### Probability & Information Theory

**Must-know concepts:**

**Expectation, Variance**:
```
E[X] = Σ x * P(x)
Var[X] = E[(X - E[X])²] = E[X²] - E[X]²
```

**Softmax** (convert logits → probabilities):
```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```
- **Temperature scaling**: softmax(z/T)
  - T<1: sharpens (more deterministic)
  - T>1: flattens (more random/creative)

**Cross-Entropy Loss** (classification):
```
L = -log p_model(y_true)
  = -z[y] + log(Σⱼ exp(zⱼ))
```
- **Interpretation**: Negative log-likelihood (maximize probability of correct class)

**KL Divergence** (distribution distance):
```
KL(p || q) = Σ p(x) log(p(x)/q(x))
```
- **Not symmetric**: KL(p||q) ≠ KL(q||p)
- **Used in**: DPO, RLHF (policy regularization)

**Practice:**
- [ ] Implement softmax + cross-entropy in NumPy (numerical stability: subtract max before exp)
- [ ] Visualize temperature effect (T=0.1, 1.0, 2.0) on distribution
- [ ] Derive why cross-entropy = -log p(correct class)

**Why:** Loss functions are your training signal. Understand them deeply.

---

### Practical Cheat Sheet (Memorize & Print)

```python
# SHAPES (Attention)
Q: (B, T, d_k)
K: (B, T, d_k)
V: (B, T, d_v)
Scores: Q @ K.T → (B, T, T)
Attention: softmax(Scores) @ V → (B, T, d_v)

# SOFTMAX
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)

# CROSS-ENTROPY (logits z, label y)
L = -z[y] + log(Σⱼ exp(zⱼ))

# ADAMW (simplified)
m ← β1*m + (1-β1)*g
v ← β2*v + (1-β2)*g²
θ ← θ - lr*(m/√v + wd*θ)

# LORA
W ≈ W₀ + ΔW
ΔW ≈ A @ B  (A: d×r, B: r×k, r<<min(d,k))

# GRADIENT CLIPPING
if ||∇L|| > max_norm:
    ∇L ← ∇L * (max_norm / ||∇L||)

# LEARNING RATE SCHEDULE
Warmup: lr = max_lr * (step / warmup_steps)
Cosine: lr = min_lr + 0.5*(max_lr - min_lr)*(1 + cos(π*step/total_steps))
```

---

### Resources

**Visual intuition:**
- 3Blue1Brown: *Essence of Linear Algebra* (series, watch chapters 1, 3, 4, 7, 10)
- 3Blue1Brown: *Essence of Calculus* (series, watch chapters 2-4 on derivatives)

**Hands-on:**
- Karpathy: *Neural Networks: Zero to Hero* - Lecture 1 (micrograd)
- Implement micrograd Value class from scratch

**Reference:**
- *The Matrix Cookbook* (Petersen & Pedersen) - grad derivatives cheat sheet
- *Deep Learning* book (Goodfellow et al.) - Chapters 2-4 (math foundations)

---

### Deliverable (End of Week 0)

**Math debugging kit:**
- [ ] `gradient_checker.py` (numerical vs analytical gradients)
- [ ] `shape_validator.py` (assert correct shapes, helpful error messages)
- [ ] `adamw.py` (AdamW implementation from scratch)
- [ ] `math_cheat_sheet.pdf` (1-page printable reference)

---

## PAPER READING STRATEGY

### Tier 1: CRITICAL PATH (Must implement core idea)

**Weeks 1-4: Foundations**
1. *Attention Is All You Need* (Vaswani et al., 2017) - Week 1
   - **Focus**: Section 3 (model architecture), scaled dot-product attention, multi-head attention
   - **Implement**: Full transformer block

2. *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021) - Week 4
   - **Focus**: Section 2-3 (RoPE formulation, complex number representation)
   - **Implement**: RoPE in attention (replace absolute positional encoding)

3. *GQA: Training Generalized Multi-Query Transformer Models* (Ainslie et al., 2023) - Week 4
   - **Focus**: Sections 1-3 (grouped KV heads)
   - **Implement**: Modify attention to group K,V heads

**Weeks 5-6: Efficiency & PEFT**
4. *Scaling Laws for Neural Language Models* (Kaplan et al., 2020) - Week 5
   - **Focus**: Sections 1-2, 6 (power laws, optimal model size vs data)
   - **Understand**: Loss scales as compute^(-α)

5. *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021) - Week 6
   - **Focus**: Sections 1-4 (low-rank math, where to apply, hyperparameters)
   - **Implement**: LoRA on all linear layers, merging/unmerging

6. *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023) - Week 6
   - **Focus**: 4-bit quantization + LoRA integration
   - **Implement**: Quantize model, train LoRA adapters

**Week 7: Continual Learning**
7. *Overcoming catastrophic forgetting in neural networks* (Kirkpatrick et al., 2016) - EWC - Week 7
   - **Focus**: Fisher information matrix, quadratic penalty
   - **Implement**: EWC regularization term

8. *Thinking Machines "LoRA Without Regret"* blog - Week 6-7
   - **Focus**: Practical LoRA hyperparameters (rank, LR, layers)
   - **Apply**: Use their recommendations in your experiments

**Weeks 8-9: SOTA 2025 & Alignment**
9. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* - Week 8
   - **Focus**: MLA (Multi-Head Latent Attention) architecture
   - **Implement**: Latent KV compression

10. *FlashAttention-2: Faster Attention with Better Parallelism* (Dao, 2023) - Week 8
    - **Focus**: Algorithm 1 (tiled attention), IO-aware optimization
    - **Use**: Library implementation, understand why it's fast

11. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (Rafailov et al., 2023) - Week 9
    - **Focus**: DPO loss derivation, no separate reward model
    - **Implement**: DPO training loop

12. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* - Week 9
    - **Focus**: GRPO (Group Relative Policy Optimization), group sampling
    - **Implement**: GRPO algorithm

**Weeks 10-11: Tools & Scale**
13. *ReAct: Synergizing Reasoning and Acting in Language Models* (Yao et al., 2022) - Week 10
    - **Focus**: Thought-Action-Observation loop
    - **Implement**: ReAct pattern with calculator/search tools

14. *Megatron-LM: Training Multi-Billion Parameter Language Models* (Shoeybi et al., 2019) - Week 11
    - **Focus**: Model parallelism, pipeline parallelism
    - **Use**: Reference for distributed training

15. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (Rajbhandari et al., 2020) - Week 11
    - **Focus**: ZeRO-1, ZeRO-2 (optimizer state sharding, gradient sharding)
    - **Use**: DeepSpeed library

---

### Tier 2: IMPORTANT (Read + notes, selective implementation)

16. *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018) - Context
17. *Language Models are Unsupervised Multitask Learners* (GPT-2, Radford et al., 2019) - Context
18. *Language Models are Few-Shot Learners* (GPT-3, Brown et al., 2020) - Scaling insights
19. *Switch Transformers: Scaling to Trillion Parameter Models* (Fedus et al., 2021) - MoE at scale
20. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (ALiBi, Press et al., 2022)
21. *GLU Variants Improve Transformer* (Shazeer, 2020) - SwiGLU activation
22. *Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019) - RMSNorm
23. *Training language models to follow instructions with human feedback* (InstructGPT, Ouyang et al., 2022)
24. *Toolformer: Language Models Can Teach Themselves to Use Tools* (Schick et al., 2023)
25. *Progressive Neural Networks* (Rusu et al., 2016) - Continual learning architecture
26. *PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning* (Mallya & Lazebnik, 2018)
27. *Training Compute-Optimal Large Language Models* (Chinchilla, Hoffmann et al., 2022) - Updated scaling laws
28. *LLaMA: Open and Efficient Foundation Language Models* (Touvron et al., 2023)
29. *Llama 2: Open Foundation and Fine-Tuned Chat Models* (Touvron et al., 2023)
30. *Mixtral of Experts* (Jiang et al., 2024) - Mistral MoE

---

### Tier 3: CONTEXT (Skim, historical understanding)

31. *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet, Krizhevsky et al., 2012)
32. *Deep Residual Learning for Image Recognition* (ResNet, He et al., 2015)
33. *Batch Normalization: Accelerating Deep Network Training* (Ioffe & Szegedy, 2015)
34. *Generative Adversarial Networks* (Goodfellow et al., 2014)
35. *Auto-Encoding Variational Bayes* (VAE, Kingma & Welling, 2013)
36. *Efficient Estimation of Word Representations in Vector Space* (Word2Vec, Mikolov et al., 2013)
37. *Sequence to Sequence Learning with Neural Networks* (Sutskever et al., 2014)
38. *Neural Machine Translation by Jointly Learning to Align and Translate* (Bahdanau attention, 2014)
39. *Universal Language Model Fine-tuning for Text Classification* (ULMFiT, Howard & Ruder, 2018)
40. *Deep contextualized word representations* (ELMo, Peters et al., 2018)

---

### Reading Protocol

**Tier 1 (Critical):**
1. Read paper (2-3 hours)
2. Implement core idea in your codebase (1 day)
3. Write 2-page summary:
   - Problem & motivation (2 paragraphs)
   - Key idea (3-5 bullet points)
   - Implementation details (equations, pseudocode)
   - Results & takeaways (1 paragraph)
   - Code snippet (10-20 lines)

**Tier 2 (Important):**
1. Read paper (1-2 hours)
2. Code key function (2-3 hours)
3. Write 1-page notes:
   - Key idea (2-3 sentences)
   - Equations (if any)
   - Code snippet (5-10 lines)
   - When to use (1 sentence)

**Tier 3 (Context):**
1. Skim abstract, intro, conclusion (15-30 min)
2. Bullet points (5-7 bullets):
   - What problem did it solve?
   - Key innovation
   - Historical context (why it mattered)

---

## TOOLS & RESOURCES

### Core Stack

**Deep Learning Framework:**
- PyTorch 2.0+ (torch.compile for speed, FSDP for distributed)
- CUDA 11.8+ or ROCm (for AMD GPUs)

**HuggingFace Ecosystem:**
- `transformers` (reference implementations, don't copy but study)
- `datasets` (efficient data loading, streaming)
- `tokenizers` (fast BPE, compare to your implementation)
- `peft` (LoRA/QLoRA, compare to your implementation)
- `accelerate` (multi-GPU abstraction)

**Efficiency & Quantization:**
- `bitsandbytes` (4-bit/8-bit quantization for QLoRA)
- `flash-attn` (FlashAttention-2 implementation)
- DeepSpeed (ZeRO optimization, distributed training)

**Retrieval & Tools:**
- `faiss` (vector search for RAG)
- `sentence-transformers` (embedding models for RAG)

**Experiment Tracking:**
- Weights & Biases (wandb) - cloud-based, great visualizations
- TensorBoard - local, simple logging

**Development:**
- Docker (reproducible environments, deployment)
- Git + GitHub (version control, portfolio)
- VSCode + Jupyter (coding + exploration)

---

### Data Sources

**Pretraining:**
- TinyStories (650MB, fast experiments, good quality)
- OpenWebText (15GB, GPT-2 replication dataset)
- FineWeb-EDU (1.3TB, high-quality educational web pages)
- The Pile (825GB, diverse domains - reference only, too large)

**Fine-Tuning (SFT):**
- SmolTalk (10k samples, chat format)
- Alpaca (52k instruction-following samples)
- UltraChat (1.5M multi-turn conversations)

**Evaluation:**
- C4 (Colossal Clean Crawled Corpus) - perplexity benchmark
- WikiText-2, WikiText-103 - perplexity benchmark
- ARC-easy, ARC-challenge (reasoning, 25-question science QA)
- GSM8K (grade school math, 8.5k problems)
- MMLU (Massive Multitask Language Understanding, 57 tasks)
- HumanEval (code generation, 164 Python problems)

---

### Code References (Study, Don't Copy)

**Karpathy's Repositories (Essential):**
- `karpathy/micrograd` - Tiny autograd engine (150 lines)
- `karpathy/nanoGPT` - Minimal GPT implementation (300 lines model + 300 lines train)
- `karpathy/nanochat` - Full ChatGPT stack (tokenizer → pretrain → SFT → UI)
- `karpathy/minbpe` - Minimal BPE tokenizer (100 lines)

**HuggingFace (Production Reference):**
- `huggingface/transformers` - Study GPT-2, Llama implementations
- `huggingface/peft` - LoRA, QLoRA implementations
- `huggingface/trl` - RLHF, DPO, GRPO trainers

**SOTA Models:**
- `deepseek-ai/DeepSeek-V2` - MLA implementation, MoE
- `meta-llama/llama` - Llama 2/3 implementations
- `mistralai/mistral-src` - Mixtral (MoE)

**Distributed Training:**
- `NVIDIA/Megatron-LM` - Model/pipeline parallelism
- `microsoft/DeepSpeed` - ZeRO optimization

---

### Communities (Ask Questions, Share Progress)

**Reddit:**
- r/MachineLearning (research discussions, paper releases)
- r/LocalLLaMA (practical LLM training/inference, hardware advice)

**Discord:**
- EleutherAI (open-source LLM community, very helpful)
- HuggingFace (library support)
- Karpathy's streams (watch live coding sessions)

**Twitter/X (Follow for latest papers):**
- @karpathy (Andrej Karpathy - pedagogy, project updates)
- @ylecun (Yann LeCun - JEPA, AI philosophy)
- @_jasonwei (Jason Wei - prompting, reasoning)
- @rasbt (Sebastian Raschka - PyTorch tips, LLM training)
- @DeepSeek_AI (DeepSeek team - MLA, MoE updates)

**GitHub Discussions:**
- Karpathy's repos (Q&A, implementation help)
- HuggingFace forums (technical support)

---

## EVALUATION FRAMEWORK

### Per-Week Metrics Dashboard

**Training Metrics (Every run):**
- **Loss curve** (train, val) - should decrease smoothly
- **Perplexity** (train, val) - exp(loss), interpretable
- **Throughput** (tokens/sec, samples/sec) - efficiency
- **GPU utilization** (%, memory GB) - is GPU bottleneck?
- **Wall-clock time** per epoch/1000 steps

**Model Quality (Weekly checkpoints):**
- **Perplexity on held-out datasets**:
  - C4 (general web text)
  - WikiText-2 (Wikipedia)
  - Domain-specific (if applicable)
- **Sample generation** (100 samples, qualitative check):
  - Coherence (1-5 scale)
  - Diversity (unique n-grams)
  - Factuality (if verifiable claims)
- **Reasoning**:
  - ARC-easy accuracy (target: >30% for 124M model)
- **Math**:
  - GSM8K accuracy (target: >5% for 124M, >20% for 1B)
- **Code** (if applicable):
  - HumanEval pass@1 (target: >5% for 1B model)

**Continual Learning Metrics (Week 7+ only):**
- **Forgetting**:
  - Δ perplexity on Task 1 after training Task 2, 3
  - Target: <20% increase (good continual learning)
- **Forward transfer**:
  - Task N performance with vs without Task N-1 pretraining
  - Positive transfer: improvement
- **Backward transfer**:
  - Task 1 performance after Task N vs initial Task 1 performance
  - Target: minimal degradation (<10%)
- **Memory overhead**:
  - Bytes per task (adapters, replay buffer, Fisher diagonal)

**Efficiency Metrics:**
- **Parameters**:
  - Total (e.g., 124M)
  - Trainable (for LoRA: 1-2% of total)
- **Memory** (GB per GPU):
  - Model weights
  - Optimizer state (AdamW = 2x weights for FP32, 1x for mixed precision)
  - Activations (depends on batch size, sequence length)
  - KV cache (for inference)
- **Inference latency**:
  - Time to first token (TTFT, target: <500ms)
  - Time per token (target: <50ms)
  - Full sequence (100 tokens, target: <5 seconds)
- **KV cache size** (bytes for 2048 context):
  - Standard attention: baseline
  - GQA: ~50% reduction
  - MLA: ~70% reduction

**Tool Use Metrics (Week 4+ only):**
- **Tool call accuracy**: % correct tool invocations (right tool, right args)
- **Task success rate**: % correct final answers (what matters)
- **False positive rate**: unnecessary tool calls (hallucinated tool use)
- **Latency**: overhead from tool invocation (target: <100ms)

**Alignment Metrics (Week 9+ only):**
- **Win rate**: Pairwise comparison vs baseline (target: >60%)
- **Helpfulness score**: Human eval on 10 prompts, 1-5 scale (target: >3.5)
- **Harmlessness score**: Toxicity filter + adversarial prompts (target: <5% toxic)
- **Alignment tax**: Perplexity shift after GRPO/DPO (target: <10% increase)

---

### Evaluation Suite Setup (Week 5)

**Build once, reuse forever:**

```python
# eval.py - Core evaluation suite

class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def perplexity(self, dataset_name):
        # Load dataset (C4, WikiText, etc.)
        # Compute loss, return exp(loss)
        pass

    def arc_easy(self, num_samples=25):
        # Multiple choice reasoning
        # Return accuracy
        pass

    def gsm8k(self, num_samples=100):
        # Grade school math
        # Return accuracy (exact match)
        pass

    def humaneval(self, num_samples=164):
        # Code generation
        # Return pass@1, pass@10
        pass

    def generate_samples(self, num=100, max_length=200):
        # Qualitative check
        # Return samples + diversity metrics
        pass

    def continual_forgetting(self, task_checkpoints):
        # Measure perplexity on Task 1 across all checkpoints
        # Plot forgetting curve
        pass
```

**Usage:**
```python
# Every week, run full eval suite on new checkpoint
evaluator = Evaluator(model, tokenizer)
results = {
    'perplexity_c4': evaluator.perplexity('c4'),
    'perplexity_wikitext': evaluator.perplexity('wikitext-2'),
    'arc_easy_acc': evaluator.arc_easy(25),
    'gsm8k_acc': evaluator.gsm8k(100),
    'samples': evaluator.generate_samples(100),
}
# Log to wandb or save to JSON
```

---

## WEEKEND REFLECTION PROTOCOL

**Every Sunday, 5pm, 2-hour block:**

### 1. Build Log Review (30 min)

```markdown
# Week X Build Log

## Completed
- [ ] Core (yes/no + what specifically)
- [ ] Stretch goals (list which ones, skip which ones)
- [ ] Code quality (tests written? docs? clean commit history?)

## Metrics (compared to targets)
- Training loss: [X.XX] (target: [Y.YY])
- Validation perplexity: [X.XX] (target: [Y.YY])
- Throughput: [X] tokens/sec (target: [Y])
- GPU utilization: [X]% (target: >80%)
- Hours spent: [breakdown by task]

## Best moment this week
[1-2 sentences: what clicked, what breakthrough]

## Worst bug this week
[1-2 sentences: what broke, how you fixed it, what you learned]

## Code snippets to remember
[1-3 snippets with annotations - add to your personal cheat sheet]
```python
# Example: Attention score computation
scores = (query @ key.T) / math.sqrt(d_k)  # Scale by √d_k
scores = scores.masked_fill(mask == 0, float('-inf'))  # Causal mask
attn = F.softmax(scores, dim=-1)
```

---

### 2. Pivot Decision Tree (20 min)

```
Q1: Did I complete Core?
  → NO: Why not?
      → Knowledge gap: Add 2-3 days deep-dive (read papers, reimplement)
      → Time misestimate: Repeat week with reduced scope (skip stretch)
      → Motivation issue: Take 2-day break, reassess goals
  → YES: Continue to Q2

Q2: Do I deeply understand the concepts? (Can I explain without notes?)
  → NO:
      → Add "deep-dive mini-week" (3-4 days):
          - Read 2-3 related papers
          - Reimplement from scratch (no looking at your old code)
          - Write tutorial blog post (forces understanding)
  → YES: Continue to Q3

Q3: Are there blocking gaps for next week?
  → YES:
      → List gaps (e.g., "don't understand FSDP", "need to learn Docker")
      → Insert "catch-up mini-week" (3-4 days) to fill gaps
  → NO: Continue to Q4

Q4: Should I compress (skip stretch) or extend (add mega-stretch)?
  → Compress if:
      - Falling behind schedule (>1 week behind)
      - Losing motivation (energy/interest <7/10)
      - Compute budget tight (can't afford GPU rental)
  → Extend if:
      - Ahead of schedule (finished Core + Stretch in <5 days)
      - High energy/interest (>8/10)
      - Want depth over breadth (prefer mastery to coverage)

DECISION: [Proceed as planned / Repeat week / Deep-dive / Catch-up / Compress / Extend]
```

**Action items for next week:**
- [ ] [Specific task 1]
- [ ] [Specific task 2]
- [ ] [Specific task 3]

---

### 3. Next Week Planning (40 min)

**Read next week's goals carefully:**
- What is the Core?
- What are the Stretch goals?
- Which papers must I read?

**List prerequisites (do I have them?):**
- Knowledge: [e.g., "understand RoPE", "know PyTorch DDP"]
- Compute: [e.g., "need 2 GPUs for DDP", "need to rent A100s"]
- Data: [e.g., "need to download FineWeb-EDU"]
- Tools: [e.g., "need to install DeepSpeed"]

**Estimate time per task:**
- Core: [X] days
- Stretch: [Y] days
- Mega-stretch: [Z] days (optional)
- Buffer: 1 day for debugging

**Identify risks:**
- Knowledge gaps: [what do I not know?]
- Compute bottlenecks: [will my GPU be enough?]
- Data issues: [can I download the data in time?]

**Schedule (which days for which tasks):**
- Monday-Tuesday: [task A]
- Wednesday-Thursday: [task B]
- Friday-Saturday: [task C]
- Sunday: Reflection + next week prep

---

### 4. Paper Queue Update (10 min)

**This week:**
- [ ] Paper X (read, implemented, 2-page summary written)
- [ ] Paper Y (read, notes taken, key idea coded)

**Next week priority:**
- [ ] Paper Z (Tier 1 critical, must read + implement)
- [ ] Paper W (Tier 2, read + notes)

**New papers released?** (check ArXiv, Twitter)
- [ ] [New paper title] - relevant? Priority?

---

### 5. Motivation Check (20 min)

**Energy level (1-10)**: [X]
- If <7: Why? (burnout, life stress, boredom?)
  - Action: Take 2-day break, reduce scope, or add more fun projects

**Interest level (1-10)**: [X]
- If <7: Why? (too much theory, not enough building, wrong focus?)
  - Action: Pivot to more interesting topic, skip boring stretch goals

**Momentum (1-10)**: [X] (am I making progress?)
- If <7: Why? (stuck on bugs, unclear goals, overwhelmed?)
  - Action: Ask for help (Discord, Twitter), reduce scope, focus on one thing

**Overall assessment**:
- Green (>7 on all): Continue as planned
- Yellow (5-7 on any): Adjust scope or take break
- Red (<5 on any): STOP. Reassess goals, consider compression path or pause

---

### 6. Reflection Journal (Optional, 10 min)

**Freeform writing** (stream of consciousness):
- What did I learn about myself this week?
- What surprised me?
- What frustrated me?
- What would I do differently?
- What am I excited about next week?

---

## FLEXIBILITY MECHANISM

### Extension Path: 16-Week Deep Dive

**For**: If ahead of schedule, high motivation, want mastery over breadth

**Strategy**: Core + All stretch + All mega-stretch + Research tracks

**Additional weeks:**
- **Week 6.5 (LoRA Deep-Dive)**:
  - DoRA, LoHa, LoKr variants
  - LoRA merging strategies (TIES, DARE, weighted sum)
  - Ablation studies (rank, layers, learning rate)

- **Week 8.5 (MoE Mastery)**:
  - Scale to 16-32 experts (billions of params, sparse)
  - Expert specialization analysis
  - Soft vs hard routing (top-k, softmax, learned routing)

- **Week 10.5 (Advanced RAG)**:
  - Reranking (cross-encoder)
  - Hybrid search (dense + BM25)
  - Query rewriting (expand user query)
  - Multi-hop retrieval (chain multiple searches)

- **Week 13 (HTM Experiments)**:
  - Study Jeff Hawkins' *Thousand Brains Theory*
  - Implement spatial pooler + temporal memory
  - Hybrid HTM + Transformer architecture

- **Week 14 (JEPA Exploration)**:
  - Study Yann LeCun's I-JEPA, V-JEPA
  - Implement JEPA on simple task (masked prediction)
  - Hybrid Transformer + JEPA objective

- **Week 15 (Advanced Continual)**:
  - Progressive Neural Networks (lateral connections)
  - Meta-learning for fast adaptation (MAML, Reptile)
  - Lifelong learning benchmarks (CLOC, CTrL)

- **Week 16 (50 Papers + Publication)**:
  - Complete 50-paper analysis sprint
  - Write comprehensive blog post series (10 posts)
  - Optional: Submit findings to arXiv or blog (build public portfolio)

**Result: 16-week extended path**
```
Weeks 1-12: Standard plan (Core + most Stretch)
Week 13: HTM (research track A)
Week 14: JEPA (research track B)
Week 15: Advanced continual learning (research track C)
Week 16: 50-paper analysis + blog publication (research track E)
```

**What you gain:**
- Deep expertise in continual learning (research frontier)
- Exposure to alternative architectures (HTM, JEPA)
- 50 papers analyzed (vs 15-20 in base plan)
- Public portfolio (blog posts, arXiv submission)
- Mastery over breadth (prefer depth)

---

### Modular Week Structure (How to use MVP/Stretch/Mega-stretch)

**Each week has 3 tiers:**

**Core - MANDATORY**
- 4-6 days of work
- Essential skills for final goal
- If you skip this, you'll be blocked later
- **You MUST complete this to proceed**

**Stretch - OPTIONAL (Recommended)**
- 2-3 days of work
- Deepens understanding
- Adds nice-to-have features
- **Complete if:**
  - You finish Core in <5 days
  - You have energy/interest >7/10
  - Next week doesn't depend on this

**Mega-stretch - OPTIONAL (Advanced)**
- 2-3 days of work
- Bleeding-edge techniques
- Research-level depth
- **Complete if:**
  - You finish Core + Stretch in <6 days
  - You want mastery, not breadth
  - You're ahead of schedule

**Decision tree each week:**
```
Did I finish Core in <5 days?
  → YES: Do Stretch goals
      → Finished Core + Stretch in <6 days?
          → YES: Do Mega-stretch
          → NO: Proceed to next week
  → NO: Skip Stretch, proceed to next week
```

---

### Dynamic Pivoting (When to adjust)

**Signals to COMPRESS (skip stretch goals):**
- You're >1 week behind schedule
- Energy/interest <7/10 for 2+ weeks
- Burnout risk (working >8 hours/day, not enjoying)
- Compute budget running out
- External life stress (job, family, health)

**Signals to EXTEND (add mega-stretch):**
- You're >1 week ahead of schedule
- Energy/interest >8/10 consistently
- Finishing Core in <4 days per week
- You want research-level depth
- You have extra compute budget

**Signals to PAUSE (take break):**
- Energy/interest <5/10 for 2+ weeks
- Not enjoying the process (feels like chore, not curiosity)
- Severe burnout symptoms (can't focus, dreading work)
- External crisis (job loss, health emergency)

**How to pivot:**
1. Identify signal (compression, extension, pause)
2. Sunday reflection: make decision explicitly
3. Update plan in README (document pivot)
4. Communicate to accountability partner (if any)
5. Proceed with adjusted plan

---

## FINAL CHECKLIST

### Technical Milestones (End of Week 12)

**Implementation:**
- [ ] Transformer from scratch (attention, MLP, residuals, LayerNorm)
- [ ] BPE tokenizer working (encode/decode roundtrip)
- [ ] Training loop with AMP, grad accumulation, checkpointing
- [ ] RoPE, GQA, SwiGLU, RMSNorm integrated (modern architecture)
- [ ] Evaluation suite (perplexity, ARC, GSM8K, HumanEval)
- [ ] LoRA implemented from scratch & tested
- [ ] Continual learning (EWC + rehearsal + LoRA-per-task)
- [ ] MLA or MoE explored (at least conceptually)
- [ ] SFT + GRPO/DPO working (aligned chat model)
- [ ] Tool use (calculator, search, Python REPL) integrated
- [ ] Multi-GPU training successful (DDP or FSDP)
- [ ] Nanochat UI deployed (working chat interface)

**Metrics achieved:**
- [ ] Trained 50M+ param model to convergence
- [ ] Trained 124M-560M param model (baseline)
- [ ] Perplexity <20 on C4, <25 on WikiText (124M model)
- [ ] ARC-easy >30% accuracy (124M model)
- [ ] LoRA finetune 10x faster than full FT
- [ ] Continual learning: forgetting <20% (measured)
- [ ] Tool use >80% accuracy on 20 tasks
- [ ] Inference latency <500ms TTFT (time to first token)

---

### Knowledge Mastery (Self-assessment)

**Can you do these without notes?**
- [ ] Derive attention score computation (Q @ K.T / √d_k)
- [ ] Explain backprop + chain rule with code example
- [ ] Tune AdamW hyperparameters (lr, β1, β2, wd)
- [ ] Debug PyTorch shape mismatches in <5 min
- [ ] Explain LoRA low-rank approximation (why it works)
- [ ] Implement EWC Fisher diagonal from scratch
- [ ] Explain GRPO vs DPO vs PPO (when to use each)
- [ ] Understand scaling laws (compute optimal, data scaling)
- [ ] Explain RoPE (why it's better than absolute PE)
- [ ] Explain GQA (memory savings, quality tradeoff)

---

### Papers Read & Implemented (Minimum)

**Tier 1 (Critical) - 15 papers:**
- [ ] Attention Is All You Need
- [ ] RoFormer (RoPE)
- [ ] Llama 2 (architecture)
- [ ] GQA
- [ ] Scaling Laws (Kaplan)
- [ ] LoRA
- [ ] QLoRA
- [ ] EWC
- [ ] Thinking Machines LoRA blog
- [ ] DeepSeek-V2 (MLA)
- [ ] FlashAttention-2
- [ ] DPO
- [ ] DeepSeek-R1 (GRPO)
- [ ] ReAct
- [ ] Megatron-LM or ZeRO

**Tier 2 (Important) - 10+ papers:**
- [ ] [Your choice based on interest]

**Total: 25+ papers read, 15+ core ideas implemented**

If extended to Week 16: **50 papers analyzed**


---

**Last Updated**: 2025-12-07
**Version**: 4.0

---
