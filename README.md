# Chase Karpathy: 100-Day AI/ML Challenge

**Goal**: Build nanochat from scratch with continual learning capability

---

## What Is This?

Personal 100-day challenge to master modern AI/ML by building from first principles. Build real systems from scratch and understand internals deeply.

---

## What I'm Building

### Final Artifact
- **nanochat** with continual learning capability
- **Small** (560M-1B params), **fast**, production-ready
- **Never forgets** (EWC + rehearsal + LoRA-per-task)
- **Uses tools** (calculator, search, Python REPL) for facts
- **Learns forever** (nested LoRA, adapter stacking)

**Design Principles:**
1. **Top-down with safety**: Start from nanochat goal, recurse to fundamentals
2. **Modular weeks**: Core (MVP) + Stretch + Mega-stretch
3. **Measure everything**: Metrics dashboard from Week 1
4. **Flexible timeline**: Compress to 8 weeks or extend to 16 weeks
5. **Weekend protocol**: Formalized reflection + pivot decision tree
6. **SOTA-first**: 2025 techniques (MLA, GRPO, FlashAttention-2) by Week 8
7. **Continual learning throughout**: Week 6 onward, not just at the end
8. **Tool use early**: Week 4, not Week 8
9. **Open-ended ceiling**: Week 13+ research tracks (HTM, JEPA, MoE)

---

## The Plan

**See [PLAN.md](PLAN.md) for full details.**

### Phase 1: Foundations (Weeks 0-2)
- **Week 0**: Math refresher + Autograd engine from scratch (micrograd)
- **Week 1**: Transformer architecture (attention, MLP, residuals)
- **Week 2**: Tokenization (BPE) + data pipeline

### Phase 2: Training & Modern Architecture (Weeks 3-5)
- **Week 3**: Training loop (AdamW, AMP, checkpointing)
- **Week 4**: Modern arch (RoPE, GQA, SwiGLU) + **Tool use basics** ⚠️
- **Week 5**: Evaluation framework + baseline 124M model

### Phase 3: PEFT & Continual Learning (Weeks 6-7)
- **Week 6**: LoRA mastery (implement from scratch, QLoRA)
- **Week 7**: Continual learning core (EWC, replay, LoRA-per-task) ⭐

### Phase 4: SOTA 2025 & Alignment (Weeks 8-9)
- **Week 8**: MLA (DeepSeek), FlashAttention-2, MoE
- **Week 9**: SFT + GRPO/DPO (alignment)

### Phase 5: Integration & Scale (Weeks 10-12)
- **Week 10**: Hydra model (multi-adapter + RAG + tools)
- **Week 11**: Scale to 1B params, multi-GPU (DDP/FSDP)
- **Week 12**: Nanochat integration (full speedrun + UI)

### Phase 6: Research Tracks (Week 13+)
- **Track A**: Hierarchical Temporal Memory (HTM, Jeff Hawkins)
- **Track B**: JEPA (Yann LeCun)
- **Track C**: Advanced continual learning
- **Track D**: MoE mastery
- **Track E**: 50 papers deep dive + blog publication

---

## Key Features

### Modular Week Structure
Each week has 3 tiers:
- **Core (MVP)**: MANDATORY (4-6 days) - essential for final goal
- **Stretch**: OPTIONAL (2-3 days) - deepens understanding
- **Mega-stretch**: OPTIONAL (2-3 days) - research-level depth

### Flexibility Mechanisms
- **8-week fast path**: Core MVP only, skip stretches
- **12-week balanced**: Core + most stretches
- **16-week deep**: Core + all stretches + research tracks

### Weekend Reflection Protocol
Every Sunday (2 hours):
1. Build log review (what worked, what broke)
2. Pivot decision tree (proceed / repeat / compress / extend)
3. Next week planning (prerequisites, risks, schedule)
4. Paper queue update
5. Motivation check (energy, interest, momentum)

---

## Math Prerequisites

**Time**: 5-7 days (Week 0)

**Just what's needed, no fluff:**
- **Linear Algebra**: Shapes, matmul, broadcasting, SVD (for LoRA)
- **Calculus**: Chain rule, gradients, backprop
- **Optimization**: SGD, AdamW, LR schedules, gradient clipping
- **Probability**: Softmax, cross-entropy, KL divergence

**Resources**: 3Blue1Brown (visual), Karpathy micrograd (hands-on)

---

## Critical Papers (Tier 1)

**Must implement core idea:**
1. Attention Is All You Need (Vaswani et al.)
2. RoFormer (RoPE)
3. LoRA (Hu et al.)
4. QLoRA (Dettmers et al.)
5. EWC (Kirkpatrick et al.)
6. Thinking Machines "LoRA Without Regret" blog
7. DeepSeek-V2 (MLA)
8. FlashAttention-2 (Dao)
9. DPO (Rafailov et al.)
10. DeepSeek-R1 (GRPO)
11. ReAct (Yao et al.)
12. Megatron-LM / ZeRO

**Total target**: 25+ papers read, 15+ core ideas implemented

---

## Tools & Resources

### Core Stack
- PyTorch 2.0+ (torch.compile, FSDP)
- HuggingFace (transformers, datasets, PEFT, accelerate)
- bitsandbytes (quantization)
- DeepSpeed (ZeRO)
- FlashAttention-2
- FAISS (RAG)
- wandb / TensorBoard

### Data
- TinyStories (650MB, fast)
- OpenWebText (15GB)
- FineWeb-EDU (1.3TB, high-quality)
- SmolTalk, Alpaca (SFT)
- C4, WikiText, ARC, GSM8K, MMLU (eval)

### Code References (Study, Don't Copy)
- karpathy/micrograd
- karpathy/nanoGPT
- karpathy/nanochat
- karpathy/minbpe
- HuggingFace repos (compare to yours)
- DeepSeek, Llama, Mixtral repos

---

## Progress Tracking

**Start Date**: Dec 1
**Current Week**: 0 (Planning complete)

### Weekly Log
- **Week 0**: [ ] Math + micrograd
- **Week 1**: [ ] Transformer
- **Week 2**: [ ] Tokenization
- **Week 3**: [ ] Training loop
- **Week 4**: [ ] Modern arch + tools
- **Week 5**: [ ] Eval + baseline
- **Week 6**: [ ] LoRA
- **Week 7**: [ ] Continual learning
- **Week 8**: [ ] SOTA 2025
- **Week 9**: [ ] Alignment
- **Week 10**: [ ] Hydra
- **Week 11**: [ ] Scaling
- **Week 12**: [ ] Nanochat
- **Week 13+**: [ ] Research tracks

---

---

## License

MIT License - Feel free to use this plan for your own learning journey.

---

---

**Last Updated**: 2025-11-30
**Version**: 3.0