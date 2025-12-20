# Chase Karpathy: 100-Day AI/ML Challenge

**Goal**: Build nanochat from scratch with continual learning capability

---

## What Is This?

Personal 100-day challenge to master modern AI/ML by building from first principles. Build real systems from scratch and understand internals deeply.
**Now featuring a deep-dive approach: Exercises + Papers + Implementation.**

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
2. **Deep Dive First**: Implementation + Exercises + Papers (no rushing)
3. **Modular weeks**: Core + Stretch + Mega-stretch
4. **Measure everything**: Metrics dashboard from Week 4
5. **Flexible timeline**: 14+ weeks
6. **Weekend protocol**: Formalized reflection + pivot decision tree
7. **SOTA-first**: 2025 techniques (MLA, GRPO, FlashAttention-2) by Week 11
8. **Continual learning throughout**: Week 9 onward
9. **Tool use early**: Week 7, not Week 11
10. **Open-ended ceiling**: Week 16+ research tracks

---

## The Plan

**See [PLAN.md](PLAN.md) for full details.**

### Phase 1: Foundations (Weeks 0-2) [COMPLETED]
- **Weeks 0-2**: Math + Micrograd + Makemore 1-2 (Bigram/MLP) + Bengio Paper.
- *Deep focus on exercises and manual gradient implementation.*

### Phase 2: Mastery & Architecture (Weeks 3-7)
- **Week 3**: Makemore Mastery (BatchNorm, Backprop, WaveNet)
- **Week 4**: Transformer Architecture (Attention is All You Need)
- **Week 5**: Tokenization (BPE) + Data Pipeline
- **Week 6**: Training Loop & Efficiency
- **Week 7**: Modern Architecture (Llama-style) + Tool Use

### Phase 3: Eval & PEFT (Weeks 8-9)
- **Week 8**: Evaluation Framework
- **Week 9**: LoRA Mastery & PEFT

### Phase 4: Continual Learning & SOTA (Weeks 10-11)
- **Week 10**: Continual Learning Core (EWC, Replay)
- **Week 11**: SOTA 2025 (MLA, FlashAttention-2, MoE)

### Phase 5: Alignment & Integration (Weeks 12-13)
- **Week 12**: SFT + GRPO/DPO (Alignment)
- **Week 13**: Hydra Model (Adapters + RAG + Tools)

### Phase 6: Scale & Final Assembly (Weeks 14-15)
- **Week 14**: Scale to 1B Params (Multi-GPU)
- **Week 15**: Nanochat Integration

### Phase 7: Research Tracks (Week 16+)
- **Track A-E**: HTM, JEPA, Advanced CL, MoE, Papers.

---

## Progress Tracking

**Start Date**: Dec 1
**Current Phase**: Week 3 (Makemore Mastery)

### Daily Log
Detailed daily progress is tracked in the `days/` directory.
- [**days/README.md**](days/README.md) - **GO HERE FOR DAILY UPDATES**

### High-Level Milestone Tracker
- **Weeks 0-2**: [x] Foundations (Micrograd, Makemore 1-2, Exercises)
- **Week 3**: [ ] Makemore 3-5
- **Week 4**: [ ] Transformer
- **Week 5**: [ ] Tokenization
- **Week 6**: [ ] Training Engine
- **Week 7**: [ ] Modern Arch
- **Week 8**: [ ] Evals
- **Week 9**: [ ] LoRA
- **Week 10**: [ ] Continual Learning
- **Week 11**: [ ] SOTA 2025
- **Week 12**: [ ] Alignment
- **Week 13**: [ ] Hydra
- **Week 14**: [ ] Scale
- **Week 15**: [ ] Nanochat

---

## Tools & Resources

### Core Stack
- PyTorch 2.0+
- HuggingFace (transformers, datasets, PEFT)
- bitsandbytes
- DeepSpeed / FlashAttention-2
- FAISS
- WandB

### Data
- TinyStories, OpenWebText, FineWeb-EDU

---

**Last Updated**: 2025-12-20
**Version**: 5.0 - Deep Dive Edition