# Chase Karpathy: 100-Day AI/ML Challenge

**Goal**: Build nanoGPT from scratch with continual learning capability

**Duration**: 100 days (14 weeks)

**Philosophy**: Top-down, project-driven, minimal theory, maximum building

---

## About This Challenge

This is a personal 100-day challenge to master modern AI/ML by building from first principles. The goal is not to memorize theory or chase credentials, but to build real systems from scratch and deeply understand how they work.

### What I'm Building

- **nanoGPT**: A transformer-based language model built from scratch (no black boxes)
- **Continual Learning System**: Enable the model to learn new tasks without forgetting old ones
- **Production-Ready Code**: Clean, documented, testable implementation

### Philosophy

- **Top-Down Learning**: Start from the end goal (nanoGPT) and decompose into learnable components
- **Project-Driven**: Build challenging projects every week, not just consume theory
- **No Fluff**: Only learn math and concepts directly needed for implementation
- **Skill Reuse**: Everything learned must be used multiple times in building
- **Hacker Mentality**: Build what I want, contribute in a hacker way (inspired by Karpathy, Krizhevsky)

---

## Structure

### Phase 1: Foundations (Weeks 1-2)
- Build autograd engine from scratch (micrograd)
- Implement neural networks without libraries
- Train on MNIST

### Phase 2: Sequences (Weeks 3-6)
- Character-level language models
- Deep network diagnostics
- Manual backpropagation
- WaveNet-style architectures

### Phase 3: Transformers (Weeks 7-9)
- Self-attention mechanism
- Full GPT architecture
- Byte Pair Encoding tokenization
- Text generation

### Phase 4: Scaling (Weeks 10-11)
- Train 50M-124M parameter models
- Learn scaling laws
- Pre-training techniques
- Proper evaluation

### Phase 5: Efficient Adaptation (Weeks 12-13)
- LoRA (Low-Rank Adaptation)
- Continual learning techniques
- Catastrophic forgetting prevention
- Multi-task scenarios

### Phase 6: Integration (Week 14)
- Production nanoGPT with continual learning
- Advanced features (RoPE, GQA)
- Comprehensive documentation
- Blog post

---

## Resources

### Primary Course
- **Andrej Karpathy - Neural Networks: Zero to Hero** (~16 hours)
  - From micrograd to full GPT implementation

### Key Papers
- Attention is All You Need (Transformers)
- LoRA: Low-Rank Adaptation
- Scaling Laws for Neural Language Models
- Self-Synthesized Rehearsal (Continual Learning)

### Supplementary
- Stanford CS230 (Deep Learning)
- Stanford CS231n (Deep Learning for Computer Vision)
- HuggingFace LLM Training Playbook

---

## Directory Structure

```
chase-karpathy/
├── weeks/
│   ├── week01-02/    # Autograd & basic NN
│   ├── week03-04/    # Sequence models
│   ├── week05/       # Deep MLPs & BatchNorm
│   ├── week06/       # Manual backprop
│   ├── week07-08/    # Attention & Transformers
│   ├── week09/       # Tokenization
│   ├── week10/       # Scaling & optimization
│   ├── week11/       # Pre-training
│   ├── week12/       # LoRA & PEFT
│   ├── week13/       # Continual learning
│   └── week14/       # Final integration
├── projects/         # Weekly projects
├── papers/           # Research papers
├── notes/            # Learning notes
├── checkpoints/      # Model checkpoints
├── PLAN.md          # Detailed 14-week plan
└── README.md        # This file
```

---

## Weekly Reflection

Every Sunday, I'll reflect on:
- What I built this week
- What I learned
- What challenged me
- Adjustments to the plan

The plan is **dynamic** - AI moves fast, and I'll adapt as needed.

---

## Success Criteria (Day 100)

- [ ] Built GPT from scratch (no black boxes)
- [ ] Model generates coherent text
- [ ] LoRA fine-tuning implementation works
- [ ] Continual learning reduces catastrophic forgetting
- [ ] Can explain every component
- [ ] Clean, runnable codebase on GitHub
- [ ] Comprehensive demo notebook
- [ ] Technical blog post

---

## Inspiration

> "The best way to learn is to build." - Andrej Karpathy

I'm not aiming to be Yann LeCun or Geoffrey Hinton (academia/corporate). I want to be like Andrej Karpathy or Alex Krizhevsky - builders who contribute through code and share knowledge openly.

This challenge is about mastering the craft of building AI systems from first principles.

---

## Progress

**Start Date**: TBD
**Current Week**: 0 (Planning)
**Status**: Setting up

---

## Contact & Contributions

This is a personal learning journey, but feel free to:
- Follow along
- Ask questions
- Share your own implementation
- Suggest improvements

Let's build.
