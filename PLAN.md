# 100-DAY AI/ML CHALLENGE: CHASE KARPATHY

**Goal**: Build nanoGPT from scratch with continual learning capability

**Duration**: 14 weeks (~98 days)

**Philosophy**: Top-down, project-driven, minimal theory, maximum building

---

## TABLE OF CONTENTS

1. [Dependency Graph](#dependency-graph)
2. [Math Quickstart](#math-quickstart)
3. [Week-by-Week Breakdown](#week-by-week-breakdown)
4. [Critical Papers](#critical-papers)
5. [Core Concepts Reference](#core-concepts-reference)
6. [Continual Learning Techniques](#continual-learning-techniques)
7. [Weekly Reflection Framework](#weekly-reflection-framework)
8. [Resources](#resources)
9. [Alternative Architectures](#alternative-architectures)

---

## DEPENDENCY GRAPH

### TOP-DOWN DECOMPOSITION

```
LEVEL 0: FINAL GOAL
└─ nanoGPT with Continual Learning Capability

LEVEL 1: MAJOR COMPONENTS
├─ Core GPT Architecture
│  ├─ Transformer Decoder
│  ├─ Training Loop
│  ├─ Tokenization System
│  └─ Inference Engine
│
├─ Continual Learning System
│  ├─ Parameter-Efficient Fine-Tuning (LoRA/QLoRA)
│  ├─ Rehearsal/Replay Mechanisms
│  └─ Catastrophic Forgetting Prevention
│
└─ Infrastructure & Optimization
   ├─ Efficient Attention (Flash Attention concepts)
   ├─ Mixed Precision Training
   └─ Checkpoint Management

LEVEL 2: TRANSFORMER DECODER BREAKDOWN
├─ Multi-Head Causal Self-Attention
│  ├─ Scaled Dot-Product Attention
│  ├─ Query/Key/Value Projections
│  ├─ Causal Masking
│  └─ Position Encoding (RoPE)
│
├─ Feed-Forward Network
│  ├─ MLP with GELU activation
│  └─ Residual Connections
│
├─ Layer Normalization
└─ Embedding Layer
   ├─ Token Embeddings
   └─ Position Embeddings

LEVEL 3: ATTENTION MECHANISM BREAKDOWN
├─ Linear Projections (Q, K, V)
├─ Attention Score Computation
│  ├─ Matrix Multiplication (Q @ K^T)
│  ├─ Scaling by √d_k
│  └─ Softmax Activation
│
├─ Causal Mask Application
└─ Output Projection

LEVEL 4: TRAINING LOOP BREAKDOWN
├─ Data Loading & Batching
├─ Forward Pass
│  ├─ Token to Embedding
│  ├─ Through Transformer Blocks
│  └─ Final Linear Layer + Softmax
│
├─ Loss Computation (Cross-Entropy)
├─ Backward Pass (Autograd)
└─ Optimization Step (AdamW)

LEVEL 5: FOUNDATIONAL COMPONENTS
├─ Autograd/Backpropagation
│  ├─ Computational Graph
│  ├─ Forward Pass Recording
│  └─ Backward Pass (Chain Rule)
│
├─ Basic Neural Network Layers
│  ├─ Linear Layer (Matrix Multiplication + Bias)
│  ├─ Activation Functions (ReLU, GELU, Softmax)
│  └─ Loss Functions (MSE, Cross-Entropy)
│
└─ Optimization Algorithms
   ├─ SGD
   ├─ Momentum
   ├─ RMSprop
   └─ Adam/AdamW
```

---

## MATH QUICKSTART

### Calculus (3-4 hours)

**Essential Topics:**
1. **Partial Derivatives**: ∂f/∂x - rate of change with respect to one variable
2. **Chain Rule**: df/dx = (df/dy) × (dy/dx) - critical for backpropagation
3. **Gradient**: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ] - direction of steepest ascent

**Common Derivatives to Memorize:**
- d/dx(x²) = 2x
- d/dx(eˣ) = eˣ
- d/dx(log x) = 1/x
- d/dx(1/x) = -1/x²

**Skip**: Limits, integration, differential equations

**Resource**: Karpathy's micrograd lecture covers this operationally

### Linear Algebra (6-8 hours)

**Essential Topics:**
1. **Matrix Multiplication**: C = AB where C[i,j] = Σₖ A[i,k] × B[k,j]
   - Dimensions: (m×n) @ (n×p) → (m×p)
2. **Matrix Transpose**: Aᵀ, where (AB)ᵀ = BᵀAᵀ
3. **Dot Product**: a·b = Σᵢ aᵢbᵢ = |a||b|cos(θ)
4. **Broadcasting**: numpy/torch array operations with different shapes
5. **Matrix-Vector Operations**: y = Wx + b (linear layer)
6. **Softmax**: softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)

**Skip**: Determinants, abstract vector spaces, formal proofs

**Resource**: 3Blue1Brown "Essence of Linear Algebra" chapters 1, 3, 4, 7

### Probability (2-3 hours)

**Essential Topics:**
1. **Probability Distributions**: P(x), Σ P(x) = 1
2. **Expectation**: E[X] = Σ x·P(x)
3. **Cross-Entropy**: H(p,q) = -Σ p(x) log q(x) - your loss function
4. **Maximum Likelihood**: Find parameters that make data most likely

**Skip**: Formal measure theory, complex probability proofs

---

## WEEK-BY-WEEK BREAKDOWN

### WEEKS 1-2: FOUNDATIONS - AUTOGRAD & BASIC NN

**Goal**: Build intuition for backpropagation and neural networks from scratch

**Week 1 Project**: Build micrograd
- Implement Value class with automatic differentiation
- Support operations: +, -, *, /, **
- Implement backward() with topological sort
- Build 2-layer MLP
- Train on toy dataset (circles, moons)

**Week 2 Project**: Extend to tensor-level
- Implement tensor operations (batch support)
- Common activations (ReLU, tanh, sigmoid)
- Build 3-layer MLP for MNIST
- Target: >95% accuracy

**Math Needed**:
- Chain rule (operational)
- Gradient descent
- Matrix multiplication
- Softmax and cross-entropy

**Key Skills**: Computational graphs, backpropagation, Python autodiff

**Resources**:
- Karpathy NN:0→Hero Lecture 1 (2h 30m)
- 3Blue1Brown: "What is backpropagation?"

**Deliverable**: Working autograd engine + MNIST classifier

---

### WEEKS 3-4: SEQUENCE MODELS - CHARACTER-LEVEL

**Goal**: Understand sequence prediction and language modeling

**Week 3 Project**: Bigram character model
- Build character-level tokenizer
- Implement bigram model
- Train on Shakespeare dataset
- Generate text samples
- Understand perplexity

**Week 4 Project**: MLP character model
- Extend to multi-layer perceptron
- Implement embedding layer
- Add context window (predict next char from last 3)
- Hyperparameter tuning:
  - Learning rate: 0.1, 0.01, 0.001
  - Hidden size: 100, 200, 500
  - Layers: 2, 3, 4
- Track training curves

**Math Needed**:
- Embeddings: discrete → continuous
- Cross-entropy for classification
- Train/val/test splits

**Key Skills**: Sequence modeling, embeddings, tokenization

**Resources**:
- Karpathy NN:0→Hero Lectures 2-3 (5h total)
- Shakespeare dataset

**Deliverable**: Character-level language model generating coherent text

---

### WEEK 5: DEEP MLPS & BATCH NORMALIZATION

**Goal**: Understand training dynamics of deeper networks

**Week 5 Project**: Deep MLP with diagnostics
- Build 5-6 layer MLP
- Implement BatchNorm from scratch
- Diagnose activation statistics (plot mean/std per layer)
- Plot gradient magnitudes per layer
- Try different initializations (Xavier, He)
- Compare training with/without BatchNorm
- Visualize gradient flow

**Math Needed**:
- Statistics: mean, variance, std
- Normalization: (x - μ) / σ
- Why gradients vanish/explode
- Weight initialization theory

**Papers**:
- Batch Normalization (Ioffe & Szegedy, 2015) - sections 1, 3 only

**Key Skills**: Debugging deep networks, activation analysis

**Resources**:
- Karpathy NN:0→Hero Lecture 4 (2h 15m)
- TensorBoard/wandb for logging

**Deliverable**: Deep MLP with diagnostic visualizations

---

### WEEK 6: MANUAL BACKPROP & WAVENET

**Goal**: Deeply understand backpropagation by implementing manually

**Week 6 Projects**:
1. **Manual backprop through MLP**
   - Take BatchNorm MLP from Week 5
   - Implement backward pass manually (no autograd)
   - Verify gradients match autograd
   - Go through: matmul, add, relu, softmax, cross-entropy

2. **WaveNet-style hierarchical model**
   - Build tree-structured MLP
   - Implement causal convolutions
   - Understand receptive fields
   - Compare with flat MLP

**Math Needed**:
- Backprop through each operation:
  - dL/dX for X = AB
  - dL/dX for X = relu(Y)
  - dL/dX for X = softmax(Y)
- Convolution basics

**Papers**:
- WaveNet (van den Oord et al., 2016) - architecture section

**Key Skills**: Deep backprop understanding, gradient checking

**Resources**:
- Karpathy NN:0→Hero Lectures 5-6 (4h total)

**Deliverable**: Manual backprop implementation + WaveNet model

---

### WEEKS 7-8: ATTENTION & TRANSFORMERS

**Goal**: Build a GPT from scratch

**Week 7 Part 1**: Self-Attention
- Implement scaled dot-product attention
- Build single-head attention
- Extend to multi-head attention
- Add causal masking
- Visualize attention weights

**Week 7 Part 2**: Transformer Block
- Combine attention + FFN + LayerNorm + residuals
- Stack 2-3 blocks
- Position encodings (learned → sinusoidal)

**Week 8**: Full GPT
- Implement complete GPT architecture
- Train on Shakespeare (~10M params)
- Hyperparameters:
  - n_layer: 6
  - n_head: 6
  - n_embd: 384
  - block_size: 256
  - batch_size: 64
  - learning_rate: 3e-4
  - max_iters: 5000
- Implement top-k, top-p sampling
- Generate samples every 500 steps

**Math Needed**:
- Attention formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Why scale by √d_k
- Positional encoding rationale
- Residual connections
- LayerNorm vs BatchNorm

**Papers - CRITICAL**:
- **Attention is All You Need (Vaswani et al., 2017)**
  - Read: Sections 1, 3 (model architecture), 5.4
  - Skip: 6-7 (not needed for building)
  - Companion: "The Illustrated Transformer" by Jay Alammar
  - Code reference: "The Annotated Transformer" by Harvard NLP

**Key Skills**: Attention, multi-head attention, causal masking, transformers

**Resources**:
- Karpathy NN:0→Hero Lecture 7 (1h 57m) - "Let's build GPT"
- Karpathy's build-nanogpt repository
- Alammar's "The Illustrated Transformer"

**Deliverable**: Working GPT generating coherent text (60% of final goal!)

---

### WEEK 9: TOKENIZATION

**Goal**: Build proper tokenizer (BPE)

**Week 9 Project**: Implement Byte Pair Encoding
- Understand character-level limitations
- Implement BPE training:
  1. Start with character vocabulary
  2. Count adjacent pairs
  3. Merge most frequent
  4. Repeat for N merges
- Build encode/decode functions
- Compare: char-level (256) vs BPE (50k)
- Implement special tokens: <|endoftext|>, <|pad|>
- Test on various languages/code
- Observe tokenization quirks

**Bonus**: Implement SentencePiece or WordPiece

**Math Needed**:
- Vocabulary size vs model size tradeoffs
- Compression ratio metrics

**Papers**:
- Neural Machine Translation with Subword Units (Sennrich et al., 2016) - sections 1-3

**Key Skills**: Tokenization algorithms, BPE, vocabulary construction

**Resources**:
- Karpathy NN:0→Hero Lecture 8 (1h 17m)
- Karpathy's minbpe repository

**Deliverable**: BPE tokenizer with ~50k vocabulary

---

### WEEK 10: SCALING & OPTIMIZATION

**Goal**: Train medium-sized GPT efficiently

**Week 10 Project**: Scale up
- Increase to ~50M parameters:
  - n_layer: 8
  - n_head: 8
  - n_embd: 512
- Train on OpenWebText subset (~1GB)
- Implement optimizations:
  - Gradient accumulation
  - Mixed precision (FP16)
  - Gradient clipping
  - LR warmup + cosine decay
  - Weight decay (AdamW)
- Track metrics:
  - Train/val loss
  - Tokens/second
  - GPU memory
- Implement checkpointing
- Resume from checkpoint

**Math Needed**:
- Learning rate schedules
- Gradient clipping rationale
- Mixed precision concepts
- Gradient accumulation

**Papers - CRITICAL**:
- **Scaling Laws for Neural Language Models (Kaplan et al., 2020)**
  - Read: Abstract, sections 1-2, 6
  - Understand: Loss scales as power law
  - Key: Larger models more sample-efficient

- **Chinchilla Scaling Laws (Hoffmann et al., 2022)**
  - Read: Abstract, section 1
  - Understand: Optimal model size vs dataset size
  - Takeaway: Most models overtrained

**Key Skills**: Training large models, GPU optimization, checkpointing

**Resources**:
- PyTorch Lightning
- wandb
- HuggingFace Accelerate
- nanoGPT train.py

**Deliverable**: 50M parameter model trained on OpenWebText

---

### WEEK 11: PRE-TRAINING & EVALUATION

**Goal**: Serious pre-training and proper evaluation

**Week 11 Project**: Large-scale pre-training
- Train 100M-124M params (GPT-2 small size)
- Full OpenWebText (~9GB)
- Run 100k-200k iterations
- Evaluation suite:
  - Perplexity on test set
  - Zero-shot tasks:
    - HellaSwag
    - Sentiment classification
    - Simple math
  - Human evaluation
- Study training dynamics
- Compare with GPT-2 small benchmarks

**Alternative (compute-limited)**: Fine-tune GPT-2 small on domain data

**Math Needed**:
- Perplexity: exp(cross_entropy_loss)
- Zero-shot vs few-shot
- Bias-variance in evaluation

**Papers**:
- Language Models are Unsupervised Multitask Learners (GPT-2, Radford et al., 2019)
  - Read: Sections 1-2, 4.1-4.2
  - Understand: Pre-training → zero-shot transfer

**Key Skills**: Large-scale training, evaluation, benchmarking

**Resources**:
- OpenWebText dataset
- EleutherAI evaluation harness
- HuggingFace datasets

**Deliverable**: Pre-trained GPT-2 small with evaluation results

---

### WEEK 12: PARAMETER-EFFICIENT FINE-TUNING (PEFT)

**Goal**: Implement LoRA for efficient fine-tuning

**Week 12 Project**: LoRA implementation
- Understand LoRA math:
  - Freeze W
  - Add ΔW = BA (B: d×r, A: r×d, r << d)
  - Forward: h = Wx + BAx
- Implement LoRA layers
- Replace linear layers in attention (Q, K, V, O)
- Typical rank r: 4-8
- Initialize: A with Kaiming, B with zeros
- Fine-tune on task:
  - Python code generation
  - Dialogue
  - Style imitation
- Compare:
  - Full fine-tuning
  - LoRA (1-2% params)
  - Frozen (zero-shot)
- Measure: performance, speed, memory, forgetting

**Bonus**: QLoRA (quantized LoRA)

**Math Needed**:
- Low-rank matrix approximation
- Intrinsic dimensionality
- Rank selection

**Papers - CRITICAL**:
- **LoRA: Low-Rank Adaptation (Hu et al., 2021)**
  - Read: All sections 1-4
  - Study equations in 4.1 carefully
  - Understand: Why LoRA works, where to apply

**Key Skills**: LoRA implementation, efficient fine-tuning

**Resources**:
- HuggingFace PEFT library (study after building)
- LoRA paper + supplementary

**Deliverable**: LoRA fine-tuned model on specific task

---

### WEEK 13: CONTINUAL LEARNING

**Goal**: Add continual learning capability

**Week 13 Project**: Continual learning system
- Benchmark setup:
  - Task 1: General text (OpenWebText)
  - Task 2: Python code
  - Task 3: Scientific text (arXiv)
  - Task 4: Dialogue

- Three approaches:

  1. **Naive Sequential Fine-tuning** (baseline)
     - Fine-tune Task 1 → Task 2
     - Measure Task 1 drop (catastrophic forgetting)

  2. **LoRA with Task-Specific Adapters**
     - Separate LoRA per task
     - Switch at inference
     - No forgetting, requires task ID

  3. **Self-Synthesized Rehearsal (SSR)**
     - Generate from Task 1 model
     - Mix 20-30% with Task 2 data
     - Reduced forgetting

- Metrics:
  - Forward transfer
  - Backward transfer
  - Average accuracy
  - Compute overhead

**Math Needed**:
- Catastrophic forgetting mechanisms
- Replay/rehearsal rationale
- Multi-task learning

**Papers - CRITICAL**:
- **Catastrophic Forgetting in LLMs (Luo et al., 2023)** [arXiv:2308.08747]
  - Read: Sections 1-4
  - Understand: How CL differs in LLMs

- **Self-Synthesized Rehearsal (Gu et al., 2024)** [ACL 2024]
  - Read: Full paper
  - Implement: Algorithm 1

- **Parameter Collision in CL (Yang et al., 2024)** [arXiv:2410.10179]
  - Read: Sections 1-3
  - Bonus: N-LoRA implementation

**Key Skills**: Continual learning, catastrophic forgetting prevention

**Resources**:
- Wang-ML-Lab/llm-continual-learning-survey
- Multi-domain datasets

**Deliverable**: GPT with continual learning capability

---

### WEEK 14: FINAL INTEGRATION

**Goal**: Production-ready nanoGPT with continual learning

**Week 14 Projects**:

1. **Clean Implementation**
   - Refactor into modules:
     - model.py (GPT)
     - tokenizer.py (BPE)
     - train.py
     - continual.py
     - adapters.py (LoRA)
   - Documentation
   - Unit tests

2. **Advanced Features**
   - **RoPE** (Rotary Position Embeddings)
     - Replace learned embeddings
     - Better extrapolation
   - **GQA** (Grouped Query Attention)
     - Reduce KV cache
     - Use 4-8 groups
   - **Flash Attention** (conceptual)
     - Study algorithm
     - Use PyTorch's scaled_dot_product_attention

3. **Continual Learning Demo**
   - Notebook demonstrating:
     - Pre-training
     - Task A with LoRA
     - Task B with SSR
     - Task A maintained
     - Adapter switching
   - Comparison samples
   - Forgetting curves

4. **Documentation**
   - Comprehensive evaluation
   - Architecture diagram
   - Training instructions
   - Blog post

**Papers - SKIM**:
- **RoFormer (Su et al., 2021)** [arXiv:2104.09864] - Section 3
- **GQA (Ainslie et al., 2023)** [arXiv:2305.13245] - Sections 1-3
- **FlashAttention (Dao et al., 2022)** [arXiv:2205.14135] - Abstract, section 3

**Key Skills**: Production code, modern optimizations, technical writing

**Deliverables**:
- Clean codebase on GitHub
- Full demo notebook
- Blog post
- Trained checkpoints
- Evaluation results

---

## CRITICAL PAPERS

### Must Read Fully
1. **Attention is All You Need** (Vaswani et al., 2017) - Week 7
2. **LoRA** (Hu et al., 2021) - Week 12
3. **Self-Synthesized Rehearsal** (Gu et al., 2024) - Week 13

### Read Key Sections
4. **Scaling Laws** (Kaplan et al., 2020) - Week 10
5. **Chinchilla** (Hoffmann et al., 2022) - Week 10
6. **Catastrophic Forgetting in LLMs** (Luo et al., 2023) - Week 13
7. **Parameter Collision** (Yang et al., 2024) - Week 13

### Skim for Context
8. **BERT** (Devlin et al., 2018) - Week 11
9. **GPT-2** (Radford et al., 2019) - Week 11
10. **BPE** (Sennrich et al., 2016) - Week 9

### Implementation References
11. **RoPE** (Su et al., 2021) - Week 14
12. **GQA** (Ainslie et al., 2023) - Week 14
13. **FlashAttention** (Dao et al., 2022) - Week 14
14. **BatchNorm** (Ioffe & Szegedy, 2015) - Week 5

---

## CORE CONCEPTS REFERENCE

### Essential (Build These)

**AdamW**: Adam with decoupled weight decay
- Use: `torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)`
- Default optimizer for transformers

**MLP**: Multi-Layer Perceptron
- Linear → GELU → Linear → Dropout
- Hidden size usually 4x embedding size

**Attention**: Scaled Dot-Product + Multi-Head
- Formula: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- Causal masking for GPT

**Transformer Block**: Attention + MLP + LayerNorm + Residuals
- Pre-norm: `x = x + attn(LN(x))`

**Autograd**: Automatic differentiation
- Forward: build graph
- Backward: traverse, apply chain rule

**PEFT**: Parameter-Efficient Fine-Tuning (LoRA, adapters)

**SFT**: Supervised Fine-Tuning (next-token prediction)

**SOTA**: State-Of-The-Art (reference, not to chase)

### Important (Week 14)

**RoPE**: Rotary Position Embedding
- Rotate Q,K by position-dependent angle
- Better extrapolation

**GQA**: Grouped Query Attention
- Share K,V heads across groups
- Reduce KV cache

**MQA**: Multi-Query Attention
- One K,V head, many Q heads
- Fastest inference

### Acronyms

- **RLHF**: Reinforcement Learning from Human Feedback
- **DPO**: Direct Preference Optimization
- **KL**: Kullback-Leibler divergence
- **EOS/BOS**: End/Beginning of Sequence
- **KV Cache**: Cached Key/Value tensors

---

## CONTINUAL LEARNING TECHNIQUES

### Practical Approaches

**1. LoRA with Task-Specific Adapters** (Week 12-13)
- Separate LoRA per task
- Pros: Zero forgetting
- Cons: Need task ID at inference

**2. Self-Synthesized Rehearsal (SSR)** (Week 13)
- Generate from old task, mix 20-30%
- Pros: No task labels needed
- Cons: Requires generation

**3. N-LoRA** (Week 13 bonus)
- Orthogonal LoRA parameters
- Better task separation

### Metrics

1. **Backward Transfer**: Performance on old tasks after new training
2. **Forward Transfer**: Generalization to new tasks
3. **Average Accuracy**: Mean across all tasks
4. **Forgetting**: max_acc - current_acc

---

## WEEKLY REFLECTION FRAMEWORK

### Every Sunday (1 hour)

**1. Review**
- What did I build?
- Main project complete?
- Unexpected challenges?

**2. Test Understanding**
- Can I explain key concepts without references?
- Can I implement from memory?
- What would I struggle to explain?

**3. Check Dependencies**
- Did this prepare me for next week?
- Gaps to fill?
- Adjust scope?

**4. Code Quality**
- Readable and documented?
- Bugs ignored?
- Can I run from scratch?

**5. Adjust Plan**
- Ahead/behind schedule?
- What to skip/streamline?
- What deserves more depth?

### Adjustment Triggers

**If AHEAD:**
- Add bonus projects (QLoRA, N-LoRA, DPO)
- Dive deeper into papers
- Try larger scale

**If BEHIND:**
- Skip bonuses
- Use reference implementations
- Focus on core: Weeks 1,2,3,7,8,12,13,14
- Reduce model sizes

**If STUCK:**
- Join communities (r/MachineLearning, EleutherAI Discord)
- Read others' implementations
- Ask specific questions
- Solidify fundamentals

---

## RESOURCES

### Primary Video Course
**Karpathy - Neural Networks: Zero to Hero** (~16 hours)
- Lecture 1: micrograd (2:30)
- Lecture 2: makemore bigram (1:58)
- Lecture 3: makemore MLP (1:55)
- Lecture 4: makemore BatchNorm (2:18)
- Lecture 5: makemore Backprop (2:00)
- Lecture 6: makemore WaveNet (1:25)
- Lecture 7: build GPT (1:57)
- Lecture 8: Tokenizer (1:17)

### Supplementary Courses
- **Stanford CS230**: Deep Learning
- **Stanford CS231n**: CNN for Vision
- **Berkeley CS189**: Introduction to ML

### Key Websites
1. karpathy.ai/zero-to-hero.html
2. github.com/karpathy/nanoGPT
3. jalammar.github.io/illustrated-transformer/
4. nlp.seas.harvard.edu/annotated-transformer
5. huggingface.co/learn
6. ahmadosman.com/blog/learn-llms-roadmap/

### Code References
- **nanoGPT**: github.com/karpathy/nanoGPT
- **micrograd**: github.com/karpathy/micrograd
- **minbpe**: github.com/karpathy/minbpe
- **HF Transformers**: github.com/huggingface/transformers
- **HF PEFT**: github.com/huggingface/peft

### Datasets
- **TinyShakespeare**: ~1MB (fast iteration)
- **OpenWebText**: ~9GB (serious training)
- **GitHub Code**: Code fine-tuning
- **Cornell Movie Dialogues**: Dialogue

### Tools
- PyTorch, HuggingFace, wandb, TensorBoard, Jupyter, Git

### Community
- r/MachineLearning
- r/LocalLLaMA
- EleutherAI Discord
- Karpathy's GitHub Discussions
- HuggingFace Forums

---

## ALTERNATIVE ARCHITECTURES

### Jeff Hawkins' HTM (Hierarchical Temporal Memory)

**Core Idea**: Brain-inspired learning based on neocortex
- Sparse distributed representations
- Unsupervised Hebbian learning
- Temporal sequence learning
- Continual learning by design

**Differences from Transformers**:
- No backpropagation
- Sparse binary activations
- Always online learning
- Biologically constrained

**When to Consider**: Streaming data, low-power, truly continual

**Status**: Niche (anomaly detection), not mainstream

**Resource**: Numenta.com, "Thousand Brains Theory"

### Yann LeCun's JEPA (Joint Embedding Predictive Architecture)

**Core Idea**: Predict representations, not pixels/tokens
- Non-generative
- Masked region prediction in embedding space
- Energy-based models
- Self-supervised

**Differences**:
- NOT an alternative to transformers (uses transformers as backbone)
- Alternative to AUTOREGRESSIVE generation
- Predict abstract representations
- Focus on understanding vs generation

**Implementations**: I-JEPA (images), V-JEPA (video)

**Status**: Research phase (Meta AI)

**Resource**: Meta AI blog

### Why Transformers Won for LLMs
1. Scalability (parallel training)
2. Long-range dependencies (attention)
3. Transfer learning
4. Empirical success
5. Tooling ecosystem

**Focus**: Master transformers first. Explore alternatives after Day 100.

---

## MILESTONES & SUCCESS CRITERIA

### Weekly Goals
- **Week 1-2**: Train MLP >95% on MNIST
- **Week 3-4**: Generate readable sequences
- **Week 5**: 10-layer network trains successfully
- **Week 6**: Manual gradients match autograd
- **Week 7-8**: GPT generates coherent sentences
- **Week 9**: BPE reduces vocab 100x
- **Week 10**: Train 50M model without OOM
- **Week 11**: Competitive perplexity
- **Week 12**: LoRA 10x faster than full
- **Week 13**: Forgetting reduced >50% with SSR
- **Week 14**: Complete system demo works

### Day 100 Success Criteria
- [ ] Built GPT from scratch (no black boxes)
- [ ] Model generates coherent text
- [ ] LoRA fine-tuning works
- [ ] Continual learning reduces forgetting
- [ ] Can explain every component
- [ ] Clean, runnable codebase on GitHub
- [ ] Comprehensive demo notebook
- [ ] Technical blog post written

---

## QUICK REFERENCE CHEATSHEET

### Key Equations
```
Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
Cross-Entropy: L = -Σ y_i log(ŷ_i)
AdamW: θ_t = θ_{t-1} - η*(m_t/√v_t + λθ_{t-1})
LoRA: h = Wx + s·BAx  (B: d×r, A: r×d, r<<d)
Perplexity: PPL = exp(L)
```

### nanoGPT Architecture
```python
GPT(
  vocab_size=50257,
  n_layer=12,
  n_head=12,
  n_embd=768,
  block_size=1024,
  dropout=0.1
)
# Each layer: [LN -> MultiHeadAttn -> LN -> MLP]
# MLP: Linear(768->3072) -> GELU -> Linear(3072->768)
```

### Hyperparameters (GPT-2 small)
```
model_size: 124M params
n_layer: 12
n_head: 12
n_embd: 768
context: 1024
vocab: 50257
batch_size: 20
lr: 6e-4
weight_decay: 0.1
```

---

## FINAL THOUGHTS

### Why This Plan Works
1. **Top-Down**: Start with goal, decompose
2. **Project-Driven**: Build every week
3. **Progressive**: Each week builds on previous
4. **Minimal Theory**: Only what's needed
5. **Research-Backed**: Karpathy, Stanford, papers
6. **Realistic**: 14 weeks = 98 days
7. **Measurable**: Clear deliverables

### Celebration Milestones
- Week 2: "I built backprop from scratch!"
- Week 4: "My model generates text!"
- Week 8: "I built a transformer!"
- Week 12: "I can fine-tune with LoRA!"
- Week 14: "I built nanoGPT with continual learning!"

### After Day 100
- Scale up (larger models)
- Try different domains
- Implement RLHF/DPO
- Explore alternatives (JEPA, diffusion)
- Contribute to open-source
- Write blog posts
- Interview for ML roles

### Remember
**"The best way to learn is to build."** - Andrej Karpathy

You're not just learning about transformers. You're building them from scratch. By Day 100, you'll have something real that YOU created.

---

**Last Updated**: 2025-11-21
**Duration**: 100 days
**Expected Hours**: 700-800
**Final Goal**: nanoGPT with continual learning from scratch
