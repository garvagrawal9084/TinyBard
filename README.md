# TinyBard

A 2.8M-parameter, character-level language model built from scratch in PyTorch, using the same building blocks as modern Llama-family models. Trained on Tiny Shakespeare, it learns to write Shakespeare-flavored text one character at a time.

This is a learning project: every component is written by hand in a single notebook so you can read the whole model top to bottom.

## What's inside

TinyBard is a decoder-only transformer that replaces the classic GPT components with their modern upgrades:

| Component | Classic | TinyBard |
|---|---|---|
| Normalization | LayerNorm | **RMSNorm**: no mean subtraction or shift, just a learned scale |
| Positional info | Sinusoidal / learned absolute | **RoPE**: rotates Q and K so attention depends on relative distance |
| Attention | Multi-head | **Grouped-query attention (GQA)**: 8 query heads share 2 KV heads (4:1) |
| Feed-forward | GELU MLP | **SwiGLU**: gated feed-forward with SiLU activation |
| Embeddings | Separate in/out | **Tied** input embedding and output head |

Other details: pre-norm residual blocks, causal masking, dropout (0.2) on attention weights and the feed-forward output, and gradient clipping at 1.0.

## Model configuration

| Setting | Value |
|---|---|
| Vocabulary | 65 characters |
| Embedding dim (`d_model`) | 256 |
| Layers | 4 |
| Query heads / KV heads | 8 / 2 |
| Head dim | 32 |
| FFN hidden dim | 680 |
| Context length | 256 characters |
| Parameters | 2,764,288 (~11.1 MB in float32) |

## Training

| Setting | Value |
|---|---|
| Dataset | [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (1,115,394 characters) |
| Split | 90% train (1,003,854) / 10% validation (111,540) |
| Optimizer | Adam, lr = 3e-4 |
| Batch size | 64 sequences of 256 characters |
| Steps | 3,000 |
| Hardware | Apple Silicon GPU (MPS) |
| Wall-clock time | ~81 minutes |

### Results

| Step | Train loss | Val loss |
|---|---|---|
| 250 | 2.5316 | 2.5613 |
| 500 | 2.3426 | 2.3638 |
| 1000 | 2.0912 | 2.1544 |
| 1500 | 1.9050 | 2.0205 |
| 2000 | 1.7817 | 1.9205 |
| 2500 | 1.6814 | 1.8490 |
| 3000 | **1.6161** | **1.8022** |

For reference, random guessing over 65 characters gives a loss of ln(65) ≈ 4.17. The widening gap between train and validation loss suggests the model is starting to overfit the small dataset, so more steps alone would give diminishing returns.

## Sample output

Prompt: `ROMEO:`, temperature 0.8

```
ROMEO:
Here good thou with deen his face the dies.

DUKE OF YORK:
Constle consection beother my word.

ROMEO:
That's the well thus fame the leave his loud,
```

The model picks up the play format (speaker names in capitals, colons, line breaks, verse-like lines) and many real words, but it is far too small to produce coherent sentences. That is expected at this scale.

### Temperature

Generation samples from the softmax of `logits / temperature`:

- **0.5**: conservative and repetitive
- **0.8**: balanced
- **1.0 to 1.2**: creative, increasingly chaotic

## Getting started

### Requirements

- Python 3.12+ (the notebook uses nested quotes inside f-strings)
- `torch`
- `matplotlib`
- Jupyter, or an IDE with notebook support such as PyCharm

```bash
pip install torch matplotlib jupyter
```

### Run it

1. Open `main.ipynb`.
2. Run all cells from top to bottom. The Tiny Shakespeare text is downloaded automatically to `shakespeare.txt` on first run.
3. Training prints a loss log every 50 steps and evaluates every 250 steps.
4. The last cells plot the loss curve and generate text at several temperatures.

The notebook uses the Apple Silicon GPU (`mps`) when available and falls back to CPU otherwise. To train on an NVIDIA GPU, add a `torch.cuda.is_available()` check to the device cell. Expect training on CPU to be much slower than the ~81 minutes seen on MPS.

## Notebook walkthrough

1. **Setup**: device selection and data download
2. **Character-level tokenization**: vocabulary, `encode` / `decode`, train/validation split, batching
3. **RMSNorm**
4. **Rotary positional embeddings (RoPE)**
5. **Grouped-query attention**: with `repeat_kv` to expand KV heads
6. **SwiGLU** feed-forward layer
7. **Transformer block** and the full **MiniLLM** model
8. **Training loop**, loss evaluation, and loss-curve plot
9. **Text generation** with temperature sampling

## Ideas for next steps

- Use a smaller weight init (the current std of 0.2 makes the starting loss about 8.0, well above random guessing) and compare convergence
- Switch to AdamW with a learning-rate warmup and cosine decay
- Add a KV cache to speed up generation
- Replace the manual attention with `F.scaled_dot_product_attention`
- Try a byte-pair-encoding tokenizer instead of characters
- Save and load checkpoints so training can resume
- Scale up the model or train on a larger corpus

## Acknowledgments

- Dataset: Tiny Shakespeare, from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
- Architecture inspired by the Llama family of models (RMSNorm, RoPE, GQA, SwiGLU). TinyBard is an independent educational implementation and is not affiliated with Meta.
