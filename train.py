#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


class CharacterDataset(Dataset):
    def __init__(self, text: str, block_size: int) -> None:
        if block_size < 2:
            raise ValueError("max_seq_len/block_size must be at least 2.")
        cleaned = text.replace("\r\n", "\n")
        self.text = cleaned
        self.block_size = block_size
        self.chars = sorted(set(cleaned))
        self.stoi: Dict[str, int] = {ch: idx for idx, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {idx: ch for idx, ch in enumerate(self.chars)}
        encoded = [self.stoi[ch] for ch in cleaned]
        self.data = torch.tensor(encoded, dtype=torch.long)
        if len(self.data) <= self.block_size:
            raise ValueError(
                f"Corpus is too small ({len(self.data)} tokens) for block size {self.block_size}."
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def token_count(self) -> int:
        return len(self.data)

    def encode(self, text: str) -> List[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            missing = repr(exc.args[0])
            raise ValueError(f"Character {missing} is not in the training vocabulary.") from exc

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def random_prefix(self, max_chars: int) -> str:
        prefix_len = max(1, min(max_chars, len(self.data) - 1))
        upper_bound = len(self.data) - prefix_len - 1
        start = random.randint(0, max(0, upper_bound))
        ids = self.data[start : start + prefix_len].tolist()
        return self.decode(ids)


default_seed_length = 64


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    dim_feedforward: int
    max_seq_len: int
    dropout: float


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.head_dim = config.d_model // config.n_heads
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.proj_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer(
            "mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self.mask[:, :, :seq_len, :seq_len]
        weights = weights.masked_fill(causal_mask == 0, float("-inf"))
        probs = torch.softmax(weights, dim=-1)
        probs = self.attn_dropout(probs)
        context = probs @ v
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        out = self.proj(context)
        return self.proj_dropout(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ff = FeedForward(config.d_model, config.dim_feedforward, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}."
            )
        token_emb = self.token_embedding(idx)
        pos_emb = self.positional_embedding[:, :seq_len, :]
        x = self.dropout(token_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        output = idx
        temp = max(temperature, 1e-3)
        for _ in range(max(0, max_new_tokens)):
            idx_cond = output[:, -self.config.max_seq_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temp
            if top_k is not None and 0 < top_k < logits.size(-1):
                values, indices = torch.topk(logits, top_k)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            output = torch.cat([output, next_token], dim=1)
        return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str) -> torch.device:
    name = preferred.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def run_epoch(
    model: TransformerLanguageModel,
    loader: DataLoader,
    device: torch.device,
    optimiser: Optional[torch.optim.Optimizer],
    grad_clip: float,
) -> float:
    training = optimiser is not None
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    for batch, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        if training:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def generate_text(
    model: TransformerLanguageModel,
    dataset: CharacterDataset,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int] = None,
    prompt: Optional[str] = None,
) -> str:
    prefix = prompt or dataset.random_prefix(max(24, model.config.max_seq_len // 3))
    encoded = dataset.encode(prefix)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return dataset.decode(output_ids[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer on Emily Dickinson poems")
    parser.add_argument("--data_path", type=Path, default=Path("data/dickinson_clean.txt"))
    parser.add_argument("--output_path", type=Path, default=Path("artifacts/dickinson_transformer.pt"))
    parser.add_argument("--vocab_size", type=int, default=None, help="Optional safety cap; derived from data by default.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sample_every", type=int, default=2, help="Preview interval in epochs.")
    parser.add_argument("--sample_tokens", type=int, default=256)
    parser.add_argument("--sample_top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    #set_seed(args.seed)
    device = resolve_device(args.device)
    text = Path(args.data_path).read_text(encoding="utf-8")
    dataset = CharacterDataset(text, block_size=args.max_seq_len)
    if args.vocab_size is not None and dataset.vocab_size > args.vocab_size:
        raise ValueError(
            f"Corpus has vocab size {dataset.vocab_size}, which exceeds --vocab_size={args.vocab_size}."
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    config = ModelConfig(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    model = TransformerLanguageModel(config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(
        f"Loaded {dataset.token_count:,} tokens | vocab={dataset.vocab_size} | steps/epoch={len(loader)}"
    )

    best_loss = float("inf")
    last_loss: Optional[float] = None
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        avg_loss = run_epoch(
            model=model,
            loader=loader,
            device=device,
            optimiser=optimiser,
            grad_clip=args.grad_clip,
        )
        elapsed = time.time() - start
        ppl = math.exp(min(20.0, avg_loss))
        best_loss = min(best_loss, avg_loss)
        last_loss = avg_loss
        print(
            f"Epoch {epoch:02d}/{args.num_epochs} | loss={avg_loss:.4f} | ppl={ppl:.2f} | {elapsed:.1f}s"
        )
        if args.sample_every > 0 and (epoch % args.sample_every == 0 or epoch == 1):
            sample = generate_text(
                model,
                dataset,
                device,
                max_new_tokens=args.sample_tokens,
                temperature=args.temperature,
                top_k=args.sample_top_k,
            )
            poem_preview = sample.split("<END>")[0].strip()
            print("---- Sample ----")
            print(poem_preview)
            print("----------------")

    checkpoint_path = Path(args.output_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu")
    checkpoint = {
        "config": asdict(config),
        "model_state_dict": model_cpu.state_dict(),
        "metadata": {
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            "block_size": dataset.block_size,
            "default_seed": dataset.text[: min(default_seed_length, len(dataset.text))],
        },
        "stats": {"final_loss": last_loss, "best_loss": best_loss},
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main(parse_args())
