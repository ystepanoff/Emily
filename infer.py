#!/usr/bin/env python3
"""Generate a random poem in the style of Emily Dickinson."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch

from train import ModelConfig, TransformerLanguageModel, resolve_device


class Vocabulary:
    def __init__(self, stoi: Dict[str, int], itos: Dict[int, str]) -> None:
        if not stoi or not itos:
            raise ValueError("Checkpoint is missing vocabulary mappings.")
        self.stoi = dict(stoi)
        self.itos = {int(k): v for k, v in itos.items()}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def random_char(self) -> str:
        return random.choice(list(self.stoi.keys()))

    def contains(self, ch: str) -> bool:
        return ch in self.stoi


def sanitise_seed(seed: Optional[str], fallback: str, vocab: Vocabulary) -> str:
    if seed:
        candidate = seed
    else:
        source = fallback or vocab.random_char()
        window = min(len(source), 32)
        start = random.randint(0, max(0, len(source) - window))
        candidate = source[start : start + window]
    filtered = "".join(ch for ch in candidate if vocab.contains(ch))
    if not filtered:
        filtered = vocab.random_char()
    return filtered


def generate_poem(
    model: TransformerLanguageModel,
    vocab: Vocabulary,
    device: torch.device,
    seed_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
) -> str:
    encoded = vocab.encode(seed_text)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return vocab.decode(output_ids[0].tolist())


def format_poem(
    generated: str,
    prompt: str,
    delimiter: Optional[str],
    keep_prompt: bool,
    keep_delimiter: bool,
) -> str:
    body = generated
    if not keep_prompt and generated.startswith(prompt):
        body = generated[len(prompt) :]
    if delimiter and not keep_delimiter and delimiter in body:
        body = body.split(delimiter)[0]
    cleaned = body.strip()
    return cleaned if cleaned else generated.strip()


def build_model(checkpoint: Dict[str, object], device: torch.device) -> tuple[TransformerLanguageModel, Vocabulary, Dict[str, object]]:
    config = ModelConfig(**checkpoint["config"])
    model = TransformerLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    metadata = checkpoint.get("metadata") or {}
    vocab = Vocabulary(metadata.get("stoi", {}), metadata.get("itos", {}))
    return model, vocab, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Dickinson-style poems from a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dickinson_transformer.pt"))
    parser.add_argument("--max_tokens", type=int, default=320, help="Number of characters to sample beyond the prompt.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None, help="Restrict sampling to the top-K logits (optional).")
    parser.add_argument("--seed_text", type=str, default=None, help="Starting text for generation. Defaults to a random snippet.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--poem_delimiter", type=str, default="<END>")
    parser.add_argument("--keep_prompt", action="store_true", help="Show the seed text in the final output.")
    parser.add_argument("--keep_delimiter", action="store_true", help="Preserve the delimiter in the output text.")
    parser.add_argument("--num_poems", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13, help="Random seed controlling sampling randomness.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    #random.seed(args.seed)
    #torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, vocab, metadata = build_model(checkpoint, device)
    default_seed = metadata.get("default_seed", "\n")
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    for idx in range(args.num_poems):
        prompt = sanitise_seed(args.seed_text, default_seed, vocab)
        generated = generate_poem(
            model,
            vocab,
            device,
            seed_text=prompt,
            max_new_tokens=max(1, args.max_tokens),
            temperature=max(args.temperature, 1e-3),
            top_k=top_k,
        )
        poem = format_poem(
            generated,
            prompt=prompt,
            delimiter=args.poem_delimiter,
            keep_prompt=args.keep_prompt,
            keep_delimiter=args.keep_delimiter,
        )
        header = f"--- Poem {idx + 1} ---" if args.num_poems > 1 else None
        if header:
            print(header)
        print(poem.strip())
        if args.num_poems > 1:
            print()


if __name__ == "__main__":
    main()
