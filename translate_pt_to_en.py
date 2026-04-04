import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_TRAIN = "/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/train.csv"
DEFAULT_TEST = "/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/test.csv"
DEFAULT_OUT_TRAIN = "/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/train_en.csv"
DEFAULT_OUT_TEST = "/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/test_en.csv"

FALLBACK_MODELS = [
    "Helsinki-NLP/opus-mt-ROMANCE-en",
    "facebook/nllb-200-distilled-600M",
    "facebook/m2m100_418M",
]


def split_text_for_translation(text: str, max_chars: int = 1200) -> List[str]:
    """Simple chunking by paragraph/sentence boundaries to avoid truncation."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = []
    current_len = 0

    # First split by newline, then keep punctuation naturally inside chunks.
    parts = text.split("\n")
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # +1 for joining newline/space.
        if current_len + len(part) + 1 <= max_chars:
            current.append(part)
            current_len += len(part) + 1
        else:
            if current:
                chunks.append("\n".join(current))
            # If one part is still too long, hard-cut it.
            if len(part) > max_chars:
                for i in range(0, len(part), max_chars):
                    chunks.append(part[i : i + max_chars])
                current = []
                current_len = 0
            else:
                current = [part]
                current_len = len(part)

    if current:
        chunks.append("\n".join(current))

    return chunks if chunks else [text]


@torch.no_grad()
def translate_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    max_input_tokens: int = 512,
    max_output_tokens: int = 512,
) -> List[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)

    generated = model.generate(
        **inputs,
        max_length=max_output_tokens,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def translate_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    batch_size: int,
    cache: Dict[str, str],
) -> List[str]:
    # Work on unique texts for speed.
    unique_texts = list(dict.fromkeys(texts))

    to_translate = [t for t in unique_texts if t not in cache]

    if to_translate:
        print(f"Translating {len(to_translate)} unique texts...")

    for i in tqdm(range(0, len(to_translate), batch_size), desc="Batches"):
        batch_texts = to_translate[i : i + batch_size]

        expanded_inputs = []
        mapping = []  # (original_text, start_idx, end_idx)

        # Chunk long texts to reduce truncation.
        for original in batch_texts:
            chunks = split_text_for_translation(original)
            start = len(expanded_inputs)
            expanded_inputs.extend(chunks)
            end = len(expanded_inputs)
            mapping.append((original, start, end))

        translated_chunks = translate_batch(
            expanded_inputs,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        for original, start, end in mapping:
            merged = "\n".join(translated_chunks[start:end]).strip()
            cache[original] = merged

    return [cache[t] for t in texts]


def normalize_report_column(df: pd.DataFrame, report_col: str) -> pd.Series:
    col = df[report_col].fillna("").astype(str)
    return col


def _from_pretrained_with_token(loader_fn, model_name: str, hf_token: str | None):
    if hf_token:
        try:
            return loader_fn(model_name, token=hf_token)
        except TypeError:
            return loader_fn(model_name, use_auth_token=hf_token)
    return loader_fn(model_name)


def load_translation_model(model_name: str, hf_token: str | None):
    candidates = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    last_err = None

    for cand in candidates:
        try:
            tokenizer = _from_pretrained_with_token(AutoTokenizer.from_pretrained, cand, hf_token)
            model = _from_pretrained_with_token(AutoModelForSeq2SeqLM.from_pretrained, cand, hf_token)
            print(f"Using translation model: {cand}")
            return tokenizer, model
        except Exception as e:
            print(f"Warning: failed to load '{cand}': {e}")
            last_err = e

    raise RuntimeError("Unable to load any translation model.") from last_err


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Portuguese report CSV files to English")
    parser.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN)
    parser.add_argument("--test_csv", type=str, default=DEFAULT_TEST)
    parser.add_argument("--out_train_csv", type=str, default=DEFAULT_OUT_TRAIN)
    parser.add_argument("--out_test_csv", type=str, default=DEFAULT_OUT_TEST)
    parser.add_argument("--report_col", type=str, default="report")

    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-ROMANCE-en")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token. Prefer env var HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )

    parser.add_argument(
        "--cache_path",
        type=str,
        default="/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/translation_cache_pt_en.json",
    )

    args = parser.parse_args()

    print("Loading CSV files...")
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    if args.report_col not in train_df.columns or args.report_col not in test_df.columns:
        raise ValueError(
            f"Column '{args.report_col}' must exist in both files. "
            f"Train columns={list(train_df.columns)}, test columns={list(test_df.columns)}"
        )

    train_texts = normalize_report_column(train_df, args.report_col).tolist()
    test_texts = normalize_report_column(test_df, args.report_col).tolist()

    print(f"Loading translation model: {args.model_name}")
    tokenizer, model = load_translation_model(args.model_name, args.hf_token)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Load translation cache if available.
    cache: Dict[str, str] = {}
    if os.path.exists(args.cache_path):
        try:
            with open(args.cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"Loaded cache with {len(cache)} entries: {args.cache_path}")
        except Exception as e:
            print(f"Warning: failed to load cache, starting fresh. Error: {e}")

    print("\nTranslating train set...")
    train_en = translate_texts(
        train_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        cache=cache,
    )

    print("\nTranslating test set...")
    test_en = translate_texts(
        test_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        cache=cache,
    )

    train_df[args.report_col] = train_en
    test_df[args.report_col] = test_en

    os.makedirs(os.path.dirname(args.out_train_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_test_csv), exist_ok=True)

    train_df.to_csv(args.out_train_csv, index=False)
    test_df.to_csv(args.out_test_csv, index=False)

    with open(args.cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    print(f"\nSaved translated train CSV: {args.out_train_csv}")
    print(f"Saved translated test CSV:  {args.out_test_csv}")
    print(f"Saved translation cache:    {args.cache_path}")


if __name__ == "__main__":
    main()
