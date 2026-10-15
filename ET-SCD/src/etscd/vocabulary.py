"""Build object-token sets and canonical target mappings from a fixed lexicon.

Retain complete single-token aliases with a matching tokenizer round-trip;
record the remaining aliases in the vocabulary metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build_object_map(tokenizer, concepts):
    tokens, skipped = {}, []
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for canonical, aliases in concepts.items():
        for alias in aliases:
            found = False
            for spelling in (alias, " " + alias, alias.title(), " " + alias.title()):
                ids = tokenizer.encode(spelling, add_special_tokens=False)
                if len(ids) != 1 or int(ids[0]) in special:
                    continue
                identity = int(ids[0])
                decoded = tokenizer.decode([identity], skip_special_tokens=False).strip().casefold()
                if decoded != alias.casefold():
                    continue
                if identity in tokens and tokens[identity] != canonical:
                    raise ValueError(f"Object alias collision for token {identity}")
                tokens[identity] = canonical
                found = True
            if not found:
                skipped.append(alias)
    if not tokens:
        raise ValueError("Lexicon produces no complete single-token object aliases")
    return tokens, sorted(set(skipped))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-path", required=True, help="Same tokenizer as the model; for LAVIS use its Vicuna tokenizer directory")
    p.add_argument("--qwen", action="store_true", help="Use the local Qwen-VL-Chat tokenizer's custom code")
    p.add_argument("--lexicon", default="data/coco_object_concepts.json")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error("Output already exists; choose a fresh output path")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=args.qwen,
                                               local_files_only=True, use_fast=False)
    lexicon = Path(args.lexicon).read_bytes()
    tokens, skipped = build_object_map(tokenizer, json.loads(lexicon))
    result = {"tokens": {str(k): v for k, v in sorted(tokens.items())},
              "metadata": {"tokenizer_path": str(Path(args.tokenizer_path).resolve()),
                           "tokenizer_class": type(tokenizer).__name__,
                           "lexicon_sha256": hashlib.sha256(lexicon).hexdigest(),
                           "convention": "complete single-token object aliases only",
                           "skipped_multi_token_or_unavailable_aliases": skipped}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(tokens)} object tokens; {len(skipped)} aliases unavailable as a complete single token")


if __name__ == "__main__":
    main()
