#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CHAIR evaluation for MSCOCO captions.

Metric definitions and object aliases follow CHAIR and FuzzyCD's Python-3
adaptation. NLTK provides tokenization and lemmatization. Cache keys contain
the evaluator version and COCO annotation-file signature.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

CACHE_VERSION = "chair_official_strict_nltk_release"

SYNONYMS_TXT = r'''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, foreigner, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, goat, ewe
cow, cattle, oxen, ox, calf, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iphone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''.strip()


DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
    "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear", "hair drier",
    "potted plant", "bow tie", "laptop computer", "stove top oven", "home plate",
    "train track",
]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]



def _safe_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _annotation_files(annotation_dir: Path, stem: str) -> list[Path]:
    val = annotation_dir / f"{stem}_val2014.json"
    train = annotation_dir / f"{stem}_train2014.json"
    if not val.exists():
        raise FileNotFoundError(f"Missing required COCO annotation: {val}")
    # For a val2014-only benchmark the validation files are sufficient.  When
    # train2014 files are available, include them exactly as the reference
    # CHAIR/FuzzyCD code does.
    return [val] + ([train] if train.exists() else [])


def annotation_signature(annotation_dir: str) -> tuple:
    root = Path(annotation_dir).expanduser().resolve()
    files: list[Path] = []
    for stem in ("instances", "captions"):
        files.extend(_annotation_files(root, stem))
    return tuple(
        (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(set(files))
    )


def ensure_nltk_resources() -> None:
    """Fail closed instead of changing metric semantics silently."""
    try:
        tokens = nltk.word_tokenize("Two people are riding bicycles.")
        nltk.pos_tag(tokens)
        WordNetLemmatizer().lemmatize("bicycles", pos="n")
    except LookupError as exc:
        raise RuntimeError(
            "Required NLTK resources are missing. Install/download punkt, "
            "punkt_tab, averaged_perceptron_tagger, "
            "averaged_perceptron_tagger_eng, wordnet and omw-1.4, then delete "
            "the old CHAIR cache before evaluating."
        ) from exc


def load_generated(path: str, image_id_key: str, caption_key: str):
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [
            json.loads(x)
            for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
    elif p.suffix == ".json":
        rows = _safe_json(p)
        if isinstance(rows, dict) and "annotations" in rows:
            rows = rows["annotations"]
    else:
        raise ValueError("Predictions must be .json or .jsonl")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No prediction rows found in {p}")

    caps: list[str] = []
    ids: list[int] = []
    for index, row in enumerate(rows):
        if image_id_key not in row:
            raise KeyError(f"{p}: row {index} missing {image_id_key!r}")
        if caption_key not in row:
            raise KeyError(f"{p}: row {index} missing {caption_key!r}")
        image_id = int(row[image_id_key])
        caption = str(row[caption_key]).strip()
        if not caption:
            raise RuntimeError(f"{p}: empty caption at row {index}, image_id={image_id}")
        ids.append(image_id)
        caps.append(caption)

    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{p}: duplicate image IDs are not allowed")
    return caps, ids


class ChairEvaluator:
    def __init__(self, annotation_dir: str):
        ensure_nltk_resources()
        self.cache_version = CACHE_VERSION
        self.annotation_dir = str(Path(annotation_dir).expanduser().resolve())
        self.annotation_signature = annotation_signature(self.annotation_dir)
        self.imid_to_objects = defaultdict(list)

        groups = [
            [x.strip() for x in line.split(",") if x.strip()]
            for line in SYNONYMS_TXT.splitlines()
        ]
        self.mscoco_objects: set[str] = set()
        self.inverse_synonym_dict: dict[str, str] = {}
        for group in groups:
            canonical = group[0]
            for item in group:
                item = item.lower()
                self.mscoco_objects.add(item)
                self.inverse_synonym_dict[item] = canonical

        self.double_word_dict = {x: x for x in DOUBLE_WORDS}
        for animal in ANIMAL_WORDS:
            self.double_word_dict[f"baby {animal}"] = animal
            self.double_word_dict[f"adult {animal}"] = animal
        for vehicle in VEHICLE_WORDS:
            self.double_word_dict[f"passenger {vehicle}"] = vehicle
        self.double_word_dict.update(
            {"bow tie": "tie", "toilet seat": "toilet", "wine glas": "wine glass"}
        )
        self._build_annotations()

    @staticmethod
    def _wordnet_pos(tag: str) -> str:
        if tag.startswith("V"):
            return "v"
        if tag.startswith("J"):
            return "a"
        if tag.startswith("R"):
            return "r"
        return "n"

    def _tokenize_lemma(self, caption: str) -> list[str]:
        text = caption.lower().replace("’", "'")
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        return [
            lemmatizer.lemmatize(word, pos=self._wordnet_pos(tag))
            for word, tag in tags
        ]

    def caption_to_words(self, caption: str):
        words = self._tokenize_lemma(caption)
        merged: list[str] = []
        idxs: list[int] = []
        i = 0
        while i < len(words):
            idxs.append(i)
            double = " ".join(words[i : i + 2])
            if double in self.double_word_dict:
                merged.append(self.double_word_dict[double])
                i += 2
            else:
                merged.append(words[i])
                i += 1

        if "toilet" in merged and "seat" in merged:
            merged = [x for x in merged if x != "seat"]

        kept_pairs = [(j, w) for j, w in enumerate(merged) if w in self.mscoco_objects]
        object_words = [w for _, w in kept_pairs]
        node_words = [self.inverse_synonym_dict[w] for w in object_words]
        object_idxs = [idxs[j] if j < len(idxs) else j for j, _ in kept_pairs]
        return object_words, node_words, object_idxs, merged

    def _build_annotations(self) -> None:
        ann_dir = Path(self.annotation_dir)
        for path in _annotation_files(ann_dir, "instances"):
            data = _safe_json(path)
            id_to_name = {
                int(x["id"]): str(x["name"]).lower()
                for x in data["categories"]
            }
            for ann in tqdm(data["annotations"], desc=f"CHAIR instances {path.name}"):
                name = id_to_name[int(ann["category_id"])]
                canonical = self.inverse_synonym_dict.get(name, name)
                self.imid_to_objects[int(ann["image_id"])].append(canonical)

        for path in _annotation_files(ann_dir, "captions"):
            data = _safe_json(path)
            for ann in tqdm(data["annotations"], desc=f"CHAIR captions {path.name}"):
                _, nodes, _, _ = self.caption_to_words(str(ann["caption"]))
                self.imid_to_objects[int(ann["image_id"])].extend(nodes)

        for image_id in list(self.imid_to_objects):
            self.imid_to_objects[image_id] = set(self.imid_to_objects[image_id])

    def compute(self, cap_file: str, image_id_key="image_id", caption_key="caption"):
        caps, ids = load_generated(cap_file, image_id_key, caption_key)
        num_caps = num_hall_caps = hall_count = object_count = 0
        recall_hits = recall_total = total_len = 0
        details = []

        for caption, image_id in tqdm(list(zip(caps, ids)), desc="CHAIR evaluate"):
            if image_id not in self.imid_to_objects:
                raise KeyError(f"Image id {image_id} has no COCO ground-truth annotations")
            words, nodes, idxs, raw = self.caption_to_words(caption)
            gt = self.imid_to_objects[image_id]
            hallucinations = []
            hall_idxs = []
            recalled = set()
            for word, node, idx in zip(words, nodes, idxs):
                if node not in gt:
                    hallucinations.append((word, node))
                    hall_idxs.append(idx)
                else:
                    recalled.add(node)

            num_caps += 1
            object_count += len(nodes)
            hall_count += len(hallucinations)
            total_len += len(raw)
            if hallucinations:
                num_hall_caps += 1
            recall_hits += len(recalled)
            recall_total += len(gt)
            details.append(
                {
                    "image_id": image_id,
                    "caption": caption,
                    "mscoco_hallucinated_words": hallucinations,
                    "mscoco_gt_words": sorted(gt),
                    "mscoco_generated_words": nodes,
                    "hallucination_idxs": hall_idxs,
                    "metrics": {
                        "CHAIRs": int(bool(hallucinations)),
                        "CHAIRi": len(hallucinations) / len(nodes) if nodes else 0.0,
                        "Recall": len(recalled) / len(gt) if gt else 0.0,
                        "Len": len(raw),
                    },
                }
            )

        chair_s = num_hall_caps / num_caps if num_caps else 0.0
        chair_i = hall_count / object_count if object_count else 0.0
        recall = recall_hits / recall_total if recall_total else 0.0
        denom = (1 - chair_i) + recall
        f1 = 2 * (1 - chair_i) * recall / denom if denom else 0.0
        return {
            "overall_metrics": {
                "CHAIRs": chair_s,
                "CHAIRi": chair_i,
                "Recall": recall,
                "F1": f1,
                "AvgLen": total_len / num_caps if num_caps else 0.0,
                "NumCaptions": num_caps,
                "NumObjectMentions": object_count,
                "NumHallucinatedMentions": hall_count,
                "EvaluatorVersion": CACHE_VERSION,
            },
            "sentences": details,
        }


def load_or_build(annotation_dir: str, cache: str = "") -> ChairEvaluator:
    ensure_nltk_resources()
    resolved = str(Path(annotation_dir).expanduser().resolve())
    signature = annotation_signature(resolved)
    cache_path = Path(cache).expanduser() if cache else None

    if cache_path and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                obj = pickle.load(f)
            valid = (
                getattr(obj, "cache_version", None) == CACHE_VERSION
                and getattr(obj, "annotation_dir", "") == resolved
                and getattr(obj, "annotation_signature", None) == signature
            )
            if valid:
                print(f"loaded CHAIR cache: {cache_path}")
                return obj
            print(f"[CACHE INVALID] rebuilding {cache_path}")
        except Exception as exc:
            print(f"[CACHE READ WARNING] {type(exc).__name__}: {exc}; rebuilding")

    evaluator = ChairEvaluator(resolved)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(evaluator, f)
        print(f"saved CHAIR cache: {cache_path}")
    return evaluator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cap-file", required=True)
    p.add_argument("--annotation-dir", required=True)
    p.add_argument("--image-id-key", default="image_id")
    p.add_argument("--caption-key", default="caption")
    p.add_argument("--cache", default="output_chair/cache/chair.pkl")
    p.add_argument("--save-path", default="")
    args = p.parse_args()

    evaluator = load_or_build(args.annotation_dir, args.cache)
    result = evaluator.compute(args.cap_file, args.image_id_key, args.caption_key)
    for key, value in result["overall_metrics"].items():
        if key in {"CHAIRs", "CHAIRi", "Recall", "F1"}:
            print(f"{key:24s}: {100 * value:.3f}")
        else:
            print(f"{key:24s}: {value}")
    if args.save_path:
        out = Path(args.save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
