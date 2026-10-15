"""Backbone adapters returning raw, full-vocabulary next-token logits.

The caller supplies an instruction and an exact generated-token prefix. Every
view uses the same instruction and token IDs. Each call recomputes the complete
context with ``use_cache=False``. ``visual_input_size`` gives the native visual
input dimensions as ``(width, height)`` for resizing regional views.

NumPy and Pillow are imported eagerly. Load each backbone in its own environment:
PyTorch and the local LLaVA-1.5 package; PyTorch/Transformers and a complete local
Qwen-VL-Chat ``trust_remote_code`` checkpoint; or PyTorch/LAVIS configured for
``blip2_vicuna_instruct`` / ``vicuna7b``.

InstructBLIP ``model_path`` selects a LAVIS checkpoint file and ``llm_model``
selects the local Vicuna directory. LAVIS builds the architecture from its
installed YAML with these paths, and the adapter records the resolved config.
Qwen uses its checkpoint's native ChatML ``make_context`` helper and a stable
local PNG path for view images. Backend instances are single-threaded.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)


class LogitBackend(Protocol):
    tokenizer: Any
    eos_ids: set[int]
    visual_input_size: tuple[int, int]

    def next_logits(self, image: Image.Image, question: str,
                    prefix: tuple[int, ...]) -> np.ndarray: ...

    def decode(self, token_ids: Iterable[int]) -> str: ...

    def synchronize(self) -> None: ...

    def memory_bytes(self) -> int | None: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BinaryTokenPair:
    """Complete single-token spellings for the binary answers."""

    yes_id: int
    no_id: int
    yes_text: str
    no_text: str

    @property
    def affirmative_id(self) -> int:
        return self.yes_id

    @property
    def negative_id(self) -> int:
        return self.no_id

    def __iter__(self):
        yield self.yes_id
        yield self.no_id

    def metadata(self) -> dict[str, Any]:
        return dict(yes_id=self.yes_id, no_id=self.no_id,
                    yes_text=self.yes_text, no_text=self.no_text)


def resolve_binary_token_ids(tokenizer: Any) -> BinaryTokenPair:
    """Choose disjoint IDs for verified single-token Yes/No spellings.

    Preference is deterministic and independent of labels or model scores.
    Multi-token encodings are excluded in full; taking their first token would
    accidentally count whitespace or shared word prefixes as an answer.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or [])

    def candidates(word: str):
        for text in (word, " " + word, word.lower(), " " + word.lower(),
                     "\n" + word, "\n" + word.lower()):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1 or int(ids[0]) in special:
                continue
            value = int(ids[0])
            decoded = tokenizer.decode([value], skip_special_tokens=False)
            if decoded.strip().casefold() == word.casefold():
                yield text, value

    yes = list(candidates("Yes"))
    no = list(candidates("No"))
    for yes_text, yes_id in yes:
        for no_text, no_id in no:
            if yes_id != no_id:
                return BinaryTokenPair(yes_id, no_id, yes_text, no_text)
    raise ValueError("Tokenizer has no disjoint complete single-token Yes/No "
                     "spellings; binary evaluation must not use first-token "
                     "approximations. Use a supported tokenizer or an explicit "
                     "sequence-scoring protocol.")


def _eos_ids(*values: Any) -> set[int]:
    result: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(v) for v in value if v is not None)
        else:
            result.add(int(value))
    return result


def resolve_visual_size(*configurations: Any) -> tuple[int, int]:
    """Resolve native image sizes to ``(width, height)``.

    Native scalar sizes and ``shortest_edge`` mappings denote square inputs.
    Native two-element sequences use the processor convention ``(height, width)``;
    mappings may specify ``height`` and ``width`` explicitly. Candidates are tried
    in order, allowing model configuration to supplement processor metadata.
    """
    def positive_integer(value: Any) -> int | None:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(value) if int(value) > 0 else None
        return None

    def parse(value: Any) -> tuple[int, int] | None:
        scalar = positive_integer(value)
        if scalar is not None:
            return scalar, scalar
        if isinstance(value, Mapping):
            height = positive_integer(value.get("height"))
            width = positive_integer(value.get("width"))
            if height is not None and width is not None:
                return width, height
            edge = positive_integer(value.get("shortest_edge"))
            if edge is not None:
                return edge, edge
        if isinstance(value, (list, tuple)) and len(value) == 2:
            height, width = (positive_integer(dimension) for dimension in value)
            if height is not None and width is not None:
                return width, height
        return None

    for configuration in configurations:
        resolved = parse(configuration)
        if resolved is not None:
            return resolved
    raise ValueError("Cannot resolve native visual input size from the processor "
                     "or model configuration; expected a positive image size, "
                     "(height, width), or a height/width mapping")


def _configuration_value(configuration: Any, name: str) -> Any:
    if isinstance(configuration, Mapping):
        return configuration.get(name)
    return getattr(configuration, name, None)


def _resize_sizes(processor: Any) -> list[Any]:
    """Collect image sizes from a processor's torchvision Resize transforms."""
    sizes: list[Any] = []
    transforms = getattr(processor, "transform", None)
    if transforms is None:
        return sizes
    pending = [transforms]
    while pending:
        transform = pending.pop(0)
        if type(transform).__name__ == "Resize":
            sizes.append(getattr(transform, "size", None))
        children = getattr(transform, "transforms", None)
        if children is not None:
            pending.extend(children)
    return sizes


def _append_prefix(torch: Any, prompt_ids: Any, prefix: tuple[int, ...]):
    if not prefix:
        return prompt_ids
    if any(not isinstance(v, (int, np.integer)) or int(v) < 0 for v in prefix):
        raise ValueError("prefix must contain nonnegative token IDs")
    suffix = torch.tensor([list(prefix)], dtype=torch.long, device=prompt_ids.device)
    return torch.cat([prompt_ids, suffix], dim=1)


class _TorchBackend:
    def _finish(self, family: str, model_path: str | None,
                settings: dict[str, Any]) -> None:
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self._metadata = dict(family=family, model_path=model_path,
                              device=str(self.device),
                              visual_input_size=list(self.visual_input_size),
                              cache_mode="full_context_recompute",
                              output="full_vocabulary_raw_logits",
                              memory_measurement="process_peak_allocated_cuda_bytes",
                              settings=settings)
        versions = {}
        for package in ("torch", "transformers", "numpy", "Pillow", "salesforce-lavis", "llava"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        self._metadata["package_versions"] = versions
        LOG.info("Resolved backbone: %s", self._metadata)

    @staticmethod
    def _logits(output: Any) -> np.ndarray:
        logits = output.logits[0, -1, :].detach().float().cpu().numpy().copy()
        if logits.ndim != 1 or logits.size < 2 or not np.isfinite(logits).all():
            raise RuntimeError("Backbone must return finite full-vocabulary logits")
        return logits

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)

    def memory_bytes(self) -> int | None:
        if self.device.type == "cuda":
            return int(self._torch.cuda.max_memory_allocated(self.device))
        return None

    def metadata(self) -> dict[str, Any]:
        # Return an independent JSON-compatible structure for run manifests.
        import copy
        return copy.deepcopy(self._metadata)


class LlavaBackend(_TorchBackend):
    """Official LLaVA-1.5 conversation and image-token interfaces."""

    def __init__(self, model_path: str, device: str = "cuda", *,
                 model_base: str | None = None, conv_mode: str = "llava_v1"):
        import torch
        from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                                     DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
        from llava.conversation import conv_templates
        from llava.mm_utils import (get_model_name_from_path, process_images,
                                   tokenizer_image_token)
        from llava.model.builder import load_pretrained_model

        self._torch = torch
        self._tokens = (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                        DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX)
        self._conversation = conv_templates[conv_mode]
        self._process_images = process_images
        self._tokenize_image = tokenizer_image_token
        self.tokenizer, self.model, self.processor, context = load_pretrained_model(
            model_path, model_base, get_model_name_from_path(model_path),
            device_map=device, device=device)
        self.visual_input_size = resolve_visual_size(
            getattr(self.processor, "crop_size", None),
            getattr(self.processor, "size", None),
            _configuration_value(getattr(self.model.config, "vision_config", None),
                                 "image_size"))
        self.eos_ids = _eos_ids(self.tokenizer.eos_token_id,
                               getattr(self.model.generation_config, "eos_token_id", None))
        self._finish("llava-1.5", model_path,
                     dict(model_base=model_base, conv_mode=conv_mode,
                          context_length=int(context)))

    def _prompt(self, question: str) -> str:
        image_token, start, end, _ = self._tokens
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = start + image_token + end
        conv = self._conversation.copy()
        conv.append_message(conv.roles[0], image_token + "\n" + question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def next_logits(self, image: Image.Image, question: str,
                    prefix: tuple[int, ...] = ()) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            ids = self._tokenize_image(self._prompt(question), self.tokenizer,
                                      self._tokens[3], return_tensors="pt")
            ids = _append_prefix(torch, ids.unsqueeze(0).to(self.device), prefix)
            pixels = self._process_images([image.convert("RGB")], self.processor,
                                         self.model.config)
            dtype = next(self.model.parameters()).dtype
            if isinstance(pixels, list):
                pixels = [v.to(device=self.device, dtype=dtype) for v in pixels]
            else:
                pixels = pixels.to(device=self.device, dtype=dtype)
            output = self.model(input_ids=ids, images=pixels,
                                attention_mask=torch.ones_like(ids),
                                use_cache=False, return_dict=True,
                                output_attentions=False, output_hidden_states=False)
            return self._logits(output)


class QwenVLBackend(_TorchBackend):
    """Stock local Qwen-VL-Chat checkpoint with native ChatML context."""

    def __init__(self, model_path: str, device: str = "cuda", *,
                 system: str = "You are a helpful assistant.",
                 tmp_dir: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ValueError("Qwen-VL-Chat requires a complete local checkpoint directory")
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        is_cuda = str(device).startswith("cuda")
        dtype = torch.float16 if is_cuda else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True,
            device_map=None, torch_dtype=dtype, low_cpu_mem_usage=False,
            bf16=False, fp16=is_cuda, fp32=not is_cuda).eval().to(device)
        self.model.generation_config = GenerationConfig.from_pretrained(
            str(path), local_files_only=True)
        if any(p.device.type == "meta" for p in self.model.parameters()):
            raise RuntimeError("Qwen checkpoint left unloaded meta parameters")
        visual = getattr(getattr(self.model, "transformer", None), "visual", None)
        self.visual_input_size = resolve_visual_size(
            getattr(visual, "image_size", None),
            _configuration_value(getattr(self.model.config, "visual", None), "image_size"))
        module = importlib.import_module(self.model.__class__.__module__)
        self._make_context = getattr(module, "make_context")
        self._chat_format = getattr(self.model.generation_config, "chat_format", None)
        if self._chat_format != "chatml":
            raise ValueError("Expected a Qwen-VL-Chat checkpoint with chat_format='chatml'")
        self._max_window = int(getattr(self.model.generation_config, "max_window_size", 6144))
        self._system = system
        self._temp = tempfile.TemporaryDirectory(prefix="etscd_qwen_", dir=tmp_dir)
        self._image_path = Path(self._temp.name) / "view.png"
        self.eos_ids = _eos_ids(self.tokenizer.eod_id,
                               getattr(self.tokenizer, "im_end_id", None),
                               getattr(self.tokenizer, "im_start_id", None),
                               getattr(self.model.generation_config, "eos_token_id", None))
        self._finish("qwen-vl-chat", str(path.resolve()),
                     dict(chat_format=self._chat_format, system=system,
                          max_window_size=self._max_window,
                          image_transport="stable_local_png_path",
                          dtype=str(dtype), local_files_only=True))

    def next_logits(self, image: Image.Image, question: str,
                    prefix: tuple[int, ...] = ()) -> np.ndarray:
        torch = self._torch
        # The path remains constant across all views and token positions. The
        # stock model rereads it because every call has no past_key_values.
        image.convert("RGB").save(self._image_path, format="PNG")
        query = self.tokenizer.from_list_format([
            {"image": str(self._image_path)}, {"text": question}])
        _, tokens = self._make_context(
            self.tokenizer, query, history=None, system=self._system,
            max_window_size=self._max_window, chat_format=self._chat_format)
        with torch.inference_mode():
            ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            ids = _append_prefix(torch, ids, prefix)
            output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False, return_dict=True,
                                output_attentions=False, output_hidden_states=False)
            return self._logits(output)

    def close(self) -> None:
        self._temp.cleanup()


class InstructBlipBackend(_TorchBackend):
    """LAVIS InstructBLIP with prefixes appended only to Vicuna's text input."""

    def __init__(self, model_path: str | None = None, device: str = "cuda", *,
                 llm_model: str | None = None):
        import torch
        from lavis.models import load_preprocess
        from lavis.common.registry import registry
        from omegaconf import OmegaConf

        self._torch = torch
        checkpoint = None
        if model_path:
            path = Path(model_path).expanduser()
            if not path.is_file():
                raise ValueError("InstructBLIP model_path must be a LAVIS checkpoint file")
            checkpoint = str(path.resolve())
        model_class = registry.get_model_class("blip2_vicuna_instruct")
        config_path = str(model_class.default_config_path("vicuna7b"))
        configuration = OmegaConf.load(config_path)
        vicuna_path = Path(llm_model or configuration.model.llm_model).expanduser()
        if not vicuna_path.is_dir():
            raise FileNotFoundError(
                "Vicuna checkpoint directory does not exist: " + str(vicuna_path)
                + ". Set llm_model to its local checkpoint directory.")
        configuration.model.llm_model = str(vicuna_path.resolve())
        if checkpoint:
            configuration.model.load_finetuned = True
            configuration.model.finetuned = checkpoint
            configuration.model.load_pretrained = False
        resolved_config = OmegaConf.to_container(configuration, resolve=True)
        self.model = model_class.from_config(configuration.model)
        if str(device) == "cpu":
            self.model = self.model.float()
        self.model = self.model.to(device).eval()
        processors, _ = load_preprocess(configuration.preprocess)
        self.processor = processors["eval"]
        patch_embed = getattr(getattr(self.model, "visual_encoder", None), "patch_embed", None)
        self.visual_input_size = resolve_visual_size(
            *_resize_sizes(self.processor),
            getattr(patch_embed, "img_size", None),
            _configuration_value(_configuration_value(resolved_config, "model"), "image_size"))
        self.tokenizer = self.model.llm_tokenizer
        self.tokenizer.padding_side = "left"
        self.eos_ids = _eos_ids(self.tokenizer.eos_token_id,
                               getattr(self.model.llm_model.generation_config,
                                       "eos_token_id", None))
        processor_type = type(self.processor)
        self._finish("instructblip-vicuna7b", checkpoint,
                     dict(lavis_name="blip2_vicuna_instruct", model_type="vicuna7b",
                          config_path=config_path, resolved_config=resolved_config,
                          checkpoint_override=checkpoint,
                          llm_model=str(vicuna_path.resolve()),
                          qformer_text_input=bool(self.model.qformer_text_input),
                          max_qformer_text_length=int(self.model.max_txt_len),
                          processor={
                              "module": processor_type.__module__,
                              "class_name": processor_type.__qualname__,
                              "config": _configuration_value(resolved_config, "preprocess"),
                          }))

    def next_logits(self, image: Image.Image, question: str,
                    prefix: tuple[int, ...] = ()) -> np.ndarray:
        torch, model = self._torch, self.model
        with torch.inference_mode(), model.maybe_autocast():
            pixels = self.processor(image.convert("RGB")).unsqueeze(0).to(self.device)
            image_embeds = model.ln_vision(model.visual_encoder(pixels))
            image_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long,
                                    device=self.device)
            queries = model.query_tokens.expand(1, -1, -1)
            args = dict(query_embeds=queries, encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_mask, return_dict=True)
            if model.qformer_text_input:
                # Q-Former visual features use the instruction text.
                text = model.tokenizer([question], padding="longest", truncation=True,
                                       max_length=model.max_txt_len,
                                       return_tensors="pt").to(self.device)
                query_mask = torch.ones(queries.size()[:-1], dtype=torch.long,
                                        device=self.device)
                args.update(input_ids=text.input_ids,
                            attention_mask=torch.cat([query_mask, text.attention_mask], dim=1))
            query_output = model.Qformer.bert(**args)
            visual = model.llm_proj(query_output.last_hidden_state[:, :queries.size(1), :])
            visual_mask = torch.ones(visual.size()[:-1], dtype=torch.long, device=self.device)
            prompt = self.tokenizer([question], padding="longest", return_tensors="pt").to(self.device)
            ids = _append_prefix(torch, prompt.input_ids, prefix)
            text_embeds = model.llm_model.get_input_embeddings()(ids)
            # Batch size is one, so the prompt contains no padding.
            text_mask = torch.ones_like(ids)
            output = model.llm_model(
                inputs_embeds=torch.cat([visual, text_embeds], dim=1),
                attention_mask=torch.cat([visual_mask, text_mask], dim=1),
                use_cache=False, return_dict=True,
                output_attentions=False, output_hidden_states=False)
            return self._logits(output)


def load_backend(family: str, model_path: str | None = None,
                 device: str = "cuda", **kwargs: Any) -> LogitBackend:
    """Load one backbone lazily with its constructor settings."""
    name = family.strip().lower().replace("_", "-")
    if name in {"llava", "llava-1.5", "llava-1.5-7b"}:
        if not model_path:
            raise ValueError("LLaVA requires model_path")
        return LlavaBackend(model_path, device=device, **kwargs)
    if name in {"qwen", "qwen-vl", "qwen-vl-chat"}:
        if not model_path:
            raise ValueError("Qwen-VL-Chat requires model_path")
        return QwenVLBackend(model_path, device=device, **kwargs)
    if name in {"instructblip", "instructblip-vicuna7b", "lavis"}:
        return InstructBlipBackend(model_path, device=device, **kwargs)
    raise ValueError(f"Unknown backbone family: {family!r}")
