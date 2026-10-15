"""Adapter interface, prompt, and native visual input-size tests."""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from etscd.adapters import (InstructBlipBackend, LlavaBackend, QwenVLBackend, _TorchBackend,
                            _append_prefix, _resize_sizes,
                            resolve_binary_token_ids, resolve_visual_size)
from etscd.runner import canonical_hash


class Tensor:
    """Minimal tensor double to inspect inputs without installing PyTorch."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.device = SimpleNamespace(type="cpu")

    def to(self, *args, **kwargs):
        return self

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.values, axis))

    def expand(self, *shape):
        shape = tuple(old if new == -1 else new for old, new in zip(self.values.shape, shape))
        return Tensor(np.broadcast_to(self.values, shape))

    def size(self, dim=None):
        return self.values.shape if dim is None else self.values.shape[dim]

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def detach(self):
        return self

    def float(self):
        return Tensor(self.values.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeTorch:
    long = np.int64
    inference_mode = staticmethod(nullcontext)

    @staticmethod
    def tensor(value, **kwargs):
        return Tensor(value)

    @staticmethod
    def cat(values, dim):
        return Tensor(np.concatenate([v.values for v in values], axis=dim))

    @staticmethod
    def ones(shape, **kwargs):
        return Tensor(np.ones(shape, dtype=np.int64))

    @staticmethod
    def ones_like(value):
        return Tensor(np.ones_like(value.values))


class BinaryTokenizer:
    all_special_ids = [0]

    def __init__(self, encodings, decodings):
        self.encodings, self.decodings = encodings, decodings

    def encode(self, text, add_special_tokens=False):
        return self.encodings.get(text, [40, 41])

    def decode(self, ids, **kwargs):
        return self.decodings[ids[0]]


class TextTokenizer:
    def __init__(self):
        self.questions = []

    def __call__(self, prompts, **kwargs):
        self.questions.extend(prompts)
        result = SimpleNamespace(input_ids=Tensor([[101, 102]]),
                                 attention_mask=Tensor([[1, 1]]))
        result.to = lambda *args: result
        return result


class FakeLLM:
    def __init__(self):
        self.ids, self.inputs = [], []

    def get_input_embeddings(self):
        def embed(ids):
            self.ids.append(ids.values.tolist())
            return Tensor(np.repeat(ids.values[:, :, None], 3, axis=-1))
        return embed

    def __call__(self, **kwargs):
        self.inputs.append(kwargs)
        return SimpleNamespace(logits=Tensor(np.arange(11).reshape(1, 1, 11)))


class AdapterTests(unittest.TestCase):
    def test_import_does_not_require_torch(self):
        # Test discovery imports adapters before this assertion.
        self.assertNotIn("torch", sys.modules)

    def test_binary_helper_rejects_first_token_approximations(self):
        tokenizer = BinaryTokenizer({"Yes": [9, 10], "No": [9, 11]}, {9: " "})
        with self.assertRaisesRegex(ValueError, "single-token"):
            resolve_binary_token_ids(tokenizer)

    def test_llava_loader_accepts_concrete_device_for_vision_tower(self):
        tokenizer = SimpleNamespace(eos_token_id=2)
        model = SimpleNamespace(config=SimpleNamespace(),
                                generation_config=SimpleNamespace(eos_token_id=2))
        processor = SimpleNamespace(crop_size={"height": 336, "width": 336})
        calls = []

        def load_pretrained_model(path, base, name, **kwargs):
            # The official builder passes device_map to vision_tower.to(device=...).
            if not isinstance(kwargs["device_map"], str):
                raise TypeError("vision_tower.to expects a concrete device")
            calls.append(kwargs)
            return tokenizer, model, processor, 2048

        modules = {
            "torch": SimpleNamespace(),
            "llava": SimpleNamespace(),
            "llava.constants": SimpleNamespace(
                DEFAULT_IMAGE_TOKEN="<image>", DEFAULT_IM_END_TOKEN="</im>",
                DEFAULT_IM_START_TOKEN="<im>", IMAGE_TOKEN_INDEX=-200),
            "llava.conversation": SimpleNamespace(conv_templates={"llava_v1": object()}),
            "llava.mm_utils": SimpleNamespace(
                get_model_name_from_path=lambda path: "llava-v1.5-7b",
                process_images=lambda *args: None, tokenizer_image_token=lambda *args: None),
            "llava.model": SimpleNamespace(),
            "llava.model.builder": SimpleNamespace(load_pretrained_model=load_pretrained_model),
        }
        with patch.dict(sys.modules, modules), patch.object(LlavaBackend, "_finish"):
            backend = LlavaBackend("local-llava", device="cuda:0")
        self.assertEqual(calls[0]["device_map"], "cuda:0")
        self.assertEqual(backend.visual_input_size, (336, 336))

    def test_qwen_loads_chat_config_and_explicit_device_precision(self):
        model_calls, config_calls = [], []

        class RemoteModel:
            __module__ = "etscd_test_qwen_model"
            config = SimpleNamespace(visual={"image_size": 448})
            generation_config = SimpleNamespace()

            def eval(self):
                return self

            def to(self, device):
                return self

            def parameters(self):
                return iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])

        def load_model(path, **kwargs):
            model_calls.append(kwargs)
            return RemoteModel()

        def load_generation_config(path, **kwargs):
            config_calls.append((path, kwargs))
            return SimpleNamespace(chat_format="chatml", max_window_size=6144,
                                   eos_token_id=151643)

        modules = {
            "torch": SimpleNamespace(float16="float16", float32="float32"),
            "transformers": SimpleNamespace(
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs:
                    SimpleNamespace(eod_id=151643)),
                GenerationConfig=SimpleNamespace(from_pretrained=load_generation_config)),
            "etscd_test_qwen_model": SimpleNamespace(make_context=lambda *args: None),
        }
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with patch.dict(sys.modules, modules), patch.object(QwenVLBackend, "_finish"):
                for device in ("cuda:0", "cpu"):
                    backend = QwenVLBackend(directory, device=device, tmp_dir=directory)
                    self.assertEqual(backend._chat_format, "chatml")
                    self.assertEqual(backend.visual_input_size, (448, 448))
                    backend.close()
        self.assertEqual([(call["bf16"], call["fp16"], call["fp32"])
                          for call in model_calls], [(False, True, False), (False, False, True)])
        self.assertTrue(all(call[1]["local_files_only"] for call in config_calls))

    def test_lavis_paths_are_resolved_before_single_checkpoint_load(self):
        calls = []
        configuration = SimpleNamespace(
            model=SimpleNamespace(llm_model="./llm/vicuna-7b", load_finetuned=False,
                                  load_pretrained=True, pretrained="default-url"),
            preprocess=SimpleNamespace())
        model = SimpleNamespace(
            llm_tokenizer=SimpleNamespace(eos_token_id=2),
            llm_model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2)),
            qformer_text_input=True, max_txt_len=128,
            visual_encoder=SimpleNamespace(patch_embed=SimpleNamespace(img_size=(224, 224))))
        model.to = lambda device: model
        model.eval = lambda: model
        model.float = lambda: model

        def from_config(cfg):
            calls.append(vars(cfg).copy())
            return model

        model_class = SimpleNamespace(default_config_path=lambda name: "lavis-vicuna7b.yaml",
                                      from_config=from_config)
        modules = {
            "torch": SimpleNamespace(),
            "lavis": SimpleNamespace(),
            "lavis.models": SimpleNamespace(load_preprocess=lambda cfg:
                ({"eval": SimpleNamespace()}, {})),
            "lavis.common": SimpleNamespace(),
            "lavis.common.registry": SimpleNamespace(registry=SimpleNamespace(
                get_model_class=lambda name: model_class)),
            "omegaconf": SimpleNamespace(OmegaConf=SimpleNamespace(
                load=lambda path: configuration,
                to_container=lambda cfg, **kwargs: {"model": vars(cfg.model).copy()})),
        }
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            checkpoint = Path(directory) / "instructblip.pth"
            checkpoint.touch()
            with patch.dict(sys.modules, modules), patch.object(InstructBlipBackend, "_finish"):
                backend = InstructBlipBackend(str(checkpoint), device="cpu", llm_model=directory)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["llm_model"], str(Path(directory).resolve()))
        self.assertEqual(calls[0]["finetuned"], str(checkpoint.resolve()))
        self.assertTrue(calls[0]["load_finetuned"])
        self.assertFalse(calls[0]["load_pretrained"])
        self.assertEqual(backend.visual_input_size, (224, 224))

    def test_binary_helper_selects_whole_disjoint_spelling(self):
        tokenizer = BinaryTokenizer({"Yes": [7, 8], " Yes": [12], "No": [13]},
                                    {12: " Yes", 13: "No"})
        pair = resolve_binary_token_ids(tokenizer)
        self.assertEqual(tuple(pair), (12, 13))
        self.assertEqual(pair.yes_text, " Yes")
        self.assertEqual(pair.no_text, "No")

    def test_lavis_equivalent_processors_have_stable_configuration_hashes(self):
        class EvalProcessor:
            pass

        class OtherEvalProcessor:
            pass

        processors = []

        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            checkpoint = Path(directory) / "instructblip.pth"
            checkpoint.touch()

            def make_metadata(processor_class=EvalProcessor, normalize=True):
                configuration = SimpleNamespace(
                    model=SimpleNamespace(llm_model=directory, load_finetuned=False,
                                          load_pretrained=True, image_size=224),
                    preprocess={"vis_processor": {"eval": {
                        "name": "blip_image_eval", "image_size": 224,
                        "normalize": normalize}}})
                model = SimpleNamespace(
                    llm_tokenizer=SimpleNamespace(eos_token_id=2),
                    llm_model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2)),
                    qformer_text_input=True, max_txt_len=128,
                    visual_encoder=SimpleNamespace(patch_embed=SimpleNamespace(img_size=(224, 224))),
                    parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))]))
                model.to = lambda device: model
                model.eval = lambda: model
                model.float = lambda: model

                def load_preprocess(cfg):
                    processor = processor_class()
                    processors.append(processor)
                    return {"eval": processor}, {}

                model_class = SimpleNamespace(
                    default_config_path=lambda name: "lavis-vicuna7b.yaml",
                    from_config=lambda cfg: model)
                modules = {
                    "torch": SimpleNamespace(),
                    "lavis": SimpleNamespace(),
                    "lavis.models": SimpleNamespace(load_preprocess=load_preprocess),
                    "lavis.common": SimpleNamespace(),
                    "lavis.common.registry": SimpleNamespace(registry=SimpleNamespace(
                        get_model_class=lambda name: model_class)),
                    "omegaconf": SimpleNamespace(OmegaConf=SimpleNamespace(
                        load=lambda path: configuration,
                        to_container=lambda cfg, **kwargs: {
                            "model": vars(cfg.model).copy(),
                            "preprocess": deepcopy(cfg.preprocess)})),
                }
                with patch.dict(sys.modules, modules):
                    return InstructBlipBackend(str(checkpoint), device="cpu",
                                               llm_model=directory).metadata()

            first = make_metadata()
            second = make_metadata()
            other_class = make_metadata(OtherEvalProcessor)
            other_config = make_metadata(normalize=False)

        self.assertIsNot(processors[0], processors[1])
        self.assertNotEqual(repr(processors[0]), repr(processors[1]))
        self.assertEqual(canonical_hash(first), canonical_hash(second))
        self.assertEqual(canonical_hash({"backend": first}),
                         canonical_hash({"backend": second}))
        self.assertNotEqual(canonical_hash(first), canonical_hash(other_class))
        self.assertNotEqual(canonical_hash(first), canonical_hash(other_config))
        descriptor = first["settings"]["processor"]
        self.assertEqual(descriptor["module"], EvalProcessor.__module__)
        self.assertEqual(descriptor["class_name"], EvalProcessor.__qualname__)
        self.assertEqual(descriptor["config"]["vis_processor"]["eval"]["normalize"], True)
        self.assertEqual(first["visual_input_size"], [224, 224])

    def test_binary_helper_excludes_special_and_wrong_decodings(self):
        tokenizer = BinaryTokenizer({"Yes": [0], "yes": [14], "No": [13]},
                                    {0: "Yes", 14: "Yesterday", 13: "No"})
        with self.assertRaises(ValueError):
            resolve_binary_token_ids(tokenizer)

    def test_prefix_ids_are_appended_without_retokenization(self):
        original = Tensor([[2, 3]])
        appended = _append_prefix(FakeTorch, original, (70, 8))
        self.assertEqual(appended.values.tolist(), [[2, 3, 70, 8]])
        self.assertEqual(original.values.tolist(), [[2, 3]])
        with self.assertRaises(ValueError):
            _append_prefix(FakeTorch, original, (-1,))

    def test_native_visual_size_resolves_integer_and_shortest_edge(self):
        self.assertEqual(resolve_visual_size(336), (336, 336))
        self.assertEqual(resolve_visual_size(np.int64(448)), (448, 448))
        self.assertEqual(resolve_visual_size({"shortest_edge": 224}), (224, 224))

    def test_native_visual_size_converts_height_width_to_pillow_order(self):
        self.assertEqual(resolve_visual_size((240, 320)), (320, 240))
        self.assertEqual(resolve_visual_size([480, 640]), (640, 480))
        self.assertEqual(resolve_visual_size({"height": 256, "width": 384}), (384, 256))

    def test_native_visual_size_uses_first_valid_configuration(self):
        self.assertEqual(resolve_visual_size(None, {"size": 17}, (224, 336), 448),
                         (336, 224))

    def test_native_visual_size_requires_positive_dimensions(self):
        for configuration in (None, True, np.bool_(True), 0, -1, 224.5,
                              "224", (0, 224), (224, -1), (224,),
                              {"height": 224}, {"height": 224, "width": False},
                              {"shortest_edge": 0}):
            with self.subTest(configuration=configuration):
                with self.assertRaisesRegex(ValueError, "native visual input size"):
                    resolve_visual_size(configuration)

    def test_lavis_resize_size_is_read_from_eval_transform(self):
        class Resize:
            size = (224, 336)

        processor = SimpleNamespace(transform=SimpleNamespace(transforms=[
            SimpleNamespace(size=(1, 1)),
            SimpleNamespace(transforms=[Resize()])]))
        self.assertEqual(resolve_visual_size(*_resize_sizes(processor)), (336, 224))
        self.assertEqual(_resize_sizes(SimpleNamespace()), [])

    def test_backend_metadata_records_native_visual_input_size(self):
        backend = _TorchBackend()
        backend.visual_input_size = (384, 256)
        backend.model = SimpleNamespace(
            eval=lambda: None,
            parameters=lambda: iter([SimpleNamespace(device="cpu")]))
        backend._finish("test-backbone", "checkpoint", {})
        metadata = backend.metadata()
        self.assertEqual(metadata["visual_input_size"], [384, 256])
        metadata["visual_input_size"][0] = 1
        self.assertEqual(backend.metadata()["visual_input_size"], [384, 256])

    def test_lavis_keeps_prefix_out_of_qformer_question(self):
        backend = InstructBlipBackend.__new__(InstructBlipBackend)
        backend._torch, backend.device = FakeTorch, SimpleNamespace(type="cpu")
        backend.tokenizer = TextTokenizer()
        qformer_tokenizer = TextTokenizer()
        llm = FakeLLM()
        backend.processor = lambda image: Tensor(np.asarray(image).transpose(2, 0, 1))
        backend.model = SimpleNamespace(
            maybe_autocast=nullcontext,
            visual_encoder=lambda pixels: Tensor(np.ones((1, 5, 3))),
            ln_vision=lambda embeds: embeds,
            query_tokens=Tensor(np.ones((1, 2, 3))),
            qformer_text_input=True, tokenizer=qformer_tokenizer, max_txt_len=128,
            Qformer=SimpleNamespace(bert=lambda **kwargs: SimpleNamespace(
                last_hidden_state=Tensor(np.ones((1, 4, 3))))),
            llm_proj=lambda queries: queries, llm_model=llm)
        for color in ("red", "blue"):
            logits = backend.next_logits(Image.new("RGB", (4, 4), color),
                                         "Describe the picture.", (70, 8))
            self.assertEqual(logits.shape, (11,))
        self.assertEqual(qformer_tokenizer.questions, ["Describe the picture."] * 2)
        self.assertEqual(llm.ids, [[[101, 102, 70, 8]]] * 2)
        self.assertTrue(all(call["use_cache"] is False for call in llm.inputs))

    def test_qwen_views_share_chat_context_and_exact_prefix(self):
        backend = QwenVLBackend.__new__(QwenVLBackend)
        backend._torch, backend.device = FakeTorch, SimpleNamespace(type="cpu")
        queries, contexts, calls, colors = [], [], [], []

        def make_query(items):
            queries.append(items)
            return str(items)

        def make_context(tokenizer, query, **kwargs):
            contexts.append((query, kwargs))
            return "chatml", [50, 51, 52]

        def model(**kwargs):
            calls.append(kwargs)
            with Image.open(backend._image_path) as image:
                colors.append(image.getpixel((0, 0)))
            return SimpleNamespace(logits=Tensor(np.arange(11).reshape(1, 1, 11)))

        backend.tokenizer = SimpleNamespace(from_list_format=make_query)
        backend._make_context = make_context
        backend._system, backend._max_window, backend._chat_format = "system", 6144, "chatml"
        backend.model = model
        # Write the view images beside the test module.
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            backend._image_path = Path(directory) / "view.png"
            for color in ("red", "blue"):
                logits = backend.next_logits(Image.new("RGB", (4, 4), color),
                                             "Question?", (70, 8))
                self.assertEqual(logits.shape, (11,))
        self.assertEqual(queries[0], queries[1])
        self.assertEqual(contexts[0], contexts[1])
        self.assertNotEqual(colors[0], colors[1])
        self.assertEqual(calls[0]["input_ids"].values.tolist(), [[50, 51, 52, 70, 8]])
        self.assertEqual(calls[1]["input_ids"].values.tolist(), [[50, 51, 52, 70, 8]])


if __name__ == "__main__":
    unittest.main()
