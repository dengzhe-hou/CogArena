import json
import sys
import types
from contextlib import nullcontext

import pytest

from cogarena import cli
from cogarena.llm_client import LLMClient
from cogarena.scoring import score_episode


def test_text_collection_has_exactly_one_item_per_paradigm():
    items = cli._collect_items(1, 42, None)
    paradigms = [item.metadata.paradigm for item in items]
    assert paradigms == list(cli.PARADIGM_GROUPING)


def test_full_text_dry_run_writes_all_thirteen_paradigms(tmp_path):
    code = cli.main(
        [
            "eval",
            "--dry-run",
            "--model",
            "smoke",
            "--n",
            "1",
            "--quiet",
            "--output",
            str(tmp_path),
        ]
    )
    assert code == 0
    aggregate = json.loads(
        (tmp_path / "smoke" / "text" / "aggregate.json").read_text()
    )
    assert aggregate["n_records"] == 13
    assert set(aggregate["paradigm_accuracy"]) == set(cli.PARADIGM_GROUPING)


def test_native_multiturn_scorers_are_routed():
    items = {
        item.metadata.paradigm: item
        for item in cli._collect_items(1, 42, None)
    }

    nback = items["n_back"]
    nback_responses = [
        turn["expected"] for turn in nback.metadata.parameters["turns"]
    ]
    assert score_episode(nback, nback_responses)["accuracy"] == 1.0

    ospan = items["operation_span"]
    ospan_responses = []
    for turn in ospan.metadata.parameters["turns"]:
        if turn["type"] == "operation_letter":
            ospan_responses.append(turn["math_expected"])
        else:
            ospan_responses.append(" ".join(ospan.metadata.parameters["letters"]))
    ospan_score = score_episode(ospan, ospan_responses)
    assert ospan_score["accuracy"] == 1.0
    assert ospan_score["math_accuracy"] == 1.0

    cvlt = items["cvlt_word_list"]
    cvlt_responses = [
        ", ".join(turn.get("expected_words") or [])
        for turn in cvlt.metadata.parameters["turns"]
    ]
    assert score_episode(cvlt, cvlt_responses)["accuracy"] == 1.0


def test_vlm_and_agent_dry_run_paths(tmp_path):
    pytest.importorskip("PIL")
    assert (
        cli.main(
            [
                "eval",
                "--dry-run",
                "--model",
                "smoke",
                "--mode",
                "vlm",
                "--n",
                "1",
                "--quiet",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    vlm = json.loads(
        (tmp_path / "smoke" / "vlm" / "aggregate.json").read_text()
    )
    assert set(vlm["paradigm_accuracy"]) == {
        "stroop",
        "flanker",
        "false_belief",
    }

    assert (
        cli.main(
            [
                "eval",
                "--dry-run",
                "--model",
                "smoke",
                "--mode",
                "agent",
                "--paradigms",
                "n_back",
                "false_belief",
                "--n",
                "1",
                "--quiet",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    agent = json.loads(
        (tmp_path / "smoke" / "agent" / "aggregate.json").read_text()
    )
    assert set(agent["paradigm_accuracy"]) == {"n_back", "false_belief"}


class _FakeTensor:
    def __init__(self, length):
        self.shape = (1, length)
        self.device = "cpu"

    def to(self, _device):
        return self

    def __getitem__(self, value):
        if isinstance(value, slice):
            start = value.start or 0
            return _FakeTensor(max(0, self.shape[-1] - start))
        return self


class _FakeTokenizer:
    chat_template = "{{ messages }}"
    pad_token_id = 0
    eos_token_id = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def apply_chat_template(self, *_args, **_kwargs):
        return "formatted prompt"

    def __call__(self, _text, return_tensors=None):
        assert return_tensors == "pt"
        return {"input_ids": _FakeTensor(4)}

    def decode(self, _tokens, skip_special_tokens=True):
        assert skip_special_tokens
        return "MODEL ANSWER"


class _FakeModel:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def parameters(self):
        yield _FakeTensor(0)

    def generate(self, **_kwargs):
        return [_FakeTensor(6)]


def test_huggingface_provider_loads_model_without_server(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.inference_mode = nullcontext
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.AutoModelForCausalLM = _FakeModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    client = LLMClient(
        {
            "provider": "huggingface",
            "model": "org/model",
            "device": "cpu",
            "max_retries": 1,
        }
    )
    assert client.generate("question", system_prompt="instructions") == "MODEL ANSWER"
    assert client.last_token_counts == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
    }


def test_huggingface_provider_rejects_images(monkeypatch):
    client = LLMClient(
        {
            "provider": "huggingface",
            "model": "org/model",
            "max_retries": 1,
        }
    )
    with pytest.raises(RuntimeError, match="supports text models"):
        client.generate("question", images=["image.png"])
