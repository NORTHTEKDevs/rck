"""Tests for the real PyTorch polisher pipeline."""
import json
import tempfile
from pathlib import Path

import torch

from rck.polisher import (
    NeuralPolisher, PairDataset, PolisherConfig, PolisherModel,
    PolisherTokenizer, train_polisher,
)
from rck.polisher.training import TrainConfig, save_checkpoint, load_checkpoint
from rck.polisher.dataset import collate_pad


# ---- tokenizer -----------------------------------------------------------

def test_tokenizer_special_tokens_first():
    tok = PolisherTokenizer()
    assert tok.id_to_token[0] == "<pad>"
    assert tok.id_to_token[1] == "<unk>"
    assert tok.id_to_token[2] == "<bos>"
    assert tok.id_to_token[3] == "<eos>"
    assert tok.id_to_token[4] == "<sep>"


def test_tokenizer_builds_from_corpus():
    tok = PolisherTokenizer.from_corpus(
        ["the dog is a mammal", "the cat is a mammal",
         "the dog has fur", "the elephant has tusks"],
        min_count=1,
    )
    assert tok.vocab_size > 5
    assert "dog" in tok.token_to_id
    assert "mammal" in tok.token_to_id


def test_tokenizer_encode_pair_returns_sep_index():
    tok = PolisherTokenizer.from_corpus(
        ["dog is mammal", "dog is animal"], min_count=1,
    )
    ids, sep = tok.encode_pair("dog is mammal", "dog is animal")
    assert ids[0] == 2  # <bos>
    assert ids[-1] == 3  # <eos>
    assert ids[sep] == 4  # <sep>


def test_tokenizer_save_load_roundtrip():
    tok = PolisherTokenizer.from_corpus(
        ["hello world this is a sentence"], min_count=1,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        tok.save(path)
        tok2 = PolisherTokenizer.load(path)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.token_to_id == tok.token_to_id
    finally:
        Path(path).unlink()


# ---- model ---------------------------------------------------------------

def test_model_forward_shape():
    config = PolisherConfig.tiny(vocab_size=100)
    model = PolisherModel(config)
    input_ids = torch.randint(0, 100, (2, 16))
    out = model(input_ids)
    assert out["logits"].shape == (2, 16, 100)


def test_model_loss_computes():
    config = PolisherConfig.tiny(vocab_size=100)
    model = PolisherModel(config)
    input_ids = torch.randint(1, 100, (2, 16))  # avoid pad
    targets = torch.randint(1, 100, (2, 16))
    out = model(input_ids, targets=targets)
    assert "loss" in out
    assert out["loss"].item() > 0


def test_model_loss_with_mask_ignores_draft():
    config = PolisherConfig.tiny(vocab_size=100)
    model = PolisherModel(config)
    input_ids = torch.randint(1, 100, (2, 16))
    targets = torch.randint(1, 100, (2, 16))
    # Mask zeros the first half (draft positions) -> loss only from second half.
    mask = torch.zeros(2, 16, dtype=torch.long)
    mask[:, 8:] = 1
    out = model(input_ids, targets=targets, loss_mask=mask)
    assert out["loss"].item() > 0


def test_model_num_params_grows_with_size():
    tiny = PolisherModel(PolisherConfig.tiny(vocab_size=1000))
    small = PolisherModel(PolisherConfig.small(vocab_size=1000))
    assert small.num_parameters() > tiny.num_parameters()


# ---- dataset -------------------------------------------------------------

def test_pair_dataset_encodes_examples():
    tok = PolisherTokenizer.from_corpus(
        ["the dog is a mammal", "the dog is an animal"],
        min_count=1,
    )
    ds = PairDataset(
        [("the dog is a mammal", "the dog is an animal")],
        tok, max_seq_len=64,
    )
    assert len(ds) == 1
    item = ds[0]
    assert "input_ids" in item and "targets" in item and "loss_mask" in item


def test_pair_dataset_loss_mask_zero_in_draft():
    tok = PolisherTokenizer.from_corpus(
        ["a b c d e f g h"], min_count=1,
    )
    ds = PairDataset([("a b c", "d e f")], tok, max_seq_len=32)
    item = ds[0]
    # First few positions should have loss_mask = 0 (inside draft).
    assert item["loss_mask"][0].item() == 0


def test_collate_pads_to_longest():
    tok = PolisherTokenizer.from_corpus(
        ["a b c", "a b c d e f g"], min_count=1,
    )
    ds = PairDataset(
        [("a b", "c"), ("a b c d", "e f g")],
        tok, max_seq_len=64,
    )
    batch = collate_pad([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == batch["targets"].shape[1]


# ---- training ------------------------------------------------------------

def test_training_loss_decreases_on_small_set():
    """Critical: verify the training loop actually learns."""
    torch.manual_seed(0)
    pairs = [
        ("the dog is a mammal", "the dog is a kind of mammal"),
        ("the cat is a mammal", "the cat is a kind of mammal"),
        ("the bird is an animal", "the bird is a kind of animal"),
        ("the fish is an animal", "the fish is a kind of animal"),
    ] * 30  # repeat to make a non-trivial dataset
    tok = PolisherTokenizer.from_corpus(
        [d for d, _ in pairs] + [t for _, t in pairs], min_count=1,
    )
    ds = PairDataset(pairs, tok, max_seq_len=64)
    config = PolisherConfig.tiny(vocab_size=tok.vocab_size)
    config.max_seq_len = 64
    model = PolisherModel(config)
    train_cfg = TrainConfig(
        batch_size=4, max_steps=80, log_every=100,
        warmup_steps=5, lr=3e-3, device="cpu",
    )
    result = train_polisher(model, ds, tok, config=train_cfg)
    assert result.final_step == 80
    early = sum(result.losses[:5]) / 5
    late = sum(result.losses[-5:]) / 5
    assert late < early - 0.5, f"loss did not decrease enough: {early:.2f} -> {late:.2f}"


def test_save_load_roundtrip_preserves_predictions():
    pairs = [("dog is mammal", "dog is animal")] * 10
    tok = PolisherTokenizer.from_corpus(
        [d for d, _ in pairs] + [t for _, t in pairs], min_count=1,
    )
    ds = PairDataset(pairs, tok, max_seq_len=32)
    config = PolisherConfig.tiny(vocab_size=tok.vocab_size)
    config.max_seq_len = 32
    model = PolisherModel(config)
    train_cfg = TrainConfig(batch_size=2, max_steps=10, warmup_steps=2,
                             lr=1e-3, log_every=100, device="cpu")
    train_polisher(model, ds, tok, config=train_cfg)
    with tempfile.TemporaryDirectory() as td:
        save_checkpoint(model, tok, td)
        model2, tok2 = load_checkpoint(td)
        # Same forward pass yields same logits.
        x = torch.tensor([[2, 5, 6, 4, 7, 3]], dtype=torch.long)
        # Pad to max_seq_len if needed.
        if x.size(1) > model.config.max_seq_len:
            x = x[:, :model.config.max_seq_len]
        model.eval(); model2.eval()
        with torch.no_grad():
            a = model(x)["logits"]
            b = model2(x)["logits"]
        assert torch.allclose(a, b, atol=1e-5)


# ---- inference -----------------------------------------------------------

def test_neural_polisher_produces_output():
    pairs = [("dog is mammal", "dog is animal")] * 10
    tok = PolisherTokenizer.from_corpus(
        [d for d, _ in pairs] + [t for _, t in pairs], min_count=1,
    )
    ds = PairDataset(pairs, tok, max_seq_len=32)
    config = PolisherConfig.tiny(vocab_size=tok.vocab_size)
    config.max_seq_len = 32
    model = PolisherModel(config)
    train_cfg = TrainConfig(batch_size=2, max_steps=10, warmup_steps=2,
                             lr=1e-3, log_every=100, device="cpu")
    train_polisher(model, ds, tok, config=train_cfg)
    with tempfile.TemporaryDirectory() as td:
        save_checkpoint(model, tok, td)
        polisher = NeuralPolisher(weights_path=td, device="cpu",
                                   max_new_tokens=10, temperature=0.7)
        out = polisher.polish("dog is mammal")
        assert isinstance(out, str)


def test_neural_polisher_raises_if_no_weights():
    try:
        NeuralPolisher(weights_path="/nonexistent/path")
        assert False, "should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "checkpoint" in str(e).lower()
