import shutil
from pathlib import Path

from speech_to_text_finetune.config import load_config
from speech_to_text_finetune.finetune_whisper import run_finetuning
import pytest
import numpy as np
from transformers import EvalPrediction


def test_finetune_whisper_local(example_config):
    base_results, eval_results = run_finetuning(config_path=example_config)

    cfg = load_config(example_config)
    expected_dir_path = Path(f"artifacts/{cfg.repo_name}")
    assert expected_dir_path.exists()

    assert 0 < base_results["eval_loss"] < 10
    assert 0 < base_results["eval_wer"] < 100
    assert 0 < eval_results["eval_loss"] < 10
    assert 0 < eval_results["eval_wer"] < 100

    shutil.rmtree(expected_dir_path)

def test_compute_word_error_rate_identical():
    """Test compute_word_error_rate when predictions and labels are identical, expecting WER of 0."""
    from speech_to_text_finetune.finetune_whisper import compute_word_error_rate
    class DummyTokenizer:
        pad_token_id = 0
        def batch_decode(self, ids, skip_special_tokens=True):
            return [" ".join(map(str, seq)) for seq in ids]
    class DummyMetric:
        def compute(self, predictions, references):
            return 0.0  # zero error when predictions match references
    dummy_tokenizer = DummyTokenizer()
    dummy_metric = DummyMetric()
    # predictions and labels are identical
    pred = EvalPrediction(predictions=[[1,2,3]], label_ids=np.array([[1,2,3]]))
    result = compute_word_error_rate(pred, dummy_tokenizer, dummy_metric)
    assert result["wer"] == 0.0
def test_compute_word_error_rate_non_zero():
    """Test compute_word_error_rate with non-matching predictions and labels, expecting a non-zero WER."""
    from speech_to_text_finetune.finetune_whisper import compute_word_error_rate
    class DummyTokenizer:
        pad_token_id = 0
        def batch_decode(self, ids, skip_special_tokens=True):
            return [" ".join(map(str, seq)) for seq in ids]
    class DummyMetric:
        def compute(self, predictions, references):
            return 0.25  # simulate a 25% error rate
    dummy_tokenizer = DummyTokenizer()
    dummy_metric = DummyMetric()
    # label contains -100 which should be replaced by pad_token_id (0)
    pred = EvalPrediction(predictions=[[1,2,3]], label_ids=np.array([[1,2,-100]]))
    result = compute_word_error_rate(pred, dummy_tokenizer, dummy_metric)
    assert result["wer"] == 25.0
def test_run_finetuning_invalid_dataset_source(monkeypatch, tmp_path):
    """Test run_finetuning raises a ValueError when the dataset_source is invalid."""
    from speech_to_text_finetune.finetune_whisper import run_finetuning
    from speech_to_text_finetune.config import LANGUAGES_NAME_TO_ID
    # Set a dummy language id required by run_finetuning
    LANGUAGES_NAME_TO_ID["en"] = "en_id"
    class DummyTrainingHP:
        push_to_hub = False
        hub_private_repo = False
        def model_dump(self):
            return {}
    class DummyConfig:
        language = "en"
        repo_name = "default"
        model_id = "openai/whisper-small"
        training_hp = DummyTrainingHP()
        dataset_source = "invalid"
        dataset_id = "dummy_dataset"
        n_train_samples = -1
        n_test_samples = -1
    monkeypatch.setattr("speech_to_text_finetune.finetune_whisper.load_config", lambda path: DummyConfig())
    with pytest.raises(ValueError, match="Unknown dataset source invalid"):
        run_finetuning(config_path=str(tmp_path / "dummy.yaml"))