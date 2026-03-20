from __future__ import annotations

from pathlib import Path
from types import MethodType

import pytest
import torch

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer


def _make_base_config(tmp_path: Path) -> dict:
    model_path = tmp_path / "model.safetensors"
    text_encoder_path = tmp_path / "gemma"
    model_path.write_bytes(b"")
    text_encoder_path.mkdir()

    return {
        "model": {
            "model_path": str(model_path),
            "text_encoder_path": str(text_encoder_path),
            "training_mode": "full",
        },
        "training_strategy": {
            "name": "ode_regression",
            "with_audio": True,
            "ode_layout_mode": "blockwise",
            "dual_stage_training": True,
            "stage1_loss_weight": 0.25,
            "stage2_loss_weight": 2.0,
        },
        "optimization": {
            "steps": 1,
            "batch_size": 1,
        },
        "data": {
            "preprocessed_data_root_stage1": "/tmp/ode_stage1",
            "preprocessed_data_root_stage2": "/tmp/ode_stage2",
        },
        "validation": {
            "prompts": [],
            "interval": None,
        },
        "wandb": {
            "enabled": False,
        },
        "output_dir": "/tmp/output",
    }


def test_dual_stage_ode_config_accepts_stage_specific_roots(tmp_path: Path) -> None:
    config = LtxTrainerConfig.model_validate(_make_base_config(tmp_path))

    assert config.training_strategy.dual_stage_training is True
    assert config.data.preprocessed_data_root is None
    assert config.data.preprocessed_data_root_stage1 == "/tmp/ode_stage1"
    assert config.data.preprocessed_data_root_stage2 == "/tmp/ode_stage2"


def test_dual_stage_ode_config_requires_both_stage_roots(tmp_path: Path) -> None:
    config = _make_base_config(tmp_path)
    del config["data"]["preprocessed_data_root_stage2"]

    with pytest.raises(ValueError, match="preprocessed_data_root_stage1 and preprocessed_data_root_stage2"):
        LtxTrainerConfig.model_validate(config)


def test_compute_dual_stage_ode_loss_applies_stage_weights(tmp_path: Path) -> None:
    trainer = object.__new__(LtxvTrainer)
    trainer._config = LtxTrainerConfig.model_validate(_make_base_config(tmp_path))

    def fake_training_step(self: LtxvTrainer, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["loss"]

    trainer._training_step = MethodType(fake_training_step, trainer)

    total_loss, metrics = trainer._compute_dual_stage_ode_loss(
        {"loss": torch.tensor(2.0)},
        {"loss": torch.tensor(3.0)},
    )

    assert torch.isclose(total_loss, torch.tensor(6.5))
    assert metrics == {
        "stage1_loss": 2.0,
        "stage2_loss": 3.0,
        "total_loss": 6.5,
    }
