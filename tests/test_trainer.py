import torch

from training.trainer import ProbStripTrainer


def test_checkpoint_records_training_metadata(tmp_path):
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    trainer = ProbStripTrainer(
        model=model,
        train_loader=[],
        checkpoint_dir=str(tmp_path),
        metadata={"split_seed": 42, "dataset_root_name": "example"},
    )

    trainer.save_checkpoint("model.pth")
    checkpoint = torch.load(
        tmp_path / "model.pth", map_location="cpu", weights_only=True
    )

    assert checkpoint["metadata"]["split_seed"] == 42
    assert checkpoint["metadata"]["dataset_root_name"] == "example"
    assert "saved_at" in checkpoint
