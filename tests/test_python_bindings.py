"""Tests for the data_core Python bindings."""

import os
import tempfile

import pyarrow as pa
import pytest

import data_core
from synthetic import create_test_dataset

try:
    import torch
except ImportError:  # pragma: no cover - optional in CI/runtime envs
    torch = None


class TestDataset:
    def test_version(self):
        assert hasattr(data_core, "__version__")
        assert data_core.__version__ == "0.1.0"

    def test_load_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=3, steps_per_episode=10)
            ds = data_core.Dataset.from_local(tmpdir)

            assert ds.num_episodes == 3
            assert ds.num_steps == 30
            assert ds.fps == 10.0
            assert ds.robot_type == "test_robot"

    def test_dataset_repr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir)
            ds = data_core.Dataset.from_local(tmpdir)
            r = repr(ds)
            assert "Dataset" in r
            assert "episodes=3" in r

    def test_dataset_len(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=5)
            ds = data_core.Dataset.from_local(tmpdir)
            assert len(ds) == 5

    def test_episode_access(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=3, steps_per_episode=10)
            ds = data_core.Dataset.from_local(tmpdir)

            ep = ds.episode(0)
            assert ep.num_steps == 10
            assert ep.episode_index == 0

    def test_episode_to_pyarrow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=1, steps_per_episode=5)
            ds = data_core.Dataset.from_local(tmpdir)

            ep = ds.episode(0)
            batch = ep.to_pyarrow()
            assert isinstance(batch, pa.RecordBatch)
            assert batch.num_rows == 5
            assert "action" in batch.schema.names

    def test_episodes_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=4)
            ds = data_core.Dataset.from_local(tmpdir)

            episodes = ds.episodes()
            assert len(episodes) == 4

    def test_save_lerobot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "src")
            dst = os.path.join(tmpdir, "dst")
            create_test_dataset(src, num_episodes=2, steps_per_episode=5)

            ds = data_core.Dataset.from_local(src)
            ds.save_lerobot(dst)

            assert os.path.exists(os.path.join(dst, "meta", "info.json"))
            assert os.path.exists(
                os.path.join(dst, "data", "chunk-000", "episode_000000.parquet")
            )


class TestPipeline:
    def test_empty_pipeline(self):
        pipeline = data_core.Pipeline()
        assert pipeline.num_steps == 0
        assert pipeline.step_names() == []

    def test_add_builtin(self):
        pipeline = data_core.Pipeline()
        pipeline.add_builtin("action_noise", '{"std": 0.05}')
        assert pipeline.num_steps == 1
        assert "action_noise" in pipeline.step_names()

    def test_pipeline_run_action_noise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=2, steps_per_episode=10)
            ds = data_core.Dataset.from_local(tmpdir)

            pipeline = data_core.Pipeline()
            pipeline.add_builtin("action_noise", '{"std": 0.1}')

            result = pipeline.run(ds)
            assert result.num_episodes == 2
            assert result.num_steps == 20

    def test_pipeline_run_temporal_subsample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=2, steps_per_episode=10)
            ds = data_core.Dataset.from_local(tmpdir)

            pipeline = data_core.Pipeline()
            pipeline.add_builtin("temporal_subsample", '{"step": 3}')

            result = pipeline.run(ds)
            assert result.num_episodes == 2
            # Steps 0, 3, 6, 9: 4 steps per episode
            assert result.num_steps == 8

    def test_pipeline_python_augmentation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=1, steps_per_episode=5)
            ds = data_core.Dataset.from_local(tmpdir)

            # Define a Python augmentation class
            class IdentityAugmentation:
                def __init__(self):
                    pass
                
                def run(self, batch: pa.RecordBatch) -> pa.RecordBatch:
                    # Identity - just return the batch unchanged
                    return batch

            pipeline = data_core.Pipeline()
            pipeline.add_local(IdentityAugmentation())

            result = pipeline.run(ds)
            assert result.num_episodes == 1
            assert result.num_steps == 5

    def test_pipeline_repr(self):
        pipeline = data_core.Pipeline()
        pipeline.add_builtin("action_noise", '{"std": 0.1}')
        pipeline.add_builtin("temporal_subsample", '{"step": 2}')
        r = repr(pipeline)
        assert "action_noise" in r
        assert "temporal_subsample" in r

    def test_unknown_builtin(self):
        pipeline = data_core.Pipeline()
        with pytest.raises(ValueError, match="Unknown builtin"):
            pipeline.add_builtin("nonexistent_aug", "{}")

    def test_chained_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=2, steps_per_episode=20)
            ds = data_core.Dataset.from_local(tmpdir)

            pipeline = data_core.Pipeline()
            pipeline.add_builtin("action_noise", '{"std": 0.05}')
            pipeline.add_builtin("temporal_subsample", '{"step": 2}')

            result = pipeline.run(ds)
            assert result.num_episodes == 2
            # 20 steps subsampled by 2: 10 per episode
            assert result.num_steps == 20


@pytest.mark.skipif(torch is None, reason="torch is required for LazyTorchDataset tests")
class TestLazyTorchDataset:
    def test_random_mode_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=2, steps_per_episode=4)
            lazy = data_core.LazyDataset.open(tmpdir)
            ds = lazy.to_torch_dataset(mode="random")

            assert len(ds) == 8
            sample = ds[0]
            assert isinstance(sample["action"], torch.Tensor)
            assert tuple(sample["action"].shape) == (7,)
            assert sample["episode_index"].dtype == torch.int64
            assert sample["images"] == {}

    def test_chunk_mode_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=2, steps_per_episode=5)
            lazy = data_core.LazyDataset.open(tmpdir)
            ds = lazy.to_torch_dataset(mode="chunk", chunk_size=3)

            # 2 episodes * (5 - 3 + 1)
            assert len(ds) == 6
            sample = ds[0]
            assert tuple(sample["action"].shape) == (3, 7)
            assert int(sample["valid_length"].item()) == 3
            assert tuple(sample["valid_mask"].shape) == (3,)
            assert sample["valid_mask"].dtype == torch.bool

    def test_episode_mode_variable_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, steps_per_episode=[2, 5, 3])
            lazy = data_core.LazyDataset.open(tmpdir)
            ds = lazy.to_torch_dataset(mode="episode")

            assert len(ds) == 3
            sample0 = ds[0]
            sample1 = ds[1]
            assert tuple(sample0["action"].shape) == (2, 7)
            assert tuple(sample1["action"].shape) == (5, 7)
            assert int(sample0["valid_length"].item()) == 2
            assert int(sample1["valid_length"].item()) == 5

    def test_episode_mode_pad_and_truncate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, steps_per_episode=[2, 6])
            lazy = data_core.LazyDataset.open(tmpdir)
            ds = lazy.to_torch_dataset(
                mode="episode",
                pad_to_length=4,
                truncate_to_length=3,
            )

            sample_short = ds[0]
            sample_long = ds[1]
            assert tuple(sample_short["action"].shape) == (4, 7)
            assert tuple(sample_long["action"].shape) == (4, 7)
            assert int(sample_short["valid_length"].item()) == 2
            assert int(sample_long["valid_length"].item()) == 3
            assert int(sample_short["valid_mask"].sum().item()) == 2
            assert int(sample_long["valid_mask"].sum().item()) == 3

    def test_torch_dataloader_helper_fixed_episode_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, steps_per_episode=[2, 6])
            lazy = data_core.LazyDataset.open(tmpdir)
            loader = lazy.torch_dataloader(
                mode="episode",
                batch_size=2,
                shuffle=False,
                pad_to_length=4,
                truncate_to_length=3,
            )

            batch = next(iter(loader))
            assert isinstance(batch, dict)
            assert tuple(batch["action"].shape) == (2, 4, 7)
            assert tuple(batch["valid_length"].shape) == (2,)

    def test_torch_dataloader_helper_variable_episode_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, steps_per_episode=[2, 5])
            lazy = data_core.LazyDataset.open(tmpdir)
            loader = lazy.torch_dataloader(
                mode="episode",
                batch_size=2,
                shuffle=False,
            )

            batch = next(iter(loader))
            assert isinstance(batch, list)
            assert len(batch) == 2

    def test_collate_fills_missing_nested_image_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_dataset(tmpdir, num_episodes=1, steps_per_episode=2)
            lazy = data_core.LazyDataset.open(tmpdir)
            ds = lazy.to_torch_dataset(mode="random")

            sample_with_image = {
                "episode_index": torch.tensor(0, dtype=torch.int64),
                "frame_index": torch.tensor(0, dtype=torch.int64),
                "global_index": torch.tensor(0, dtype=torch.int64),
                "timestamp": torch.tensor(0.0, dtype=torch.float64),
                "action": torch.zeros(7, dtype=torch.float32),
                "reward": torch.tensor(0.0, dtype=torch.float32),
                "done": torch.tensor(False, dtype=torch.bool),
                "images": {"image": torch.zeros((3, 8, 8), dtype=torch.float32)},
            }
            sample_without_image = {
                "episode_index": torch.tensor(0, dtype=torch.int64),
                "frame_index": torch.tensor(1, dtype=torch.int64),
                "global_index": torch.tensor(1, dtype=torch.int64),
                "timestamp": torch.tensor(0.1, dtype=torch.float64),
                "action": torch.zeros(7, dtype=torch.float32),
                "reward": torch.tensor(0.0, dtype=torch.float32),
                "done": torch.tensor(False, dtype=torch.bool),
                "images": {},
            }

            batch = ds.collate_fn([sample_with_image, sample_without_image])
            assert "images" in batch
            assert "image" in batch["images"]
            assert tuple(batch["images"]["image"].shape) == (2, 3, 8, 8)
