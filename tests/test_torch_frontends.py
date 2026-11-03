"""Tests for the PyTorch frontends: LazyTorchDataset, torch_dataloader and data_core.torch.

The whole module is skipped when torch is not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import data_core  # noqa: E402
from data_core.torch import AugmentedDataset, make_dataloader  # noqa: E402
from data_core.torch.dataset import EpisodeGroupedSampler  # noqa: E402
from synthetic import FPS, create_test_dataset, frame_pixels  # noqa: E402

H, W = 24, 32
ACTION_DIM = 2
STATE_DIM = 3
LENGTHS = [3, 5, 4]  # global frame ranges: [0, 3), [3, 8), [8, 12)
STARTS = [0, 3, 8]


@pytest.fixture
def plain_path(tmp_path):
    path = str(tmp_path / "plain")
    create_test_dataset(path, steps_per_episode=LENGTHS, action_dim=ACTION_DIM, state_dim=STATE_DIM)
    return path


@pytest.fixture
def jpeg_path(tmp_path):
    pytest.importorskip("PIL")
    path = str(tmp_path / "jpeg")
    create_test_dataset(
        path,
        steps_per_episode=LENGTHS,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        camera="cam0",
        image_mode="jpeg",
        image_hw=(H, W),
    )
    return path


@pytest.fixture
def rgb_path(tmp_path):
    path = str(tmp_path / "rgb")
    create_test_dataset(
        path,
        steps_per_episode=LENGTHS,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        camera="cam0",
        image_mode="rgb",
        image_hw=(H, W),
    )
    return path


class TestLazyTorchDatasetModes:
    def test_random_mode_fields(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path).to_torch_dataset("random")
        sample = ds[4]
        assert set(sample) == {
            "episode_index",
            "frame_index",
            "global_index",
            "timestamp",
            "action",
            "state",
            "reward",
            "done",
            "images",
        }
        assert sample["episode_index"].item() == 1
        assert sample["frame_index"].item() == 1
        assert sample["global_index"].item() == 4
        assert sample["timestamp"].dtype == torch.float64
        assert sample["timestamp"].item() == pytest.approx(1 / FPS)
        assert sample["action"].tolist() == [4.0] * ACTION_DIM
        assert sample["state"].tolist() == [1.0] * STATE_DIM
        assert torch.isnan(sample["reward"])
        assert sample["done"].item() is False
        assert sample["images"] == {}

    def test_random_mode_images(self, jpeg_path):
        ds = data_core.LazyDataset.open(jpeg_path).to_torch_dataset("random")
        image = ds[0]["images"]["cam0"]
        assert image.shape == (3, H, W)
        assert image.dtype == torch.float32
        assert 0.0 <= image.min().item() and image.max().item() <= 1.0

    def test_negative_and_out_of_range_index(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path).to_torch_dataset("random")
        assert ds[-1]["global_index"].item() == sum(LENGTHS) - 1
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_chunks_stay_inside_episodes(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path).to_torch_dataset("chunk", chunk_size=3)
        assert len(ds) == 6
        for i in range(len(ds)):
            sample = ds[i]
            episode = sample["episode_index"].item()
            start = sample["start_frame_index"].item()
            assert sample["frame_index"].tolist() == [start, start + 1, start + 2]
            expected_global = [STARTS[episode] + start + k for k in range(3)]
            assert sample["global_index"].tolist() == expected_global
            assert sample["action"].shape == (3, ACTION_DIM)
            assert sample["state"].shape == (3, STATE_DIM)
            assert sample["valid_mask"].all()

    def test_chunks_on_episode_subset(self, plain_path):
        lazy = data_core.LazyDataset.open(plain_path).with_episodes([2, 1])
        ds = lazy.to_torch_dataset("chunk", chunk_size=4)
        # Episode 2 has one 4-frame chunk, episode 1 has two.
        assert len(ds) == 3
        episodes = [ds[i]["episode_index"].item() for i in range(len(ds))]
        assert episodes == [2, 1, 1]
        assert ds[0]["global_index"].tolist() == [8, 9, 10, 11]
        assert ds[2]["global_index"].tolist() == [4, 5, 6, 7]

    def test_chunk_images_are_stacked(self, jpeg_path):
        ds = data_core.LazyDataset.open(jpeg_path).to_torch_dataset("chunk", chunk_size=2)
        assert ds[0]["images"]["cam0"].shape == (2, 3, H, W)

    def test_episode_mode_padding(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path).to_torch_dataset("episode", pad_to_length=4)
        sample = ds[0]
        assert sample["action"].shape == (4, ACTION_DIM)
        assert sample["valid_length"].item() == 3
        assert sample["valid_mask"].tolist() == [True] * 3 + [False]
        assert sample["frame_index"].tolist() == [0, 1, 2, -1]
        assert sample["global_index"].tolist() == [0, 1, 2, -1]
        assert sample["action"][3:].abs().sum().item() == 0.0
        # Padding never shortens a longer episode (length 5).
        assert ds[1]["action"].shape == (5, ACTION_DIM)
        assert ds[1]["valid_mask"].all()

    def test_episode_mode_truncation(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path).to_torch_dataset("episode", truncate_to_length=4)
        lengths = [ds[i]["action"].shape[0] for i in range(len(ds))]
        assert lengths == [3, 4, 4]
        assert ds[1]["valid_length"].item() == 4

    def test_episode_mode_images_padded_with_zeros(self, jpeg_path):
        ds = data_core.LazyDataset.open(jpeg_path).to_torch_dataset("episode", pad_to_length=5)
        images = ds[0]["images"]["cam0"]
        assert images.shape == (5, 3, H, W)
        assert images[3:].abs().sum().item() == 0.0
        assert images[:3].abs().sum().item() > 0.0


class TestTorchDataloader:
    def test_random_mode(self, plain_path):
        loader = data_core.LazyDataset.open(plain_path).torch_dataloader(
            "random", batch_size=4, shuffle=False
        )
        batches = list(loader)
        assert len(batches) == 3
        assert batches[0]["action"].shape == (4, ACTION_DIM)
        assert batches[0]["global_index"].tolist() == [0, 1, 2, 3]
        assert batches[0]["images"] == {}

    def test_random_mode_drop_last(self, plain_path):
        loader = data_core.LazyDataset.open(plain_path).torch_dataloader(
            "random", batch_size=5, shuffle=True, drop_last=True
        )
        batches = list(loader)
        assert len(batches) == 2
        seen = sorted(i for b in batches for i in b["global_index"].tolist())
        assert len(set(seen)) == 10

    def test_random_mode_with_images(self, jpeg_path):
        loader = data_core.LazyDataset.open(jpeg_path).torch_dataloader(
            "random", batch_size=3, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["images"]["cam0"].shape == (3, 3, H, W)

    def test_chunk_mode(self, plain_path):
        loader = data_core.LazyDataset.open(plain_path).torch_dataloader(
            "chunk", batch_size=2, shuffle=False, chunk_size=3
        )
        batch = next(iter(loader))
        assert batch["action"].shape == (2, 3, ACTION_DIM)
        assert batch["valid_mask"].shape == (2, 3)

    def test_episode_mode_fixed_length(self, plain_path):
        loader = data_core.LazyDataset.open(plain_path).torch_dataloader(
            "episode", batch_size=3, shuffle=False, pad_to_length=4, truncate_to_length=4
        )
        batch = next(iter(loader))
        assert batch["action"].shape == (3, 4, ACTION_DIM)
        assert batch["valid_length"].tolist() == [3, 4, 4]
        assert batch["valid_mask"].sum(dim=1).tolist() == [3, 4, 4]

    def test_episode_mode_variable_length_returns_list(self, plain_path):
        # pad_to_length < truncate_to_length leaves lengths unequal.
        loader = data_core.LazyDataset.open(plain_path).torch_dataloader(
            "episode", batch_size=3, shuffle=False, pad_to_length=2, truncate_to_length=4
        )
        batch = next(iter(loader))
        assert isinstance(batch, list)
        assert [item["action"].shape[0] for item in batch] == [3, 4, 4]

    def test_invalid_mode(self, plain_path):
        with pytest.raises(ValueError, match="Invalid mode"):
            data_core.LazyDataset.open(plain_path).torch_dataloader("bogus")


class TestAugmentedDataset:
    def test_sample_layout(self, rgb_path):
        ds = AugmentedDataset(
            rgb_path,
            delta_timestamps={"observation.images.cam0": [-0.1, 0.0], "action": [0.0, 0.1]},
        )
        assert len(ds) == sum(LENGTHS)
        assert ds.num_episodes == len(LENGTHS)
        assert ds.episode_data_index == [(0, 3), (3, 8), (8, 12)]

        sample = ds[5]
        assert set(sample) == {
            "observation.images.cam0",
            "observation.images.cam0_is_pad",
            "observation.state",
            "action",
            "action_is_pad",
            "episode_index",
            "frame_index",
            "timestamp",
        }
        images = sample["observation.images.cam0"]
        assert images.shape == (2, 3, H, W)
        assert images.dtype == torch.uint8
        # Channel-first and pixel-exact: frames 4 and 5 of the synthetic data.
        for k, global_index in enumerate([4, 5]):
            expected = torch.from_numpy(frame_pixels(global_index, H, W)).permute(2, 0, 1)
            assert torch.equal(images[k], expected)
        assert sample["observation.state"].shape == (2, STATE_DIM)
        assert sample["action"].tolist() == [[5.0] * ACTION_DIM, [6.0] * ACTION_DIM]
        assert sample["episode_index"].item() == 1
        assert sample["frame_index"].item() == 2

    def test_float_images(self, rgb_path):
        ds = AugmentedDataset(rgb_path, delta_timestamps={}, float_images=True)
        sample = ds[0]
        images = sample["observation.images.cam0"]
        assert images.shape == (1, 3, H, W)
        assert images.dtype == torch.float32
        expected = torch.from_numpy(frame_pixels(0, H, W)).permute(2, 0, 1).float() / 255.0
        assert torch.allclose(images[0], expected)
        # A single-frame window gives a 1-D state and no action.
        assert sample["observation.state"].shape == (STATE_DIM,)
        assert "action" not in sample

    def test_padding_at_episode_boundaries(self, rgb_path):
        ds = AugmentedDataset(
            rgb_path,
            delta_timestamps={"observation.images.cam0": [-0.2, 0.0], "action": [0.0, 0.1, 0.2]},
        )
        # Global frame 2 is the last frame of episode 0.
        sample = ds[2]
        assert sample["action_is_pad"].tolist() == [False, True, True]
        # Out-of-episode actions repeat the last frame instead of reading episode 1.
        assert sample["action"].tolist() == [[2.0] * ACTION_DIM] * 3
        assert sample["observation.images.cam0_is_pad"].tolist() == [False, False]

        first = ds[3]  # first frame of episode 1
        assert first["observation.images.cam0_is_pad"].tolist() == [True, False]
        images = first["observation.images.cam0"]
        assert torch.equal(images[0], images[1])
        assert first["action_is_pad"].tolist() == [False, False, False]

    def test_index_errors(self, rgb_path):
        ds = AugmentedDataset(rgb_path, delta_timestamps={})
        assert ds[-1]["frame_index"].item() == LENGTHS[-1] - 1
        with pytest.raises(IndexError):
            ds[len(ds)]
        # IndexError lets plain iteration terminate.
        assert len(list(iter(ds))) == len(ds)

    def test_random_crop_augmentation(self, rgb_path):
        ds = AugmentedDataset(
            rgb_path,
            delta_timestamps={"observation.images.cam0": [-0.1, 0.0]},
            augmentation="random_crop",
            augmentation_params={"height": 16, "width": 20, "camera_names": ["cam0"]},
        )
        images = ds[4]["observation.images.cam0"]
        assert images.shape == (2, 3, 16, 20)
        # Every cropped pixel row must appear in the source frame (HWC -> CHW intact).
        source = frame_pixels(4, H, W)
        crop = images[1].permute(1, 2, 0).numpy()
        found = any(
            np.array_equal(source[y : y + 16, x : x + 20], crop)
            for y in range(H - 16 + 1)
            for x in range(W - 20 + 1)
        )
        assert found

    def test_fused_image_augmentation(self, rgb_path):
        ds = AugmentedDataset(
            rgb_path,
            delta_timestamps={},
            augmentation="fused_image",
            augmentation_params={"camera_names": ["cam0"], "crop_height": H, "crop_width": W},
        )
        images = ds[7]["observation.images.cam0"]
        expected = torch.from_numpy(frame_pixels(7, H, W)).permute(2, 0, 1)
        assert torch.equal(images[0], expected)

    def test_unknown_camera_key_rejected(self, rgb_path):
        with pytest.raises(ValueError, match="names camera 'top'"):
            AugmentedDataset(rgb_path, delta_timestamps={"observation.images.top": [0.0]})

    def test_off_grid_timestamps_rejected(self, rgb_path):
        # 1/50 s falls between frames of the synthetic dataset.
        assert FPS % 50 != 0
        with pytest.raises(ValueError, match="multiples of 1/fps"):
            AugmentedDataset(rgb_path, delta_timestamps={"action": [i / 50 for i in range(5)]})

    def test_unused_key_warns(self, rgb_path):
        with pytest.warns(UserWarning, match="is ignored"):
            AugmentedDataset(rgb_path, delta_timestamps={"observation.velocity": [0.0]})

    def test_augmentation_without_params_is_applied(self, rgb_path):
        ds = AugmentedDataset(rgb_path, delta_timestamps={}, augmentation="action_noise")
        assert ds._ds.has_augmentations()

    def test_make_dataloader_single_process(self, rgb_path):
        ds = AugmentedDataset(
            rgb_path, delta_timestamps={"action": [0.0, 0.1]}, float_images=True
        )
        loader = make_dataloader(ds, batch_size=4, shuffle=False, drop_last=False)
        batches = list(loader)
        assert [b["action"].shape[0] for b in batches] == [4, 4, 4]
        assert batches[0]["observation.images.cam0"].shape == (4, 1, 3, H, W)
        assert batches[0]["action"].shape == (4, 2, ACTION_DIM)
        assert batches[0]["action_is_pad"].shape == (4, 2)

    def test_make_dataloader_workers(self, rgb_path):
        ds = AugmentedDataset(rgb_path, delta_timestamps={"action": [0.0]})
        loader = make_dataloader(
            ds, batch_size=2, shuffle=True, num_workers=2, drop_last=False, preload_episodes=[0, 2]
        )
        assert isinstance(loader.sampler, EpisodeGroupedSampler)
        seen = sorted(
            int(g) for batch in loader for g in batch["action"][:, 0, 0].tolist()
        )
        # Only frames of the preloaded episodes 0 ([0, 3)) and 2 ([8, 12)).
        assert seen == [0, 1, 2, 8, 9, 10, 11]


def test_episode_grouped_sampler():
    bounds = [(0, 3), (3, 8), (8, 12)]
    ordered = list(EpisodeGroupedSampler(bounds, shuffle=False))
    assert ordered == list(range(12))
    shuffled = list(EpisodeGroupedSampler(bounds, shuffle=True))
    assert sorted(shuffled) == list(range(12))
    assert len(EpisodeGroupedSampler(bounds)) == 12
    # Frames of one episode are contiguous in the output.
    owner = [next(i for i, (a, b) in enumerate(bounds) if a <= f < b) for f in shuffled]
    assert sum(1 for x, y in zip(owner, owner[1:]) if x != y) == len(bounds) - 1
