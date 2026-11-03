"""Tests for data_core.LazyDataset and data_core.Frame on synthetic data."""

import io
import json

import numpy as np
import pyarrow as pa
import pytest

import data_core
from synthetic import FPS, TASK, create_test_dataset, frame_pixels

H, W = 24, 32
ACTION_DIM = 2
STATE_DIM = 3
LENGTHS = [3, 5, 4]  # global frame ranges: [0, 3), [3, 8), [8, 12)


@pytest.fixture
def plain_path(tmp_path):
    path = str(tmp_path / "plain")
    create_test_dataset(path, steps_per_episode=LENGTHS, action_dim=ACTION_DIM, state_dim=STATE_DIM)
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


def decode_raw(entry):
    """Turn a get_sample_raw image entry into an (n, H, W, 3) array."""
    pixels, width, height, n_frames = entry
    assert isinstance(pixels, bytes)
    return np.frombuffer(pixels, dtype=np.uint8).reshape(n_frames, height, width, 3)


class TestOpenAndMetadata:
    def test_metadata(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path)
        assert ds.name == "rgb"
        assert ds.fps == FPS
        assert ds.num_frames == sum(LENGTHS)
        assert ds.num_episodes == len(LENGTHS)
        assert len(ds) == sum(LENGTHS)
        assert ds.camera_names() == ["cam0"]
        assert ds.episode_data_index() == [(0, 3), (3, 8), (8, 12)]
        assert not ds.has_augmentations()
        assert repr(ds) == "LazyDataset(name='rgb', frames=12, episodes=3)"

    def test_open_missing_directory(self, tmp_path):
        with pytest.raises(RuntimeError):
            data_core.LazyDataset.open(str(tmp_path / "does-not-exist"))


class TestFrameAccess:
    def test_frame_fields(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        frame = ds[4]
        assert isinstance(frame, data_core.Frame)
        assert frame.index == 4
        assert frame.episode_index == 1
        assert frame.frame_index == 1
        assert frame.timestamp == pytest.approx(1 / FPS)
        assert frame.action == [4.0] * ACTION_DIM
        assert frame.state == [1.0] * STATE_DIM
        assert frame.task_index == 0
        assert frame.task == TASK
        assert frame.reward is None
        assert frame.done is False
        assert frame.camera_names() == []
        assert frame.image("cam0") is None
        assert repr(frame) == f"Frame(index=4, episode=1, action_dim={ACTION_DIM})"

    def test_get_frame_matches_getitem(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        for idx in range(len(ds)):
            assert ds.get_frame(idx).index == ds[idx].index == idx

    def test_negative_index(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        assert ds[-1].index == len(ds) - 1
        assert ds[-len(ds)].index == 0

    def test_out_of_range(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        with pytest.raises(IndexError, match="out of range"):
            ds[len(ds)]
        with pytest.raises(IndexError, match="out of range"):
            ds[-len(ds) - 1]
        with pytest.raises(RuntimeError, match="out of range"):
            ds.get_frame(len(ds))

    def test_iteration(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        assert [f.index for f in ds] == list(range(len(ds)))

    def test_frame_image_jpeg(self, jpeg_path):
        ds = data_core.LazyDataset.open(jpeg_path)
        frame = ds[0]
        assert frame.camera_names() == ["cam0"]
        jpeg = frame.image("cam0")
        assert isinstance(jpeg, bytes)
        assert jpeg[:2] == b"\xff\xd8"
        assert frame.image("missing") is None


class TestEpisodes:
    def test_get_episode(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        ep = ds.get_episode(2)
        assert ep.episode_index == 2
        assert ep.num_steps == LENGTHS[2]
        batch = ep.to_pyarrow()
        assert isinstance(batch, pa.RecordBatch)
        assert batch.column("frame_index").to_pylist() == list(range(LENGTHS[2]))

    def test_get_episode_out_of_range(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        with pytest.raises(RuntimeError):
            ds.get_episode(len(LENGTHS))

    def test_load_all(self, plain_path):
        full = data_core.LazyDataset.open(plain_path).load_all()
        assert isinstance(full, data_core.Dataset)
        assert full.num_episodes == len(LENGTHS)
        assert full.num_steps == sum(LENGTHS)


class TestWithEpisodes:
    def test_subset_indexing(self, plain_path):
        sub = data_core.LazyDataset.open(plain_path).with_episodes([2, 0])
        assert sub.num_episodes == 2
        assert len(sub) == sub.num_frames == LENGTHS[2] + LENGTHS[0]
        # Subset indices follow the given episode order; Frame.index stays global.
        assert [sub[i].episode_index for i in range(len(sub))] == [2] * 4 + [0] * 3
        assert sub[0].index == 8
        assert sub[4].index == 0
        assert sub[-1].frame_index == LENGTHS[0] - 1

    def test_unselected_episode_rejected(self, plain_path):
        sub = data_core.LazyDataset.open(plain_path).with_episodes([2, 0])
        with pytest.raises(RuntimeError, match="not in the selected episode set"):
            sub.get_episode(1)

    def test_load_all_respects_selection(self, plain_path):
        full = data_core.LazyDataset.open(plain_path).with_episodes([1]).load_all()
        assert full.num_episodes == 1
        assert full.num_steps == LENGTHS[1]

    def test_selection_survives_with_cache_limit(self, plain_path):
        sub = data_core.LazyDataset.open(plain_path).with_episodes([1]).with_cache_limit(1)
        assert sub.num_episodes == 1

    def test_with_after_augmentation_raises(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        ds.add_augmentation("action_noise", '{"std": 0.01}')
        with pytest.raises(ValueError, match="before add_augmentation"):
            ds.with_cache_limit(1)

    def test_episode_data_index_follows_selection(self, plain_path):
        sub = data_core.LazyDataset.open(plain_path).with_episodes([2, 0])
        ranges = sub.episode_data_index()
        assert ranges == [(0, LENGTHS[2]), (LENGTHS[2], LENGTHS[2] + LENGTHS[0])]
        assert ranges[-1][1] == len(sub)


class TestCache:
    def test_cache_limit_still_serves_all_frames(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path).with_cache_limit(1)
        # Alternate between episodes so every access evicts the other one.
        for idx in [0, 4, 9, 1, 5, 10, 2]:
            assert ds[idx].index == idx

    def test_set_cache_limit_and_clear_cache(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        ds.set_cache_limit(2)
        assert [ds[i].episode_index for i in (0, 3, 8)] == [0, 1, 2]
        ds.clear_cache()
        assert ds[11].frame_index == 3

    def test_preload_episodes(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        ds.preload_episodes([0, 2])
        assert ds[0].episode_index == 0
        assert ds[8].episode_index == 2
        # Once a frozen cache exists, episodes outside it are not loaded.
        with pytest.raises(RuntimeError, match="not preloaded"):
            ds[3]

    def test_preload_episodes_out_of_range(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        with pytest.raises(RuntimeError):
            ds.preload_episodes([len(LENGTHS)])


class TestWindowedAccess:
    def test_window_offsets_are_clamped_to_episode(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        # Global frame 4 is frame 1 of episode 1 (length 5).
        window = ds.get_frame_windowed(4, [-5, -1, 0, 2, 10])
        assert isinstance(window, data_core.Episode)
        assert window.num_steps == 5
        batch = window.to_pyarrow()
        assert batch.column("frame_index").to_pylist() == [0, 0, 1, 3, 4]
        assert batch.column("episode_index").to_pylist() == [1] * 5

    def test_window_with_raw_pixels(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path).with_raw_pixels()
        window = ds.get_frame_windowed(4, [-1, 0])
        assert window.has_column("observation.image.cam0.rgb")
        assert window.image_dimensions("cam0") == (W, H)

    def test_window_f32(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path).with_raw_pixels()
        window = ds.get_frame_windowed_f32(4, [0])
        batch = window.to_pyarrow()
        chw = np.asarray(batch.column("observation.image.cam0.f32")[0].as_py(), dtype=np.float32)
        expected = frame_pixels(4, H, W).transpose(2, 0, 1).astype(np.float32) / 255.0
        np.testing.assert_allclose(chw.reshape(3, H, W), expected, atol=1e-6)


class TestGetSampleRaw:
    def test_raw_pixels(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path).with_raw_pixels()
        raw = ds.get_sample_raw(4, [-1, 0], [0, 1])
        assert set(raw) == {
            "images", "state", "action", "action_is_pad", "episode_index", "frame_index", "timestamp"
        }
        frames = decode_raw(raw["images"]["cam0"])
        assert frames.shape == (2, H, W, 3)
        # Pixels come back in HWC order, unchanged.
        np.testing.assert_array_equal(frames[0], frame_pixels(3, H, W))
        np.testing.assert_array_equal(frames[1], frame_pixels(4, H, W))
        assert raw["state"] == [[1.0] * STATE_DIM] * 2
        assert raw["action"] == [[4.0] * ACTION_DIM, [5.0] * ACTION_DIM]
        assert raw["episode_index"] == 1
        assert raw["frame_index"] == 1
        assert raw["timestamp"] == pytest.approx(1 / FPS)

    def test_jpeg_frames_are_decoded(self, jpeg_path):
        ds = data_core.LazyDataset.open(jpeg_path)
        raw = ds.get_sample_raw(0, [0, 1], [])
        assert "action" not in raw
        frames = decode_raw(raw["images"]["cam0"])
        assert frames.shape == (2, H, W, 3)
        # Compare with Pillow's decode of the stored JPEG; decoders may differ by rounding.
        from PIL import Image

        reference = np.asarray(Image.open(io.BytesIO(ds[1].image("cam0"))).convert("RGB"))
        diff = np.abs(frames[1].astype(int) - reference.astype(int))
        assert diff.max() <= 8
        assert diff.mean() < 1.0

    def test_no_images_without_raw_pixels(self, rgb_path):
        # Without with_raw_pixels() the large .rgb column is left out of the window.
        raw = data_core.LazyDataset.open(rgb_path).get_sample_raw(0, [0], [])
        assert raw["images"] == {}

    def test_empty_image_offsets(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path).with_raw_pixels()
        with pytest.raises(RuntimeError):
            ds.get_sample_raw(0, [], [])

    def test_action_offsets_at_dataset_edges(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        assert ds.get_sample_raw(0, [0], [-2, 0])["action"] == [[0.0] * ACTION_DIM] * 2
        last = len(ds) - 1
        raw = ds.get_sample_raw(last, [0], [0, 3])
        assert raw["action"] == [[float(last)] * ACTION_DIM] * 2

    def test_action_offsets_stay_in_episode(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        # Global frame 2 is the last frame of episode 0.
        raw = ds.get_sample_raw(2, [0], [0, 1])
        assert raw["action"] == [[2.0] * ACTION_DIM] * 2
        assert raw["action_is_pad"] == [False, True]
        # Global frame 3 is the first frame of episode 1.
        raw = ds.get_sample_raw(3, [0], [-1, 0, 1])
        assert raw["action"] == [[3.0] * ACTION_DIM, [3.0] * ACTION_DIM, [4.0] * ACTION_DIM]
        assert raw["action_is_pad"] == [True, False, False]

    def test_action_offsets_with_selection(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path).with_episodes([1])
        # Frame 4 of the selection is the last frame of episode 1 (global frame 7).
        raw = ds.get_sample_raw(4, [0], [0, 1])
        assert raw["action"] == [[7.0] * ACTION_DIM] * 2
        assert raw["action_is_pad"] == [False, True]

    def test_random_crop_augmentation(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path).with_raw_pixels()
        ds.add_augmentation(
            "random_crop",
            json.dumps({"height": 16, "width": 20, "camera_names": ["cam0"], "window_size": 2}),
        )
        assert ds.has_augmentations()
        frames = decode_raw(ds.get_sample_raw(5, [-1, 0], [])["images"]["cam0"])
        assert frames.shape == (2, 16, 20, 3)

    def test_frame_dropout_zeroes_raw_pixels(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path).with_raw_pixels()
        ds.add_augmentation(
            "frame_dropout", json.dumps({"camera_names": ["cam0"], "drop_prob": 1.0})
        )
        frames = decode_raw(ds.get_sample_raw(5, [-1, 0], [])["images"]["cam0"])
        assert frames.shape == (2, H, W, 3)
        assert not frames.any()

    def test_unknown_camera_rejected(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path).with_raw_pixels()
        with pytest.raises(ValueError, match="unknown camera"):
            ds.add_augmentation("color_jitter", json.dumps({"camera_names": ["cam_high"]}))
        assert not ds.has_augmentations()

    def test_fused_image_chw_output(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path).with_raw_pixels()
        params = {"camera_names": ["cam0"], "crop_height": H, "crop_width": W, "chw_output": True}
        ds.add_augmentation("fused_image", json.dumps(params))
        pixels, width, height, n_frames = ds.get_sample_raw(6, [0], [])["images"]["cam0"]
        chw = np.frombuffer(pixels, dtype=np.uint8).reshape(n_frames, 3, height, width)
        np.testing.assert_array_equal(chw[0], frame_pixels(6, H, W).transpose(2, 0, 1))

    def test_unknown_augmentation(self, rgb_path):
        ds = data_core.LazyDataset.open(rgb_path)
        with pytest.raises(ValueError, match="Unknown builtin"):
            ds.add_augmentation("nope", "{}")
        with pytest.raises(ValueError):
            ds.add_augmentation("random_crop", "{not json")


class TestTorchDatasetArgumentChecks:
    """Argument validation in to_torch_dataset runs before torch is imported."""

    def test_invalid_mode(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        with pytest.raises(ValueError, match="Invalid mode"):
            ds.to_torch_dataset("frames")

    def test_chunk_size_larger_than_every_episode(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        with pytest.raises(ValueError, match="No valid chunks"):
            ds.to_torch_dataset("chunk", chunk_size=max(LENGTHS) + 1)

    def test_zero_lengths_rejected(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        with pytest.raises(ValueError):
            ds.to_torch_dataset("chunk", chunk_size=0)
        with pytest.raises(ValueError):
            ds.to_torch_dataset("episode", pad_to_length=0)
        with pytest.raises(ValueError):
            ds.to_torch_dataset("episode", truncate_to_length=0)

    def test_properties(self, plain_path):
        ds = data_core.LazyDataset.open(plain_path)
        chunked = ds.to_torch_dataset("chunk", chunk_size=3)
        assert chunked.mode == "chunk"
        assert chunked.chunk_size == 3
        # Valid starts per episode: max(0, length - 3 + 1) -> 1 + 3 + 2.
        assert len(chunked) == 6
        assert repr(chunked) == "LazyTorchDataset(mode='chunk', samples=6)"
        assert len(ds.to_torch_dataset("random")) == sum(LENGTHS)
        assert len(ds.to_torch_dataset("episode")) == len(LENGTHS)
        assert len(ds.with_episodes([1]).to_torch_dataset("episode")) == 1


def test_frame_is_not_constructible():
    with pytest.raises(TypeError):
        data_core.Frame()

