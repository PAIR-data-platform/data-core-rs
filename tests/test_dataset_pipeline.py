"""Tests for data_core.Dataset, Episode, ParallelConfig and the Pipeline built-ins."""

import json
import os

import numpy as np
import pyarrow as pa
import pytest

import data_core
from synthetic import FPS, create_test_dataset

H, W = 24, 32
ACTION_DIM = 4
STATE_DIM = 3
LENGTHS = [6, 8]

IMAGE_PARAMS = {"camera_names": ["cam0"]}

# One entry per built-in name accepted by Pipeline.add_builtin. Image
# augmentations get the synthetic camera; the rest use their defaults or
# point at columns that exist in the synthetic data.
BUILTIN_PARAMS = {
    "action_noise": {"std": 0.1},
    "temporal_subsample": {"step": 2},
    "state_scaling": {"scale": 2.0, "column": "observation.state"},
    "state_noise": {"std": 0.05},
    "color_jitter": {**IMAGE_PARAMS, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
    "random_crop": {**IMAGE_PARAMS, "height": 16, "width": 20},
    "gaussian_blur": {**IMAGE_PARAMS, "sigma": 1.0},
    "gaussian_noise": {**IMAGE_PARAMS, "std": 0.05},
    "horizontal_flip": {**IMAGE_PARAMS, "p": 1.0, "action_mirror_dims": [0]},
    "frame_dropout": {**IMAGE_PARAMS, "drop_prob": 0.5},
    "fused_image": {
        **IMAGE_PARAMS,
        "crop_height": 16,
        "crop_width": 20,
        "brightness": 0.1,
        "noise_std": 0.01,
        "flip_p": 0.5,
    },
}


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


def column(dataset, name):
    """Concatenate one column over all episodes as a Python list."""
    out = []
    for ep in dataset.episodes():
        out.extend(ep.to_pyarrow().column(name).to_pylist())
    return out


class TestDataset:
    def test_metadata(self, jpeg_path):
        ds = data_core.Dataset.from_local(jpeg_path)
        assert ds.num_episodes == len(ds) == len(LENGTHS)
        assert ds.num_steps == sum(LENGTHS)
        assert ds.fps == FPS
        assert ds.robot_type == "test_robot"
        assert ds.name == jpeg_path
        assert ds.camera_names() == ["cam0"]

    def test_episode_index_out_of_range(self, plain_path):
        ds = data_core.Dataset.from_local(plain_path)
        with pytest.raises(RuntimeError):
            ds.episode(len(LENGTHS))

    def test_from_local_missing_directory(self, tmp_path):
        with pytest.raises(RuntimeError):
            data_core.Dataset.from_local(str(tmp_path / "missing"))

    def test_not_constructible(self):
        with pytest.raises(TypeError):
            data_core.Dataset()


class TestSaveLerobot:
    def test_round_trip(self, plain_path, tmp_path):
        ds = data_core.Dataset.from_local(plain_path)
        out = str(tmp_path / "saved")
        ds.save_lerobot(out)

        with open(os.path.join(out, "meta", "info.json")) as f:
            info = json.load(f)
        assert info["total_episodes"] == len(LENGTHS)
        assert info["total_frames"] == sum(LENGTHS)

        reloaded = data_core.Dataset.from_local(out)
        assert reloaded.num_episodes == ds.num_episodes
        assert reloaded.num_steps == ds.num_steps
        assert reloaded.fps == ds.fps
        assert reloaded.robot_type == ds.robot_type
        for name in ("action", "observation.state", "frame_index", "episode_index", "timestamp"):
            assert column(reloaded, name) == column(ds, name), name

    def test_round_trip_after_pipeline(self, plain_path, tmp_path):
        pipeline = data_core.Pipeline()
        pipeline.add_builtin("temporal_subsample", '{"step": 2}')
        augmented = pipeline.run(data_core.Dataset.from_local(plain_path))
        out = str(tmp_path / "saved")
        augmented.save_lerobot(out)

        reloaded = data_core.Dataset.from_local(out)
        assert reloaded.num_steps == sum((n + 1) // 2 for n in LENGTHS)
        assert column(reloaded, "action") == column(augmented, "action")

    def test_round_trip_keeps_jpeg_images(self, jpeg_path, tmp_path):
        ds = data_core.Dataset.from_local(jpeg_path)
        out = str(tmp_path / "saved")
        ds.save_lerobot(out)
        reloaded = data_core.Dataset.from_local(out)
        assert column(reloaded, "observation.image.cam0") == column(ds, "observation.image.cam0")


class TestEpisode:
    def test_from_pyarrow_round_trip(self):
        batch = pa.RecordBatch.from_pydict(
            {
                "episode_index": pa.array([7, 7, 7], pa.int64()),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.arange(6, dtype=np.float32)), 2
                ),
            }
        )
        ep = data_core.Episode.from_pyarrow(batch)
        assert ep.num_steps == 3
        assert ep.episode_index == 7
        assert ep.has_column("action")
        assert not ep.has_column("observation.state")
        assert ep.to_pyarrow().equals(batch)
        assert repr(ep) == "Episode(index=7, steps=3)"

    def test_from_pyarrow_rejects_empty_batch(self):
        batch = pa.RecordBatch.from_pydict({"action": pa.array([], pa.float32())})
        with pytest.raises(RuntimeError, match="at least one timestep"):
            data_core.Episode.from_pyarrow(batch)

    def test_from_pyarrow_rejects_empty_batch_with_episode_index(self):
        batch = pa.RecordBatch.from_pydict({"episode_index": pa.array([], pa.int64())})
        with pytest.raises(RuntimeError):
            data_core.Episode.from_pyarrow(batch)

    def test_decode_images(self, jpeg_path):
        ep = data_core.Dataset.from_local(jpeg_path).episode(0)
        with pytest.raises(RuntimeError):
            ep.image_dimensions("cam0")
        decoded = ep.decode_images(["cam0"])
        assert decoded.num_steps == ep.num_steps
        assert decoded.image_dimensions("cam0") == (W, H)
        for suffix in (".rgb", ".width", ".height"):
            assert decoded.has_column("observation.image.cam0" + suffix)

    def test_decode_images_skips_missing_camera(self, jpeg_path):
        ep = data_core.Dataset.from_local(jpeg_path).episode(0)
        decoded = ep.decode_images(["missing"])
        assert not decoded.has_column("observation.image.missing.rgb")
        assert decoded.to_pyarrow().schema.names == ep.to_pyarrow().schema.names


class TestParallelConfig:
    def test_defaults_and_setters(self):
        cfg = data_core.ParallelConfig()
        assert cfg.num_threads is None
        assert cfg.chunk_size == 1
        threaded = cfg.set_num_threads(2)
        chunked = cfg.set_chunk_size(4)
        assert threaded.num_threads == 2
        assert chunked.chunk_size == 4
        # Setters return new objects.
        assert cfg.num_threads is None
        assert cfg.chunk_size == 1
        assert cfg.set_chunk_size(0).chunk_size == 1

    def test_dataset_parallel_settings(self, plain_path):
        ds = data_core.Dataset.from_local(plain_path)
        cfg = data_core.ParallelConfig().set_num_threads(2).set_chunk_size(2)
        pipeline = data_core.Pipeline()
        pipeline.add_builtin("temporal_subsample", '{"step": 3}')
        expected = column(pipeline.run(ds), "action")
        for configured in (
            ds.with_parallel_config(cfg),
            ds.set_num_threads(2),
            ds.set_chunk_size(2),
            ds.set_chunk_size(0),
        ):
            result = pipeline.run_parallel(configured)
            assert column(result, "action") == expected


class TestBuiltins:
    def test_every_builtin_is_covered(self):
        doc = data_core.Pipeline.__doc__ or ""
        for name in BUILTIN_PARAMS:
            pipeline = data_core.Pipeline()
            # Defaults for everything except the required camera names.
            params = {"camera_names": IMAGE_PARAMS["camera_names"]} if "camera_names" in BUILTIN_PARAMS[name] else {}
            pipeline.add_builtin(name, json.dumps(params))
            assert pipeline.step_names() == [name]
            assert name in doc

    @pytest.mark.parametrize("name", sorted(n for n, p in BUILTIN_PARAMS.items() if "camera_names" in p))
    def test_image_builtins_require_camera_names(self, name):
        with pytest.raises(ValueError, match="camera_names"):
            data_core.Pipeline().add_builtin(name, "{}")

    @pytest.mark.parametrize("parallel", [False, True], ids=["run", "run_parallel"])
    def test_unknown_camera_rejected_at_run(self, jpeg_path, parallel):
        ds = data_core.Dataset.from_local(jpeg_path)
        pipeline = data_core.Pipeline()
        pipeline.add_builtin("color_jitter", json.dumps({"camera_names": ["cam_high"]}))
        run = pipeline.run_parallel if parallel else pipeline.run
        with pytest.raises(ValueError, match="unknown camera"):
            run(ds)

    def test_state_scaling_defaults_to_observation_state(self, jpeg_path):
        ds = data_core.Dataset.from_local(jpeg_path)
        pipeline = data_core.Pipeline()
        pipeline.add_builtin("state_scaling", '{"scale": 2.0}')
        pipeline.run(ds)

    @pytest.mark.parametrize("parallel", [False, True], ids=["run", "run_parallel"])
    @pytest.mark.parametrize("name", sorted(BUILTIN_PARAMS))
    def test_builtin_runs(self, name, parallel, jpeg_path, rgb_path):
        # fused_image works only on raw .rgb pixel columns; the others are run
        # on JPEG images, the layout Dataset.from_local produces from video.
        ds = data_core.Dataset.from_local(rgb_path if name == "fused_image" else jpeg_path)
        pipeline = data_core.Pipeline()
        pipeline.add_builtin(name, json.dumps(BUILTIN_PARAMS[name]))
        assert pipeline.num_steps == 1
        result = pipeline.run_parallel(ds) if parallel else pipeline.run(ds)

        assert result.num_episodes == len(LENGTHS)
        if name == "temporal_subsample":
            assert result.num_steps == sum((n + 1) // 2 for n in LENGTHS)
        else:
            assert result.num_steps == sum(LENGTHS)

        if name == "random_crop":
            decoded = result.episode(0).decode_images(["cam0"])
            assert decoded.image_dimensions("cam0") == (20, 16)
        if name == "fused_image":
            assert result.episode(0).image_dimensions("cam0") == (20, 16)
        if name == "state_scaling":
            assert column(result, "observation.state")[-1] == [2.0] * STATE_DIM
        if name == "action_noise":
            assert column(result, "action") != column(ds, "action")

    @pytest.mark.parametrize("name", ["random_crop", "gaussian_blur"])
    def test_raw_pixel_builtins(self, name, rgb_path):
        ds = data_core.Dataset.from_local(rgb_path)
        pipeline = data_core.Pipeline()
        pipeline.add_builtin(name, json.dumps(BUILTIN_PARAMS[name]))
        result = pipeline.run(ds)
        assert result.num_steps == sum(LENGTHS)

    @pytest.mark.parametrize(
        "name", ["color_jitter", "gaussian_noise", "horizontal_flip", "frame_dropout"]
    )
    def test_raw_pixel_builtins_from_parquet(self, name, rgb_path):
        ds = data_core.Dataset.from_local(rgb_path)
        pipeline = data_core.Pipeline()
        pipeline.add_builtin(name, json.dumps(BUILTIN_PARAMS[name]))
        pipeline.run(ds)

    def test_invalid_params(self):
        pipeline = data_core.Pipeline()
        with pytest.raises(ValueError):
            pipeline.add_builtin("action_noise", "not json")
        with pytest.raises(ValueError, match="Unknown builtin"):
            pipeline.add_builtin("nope", "{}")
        assert pipeline.num_steps == 0

    def test_missing_column_raises_runtime_error(self, plain_path):
        pipeline = data_core.Pipeline()
        pipeline.add_builtin("state_scaling", '{"column": "no_such_column"}')
        with pytest.raises(RuntimeError, match="Column not found"):
            pipeline.run(data_core.Dataset.from_local(plain_path))

    def test_source_dataset_is_unchanged(self, plain_path):
        ds = data_core.Dataset.from_local(plain_path)
        before = column(ds, "action")
        pipeline = data_core.Pipeline()
        pipeline.add_builtin("action_noise", '{"std": 1.0}')
        pipeline.run(ds)
        assert column(ds, "action") == before

