"""PyTorch ``Dataset`` over ``data_core.LazyDataset`` with LeRobot-style samples.

Each sample is produced by one call to ``LazyDataset.get_sample_raw``. Rust
loads the episode (from cache when possible), applies the configured
augmentation and returns the image window as raw uint8 bytes, which are
wrapped with ``torch.frombuffer``.
"""

from __future__ import annotations

import bisect
import json
import random
import warnings
from collections.abc import Iterator
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import data_core


class EpisodeGroupedSampler(Sampler[int]):
    """Sampler that yields all frames of one episode before the next.

    Episode order is shuffled each epoch when ``shuffle`` is True, and so is
    the frame order inside each episode. Keeping one episode's frames
    together keeps the episode cache hot.

    Args:
        episode_boundaries: ``(from_index, to_index)`` frame range of each
            episode, as returned by ``LazyDataset.episode_data_index``.
        shuffle: Shuffle episodes and the frames within them.
    """

    def __init__(self, episode_boundaries: list[tuple[int, int]], shuffle: bool = True):
        self._boundaries = episode_boundaries
        self._shuffle = shuffle
        self._total = sum(end - start for start, end in episode_boundaries)

    def __iter__(self) -> Iterator[int]:
        ep_order = list(range(len(self._boundaries)))
        if self._shuffle:
            random.shuffle(ep_order)
        for ep_idx in ep_order:
            start, end = self._boundaries[ep_idx]
            indices = list(range(start, end))
            if self._shuffle:
                random.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return self._total


def _fps_to_frame_offsets(
    key: str, delta_timestamps: list[float], fps: float, tolerance_s: float
) -> list[int]:
    """Convert timestamps to frame offsets, rejecting ones between frames.

    As in ``LeRobotDataset``, every timestamp must be within ``tolerance_s``
    of a multiple of ``1 / fps``.
    """
    offsets = [round(dt * fps) for dt in delta_timestamps]
    off_grid = [dt for dt, off in zip(delta_timestamps, offsets) if abs(dt - off / fps) > tolerance_s]
    if off_grid:
        raise ValueError(
            f"delta_timestamps[{key!r}] has timestamps that are not multiples of 1/fps "
            f"(fps={fps}, tolerance_s={tolerance_s}): {off_grid[:5]}"
        )
    return offsets


class AugmentedDataset(Dataset):
    """Map-style dataset that returns LeRobot-style sample dicts.

    The dataset is opened in raw-pixel mode, so video is decoded to RGB once
    per episode and cached.

    Args:
        dataset_path: Local LeRobot dataset directory.
        delta_timestamps: LeRobot-style relative timestamps in seconds. The
            first key containing ``"image"`` sets the image window for every
            camera (default ``[0.0]``), and the ``"action"`` key sets the
            action window (default: no actions returned).
            ``observation.state`` follows the image window; other keys are
            ignored with a warning. An ``observation.images.{cam}`` key must
            name a camera of the dataset, and every timestamp must be a
            multiple of ``1 / fps``; otherwise ValueError is raised.
        fps: Frame rate used to convert timestamps to frame offsets. Defaults
            to the dataset's fps.
        augmentation: Optional name of a built-in augmentation, as accepted by
            ``data_core.Pipeline.add_builtin``. It applies to every camera
            unless ``augmentation_params["camera_names"]`` names a subset.
        augmentation_params: JSON-serializable parameters for
            ``augmentation``. Missing parameters take their defaults.
        cache_limit: Maximum number of episodes kept in the cache; 0 means
            unlimited.
        float_images: Return float32 images in [0, 1] instead of uint8.
        tolerance_s: Largest allowed distance, in seconds, between a timestamp
            and the nearest frame. Larger gaps raise ValueError.

    Each sample contains, for every camera ``cam``:
    ``observation.images.{cam}`` with shape ``(T, 3, H, W)`` and
    ``observation.images.{cam}_is_pad`` with shape ``(T,)``. When available
    it also has ``observation.state`` (``(D,)`` for a single-frame window,
    else ``(T, D)``), ``action`` and ``action_is_pad``, ``episode_index``,
    ``frame_index`` and ``timestamp``. Offsets that fall outside the episode
    are clamped to its first or last frame and marked True in the matching
    ``_is_pad`` tensor, as ``LeRobotDataset`` does.
    """

    def __init__(
        self,
        dataset_path: str,
        delta_timestamps: dict[str, list[float]],
        fps: Optional[float] = None,
        augmentation: Optional[str] = None,
        augmentation_params: Optional[dict[str, Any]] = None,
        cache_limit: int = 3,
        float_images: bool = False,
        tolerance_s: float = 1e-4,
    ):
        self._ds = data_core.LazyDataset.open(str(dataset_path)).with_raw_pixels()
        if cache_limit > 0:
            self._ds.set_cache_limit(cache_limit)

        self._fps = fps or self._ds.fps
        self._camera_names = self._ds.camera_names()
        self._float_images = float_images

        image_offsets: Optional[list[int]] = None
        action_offsets: Optional[list[int]] = None
        for key, dts in delta_timestamps.items():
            offsets = _fps_to_frame_offsets(key, dts, self._fps, tolerance_s)
            if "image" in key:
                camera = key.rsplit(".", 1)[-1]
                if key.startswith("observation.images.") and camera not in self._camera_names:
                    raise ValueError(
                        f"delta_timestamps key {key!r} names camera {camera!r}, "
                        f"but this dataset has {self._camera_names}"
                    )
                if image_offsets is None:
                    image_offsets = offsets
            elif key == "action":
                action_offsets = offsets
            elif key != "observation.state":
                warnings.warn(
                    f"delta_timestamps key {key!r} is ignored: only image keys and "
                    f"'action' set windows (observation.state follows the image window)",
                    stacklevel=2,
                )
        self._image_offsets: list[int] = image_offsets if image_offsets is not None else [0]
        self._action_offsets: list[int] = action_offsets if action_offsets is not None else []

        # Rust returns HWC bytes, except for fused_image with chw_output=True.
        self._images_chw = False
        if augmentation:
            params = dict(augmentation_params or {})
            # Image augmentations need target cameras; default to all of them.
            params.setdefault("camera_names", list(self._camera_names))
            if augmentation == "fused_image":
                params["chw_output"] = True
                self._images_chw = True
            self._ds.add_augmentation(augmentation, json.dumps(params))

        self._n_frames = self._ds.num_frames
        self._episode_bounds = self._ds.episode_data_index()
        self._episode_starts = [start for start, _ in self._episode_bounds]

    @property
    def num_episodes(self) -> int:
        """Number of episodes in the dataset."""
        return self._ds.num_episodes

    @property
    def episode_data_index(self) -> list[tuple[int, int]]:
        """``(from_index, to_index)`` frame range of each episode."""
        return list(self._episode_bounds)

    def preload_episodes(self, episode_indices: Optional[list[int]] = None) -> None:
        """Load episodes into a read-only cache shared with forked workers.

        Call this before creating a DataLoader with ``num_workers > 0``.
        Forked workers share the cache copy-on-write. Afterwards, frames from
        episodes that were not preloaded can no longer be read.

        Args:
            episode_indices: Episodes to load; None loads all of them.
        """
        if episode_indices is None:
            episode_indices = list(range(self._ds.num_episodes))
        self._ds.preload_episodes(episode_indices)

    def __len__(self) -> int:
        return self._n_frames

    def _episode_bounds_for(self, idx: int) -> tuple[int, int]:
        ep = bisect.bisect_right(self._episode_starts, idx) - 1
        return self._episode_bounds[ep]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += self._n_frames
        if not 0 <= idx < self._n_frames:
            raise IndexError(f"index {idx} out of range for dataset with {self._n_frames} frames")

        start, end = self._episode_bounds_for(idx)
        image_is_pad = torch.tensor(
            [not start <= idx + off < end for off in self._image_offsets], dtype=torch.bool
        )
        # get_sample_raw clamps action offsets to this episode and returns
        # the matching padding mask.
        raw = self._ds.get_sample_raw(idx, self._image_offsets, self._action_offsets)
        result: dict[str, torch.Tensor] = {}

        for cam, (img_bytes, w, h, n_frames) in raw["images"].items():
            t = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8)
            if self._images_chw:
                t = t.reshape(n_frames, 3, h, w)
            else:
                t = t.reshape(n_frames, h, w, 3).permute(0, 3, 1, 2).contiguous()
            if self._float_images:
                t = t.float().div_(255.0)
            result[f"observation.images.{cam}"] = t
            result[f"observation.images.{cam}_is_pad"] = image_is_pad.clone()

        if "state" in raw:
            states = raw["state"]
            if len(states) == 1:
                result["observation.state"] = torch.tensor(states[0], dtype=torch.float32)
            else:
                result["observation.state"] = torch.tensor(states, dtype=torch.float32)

        if "action" in raw:
            result["action"] = torch.tensor(raw["action"], dtype=torch.float32)
            result["action_is_pad"] = torch.tensor(raw["action_is_pad"], dtype=torch.bool)

        if "episode_index" in raw:
            result["episode_index"] = torch.tensor(raw["episode_index"], dtype=torch.int64)
        if "frame_index" in raw:
            result["frame_index"] = torch.tensor(raw["frame_index"], dtype=torch.int64)
        if "timestamp" in raw:
            result["timestamp"] = torch.tensor(raw["timestamp"], dtype=torch.float32)

        return result


def make_dataloader(
    dataset: AugmentedDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    preload_episodes: Optional[list[int]] = None,
    episode_grouped: bool = True,
) -> DataLoader:
    """Create a ``DataLoader`` for an ``AugmentedDataset``.

    With ``num_workers > 0`` this also:

    - preloads ``preload_episodes`` (all episodes when None) so forked
      workers share the decoded frames copy-on-write;
    - uses ``EpisodeGroupedSampler`` over the preloaded episodes when
      ``episode_grouped`` is True, so samples come only from those episodes;
    - starts workers with the ``"fork"`` multiprocessing context, which the
      preloaded cache relies on (Linux only).

    With ``num_workers == 0`` episodes are read on demand and
    ``prefetch_factor``, ``preload_episodes`` and ``episode_grouped`` are
    ignored.
    """
    sampler = None
    if num_workers > 0:
        dataset.preload_episodes(preload_episodes)

        if episode_grouped:
            all_boundaries = dataset.episode_data_index
            if preload_episodes is not None:
                boundaries = [all_boundaries[i] for i in preload_episodes]
            else:
                boundaries = all_boundaries
            sampler = EpisodeGroupedSampler(boundaries, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
