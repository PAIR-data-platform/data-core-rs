"""Type stubs for ``data_core._native``, the Rust extension module.

data-core is a Rust data engine for LeRobot robot-learning datasets. The
classes below are implemented in ``crates/python/src/lib.rs`` with PyO3.
None of them can be subclassed. ``Episode``, ``Dataset``, ``Frame``,
``LazyDataset`` and ``LazyTorchDataset`` have no public constructor: obtain
them from the factory methods documented on each class.

The public names are re-exported from the ``data_core`` package, which ships
a ``py.typed`` marker.
"""

from collections.abc import Sequence
from typing import Any, Literal, Protocol, final

import pyarrow as pa
from typing_extensions import Required, TypeAlias, TypedDict

__all__ = [
    "__version__",
    "Episode",
    "Dataset",
    "Pipeline",
    "ParallelConfig",
    "LazyDataset",
    "LazyTorchDataset",
    "Frame",
]

__version__: str

_TorchSampleMode: TypeAlias = Literal["random", "chunk", "episode"]
"""Sampling mode accepted by ``LazyDataset.to_torch_dataset``."""

class _ArrowArrayExportable(Protocol):
    """Any object implementing the Arrow PyCapsule array interface,
    for example ``pyarrow.RecordBatch``."""

    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...

class _LocalAugmentation(Protocol):
    """Interface for Python objects passed to ``Pipeline.add_local``.

    ``run`` receives one episode as a ``pyarrow.RecordBatch`` and must return
    a ``pyarrow.RecordBatch``. It is called positionally. Like the other
    underscore names here, it exists only for type checking.
    """

    def run(self, episode: pa.RecordBatch, /) -> pa.RecordBatch: ...

class _GlobalAugmentation(Protocol):
    """Interface for Python objects passed to ``Pipeline.add_global``.

    ``run`` receives every episode as a list of ``pyarrow.RecordBatch`` and a
    metadata dict, and must return a list of ``pyarrow.RecordBatch``. It is
    called positionally. It exists only for type checking.
    """

    def run(
        self, episodes: list[pa.RecordBatch], metadata: dict[str, Any], /
    ) -> list[pa.RecordBatch]: ...

class _RawSample(TypedDict, total=False):
    """Return value of ``LazyDataset.get_sample_raw`` (a plain dict at runtime).

    ``images`` is always present. Each entry maps a camera name to
    ``(pixels, width, height, n_frames)``, where ``pixels`` holds
    ``n_frames * height * width * 3`` uint8 values, HWC order per frame.
    ``state`` is present when the dataset has an ``observation.state``
    column, ``action`` and ``action_is_pad`` only when ``action_offsets`` is
    non-empty. ``action_is_pad`` is True where an action offset fell outside
    the episode and the boundary action was repeated.
    ``episode_index``, ``frame_index`` and ``timestamp`` are present when the
    matching column exists; ``frame_index`` and ``timestamp`` are read from
    the last row of the image window.
    """

    images: Required[dict[str, tuple[bytes, int, int, int]]]
    state: list[list[float]]
    action: list[list[float]]
    action_is_pad: list[bool]
    episode_index: int
    frame_index: int
    timestamp: float

@final
class Episode:
    """A single episode (trajectory) backed by an Arrow RecordBatch.

    Obtain episodes from ``Dataset.episode``, ``LazyDataset.get_episode`` or
    ``Episode.from_pyarrow``.
    """

    @property
    def num_steps(self) -> int:
        """Number of timesteps (rows) in this episode."""
    @property
    def episode_index(self) -> int:
        """Index of this episode in its dataset."""
    def to_pyarrow(self) -> pa.RecordBatch:
        """Return the episode data as a ``pyarrow.RecordBatch`` without copying."""
    @classmethod
    def from_pyarrow(cls, pyarrow_batch: _ArrowArrayExportable) -> Episode:
        """Create an episode from a ``pyarrow.RecordBatch`` without copying.

        The episode index is read from the first row of an ``episode_index``
        int64 column, or 0 if there is none. Raises ``RuntimeError`` if the
        batch has no rows.
        """
    def decode_images(self, camera_names: Sequence[str]) -> Episode:
        """Decode JPEG columns ``observation.image.{camera}`` to raw RGB.

        Returns a new episode with ``.rgb``, ``.width`` and ``.height``
        columns added for each camera. Cameras without a JPEG column are
        skipped. Raises ``RuntimeError`` if a frame fails to decode.
        """
    def has_column(self, name: str) -> bool:
        """Return True if the episode has a column called ``name``."""
    def image_dimensions(self, camera_name: str) -> tuple[int, int]:
        """Return ``(width, height)`` for a camera with raw pixel columns.

        Raises ``RuntimeError`` if the camera has no ``.width`` and
        ``.height`` columns, for example before ``decode_images``.
        """

@final
class Dataset:
    """An in-memory robotics dataset: a list of episodes plus metadata.

    Create one with ``Dataset.from_local`` or ``LazyDataset.load_all``.
    """

    @classmethod
    def from_local(cls, path: str) -> Dataset:
        """Load every episode of a LeRobot v2 or v3 dataset directory into memory."""
    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
    @property
    def num_steps(self) -> int:
        """Total number of steps across all episodes."""
    @property
    def name(self) -> str:
        """Dataset name (the source path for datasets loaded from disk)."""
    @property
    def robot_type(self) -> str | None:
        """Robot type from ``info.json``, if set."""
    @property
    def fps(self) -> float | None:
        """Frames per second from ``info.json``, if set."""
    def camera_names(self) -> list[str]:
        """Camera names declared in the ``info.json`` features."""
    def episode(self, index: int) -> Episode:
        """Return the episode at position ``index``.

        Raises ``RuntimeError`` if ``index`` is out of range.
        """
    def episodes(self) -> list[Episode]:
        """Return all episodes as a list."""
    def save_lerobot(self, output_dir: str) -> None:
        """Write the dataset to ``output_dir`` in the LeRobot v2 layout."""
    def set_num_threads(self, num_threads: int) -> Dataset:
        """Return a copy that uses ``num_threads`` threads for parallel operations.

        Other parallel settings, such as the chunk size, are kept.
        """
    def set_chunk_size(self, chunk_size: int) -> Dataset:
        """Return a copy that groups ``chunk_size`` episodes per parallel task."""
    def with_parallel_config(self, config: ParallelConfig) -> Dataset:
        """Return a copy that uses the given ``ParallelConfig``."""
    def __len__(self) -> int:
        """Number of episodes."""

@final
class ParallelConfig:
    """Thread count and chunk size used by ``Pipeline.run_parallel``.

    The setters return a new object and leave the original unchanged.
    """

    def __init__(self) -> None:
        """Create a config that uses Rayon's default thread pool and a chunk size of 1."""
    def set_num_threads(self, num_threads: int) -> ParallelConfig:
        """Return a copy that uses ``num_threads`` worker threads."""
    def set_chunk_size(self, chunk_size: int) -> ParallelConfig:
        """Return a copy with the given chunk size (values below 1 become 1)."""
    @property
    def num_threads(self) -> int | None:
        """Configured thread count, or None for Rayon's default pool."""
    @property
    def chunk_size(self) -> int:
        """Number of episodes per parallel task."""

@final
class Pipeline:
    """An ordered list of augmentation steps applied to a ``Dataset``.

    Built-in step names accepted by ``add_builtin``: ``action_noise``,
    ``temporal_subsample``, ``state_scaling``, ``state_noise``,
    ``color_jitter``, ``random_crop``, ``gaussian_blur``, ``gaussian_noise``,
    ``horizontal_flip``, ``frame_dropout`` and ``fused_image``.
    """

    def __init__(self) -> None:
        """Create an empty pipeline."""
    def add_builtin(self, name: str, params_json: str) -> None:
        """Append a built-in Rust augmentation.

        ``params_json`` is a JSON object, for example ``'{"std": 0.05}'``.
        Missing parameters take their defaults. Image augmentations require
        a ``camera_names`` list (names as returned by ``camera_names()``);
        ``gaussian_noise``, ``horizontal_flip``, ``frame_dropout`` and
        ``fused_image`` change only raw ``.rgb`` pixel columns;
        ``gaussian_blur`` changes only JPEG columns. Raises
        ``ValueError`` for an unknown name, invalid JSON, or an image
        augmentation without camera names. ``run`` and ``run_parallel`` raise
        ``ValueError`` if a step names a camera the dataset does not have.
        """
    def add_local(self, instance: _LocalAugmentation) -> None:
        """Append a Python per-episode augmentation (see ``_LocalAugmentation``).

        Raises ``ValueError`` if ``instance`` has no callable ``run``.
        """
    def add_global(self, instance: _GlobalAugmentation) -> None:
        """Append a Python whole-dataset augmentation (see ``_GlobalAugmentation``).

        Raises ``ValueError`` if ``instance`` has no callable ``run``.
        """
    @property
    def num_steps(self) -> int:
        """Number of steps in the pipeline."""
    def step_names(self) -> list[str]:
        """Names of the steps, in execution order."""
    def run(self, dataset: Dataset) -> Dataset:
        """Apply every step in order and return a new dataset.

        Raises ``ValueError`` if a built-in step names a camera the dataset
        does not have.
        """
    def run_parallel(self, dataset: Dataset) -> Dataset:
        """Like ``run``, but local steps run in parallel across episodes.

        Uses the dataset's ``ParallelConfig``. Global steps still run
        sequentially.
        """

@final
class Frame:
    """A single timestep returned by ``LazyDataset.get_frame`` or ``LazyDataset[idx]``."""

    @property
    def index(self) -> int:
        """Global frame index in the full dataset."""
    @property
    def episode_index(self) -> int:
        """Index of the episode this frame belongs to."""
    @property
    def frame_index(self) -> int:
        """Frame index within the episode (0-based)."""
    @property
    def timestamp(self) -> float:
        """Timestamp in seconds."""
    @property
    def action(self) -> list[float]:
        """Action vector."""
    @property
    def state(self) -> list[float] | None:
        """``observation.state`` vector, or None if the dataset has none."""
    @property
    def reward(self) -> float | None:
        """Reward, or None if the dataset has no ``reward`` column."""
    @property
    def done(self) -> bool:
        """True for a terminal frame; False if the dataset has no ``done`` column."""
    @property
    def task_index(self) -> int:
        """Task index."""
    @property
    def task(self) -> str | None:
        """Task description from ``tasks.jsonl``, if available."""
    def image(self, camera: str) -> bytes | None:
        """JPEG bytes for ``camera``, or None if this frame has no JPEG image for it."""
    def camera_names(self) -> list[str]:
        """Names of the cameras that have a JPEG image in this frame."""

@final
class LazyDataset:
    """A dataset that reads metadata up front and loads episodes on demand.

    Frame indices passed to ``get_frame``, ``__getitem__``,
    ``get_frame_windowed`` and ``get_sample_raw`` count frames across the
    selected episodes (see ``with_episodes``). ``get_episode`` and
    ``preload_episodes`` take dataset episode indices.

    ``with_episodes``, ``with_cache_limit`` and ``with_raw_pixels`` return a
    new dataset that keeps the episode selection, raw-pixel mode and cache
    limit of the original. Call them before ``add_augmentation``: calling one
    after an augmentation was added raises ``ValueError``.
    """

    @classmethod
    def open(cls, path: str) -> LazyDataset:
        """Open a LeRobot dataset directory, reading only the metadata files."""
    def with_episodes(self, episodes: Sequence[int]) -> LazyDataset:
        """Return a new dataset restricted to the given episode indices, in that order."""
    def with_cache_limit(self, limit: int) -> LazyDataset:
        """Return a new dataset that caches at most ``limit`` episodes (0 means unlimited)."""
    def with_raw_pixels(self) -> LazyDataset:
        """Return a new dataset that decodes video to raw RGB instead of JPEG bytes."""
    def set_cache_limit(self, limit: int) -> None:
        """Set the episode cache limit in place (0 means unlimited)."""
    def clear_cache(self) -> None:
        """Drop all cached episodes."""
    def preload_episodes(self, episode_indices: Sequence[int]) -> None:
        """Load episodes into a read-only cache shared with forked DataLoader workers.

        After this call, accessing an episode that was not preloaded raises
        ``RuntimeError``.
        """
    @property
    def name(self) -> str:
        """Dataset name (the directory name)."""
    @property
    def fps(self) -> float:
        """Frames per second."""
    @property
    def num_frames(self) -> int:
        """Number of frames in the selected episodes."""
    @property
    def num_episodes(self) -> int:
        """Number of selected episodes."""
    def camera_names(self) -> list[str]:
        """Camera names declared in the ``info.json`` features."""
    def episode_data_index(self) -> list[tuple[int, int]]:
        """``(from_index, to_index)`` frame range of each selected episode.

        Ranges use the frame index space of ``get_frame`` and ``__getitem__``:
        the dataset's global ranges without ``with_episodes``, and contiguous
        ranges from 0 across the selected episodes with it.
        """
    def get_frame(self, idx: int) -> Frame:
        """Return frame ``idx``, loading its episode if needed.

        Raises ``RuntimeError`` if ``idx`` is out of range.
        """
    def get_episode(self, idx: int) -> Episode:
        """Return episode ``idx`` (a dataset episode index), loading and caching it if needed."""
    def get_frame_windowed(self, idx: int, window_offsets: Sequence[int]) -> Episode:
        """Return the frames at ``idx + offset`` for each offset, with augmentations applied.

        Offsets are clamped to the episode that contains ``idx``.
        ``window_offsets`` must not be empty.
        """
    def add_augmentation(self, name: str, params_json: str) -> None:
        """Add a built-in augmentation applied by the windowed accessors.

        Takes the same names and JSON parameters as ``Pipeline.add_builtin``.
        The ``window_size`` parameter keeps random parameters the same across
        the frames of one window (``gaussian_noise`` still draws new noise for
        every frame). Raises ``ValueError`` if ``camera_names`` names a camera
        that is not in ``camera_names()``.
        """
    def get_frame_windowed_f32(self, idx: int, window_offsets: Sequence[int]) -> Episode:
        """Like ``get_frame_windowed``, plus float32 CHW ``.f32`` image columns in [0, 1]."""
    def get_sample_raw(
        self, idx: int, image_offsets: Sequence[int], action_offsets: Sequence[int]
    ) -> _RawSample:
        """Return one training sample with images as raw uint8 bytes.

        Image offsets are clamped to the episode that contains ``idx``.
        Action offsets are clamped to the same episode; ``action_is_pad``
        marks the clamped ones.
        ``image_offsets`` must not be empty.
        """
    def has_augmentations(self) -> bool:
        """Return True if ``add_augmentation`` has been called."""
    def load_all(self) -> Dataset:
        """Load the selected episodes into an in-memory ``Dataset``."""
    def to_torch_dataset(
        self,
        mode: _TorchSampleMode,
        chunk_size: int | None = None,
        pad_to_length: int | None = None,
        truncate_to_length: int | None = None,
    ) -> LazyTorchDataset:
        """Wrap this dataset as a PyTorch map-style dataset.

        ``mode`` is ``"random"`` (one frame per sample), ``"chunk"``
        (``chunk_size`` contiguous frames from one episode) or ``"episode"``
        (one whole episode, first truncated to ``truncate_to_length`` and
        then zero-padded to ``pad_to_length``). The episode selection is kept;
        augmentations, cache limit and raw-pixel mode are not. torch must be
        installed when samples are read.
        """
    def torch_dataloader(
        self,
        mode: _TorchSampleMode,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        chunk_size: int | None = None,
        pad_to_length: int | None = None,
        truncate_to_length: int | None = None,
    ) -> Any:
        """Return a ``torch.utils.data.DataLoader`` over ``to_torch_dataset(...)``.

        The loader uses ``LazyTorchDataset.collate_fn``. Requires torch.
        """
    def __getitem__(self, idx: int, /) -> Frame:
        """Return frame ``idx``; negative indices count from the end.

        Raises ``IndexError`` when ``idx`` is out of range.
        """
    def __len__(self) -> int:
        """Number of frames in the selected episodes."""

@final
class LazyTorchDataset:
    """A PyTorch map-style dataset created by ``LazyDataset.to_torch_dataset``.

    Each item is a dict of torch tensors. In ``"random"`` mode it has
    ``episode_index``, ``frame_index``, ``global_index``, ``timestamp``,
    ``action``, ``reward`` (NaN when absent), ``done``, ``images`` (a dict of
    float32 CHW tensors in [0, 1]) and ``state`` when available. In
    ``"chunk"`` and ``"episode"`` modes the per-frame fields gain a leading
    time dimension and the dict also has ``start_frame_index``,
    ``valid_length`` and ``valid_mask``.
    """

    @property
    def mode(self) -> str:
        """Sampling mode: ``"random"``, ``"chunk"`` or ``"episode"``."""
    @property
    def chunk_size(self) -> int:
        """Frames per sample in chunk mode (1 if not set)."""
    def __len__(self) -> int:
        """Number of samples: frames, valid chunk starts or episodes, depending on the mode."""
    def __getitem__(self, idx: int, /) -> dict[str, Any]:
        """Return sample ``idx`` as a dict of tensors.

        Raises ``IndexError`` when ``idx`` is out of range.
        """
    def collate_fn(self, batch: list[dict[str, Any]]) -> Any:
        """Collate a list of samples for ``torch.utils.data.DataLoader``.

        In episode mode the list is returned unchanged unless both
        ``pad_to_length`` and ``truncate_to_length`` are set with
        ``pad_to_length >= truncate_to_length``. Otherwise missing camera
        images are filled with zeros and torch's ``default_collate`` is applied.
        """
