"""data-core: a Rust data engine for LeRobot robot-learning datasets.

The classes are implemented in Rust (``data_core._native``) and re-exported
here. ``data_core.torch`` adds a PyTorch ``Dataset`` that returns
LeRobot-style samples; it needs ``torch``.
"""

from ._native import (
    Dataset as Dataset,
    Episode as Episode,
    Frame as Frame,
    LazyDataset as LazyDataset,
    LazyTorchDataset as LazyTorchDataset,
    ParallelConfig as ParallelConfig,
    Pipeline as Pipeline,
    __version__ as __version__,
)

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
