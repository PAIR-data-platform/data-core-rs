"""PyTorch frontend for data-core (``data_core.torch``).

``AugmentedDataset`` wraps ``data_core.LazyDataset`` and returns
LeRobot-style sample dicts, so it can stand in for ``LeRobotDataset`` in an
existing training loop. Requires ``torch``.
"""

try:
    import torch as _torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "data_core.torch needs PyTorch (pip install torch)"
    ) from exc

from .dataset import AugmentedDataset, EpisodeGroupedSampler, make_dataloader

__all__ = ["AugmentedDataset", "EpisodeGroupedSampler", "make_dataloader"]
