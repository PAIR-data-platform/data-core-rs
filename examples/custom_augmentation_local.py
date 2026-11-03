"""
Custom Local Augmentation Example

Demonstrates how to create a custom local augmentation class that processes
one episode at a time. Local augmentations transform a single trajectory
(episode) into a modified version of that same trajectory.

This example shows:
1. How to define a local augmentation class with __init__ and run() methods
2. How to work with PyArrow RecordBatch for zero-copy efficiency
3. How to add custom augmentations to a pipeline
4. Multiple example augmentations: action smoothing, episode truncation, custom noise

Usage:
    python examples/custom_augmentation_local.py

Requirements:
    pip install pyarrow numpy
"""

import numpy as np
import pyarrow as pa
from typing import Optional

import data_core


# ============================================================================
# Example 1: Action Smoothing via Exponential Moving Average
# ============================================================================

class ActionSmoothingAugmentation:
    """
    Smooth action sequences using an exponential moving average.
    This can help reduce jitter in recorded demonstrations.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize the smoothing augmentation.
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Lower values = more smoothing.
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in range (0, 1]")
        self.alpha = alpha
    
    def run(self, episode: pa.RecordBatch) -> pa.RecordBatch:
        """
        Apply exponential moving average smoothing to actions.
        
        Args:
            episode: Input episode as PyArrow RecordBatch
            
        Returns:
            Episode with smoothed actions
        """
        # Extract action column (FixedSizeList of floats)
        actions_col = episode.column("action")
        
        # Convert to numpy for easier manipulation
        # actions is a FixedSizeListArray, we need to flatten it
        num_steps = len(actions_col)
        action_dim = actions_col.type.list_size
        
        # Convert to 2D numpy array (num_steps, action_dim)
        actions_np = np.array([
            actions_col[i].as_py() for i in range(num_steps)
        ], dtype=np.float32)
        
        # Apply exponential moving average
        smoothed = np.zeros_like(actions_np)
        smoothed[0] = actions_np[0]
        
        for t in range(1, num_steps):
            smoothed[t] = self.alpha * actions_np[t] + (1 - self.alpha) * smoothed[t-1]
        
        # Convert back to PyArrow FixedSizeList
        smoothed_list = pa.array(
            [smoothed[i].tolist() for i in range(num_steps)],
            type=pa.list_(pa.float32(), action_dim)
        )
        
        # Create new RecordBatch with smoothed actions
        schema = episode.schema
        columns = []
        for i, field in enumerate(schema):
            if field.name == "action":
                columns.append(smoothed_list)
            else:
                columns.append(episode.column(i))
        
        return pa.RecordBatch.from_arrays(columns, schema=schema)


# ============================================================================
# Example 2: Episode Truncation
# ============================================================================

class TruncateEpisodeAugmentation:
    """
    Truncate episodes to a maximum length. Useful for removing redundant
    post-task completion steps or standardizing episode lengths.
    """
    
    def __init__(self, max_steps: int = 100):
        """
        Initialize the truncation augmentation.
        
        Args:
            max_steps: Maximum number of steps to keep from the beginning
        """
        self.max_steps = max_steps
    
    def run(self, episode: pa.RecordBatch) -> pa.RecordBatch:
        """
        Truncate episode to max_steps.
        
        Args:
            episode: Input episode as PyArrow RecordBatch
            
        Returns:
            Truncated episode
        """
        num_steps = len(episode)
        
        if num_steps <= self.max_steps:
            # No truncation needed
            return episode
        
        # Slice the RecordBatch to keep only first max_steps
        return episode.slice(0, self.max_steps)


# ============================================================================
# Example 3: Custom Action Noise with Per-Dimension Scaling
# ============================================================================

class PerDimensionActionNoiseAugmentation:
    """
    Add Gaussian noise to actions with different scales per dimension.
    This is useful when different action dimensions have different sensitivities.
    """
    
    def __init__(self, noise_scales: list[float], seed: Optional[int] = None):
        """
        Initialize per-dimension action noise.
        
        Args:
            noise_scales: List of noise standard deviations, one per action dimension
            seed: Random seed for reproducibility
        """
        self.noise_scales = np.array(noise_scales, dtype=np.float32)
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def run(self, episode: pa.RecordBatch) -> pa.RecordBatch:
        """
        Add dimension-specific Gaussian noise to actions.
        
        Args:
            episode: Input episode as PyArrow RecordBatch
            
        Returns:
            Episode with noisy actions
        """
        # Extract actions
        actions_col = episode.column("action")
        num_steps = len(actions_col)
        action_dim = actions_col.type.list_size
        
        # Validate noise_scales matches action_dim
        if len(self.noise_scales) != action_dim:
            raise ValueError(
                f"noise_scales length ({len(self.noise_scales)}) must match "
                f"action_dim ({action_dim})"
            )
        
        # Convert to numpy
        actions_np = np.array([
            actions_col[i].as_py() for i in range(num_steps)
        ], dtype=np.float32)
        
        # Generate and add noise
        noise = self._rng.randn(num_steps, action_dim).astype(np.float32)
        noise = noise * self.noise_scales[np.newaxis, :]  # Scale per dimension
        noisy_actions = actions_np + noise
        
        # Convert back to PyArrow
        noisy_list = pa.array(
            [noisy_actions[i].tolist() for i in range(num_steps)],
            type=pa.list_(pa.float32(), action_dim)
        )
        
        # Create new RecordBatch
        schema = episode.schema
        columns = []
        for i, field in enumerate(schema):
            if field.name == "action":
                columns.append(noisy_list)
            else:
                columns.append(episode.column(i))
        
        return pa.RecordBatch.from_arrays(columns, schema=schema)


# ============================================================================
# Example 4: State Velocity Augmentation
# ============================================================================

class StateVelocityAugmentation:
    """
    Add velocity information to the state by computing finite differences.
    This augments the state space with velocity features.
    """
    
    def __init__(self, state_column: str = "state", dt: Optional[float] = None):
        """
        Initialize state velocity augmentation.
        
        Args:
            state_column: Name of the state column to compute velocities for
            dt: Time delta for velocity computation. If None, uses unit time.
        """
        self.state_column = state_column
        self.dt = dt
    
    def run(self, episode: pa.RecordBatch) -> pa.RecordBatch:
        """
        Augment state with velocity features.
        
        Args:
            episode: Input episode as PyArrow RecordBatch
            
        Returns:
            Episode with velocity-augmented state
        """
        # Check if state column exists
        if self.state_column not in episode.schema.names:
            # If no state column, just return as-is
            return episode
        
        # Extract state
        state_col = episode.column(self.state_column)
        num_steps = len(state_col)
        state_dim = state_col.type.list_size
        
        # Convert to numpy
        states_np = np.array([
            state_col[i].as_py() for i in range(num_steps)
        ], dtype=np.float32)
        
        # Compute velocities (finite difference)
        velocities = np.zeros_like(states_np)
        velocities[0] = 0  # First step has zero velocity
        
        for t in range(1, num_steps):
            if self.dt is not None:
                velocities[t] = (states_np[t] - states_np[t-1]) / self.dt
            else:
                velocities[t] = states_np[t] - states_np[t-1]
        
        # Concatenate state and velocity
        augmented_state = np.concatenate([states_np, velocities], axis=1)
        new_state_dim = state_dim * 2
        
        # Convert back to PyArrow
        augmented_list = pa.array(
            [augmented_state[i].tolist() for i in range(num_steps)],
            type=pa.list_(pa.float32(), new_state_dim)
        )
        
        # Create new RecordBatch with augmented state and updated schema
        old_schema = episode.schema
        new_fields = []
        columns = []
        
        for i, field in enumerate(old_schema):
            if field.name == self.state_column:
                # Update field type to reflect new dimension
                new_field = pa.field(field.name, pa.list_(pa.float32(), new_state_dim))
                new_fields.append(new_field)
                columns.append(augmented_list)
            else:
                new_fields.append(field)
                columns.append(episode.column(i))
        
        new_schema = pa.schema(new_fields)
        return pa.RecordBatch.from_arrays(columns, schema=new_schema)


# ============================================================================
# Main Demo
# ============================================================================

def create_synthetic_episode(num_steps: int = 50, action_dim: int = 7) -> pa.RecordBatch:
    """Create a synthetic episode for demonstration purposes."""
    
    # Create synthetic data
    actions = np.random.randn(num_steps, action_dim).astype(np.float32) * 0.5
    states = np.cumsum(actions, axis=0)  # Integrate actions to get state
    timestamps = np.linspace(0, 5, num_steps)
    step_indices = np.arange(num_steps, dtype=np.int64)
    episode_indices = np.full(num_steps, 0, dtype=np.int64)
    rewards = np.random.rand(num_steps).astype(np.float32)
    dones = np.zeros(num_steps, dtype=bool)
    dones[-1] = True
    
    # Build PyArrow RecordBatch
    batch = pa.RecordBatch.from_arrays([
        pa.array(step_indices),
        pa.array(timestamps),
        pa.array([actions[i].tolist() for i in range(num_steps)], 
                 type=pa.list_(pa.float32(), action_dim)),
        pa.array([states[i].tolist() for i in range(num_steps)],
                 type=pa.list_(pa.float32(), action_dim)),
        pa.array(rewards),
        pa.array(dones),
        pa.array(episode_indices),
    ], names=[
        "step_index",
        "timestamp", 
        "action",
        "state",
        "reward",
        "done",
        "episode_index"
    ])
    
    return batch


def main():
    """Demonstrate custom local augmentations."""
    
    print("=" * 70)
    print("Custom Local Augmentation Example")
    print("=" * 70)
    
    # Create synthetic episode
    print("\n1. Creating synthetic episode...")
    episode_batch = create_synthetic_episode(num_steps=50, action_dim=7)
    print(f"   Original episode: {len(episode_batch)} steps, "
          f"action_dim={episode_batch.column('action').type.list_size}")
    
    # Create a dataset with the episode
    episode = data_core.Episode.from_pyarrow(episode_batch)
    
    # Example 1: Action Smoothing
    print("\n2. Testing ActionSmoothingAugmentation...")
    pipeline1 = data_core.Pipeline()
    pipeline1.add_local(ActionSmoothingAugmentation(alpha=0.3))
    
    # Create a dataset and run pipeline
    # Note: Can't easily create a Dataset in Python, so we'll test the class directly
    
    # Test the augmentation directly on the RecordBatch
    smoother = ActionSmoothingAugmentation(alpha=0.3)
    smoothed = smoother.run(episode_batch)
    
    # Compare action magnitudes
    orig_actions = np.array([episode_batch.column('action')[i].as_py() for i in range(len(episode_batch))])
    smoothed_actions = np.array([smoothed.column('action')[i].as_py() for i in range(len(smoothed))])
    
    orig_std = np.std(np.diff(orig_actions, axis=0))
    smoothed_std = np.std(np.diff(smoothed_actions, axis=0))
    
    print(f"   Original action jitter (std of diffs): {orig_std:.4f}")
    print(f"   Smoothed action jitter (std of diffs): {smoothed_std:.4f}")
    print(f"   Reduction: {((orig_std - smoothed_std) / orig_std * 100):.1f}%")
    
    # Example 2: Truncation
    print("\n3. Testing TruncateEpisodeAugmentation...")
    truncator = TruncateEpisodeAugmentation(max_steps=30)
    truncated = truncator.run(episode_batch)
    print(f"   Original length: {len(episode_batch)} steps")
    print(f"   Truncated length: {len(truncated)} steps")
    
    # Example 3: Per-dimension noise
    print("\n4. Testing PerDimensionActionNoiseAugmentation...")
    noise_scales = [0.1, 0.1, 0.1, 0.05, 0.05, 0.2, 0.2]  # Different noise per dimension
    noisy_aug = PerDimensionActionNoiseAugmentation(noise_scales, seed=42)
    noisy = noisy_aug.run(episode_batch)
    
    noisy_actions = np.array([noisy.column('action')[i].as_py() for i in range(len(noisy))])
    noise_added = noisy_actions - orig_actions
    per_dim_std = np.std(noise_added, axis=0)
    
    print(f"   Target noise scales: {noise_scales}")
    print(f"   Actual noise std:    {per_dim_std.tolist()}")
    
    # Example 4: State velocity
    print("\n5. Testing StateVelocityAugmentation...")
    velocity_aug = StateVelocityAugmentation(dt=0.1)
    velocity_augmented = velocity_aug.run(episode_batch)
    
    orig_state_dim = episode_batch.column('state').type.list_size
    new_state_dim = velocity_augmented.column('state').type.list_size
    
    print(f"   Original state dim: {orig_state_dim}")
    print(f"   Augmented state dim: {new_state_dim} (added {new_state_dim - orig_state_dim} velocity features)")
    
    print("\n" + "=" * 70)
    print("All local augmentation examples completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Create your own augmentation classes with __init__ and run() methods")
    print("  2. Add them to a Pipeline with pipeline.add_local(YourAugmentation(...))")
    print("  3. Run the pipeline on a real dataset with pipeline.run(dataset)")
    print("  4. See custom_augmentation_global.py for global augmentation examples")


if __name__ == "__main__":
    main()
