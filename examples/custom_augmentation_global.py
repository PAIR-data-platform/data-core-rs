"""
Custom Global Augmentation Example

Demonstrates how to create custom global augmentation classes that process
entire datasets to generate new trajectories. Global augmentations can:
- Generate new episodes from existing ones (like MimicGen)
- Combine multiple episodes into new trajectories
- Filter or rebalance datasets
- Create variations by stitching trajectory segments

This example shows:
1. How to define a global augmentation class with __init__ and run() methods
2. How to work with lists of PyArrow RecordBatches and dataset metadata
3. Multiple example augmentations: episode duplication, trajectory stitching, dataset balancing

Usage:
    python examples/custom_augmentation_global.py

Requirements:
    pip install pyarrow numpy
"""

import numpy as np
import pyarrow as pa
from typing import List, Dict, Any, Optional



# ============================================================================
# Example 1: Episode Duplication (Simple Data Augmentation)
# ============================================================================

class DuplicateEpisodesAugmentation:
    """
    Duplicate each episode N times to increase dataset size.
    Each duplicate gets a new episode_index.
    """
    
    def __init__(self, num_duplicates: int = 2):
        """
        Initialize episode duplication.
        
        Args:
            num_duplicates: Number of duplicates to create per episode
        """
        self.num_duplicates = num_duplicates
    
    def run(
        self, 
        episodes: List[pa.RecordBatch], 
        metadata: Dict[str, Any]
    ) -> List[pa.RecordBatch]:
        """
        Duplicate each episode N times.
        
        Args:
            episodes: List of input episodes as PyArrow RecordBatches
            metadata: Dataset metadata (fps, camera_names, etc.)
            
        Returns:
            List with original + duplicated episodes
        """
        new_episodes = []
        next_episode_idx = len(episodes)
        
        for orig_episode in episodes:
            # Keep original
            new_episodes.append(orig_episode)
            
            # Create duplicates
            for _ in range(self.num_duplicates - 1):
                # Update episode_index column
                episode_indices = pa.array(
                    [next_episode_idx] * len(orig_episode),
                    type=pa.int64()
                )
                
                # Rebuild RecordBatch with new episode_index
                columns = []
                schema = orig_episode.schema
                
                for i, field in enumerate(schema):
                    if field.name == "episode_index":
                        columns.append(episode_indices)
                    else:
                        columns.append(orig_episode.column(i))
                
                duplicate = pa.RecordBatch.from_arrays(columns, schema=schema)
                new_episodes.append(duplicate)
                next_episode_idx += 1
        
        return new_episodes


# ============================================================================
# Example 2: Trajectory Stitching (MimicGen-inspired)
# ============================================================================

class TrajectoryStitchingAugmentation:
    """
    Stitch together segments from different episodes to create new trajectories.
    Inspired by MimicGen's trajectory composition approach.
    
    This creates new episodes by:
    1. Selecting a source episode
    2. Finding a keyframe (e.g., when reward > threshold)
    3. Stitching the pre-keyframe segment from one episode with 
       the post-keyframe segment from another
    """
    
    def __init__(
        self, 
        num_new_episodes: int = 10,
        reward_threshold: float = 0.8,
        seed: Optional[int] = None
    ):
        """
        Initialize trajectory stitching.
        
        Args:
            num_new_episodes: Number of new episodes to generate
            reward_threshold: Reward threshold for keyframe detection
            seed: Random seed for reproducibility
        """
        self.num_new_episodes = num_new_episodes
        self.reward_threshold = reward_threshold
        self._rng = np.random.RandomState(seed)
    
    def _find_keyframe(self, episode: pa.RecordBatch) -> Optional[int]:
        """Find the first step where reward exceeds threshold."""
        if "reward" not in episode.schema.names:
            return None
        
        rewards = episode.column("reward").to_numpy()
        keyframe_indices = np.where(rewards > self.reward_threshold)[0]
        
        return int(keyframe_indices[0]) if len(keyframe_indices) > 0 else None
    
    def run(
        self, 
        episodes: List[pa.RecordBatch], 
        metadata: Dict[str, Any]
    ) -> List[pa.RecordBatch]:
        """
        Generate new episodes by stitching trajectory segments.
        
        Args:
            episodes: List of input episodes
            metadata: Dataset metadata
            
        Returns:
            Original episodes + newly stitched episodes
        """
        if len(episodes) < 2:
            # Need at least 2 episodes to stitch
            return episodes
        
        new_episodes = list(episodes)  # Keep originals
        next_episode_idx = len(episodes)
        
        for _ in range(self.num_new_episodes):
            # Randomly select two episodes
            ep1_idx, ep2_idx = self._rng.choice(len(episodes), size=2, replace=False)
            ep1 = episodes[ep1_idx]
            ep2 = episodes[ep2_idx]
            
            # Find keyframes
            keyframe1 = self._find_keyframe(ep1)
            keyframe2 = self._find_keyframe(ep2)
            
            if keyframe1 is None or keyframe2 is None:
                # No clear keyframe, use midpoint
                keyframe1 = len(ep1) // 2
                keyframe2 = len(ep2) // 2
            
            # Stitch: first half of ep1 + second half of ep2
            segment1 = ep1.slice(0, keyframe1)
            segment2 = ep2.slice(keyframe2, len(ep2) - keyframe2)
            
            # Concatenate segments
            try:
                # Concatenate the two segments
                stitched_tables = []
                
                for seg in [segment1, segment2]:
                    # Convert to Table for easier manipulation
                    stitched_tables.append(pa.Table.from_batches([seg]))
                
                # Concatenate tables
                combined_table = pa.concat_tables(stitched_tables)
                
                # Update metadata columns
                num_steps = len(combined_table)
                
                # Update episode_index
                episode_indices = pa.array([next_episode_idx] * num_steps, type=pa.int64())
                
                # Update step_index to be sequential
                step_indices = pa.array(np.arange(num_steps, dtype=np.int64))
                
                # Rebuild table with updated indices
                schema = combined_table.schema
                columns = []
                
                for field in schema:
                    if field.name == "episode_index":
                        columns.append(episode_indices)
                    elif field.name == "step_index":
                        columns.append(step_indices)
                    else:
                        columns.append(combined_table.column(field.name).combine_chunks())
                
                stitched_batch = pa.RecordBatch.from_arrays(columns, schema=schema)
                new_episodes.append(stitched_batch)
                next_episode_idx += 1
                
            except Exception as e:
                # If stitching fails (e.g., schema mismatch), skip
                print(f"Warning: Failed to stitch episodes {ep1_idx} and {ep2_idx}: {e}")
                continue
        
        return new_episodes


# ============================================================================
# Example 3: Dataset Balancing by Task
# ============================================================================

class BalanceTasksAugmentation:
    """
    Balance a multi-task dataset by ensuring each task has similar numbers of episodes.
    Uses duplication for underrepresented tasks.
    """
    
    def __init__(self, target_episodes_per_task: Optional[int] = None):
        """
        Initialize task balancing.
        
        Args:
            target_episodes_per_task: Target number of episodes per task.
                                      If None, uses the max count.
        """
        self.target_episodes_per_task = target_episodes_per_task
    
    def run(
        self, 
        episodes: List[pa.RecordBatch], 
        metadata: Dict[str, Any]
    ) -> List[pa.RecordBatch]:
        """
        Balance dataset by duplicating underrepresented tasks.
        
        Args:
            episodes: List of input episodes
            metadata: Dataset metadata with 'tasks' field
            
        Returns:
            Balanced list of episodes
        """
        # Check if episodes have task information
        if len(episodes) == 0:
            return episodes
        
        first_episode = episodes[0]
        if "task_index" not in first_episode.schema.names:
            # No task info, return as-is
            return episodes
        
        # Group episodes by task
        task_to_episodes = {}
        for ep in episodes:
            # Get task index (assuming constant within episode)
            task_idx = ep.column("task_index")[0].as_py()
            if task_idx not in task_to_episodes:
                task_to_episodes[task_idx] = []
            task_to_episodes[task_idx].append(ep)
        
        # Determine target count
        if self.target_episodes_per_task is None:
            target = max(len(eps) for eps in task_to_episodes.values())
        else:
            target = self.target_episodes_per_task
        
        # Balance by duplicating
        balanced_episodes = []
        next_episode_idx = len(episodes)
        
        for task_idx, task_episodes in task_to_episodes.items():
            # Add original episodes
            balanced_episodes.extend(task_episodes)
            
            # Duplicate to reach target
            num_needed = target - len(task_episodes)
            if num_needed > 0:
                # Duplicate randomly selected episodes
                rng = np.random.RandomState(task_idx)
                for _ in range(num_needed):
                    # Pick a random episode from this task
                    source_ep = task_episodes[rng.randint(len(task_episodes))]
                    
                    # Update episode_index
                    episode_indices = pa.array(
                        [next_episode_idx] * len(source_ep),
                        type=pa.int64()
                    )
                    
                    # Rebuild
                    columns = []
                    schema = source_ep.schema
                    for i, field in enumerate(schema):
                        if field.name == "episode_index":
                            columns.append(episode_indices)
                        else:
                            columns.append(source_ep.column(i))
                    
                    duplicate = pa.RecordBatch.from_arrays(columns, schema=schema)
                    balanced_episodes.append(duplicate)
                    next_episode_idx += 1
        
        return balanced_episodes


# ============================================================================
# Example 4: Temporal Subsampling at Dataset Level
# ============================================================================

class DatasetTemporalSubsampleAugmentation:
    """
    Create a new dataset by subsampling all episodes at a different frequency.
    This is a global operation because it needs fps from metadata.
    """
    
    def __init__(self, target_fps: float):
        """
        Initialize temporal subsampling.
        
        Args:
            target_fps: Target frames per second for the output dataset
        """
        self.target_fps = target_fps
    
    def run(
        self, 
        episodes: List[pa.RecordBatch], 
        metadata: Dict[str, Any]
    ) -> List[pa.RecordBatch]:
        """
        Subsample all episodes to target FPS.
        
        Args:
            episodes: List of input episodes
            metadata: Dataset metadata with 'fps' field
            
        Returns:
            Subsampled episodes
        """
        original_fps = metadata.get("fps")
        if original_fps is None:
            # No FPS info, can't subsample intelligently
            return episodes
        
        if self.target_fps >= original_fps:
            # No subsampling needed
            return episodes
        
        # Calculate subsampling factor
        subsample_factor = int(original_fps / self.target_fps)
        if subsample_factor < 2:
            return episodes
        
        # Subsample each episode
        subsampled_episodes = []
        
        for ep in episodes:
            # Take every Nth step
            indices = list(range(0, len(ep), subsample_factor))
            
            # Slice and combine
            slices = [ep.slice(i, 1) for i in indices]
            
            if len(slices) == 0:
                continue
            
            # Concatenate slices
            table = pa.Table.from_batches(slices)
            
            # Update step_index to be sequential
            step_indices = pa.array(np.arange(len(table), dtype=np.int64))
            
            # Rebuild
            columns = []
            schema = table.schema
            for field in schema:
                if field.name == "step_index":
                    columns.append(step_indices)
                else:
                    columns.append(table.column(field.name).combine_chunks())
            
            subsampled = pa.RecordBatch.from_arrays(columns, schema=schema)
            subsampled_episodes.append(subsampled)
        
        return subsampled_episodes


# ============================================================================
# Example 5: MimicGen-Style Keypoint-Based Trajectory Generation
# ============================================================================

class KeypointInterpolationAugmentation:
    """
    Generate new episodes by extracting keypoints and interpolating between them.
    This mimics MimicGen's approach of keyframe-based trajectory synthesis.
    """
    
    def __init__(
        self,
        num_keypoints: int = 5,
        interpolation_steps: int = 10,
        num_new_episodes: int = 5,
        seed: Optional[int] = None
    ):
        """
        Initialize keypoint interpolation.
        
        Args:
            num_keypoints: Number of keypoints to extract from each episode
            interpolation_steps: Steps to interpolate between keypoints
            num_new_episodes: Number of new episodes to generate
            seed: Random seed
        """
        self.num_keypoints = num_keypoints
        self.interpolation_steps = interpolation_steps
        self.num_new_episodes = num_new_episodes
        self._rng = np.random.RandomState(seed)
    
    def _extract_keypoints(self, episode: pa.RecordBatch) -> List[int]:
        """Extract keypoint indices from an episode."""
        num_steps = len(episode)
        if num_steps <= self.num_keypoints:
            return list(range(num_steps))
        
        # Evenly space keypoints
        indices = np.linspace(0, num_steps - 1, self.num_keypoints, dtype=int)
        return indices.tolist()
    
    def _interpolate_actions(
        self, 
        action1: np.ndarray, 
        action2: np.ndarray
    ) -> np.ndarray:
        """Linear interpolation between two actions."""
        steps = self.interpolation_steps
        alphas = np.linspace(0, 1, steps + 2)[1:-1]  # Exclude endpoints
        
        interpolated = []
        for alpha in alphas:
            interp_action = (1 - alpha) * action1 + alpha * action2
            interpolated.append(interp_action)
        
        return np.array(interpolated)
    
    def run(
        self, 
        episodes: List[pa.RecordBatch], 
        metadata: Dict[str, Any]
    ) -> List[pa.RecordBatch]:
        """
        Generate new episodes via keypoint interpolation.
        
        Args:
            episodes: List of input episodes
            metadata: Dataset metadata
            
        Returns:
            Original + synthesized episodes
        """
        if len(episodes) == 0:
            return episodes
        
        new_episodes = list(episodes)  # Keep originals
        next_episode_idx = len(episodes)
        action_dim = metadata.get("action_dim", 7)
        
        for _ in range(self.num_new_episodes):
            # Randomly select source episode
            source_ep = episodes[self._rng.randint(len(episodes))]
            
            # Extract keypoints
            keypoint_indices = self._extract_keypoints(source_ep)
            
            # Extract keypoint data
            keypoint_batches = [source_ep.slice(i, 1) for i in keypoint_indices]
            
            # Build new episode by interpolating between keypoints
            synthesized_steps = []
            
            for i in range(len(keypoint_indices) - 1):
                # Add keypoint
                synthesized_steps.append(keypoint_batches[i])
                
                # Interpolate to next keypoint
                action1 = np.array(keypoint_batches[i].column("action")[0].as_py())
                action2 = np.array(keypoint_batches[i + 1].column("action")[0].as_py())
                
                interpolated_actions = self._interpolate_actions(action1, action2)
                
                # Create intermediate steps (simplified - just copy keypoint and update action)
                for interp_action in interpolated_actions:
                    step = keypoint_batches[i]
                    
                    # Update action
                    action_col = pa.array(
                        [interp_action.tolist()],
                        type=pa.list_(pa.float32(), action_dim)
                    )
                    
                    # Rebuild step
                    columns = []
                    for field in step.schema:
                        if field.name == "action":
                            columns.append(action_col)
                        else:
                            columns.append(step.column(field.name))
                    
                    interp_step = pa.RecordBatch.from_arrays(columns, schema=step.schema)
                    synthesized_steps.append(interp_step)
            
            # Add final keypoint
            synthesized_steps.append(keypoint_batches[-1])
            
            # Concatenate into episode
            try:
                synth_table = pa.Table.from_batches(synthesized_steps)
                num_steps = len(synth_table)
                
                # Update indices
                episode_indices = pa.array([next_episode_idx] * num_steps, type=pa.int64())
                step_indices = pa.array(np.arange(num_steps, dtype=np.int64))
                
                # Rebuild
                columns = []
                for field in synth_table.schema:
                    if field.name == "episode_index":
                        columns.append(episode_indices)
                    elif field.name == "step_index":
                        columns.append(step_indices)
                    else:
                        columns.append(synth_table.column(field.name).combine_chunks())
                
                synth_episode = pa.RecordBatch.from_arrays(columns, schema=synth_table.schema)
                new_episodes.append(synth_episode)
                next_episode_idx += 1
                
            except Exception as e:
                print(f"Warning: Failed to create synthesized episode: {e}")
                continue
        
        return new_episodes


# ============================================================================
# Main Demo
# ============================================================================

def create_synthetic_episodes(
    num_episodes: int = 10, 
    steps_per_episode: int = 30,
    action_dim: int = 7
) -> List[pa.RecordBatch]:
    """Create synthetic episodes for demonstration."""
    
    episodes = []
    
    for ep_idx in range(num_episodes):
        num_steps = steps_per_episode + np.random.randint(-5, 5)
        
        # Create data
        actions = np.random.randn(num_steps, action_dim).astype(np.float32) * 0.5
        states = np.cumsum(actions, axis=0)
        timestamps = np.linspace(0, 3, num_steps)
        step_indices = np.arange(num_steps, dtype=np.int64)
        episode_indices = np.full(num_steps, ep_idx, dtype=np.int64)
        
        # Simulate rewards peaking near the end
        rewards = np.linspace(0, 1, num_steps).astype(np.float32)
        rewards += np.random.rand(num_steps).astype(np.float32) * 0.1
        
        dones = np.zeros(num_steps, dtype=bool)
        dones[-1] = True
        
        # Task index (simulate 3 tasks)
        task_idx = ep_idx % 3
        task_indices = np.full(num_steps, task_idx, dtype=np.int64)
        
        # Build RecordBatch
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
            pa.array(task_indices),
        ], names=[
            "step_index", "timestamp", "action", "state", 
            "reward", "done", "episode_index", "task_index"
        ])
        
        episodes.append(batch)
    
    return episodes


def main():
    """Demonstrate custom global augmentations."""
    
    print("=" * 70)
    print("Custom Global Augmentation Example")
    print("=" * 70)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    episodes = create_synthetic_episodes(num_episodes=10, steps_per_episode=30)
    
    metadata = {
        "name": "synthetic_dataset",
        "fps": 30.0,
        "action_dim": 7,
        "num_episodes": len(episodes),
        "num_steps": sum(len(ep) for ep in episodes),
        "camera_names": ["cam_0"],
        "tasks": ["task_0", "task_1", "task_2"],
        "robot_type": "test_robot"
    }
    
    print(f"   Created {len(episodes)} episodes, "
          f"{metadata['num_steps']} total steps")
    
    # Example 1: Duplication
    print("\n2. Testing DuplicateEpisodesAugmentation...")
    duplicator = DuplicateEpisodesAugmentation(num_duplicates=3)
    duplicated = duplicator.run(episodes, metadata)
    print(f"   Original: {len(episodes)} episodes")
    print(f"   After duplication (3x): {len(duplicated)} episodes")
    
    # Example 2: Trajectory Stitching
    print("\n3. Testing TrajectoryStitchingAugmentation...")
    stitcher = TrajectoryStitchingAugmentation(num_new_episodes=5, seed=42)
    stitched = stitcher.run(episodes, metadata)
    print(f"   Original: {len(episodes)} episodes")
    print(f"   After stitching (+5 new): {len(stitched)} episodes")
    
    # Example 3: Task Balancing
    print("\n4. Testing BalanceTasksAugmentation...")
    balancer = BalanceTasksAugmentation()
    balanced = balancer.run(episodes, metadata)
    
    # Count episodes per task
    task_counts = {}
    for ep in balanced:
        task_idx = ep.column("task_index")[0].as_py()
        task_counts[task_idx] = task_counts.get(task_idx, 0) + 1
    
    print(f"   Original: {len(episodes)} episodes")
    print(f"   After balancing: {len(balanced)} episodes")
    print(f"   Episodes per task: {task_counts}")
    
    # Example 4: Temporal Subsampling
    print("\n5. Testing DatasetTemporalSubsampleAugmentation...")
    subsampler = DatasetTemporalSubsampleAugmentation(target_fps=10.0)
    subsampled = subsampler.run(episodes, metadata)
    
    orig_avg_len = np.mean([len(ep) for ep in episodes])
    subsampled_avg_len = np.mean([len(ep) for ep in subsampled])
    
    print(f"   Original FPS: {metadata['fps']}, Target FPS: 10.0")
    print(f"   Original avg episode length: {orig_avg_len:.1f} steps")
    print(f"   Subsampled avg episode length: {subsampled_avg_len:.1f} steps")
    
    # Example 5: Keypoint Interpolation
    print("\n6. Testing KeypointInterpolationAugmentation...")
    interpolator = KeypointInterpolationAugmentation(
        num_keypoints=5, 
        interpolation_steps=8,
        num_new_episodes=3,
        seed=42
    )
    interpolated = interpolator.run(episodes, metadata)
    print(f"   Original: {len(episodes)} episodes")
    print(f"   After keypoint interpolation (+3 new): {len(interpolated)} episodes")
    
    print("\n" + "=" * 70)
    print("All global augmentation examples completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  - Global augmentations receive ALL episodes + metadata dict")
    print("  - They can generate new episodes (like MimicGen)")
    print("  - Use metadata for context (fps, action_dim, tasks, etc.)")
    print("  - Return a list of RecordBatches (original + new episodes)")
    print("\nNext steps:")
    print("  1. Create your own global augmentation with __init__ and run() methods")
    print("  2. Add to pipeline with pipeline.add_global(YourAugmentation(...))")
    print("  3. Combine with local augmentations for powerful data pipelines")


if __name__ == "__main__":
    main()
