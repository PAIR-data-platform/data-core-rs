"""
End-to-End Example: Custom Augmentation Pipeline

This example demonstrates the complete workflow of using custom augmentations
in a realistic scenario, combining local and global augmentations.
"""

import numpy as np
import pyarrow as pa
from typing import List, Dict, Any

import data_core


# ============================================================================
# Custom Augmentations
# ============================================================================

class ExplorationNoiseAugmentation:
    """Add exploration noise to encourage robustness."""
    
    def __init__(self, noise_std: float = 0.05, seed: int = 42):
        self.noise_std = noise_std
        self._rng = np.random.RandomState(seed)
    
    def run(self, episode: pa.RecordBatch) -> pa.RecordBatch:
        """Add Gaussian noise to actions."""
        actions_col = episode.column("action")
        num_steps = len(actions_col)
        action_dim = actions_col.type.list_size
        
        actions_np = np.array([actions_col[i].as_py() for i in range(num_steps)], dtype=np.float32)
        noise = self._rng.randn(num_steps, action_dim).astype(np.float32) * self.noise_std
        noisy_actions = actions_np + noise
        
        noisy_list = pa.array(
            [noisy_actions[i].tolist() for i in range(num_steps)],
            type=pa.list_(pa.float32(), action_dim)
        )
        
        columns = []
        for i, field in enumerate(episode.schema):
            if field.name == "action":
                columns.append(noisy_list)
            else:
                columns.append(episode.column(i))
        
        return pa.RecordBatch.from_arrays(columns, schema=episode.schema)


class MimicGenStyleAugmentation:
    """Generate new trajectories by stitching successful segments."""
    
    def __init__(self, num_new_episodes: int = 5, success_threshold: float = 0.9):
        self.num_new_episodes = num_new_episodes
        self.success_threshold = success_threshold
    
    def run(self, episodes: List[pa.RecordBatch], metadata: Dict[str, Any]) -> List[pa.RecordBatch]:
        """Generate new episodes via trajectory stitching."""
        if len(episodes) < 2:
            return episodes
        
        # Find successful episodes (high final reward)
        successful_episodes = []
        for ep in episodes:
            if "reward" in ep.schema.names:
                final_reward = ep.column("reward")[-1].as_py()
                if final_reward > self.success_threshold:
                    successful_episodes.append(ep)
        
        if len(successful_episodes) < 2:
            successful_episodes = episodes[:min(2, len(episodes))]
        
        new_episodes = list(episodes)
        next_idx = len(episodes)
        
        for i in range(self.num_new_episodes):
            # Randomly select two successful episodes
            idx1, idx2 = np.random.choice(len(successful_episodes), size=2, replace=False)
            ep1, ep2 = successful_episodes[idx1], successful_episodes[idx2]
            
            # Stitch at midpoints
            mid1 = len(ep1) // 2
            mid2 = len(ep2) // 2
            
            segment1 = ep1.slice(0, mid1)
            segment2 = ep2.slice(mid2, len(ep2) - mid2)
            
            try:
                # Combine
                combined = pa.concat_tables([
                    pa.Table.from_batches([segment1]),
                    pa.Table.from_batches([segment2])
                ])
                
                num_steps = len(combined)
                episode_indices = pa.array([next_idx] * num_steps, type=pa.int64())
                step_indices = pa.array(np.arange(num_steps, dtype=np.int64))
                
                columns = []
                for field in combined.schema:
                    if field.name == "episode_index":
                        columns.append(episode_indices)
                    elif field.name == "step_index":
                        columns.append(step_indices)
                    else:
                        columns.append(combined.column(field.name).combine_chunks())
                
                stitched = pa.RecordBatch.from_arrays(columns, schema=combined.schema)
                new_episodes.append(stitched)
                next_idx += 1
            except:
                continue
        
        return new_episodes


# ============================================================================
# Main Demo
# ============================================================================

def create_synthetic_episodes(num_episodes: int = 5) -> List[pa.RecordBatch]:
    """Create synthetic episodes with varying quality."""
    episodes = []
    
    for ep_idx in range(num_episodes):
        num_steps = 20 + np.random.randint(-5, 5)
        action_dim = 7
        
        actions = np.random.randn(num_steps, action_dim).astype(np.float32) * 0.5
        states = np.cumsum(actions, axis=0)
        timestamps = np.linspace(0, 2, num_steps)
        step_indices = np.arange(num_steps, dtype=np.int64)
        episode_indices = np.full(num_steps, ep_idx, dtype=np.int64)
        
        # Simulate varying success rates
        success = np.random.rand() > 0.3  # 70% success rate
        rewards = np.linspace(0, 1.0 if success else 0.5, num_steps).astype(np.float32)
        
        dones = np.zeros(num_steps, dtype=bool)
        dones[-1] = True
        
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
        ], names=["step_index", "timestamp", "action", "state", "reward", "done", "episode_index"])
        
        episodes.append(batch)
    
    return episodes


def main():
    """Demonstrate end-to-end custom augmentation pipeline."""
    
    print("=" * 70)
    print("End-to-End Custom Augmentation Pipeline Example")
    print("=" * 70)
    
    # 1. Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    episodes = create_synthetic_episodes(num_episodes=5)
    print(f"   Created {len(episodes)} episodes")
    
    total_steps = sum(len(ep) for ep in episodes)
    successful = sum(1 for ep in episodes if ep.column("reward")[-1].as_py() > 0.9)
    print(f"   Total steps: {total_steps}")
    print(f"   Successful episodes: {successful}/{len(episodes)}")
    
    # 2. Build augmentation pipeline
    print("\n2. Building augmentation pipeline...")
    print("   - Local: ExplorationNoiseAugmentation (noise_std=0.05)")
    print("   - Global: MimicGenStyleAugmentation (generate 10 new episodes)")
    
    pipeline = data_core.Pipeline()
    
    # Add local augmentation (applies to each episode)
    pipeline.add_local(ExplorationNoiseAugmentation(noise_std=0.05, seed=42))
    
    # Add global augmentation (generates new episodes)
    pipeline.add_global(MimicGenStyleAugmentation(num_new_episodes=10, success_threshold=0.9))
    
    print(f"   Pipeline has {pipeline.num_steps} steps")
    print(f"   Step names: {pipeline.step_names()}")
    
    # 3. Apply augmentations manually (simulating pipeline.run())
    print("\n3. Applying augmentations...")
    
    # Apply local augmentation
    local_aug = ExplorationNoiseAugmentation(noise_std=0.05, seed=42)
    augmented_episodes = [local_aug.run(ep) for ep in episodes]
    print(f"   After local augmentation: {len(augmented_episodes)} episodes")
    
    # Apply global augmentation
    metadata = {
        "name": "synthetic_dataset",
        "fps": 10.0,
        "action_dim": 7,
        "num_episodes": len(episodes),
        "num_steps": total_steps,
        "camera_names": ["cam_0"],
        "tasks": ["pick_and_place"],
    }
    
    global_aug = MimicGenStyleAugmentation(num_new_episodes=10, success_threshold=0.9)
    final_episodes = global_aug.run(augmented_episodes, metadata)
    print(f"   After global augmentation: {len(final_episodes)} episodes")
    
    # 4. Analyze results
    print("\n4. Results:")
    print(f"   Original dataset: {len(episodes)} episodes, {total_steps} steps")
    
    final_steps = sum(len(ep) for ep in final_episodes)
    print(f"   Augmented dataset: {len(final_episodes)} episodes, {final_steps} steps")
    
    increase = ((len(final_episodes) - len(episodes)) / len(episodes)) * 100
    print(f"   Episode count increase: {increase:.1f}%")
    
    # Check action noise was applied
    orig_actions = np.array([episodes[0].column("action")[0].as_py()])
    aug_actions = np.array([augmented_episodes[0].column("action")[0].as_py()])
    noise_magnitude = np.linalg.norm(orig_actions - aug_actions)
    print(f"   Noise magnitude (first action): {noise_magnitude:.4f}")
    
    print("\n" + "=" * 70)
    print("End-to-end pipeline completed successfully!")
    print("=" * 70)
    print("\nKey Points:")
    print("  - Local augmentations transform each episode independently")
    print("  - Global augmentations can generate entirely new episodes")
    print("  - Combine both for powerful data augmentation strategies")
    print("  - MimicGen-style stitching increases dataset diversity")


if __name__ == "__main__":
    main()
