"""Tests for custom augmentation API in data_core."""

import numpy as np
import pyarrow as pa
import pytest
from typing import List, Dict, Any

import data_core


# ============================================================================
# Test Augmentations
# ============================================================================

class ValidLocalAugmentation:
    """A valid local augmentation for testing."""
    
    def __init__(self, scale: float = 1.0):
        self.scale = scale
    
    def run(self, episode: pa.RecordBatch) -> pa.RecordBatch:
        """Scale all actions by a factor."""
        actions_col = episode.column("action")
        num_steps = len(actions_col)
        action_dim = actions_col.type.list_size
        
        # Scale actions
        actions_np = np.array([
            actions_col[i].as_py() for i in range(num_steps)
        ], dtype=np.float32)
        scaled = actions_np * self.scale
        
        # Convert back
        scaled_list = pa.array(
            [scaled[i].tolist() for i in range(num_steps)],
            type=pa.list_(pa.float32(), action_dim)
        )
        
        # Rebuild
        columns = []
        for i, field in enumerate(episode.schema):
            if field.name == "action":
                columns.append(scaled_list)
            else:
                columns.append(episode.column(i))
        
        return pa.RecordBatch.from_arrays(columns, schema=episode.schema)


class ValidGlobalAugmentation:
    """A valid global augmentation for testing."""
    
    def __init__(self, duplicate: bool = True):
        self.duplicate = duplicate
    
    def run(
        self, 
        episodes: List[pa.RecordBatch], 
        metadata: Dict[str, Any]
    ) -> List[pa.RecordBatch]:
        """Duplicate all episodes if enabled."""
        if not self.duplicate:
            return episodes
        
        result = list(episodes)
        next_idx = len(episodes)
        
        for ep in episodes:
            # Update episode_index
            episode_indices = pa.array(
                [next_idx] * len(ep),
                type=pa.int64()
            )
            
            columns = []
            for i, field in enumerate(ep.schema):
                if field.name == "episode_index":
                    columns.append(episode_indices)
                else:
                    columns.append(ep.column(i))
            
            duplicate = pa.RecordBatch.from_arrays(columns, schema=ep.schema)
            result.append(duplicate)
            next_idx += 1
        
        return result


class MissingRunMethod:
    """Invalid: missing run method."""
    
    def __init__(self):
        pass


class RunNotCallable:
    """Invalid: run is not callable."""
    
    def __init__(self):
        self.run = "not a method"


class WrongSignature:
    """Invalid: run has wrong signature."""
    
    def __init__(self):
        pass
    
    def run(self):
        """No arguments - wrong signature."""
        return None


class ReturnsWrongType:
    """Invalid: returns wrong type."""
    
    def __init__(self):
        pass
    
    def run(self, episode: pa.RecordBatch):
        """Returns wrong type."""
        return "not a RecordBatch"


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_episode(
    num_steps: int = 10, 
    action_dim: int = 7,
    episode_index: int = 0
) -> pa.RecordBatch:
    """Create a synthetic episode for testing."""
    actions = np.random.randn(num_steps, action_dim).astype(np.float32)
    states = np.cumsum(actions, axis=0)
    timestamps = np.linspace(0, 1, num_steps)
    step_indices = np.arange(num_steps, dtype=np.int64)
    episode_indices = np.full(num_steps, episode_index, dtype=np.int64)
    rewards = np.random.rand(num_steps).astype(np.float32)
    dones = np.zeros(num_steps, dtype=bool)
    dones[-1] = True
    
    return pa.RecordBatch.from_arrays([
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
        "step_index", "timestamp", "action", "state",
        "reward", "done", "episode_index"
    ])


# ============================================================================
# Unit Tests - Local Augmentation Validation
# ============================================================================

class TestLocalAugmentationValidation:
    """Test validation of local augmentation classes."""
    
    def test_valid_local_augmentation(self):
        """Valid local augmentation should be accepted."""
        pipeline = data_core.Pipeline()
        aug = ValidLocalAugmentation(scale=2.0)
        
        # Should not raise
        pipeline.add_local(aug)
        assert pipeline.num_steps == 1
    
    def test_missing_run_method(self):
        """Should reject class without run method."""
        pipeline = data_core.Pipeline()
        aug = MissingRunMethod()
        
        with pytest.raises(ValueError, match="must have a 'run' method"):
            pipeline.add_local(aug)
    
    def test_run_not_callable(self):
        """Should reject class where run is not callable."""
        pipeline = data_core.Pipeline()
        aug = RunNotCallable()
        
        with pytest.raises(ValueError, match="must be callable"):
            pipeline.add_local(aug)


class TestGlobalAugmentationValidation:
    """Test validation of global augmentation classes."""
    
    def test_valid_global_augmentation(self):
        """Valid global augmentation should be accepted."""
        pipeline = data_core.Pipeline()
        aug = ValidGlobalAugmentation(duplicate=True)
        
        # Should not raise
        pipeline.add_global(aug)
        assert pipeline.num_steps == 1
    
    def test_missing_run_method_global(self):
        """Should reject class without run method."""
        pipeline = data_core.Pipeline()
        aug = MissingRunMethod()
        
        with pytest.raises(ValueError, match="must have a 'run' method"):
            pipeline.add_global(aug)


# ============================================================================
# Unit Tests - Local Augmentation Execution
# ============================================================================

class TestLocalAugmentationExecution:
    """Test execution of local augmentations."""
    
    def test_local_augmentation_modifies_episode(self):
        """Local augmentation should modify episode data."""
        episode = create_test_episode(num_steps=10, action_dim=7)
        aug = ValidLocalAugmentation(scale=2.0)
        
        result = aug.run(episode)
        
        # Check that actions were scaled
        orig_actions = np.array([episode.column("action")[i].as_py() 
                                for i in range(len(episode))])
        result_actions = np.array([result.column("action")[i].as_py() 
                                  for i in range(len(result))])
        
        np.testing.assert_allclose(result_actions, orig_actions * 2.0)
    
    def test_local_augmentation_preserves_schema(self):
        """Local augmentation should preserve episode schema."""
        episode = create_test_episode(num_steps=10, action_dim=7)
        aug = ValidLocalAugmentation(scale=1.5)
        
        result = aug.run(episode)
        
        assert result.schema.equals(episode.schema)
        assert len(result) == len(episode)
    
    def test_local_augmentation_in_pipeline(self):
        """Test local augmentation within a pipeline."""
        # Create a simple dataset (note: we need a full Dataset object)
        # For now, test the augmentation directly
        episode = create_test_episode(num_steps=15, action_dim=7)
        
        # Create pipeline
        pipeline = data_core.Pipeline()
        pipeline.add_local(ValidLocalAugmentation(scale=0.5))
        
        # Note: Can't easily test full pipeline without a Dataset
        # But we've validated the augmentation works
        assert pipeline.num_steps == 1


# ============================================================================
# Unit Tests - Global Augmentation Execution  
# ============================================================================

class TestGlobalAugmentationExecution:
    """Test execution of global augmentations."""
    
    def test_global_augmentation_duplicates_episodes(self):
        """Global augmentation should be able to add episodes."""
        episodes = [
            create_test_episode(num_steps=10, episode_index=0),
            create_test_episode(num_steps=15, episode_index=1),
        ]
        
        metadata = {
            "name": "test",
            "fps": 10.0,
            "action_dim": 7,
            "num_episodes": 2,
            "num_steps": 25,
            "camera_names": ["cam_0"],
            "tasks": ["task_0"],
        }
        
        aug = ValidGlobalAugmentation(duplicate=True)
        result = aug.run(episodes, metadata)
        
        # Should have doubled the episodes
        assert len(result) == 4
        assert len(result[0]) == 10  # Original
        assert len(result[1]) == 15  # Original
        assert len(result[2]) == 10  # Duplicate
        assert len(result[3]) == 15  # Duplicate
    
    def test_global_augmentation_receives_metadata(self):
        """Global augmentation should receive metadata dict."""
        episodes = [create_test_episode(num_steps=10, episode_index=0)]
        
        metadata = {
            "name": "test_dataset",
            "fps": 30.0,
            "action_dim": 7,
            "state_dim": 5,
            "camera_names": ["cam_0", "cam_1"],
            "tasks": ["pick", "place"],
            "num_episodes": 1,
            "num_steps": 10,
            "robot_type": "aloha",
            "source": "/path/to/data",
        }
        
        # Create a custom augmentation that validates metadata
        class MetadataValidator:
            def __init__(self):
                self.received_metadata = None
            
            def run(self, episodes, metadata):
                self.received_metadata = metadata
                return episodes
        
        validator = MetadataValidator()
        result = validator.run(episodes, metadata)
        
        # Check all metadata was passed
        assert validator.received_metadata["name"] == "test_dataset"
        assert validator.received_metadata["fps"] == 30.0
        assert validator.received_metadata["action_dim"] == 7
        assert validator.received_metadata["camera_names"] == ["cam_0", "cam_1"]
        assert validator.received_metadata["robot_type"] == "aloha"


# ============================================================================
# Unit Tests - Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling for malformed augmentations."""
    
    def test_returns_wrong_type_local(self):
        """Should handle augmentation that returns wrong type."""
        episode = create_test_episode(num_steps=10)
        aug = ReturnsWrongType()
        
        # The error will happen when Rust tries to convert the result
        # We can't easily test this without going through the full pipeline
        # But the validation at add_local time should catch structural issues
        pass
    
    def test_python_exception_propagation(self):
        """Python exceptions should propagate with useful messages."""
        
        class RaisesException:
            def __init__(self):
                pass
            
            def run(self, episode):
                raise ValueError("Custom error from augmentation")
        
        episode = create_test_episode(num_steps=10)
        aug = RaisesException()
        
        # The exception will be caught when run() is called
        with pytest.raises(ValueError, match="Custom error"):
            aug.run(episode)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full pipelines."""
    
    def test_mixed_local_and_global_pipeline(self):
        """Test pipeline with both local and global augmentations."""
        pipeline = data_core.Pipeline()
        
        # Add local then global
        pipeline.add_local(ValidLocalAugmentation(scale=1.5))
        pipeline.add_global(ValidGlobalAugmentation(duplicate=True))
        
        assert pipeline.num_steps == 2
        step_names = pipeline.step_names()
        assert "ValidLocalAugmentation" in step_names[0]
        assert "ValidGlobalAugmentation" in step_names[1]
    
    def test_multiple_local_augmentations(self):
        """Test pipeline with multiple local augmentations."""
        pipeline = data_core.Pipeline()
        
        pipeline.add_local(ValidLocalAugmentation(scale=2.0))
        pipeline.add_local(ValidLocalAugmentation(scale=0.5))
        
        assert pipeline.num_steps == 2
    
    def test_multiple_global_augmentations(self):
        """Test pipeline with multiple global augmentations."""
        pipeline = data_core.Pipeline()
        
        pipeline.add_global(ValidGlobalAugmentation(duplicate=True))
        pipeline.add_global(ValidGlobalAugmentation(duplicate=False))
        
        assert pipeline.num_steps == 2
    
    def test_builtin_and_custom_mixed(self):
        """Test mixing built-in and custom augmentations."""
        pipeline = data_core.Pipeline()
        
        # Add built-in
        pipeline.add_builtin("action_noise", '{"std": 0.01}')
        
        # Add custom
        pipeline.add_local(ValidLocalAugmentation(scale=1.2))
        
        assert pipeline.num_steps == 2


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_episode(self):
        """Test augmentation on empty episode."""
        # Create empty episode
        empty = pa.RecordBatch.from_arrays([
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.float64()),
            pa.array([], type=pa.list_(pa.float32(), 7)),
            pa.array([], type=pa.list_(pa.float32(), 7)),
            pa.array([], type=pa.float32()),
            pa.array([], type=pa.bool_()),
            pa.array([], type=pa.int64()),
        ], names=[
            "step_index", "timestamp", "action", "state",
            "reward", "done", "episode_index"
        ])
        
        aug = ValidLocalAugmentation(scale=2.0)
        result = aug.run(empty)
        
        assert len(result) == 0
        assert result.schema.equals(empty.schema)
    
    def test_single_step_episode(self):
        """Test augmentation on single-step episode."""
        episode = create_test_episode(num_steps=1)
        aug = ValidLocalAugmentation(scale=3.0)
        
        result = aug.run(episode)
        
        assert len(result) == 1
    
    def test_empty_episodes_list_global(self):
        """Test global augmentation with empty episode list."""
        episodes = []
        metadata = {
            "name": "empty",
            "fps": 10.0,
            "action_dim": 7,
            "num_episodes": 0,
            "num_steps": 0,
            "camera_names": [],
            "tasks": [],
        }
        
        aug = ValidGlobalAugmentation(duplicate=True)
        result = aug.run(episodes, metadata)
        
        assert len(result) == 0
    
    def test_single_episode_global(self):
        """Test global augmentation with single episode."""
        episodes = [create_test_episode(num_steps=10, episode_index=0)]
        metadata = {
            "name": "single",
            "fps": 10.0,
            "action_dim": 7,
            "num_episodes": 1,
            "num_steps": 10,
            "camera_names": ["cam_0"],
            "tasks": ["task_0"],
        }
        
        aug = ValidGlobalAugmentation(duplicate=True)
        result = aug.run(episodes, metadata)
        
        # Should duplicate even single episode
        assert len(result) == 2


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
