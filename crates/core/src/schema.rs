//! Canonical Arrow schema for episodes and a compatibility check.
//!
//! The canonical schema is what [`crate::Dataset::empty`] and the eager LeRobot
//! loader attach to a dataset. Episodes read from LeRobot files keep the
//! columns found on disk, which may be a superset of (or differ from) this
//! schema; use [`validate_schema_compatibility`] to check a concrete schema
//! against it.

use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

/// Build the canonical Arrow schema for an episode.
///
/// Columns, in order: `step_index`, `timestamp`, `action`, optional `state`,
/// one `observation.image.{camera}` column per camera (`LargeBinary`, encoded
/// image bytes), optional `observation.state`, `reward`, `done`,
/// `episode_index`, and `task_index`. Vector columns are
/// `FixedSizeList<Float32>` with the lengths given in `config`.
///
/// ```
/// use data_core::{build_episode_schema, SchemaConfig};
///
/// let config = SchemaConfig {
///     action_dim: 14,
///     state_dim: None,
///     observation_state_dim: Some(14),
///     camera_names: vec!["top".into()],
/// };
/// let schema = build_episode_schema(&config);
/// assert!(schema.field_with_name("observation.image.top").is_ok());
/// assert!(schema.field_with_name("state").is_err());
/// ```
pub fn build_episode_schema(config: &SchemaConfig) -> Schema {
    let mut fields = vec![
        Field::new("step_index", DataType::Int64, false),
        Field::new("timestamp", DataType::Float64, true),
        Field::new(
            "action",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                config.action_dim as i32,
            ),
            false,
        ),
    ];

    // State vector (optional)
    if let Some(state_dim) = config.state_dim {
        fields.push(Field::new(
            "state",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                state_dim as i32,
            ),
            true,
        ));
    }

    // Image observation columns, one per camera
    for camera_name in &config.camera_names {
        fields.push(Field::new(
            format!("observation.image.{camera_name}"),
            DataType::LargeBinary,
            true,
        ));
    }

    // Observation state (e.g., proprioception fed as observation)
    if let Some(obs_state_dim) = config.observation_state_dim {
        fields.push(Field::new(
            "observation.state",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                obs_state_dim as i32,
            ),
            true,
        ));
    }

    // Reward and done flags
    fields.push(Field::new("reward", DataType::Float32, true));
    fields.push(Field::new("done", DataType::Boolean, true));

    // Episode index (which episode this step belongs to)
    fields.push(Field::new("episode_index", DataType::Int64, false));

    // Task index (for multi-task datasets)
    fields.push(Field::new("task_index", DataType::Int64, true));

    Schema::new(fields)
}

/// Configuration for building an episode schema.
///
/// Usually derived from the dataset's `info.json` via
/// [`crate::DatasetMetadata::to_schema_config`].
#[derive(Debug, Clone)]
pub struct SchemaConfig {
    /// Length of the `action` vector.
    pub action_dim: usize,
    /// Length of the optional `state` vector; `None` omits the column.
    pub state_dim: Option<usize>,
    /// Length of the optional `observation.state` vector; `None` omits the column.
    pub observation_state_dim: Option<usize>,
    /// Camera names; each adds an `observation.image.{name}` JPEG column.
    pub camera_names: Vec<String>,
}

impl Default for SchemaConfig {
    fn default() -> Self {
        Self {
            action_dim: 7, // common for 7-DoF arms
            state_dim: None,
            observation_state_dim: None,
            camera_names: vec!["cam_0".to_string()],
        }
    }
}

/// Check that `actual` is compatible with `expected`.
///
/// Every field of `expected` that is present in `actual` must have the same
/// data type. Fields missing from `actual` are allowed only if they are
/// nullable in `expected`. Extra fields in `actual` are ignored.
///
/// # Errors
///
/// Returns [`crate::DataCoreError::SchemaMismatch`] describing the first
/// incompatible field.
pub fn validate_schema_compatibility(actual: &Schema, expected: &Schema) -> crate::Result<()> {
    // Check that all expected fields exist in actual (actual may have extra fields)
    for expected_field in expected.fields() {
        match actual.field_with_name(expected_field.name()) {
            Ok(actual_field) => {
                if actual_field.data_type() != expected_field.data_type() {
                    return Err(crate::DataCoreError::SchemaMismatch(format!(
                        "Field '{}': expected type {:?}, got {:?}",
                        expected_field.name(),
                        expected_field.data_type(),
                        actual_field.data_type()
                    )));
                }
            }
            Err(_) => {
                if !expected_field.is_nullable() {
                    return Err(crate::DataCoreError::SchemaMismatch(format!(
                        "Required field '{}' not found in schema",
                        expected_field.name()
                    )));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_schema() {
        let config = SchemaConfig::default();
        let schema = build_episode_schema(&config);

        assert!(schema.field_with_name("step_index").is_ok());
        assert!(schema.field_with_name("action").is_ok());
        assert!(schema.field_with_name("observation.image.cam_0").is_ok());
        assert!(schema.field_with_name("reward").is_ok());
        assert!(schema.field_with_name("done").is_ok());
        assert!(schema.field_with_name("episode_index").is_ok());
    }

    #[test]
    fn test_multi_camera_schema() {
        let config = SchemaConfig {
            action_dim: 7,
            state_dim: Some(14),
            observation_state_dim: Some(14),
            camera_names: vec!["top".to_string(), "wrist".to_string()],
        };
        let schema = build_episode_schema(&config);

        assert!(schema.field_with_name("observation.image.top").is_ok());
        assert!(schema.field_with_name("observation.image.wrist").is_ok());
        assert!(schema.field_with_name("state").is_ok());
        assert!(schema.field_with_name("observation.state").is_ok());
    }

    #[test]
    fn test_schema_compatibility_valid() {
        let config = SchemaConfig::default();
        let schema = build_episode_schema(&config);
        assert!(validate_schema_compatibility(&schema, &schema).is_ok());
    }

    #[test]
    fn test_schema_compatibility_missing_required() {
        let config = SchemaConfig::default();
        let expected = build_episode_schema(&config);

        // Create a schema missing the required "action" field
        let actual = Schema::new(vec![Field::new("step_index", DataType::Int64, false)]);

        assert!(validate_schema_compatibility(&actual, &expected).is_err());
    }
}
