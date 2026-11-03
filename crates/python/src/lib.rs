//! PyO3 bindings for data-core, built into the `data_core._native` extension
//! module and re-exported by the `data_core` Python package.
//!
//! Exposes `Episode`, `Dataset`, `ParallelConfig`, `Pipeline` (built-in and
//! Python-defined augmentations), `LazyDataset` with its copy-on-write
//! preload and raw-sample accessors, and the `LazyTorchDataset` PyTorch
//! frontend. The Python-facing API is documented in
//! `python/data_core/_native.pyi`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyType};

use ::data_core::{DataCoreError, Dataset as RustDataset, Episode as RustEpisode};
use arrow::record_batch::RecordBatch;
use data_core_augmentations::{AugmentationStep, GlobalAugmentation, LocalAugmentation};
use data_core_io::{
    Frame as RustFrame, LazyDataset as RustLazyDataset, LeRobotLoader, LeRobotWriter,
};
use data_core_pipeline::Pipeline as RustPipeline;
use pyo3_arrow::PyRecordBatch;

fn to_py_err(e: DataCoreError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn normalize_index(idx: isize, len: usize, label: &str) -> PyResult<usize> {
    if idx < 0 {
        let len_i = len as isize;
        if idx < -len_i {
            return Err(PyIndexError::new_err(format!(
                "Index {} out of range for {} with length {}",
                idx, label, len
            )));
        }
        Ok((len_i + idx) as usize)
    } else {
        let idx_u = idx as usize;
        if idx_u >= len {
            return Err(PyIndexError::new_err(format!(
                "Index {} out of range for {} with length {}",
                idx_u, label, len
            )));
        }
        Ok(idx_u)
    }
}

fn torch_tensor_f32_vec(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    values: Vec<f32>,
) -> PyResult<PyObject> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("float32")?)?;
    Ok(torch
        .call_method("tensor", (values,), Some(&kwargs))?
        .unbind())
}

fn torch_tensor_f32_2d(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    values: Vec<Vec<f32>>,
) -> PyResult<PyObject> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("float32")?)?;
    Ok(torch
        .call_method("tensor", (values,), Some(&kwargs))?
        .unbind())
}

fn torch_tensor_f64_scalar(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    value: f64,
) -> PyResult<PyObject> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("float64")?)?;
    Ok(torch
        .call_method("tensor", (value,), Some(&kwargs))?
        .unbind())
}

fn torch_tensor_f32_scalar(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    value: f32,
) -> PyResult<PyObject> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("float32")?)?;
    Ok(torch
        .call_method("tensor", (value,), Some(&kwargs))?
        .unbind())
}

fn torch_tensor_i64_scalar(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    value: i64,
) -> PyResult<PyObject> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("int64")?)?;
    Ok(torch
        .call_method("tensor", (value,), Some(&kwargs))?
        .unbind())
}

fn torch_tensor_i64_vec(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    values: Vec<i64>,
) -> PyResult<PyObject> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("int64")?)?;
    Ok(torch
        .call_method("tensor", (values,), Some(&kwargs))?
        .unbind())
}

fn torch_tensor_bool_scalar(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    value: bool,
) -> PyResult<PyObject> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("bool")?)?;
    Ok(torch
        .call_method("tensor", (value,), Some(&kwargs))?
        .unbind())
}

fn torch_tensor_bool_vec(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    values: Vec<bool>,
) -> PyResult<PyObject> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("bool")?)?;
    Ok(torch
        .call_method("tensor", (values,), Some(&kwargs))?
        .unbind())
}

fn decode_jpeg_to_chw_tensor(
    py: Python<'_>,
    torch: &Bound<'_, PyAny>,
    jpeg: &[u8],
) -> PyResult<PyObject> {
    let rgb = image::load_from_memory(jpeg)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to decode JPEG bytes: {e}")))?
        .to_rgb8();
    let (w, h) = rgb.dimensions();
    let raw = rgb.into_raw();

    // A bytearray is writable, so torch.frombuffer does not warn.
    let py_bytes = PyByteArray::new(py, &raw);
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("uint8")?)?;

    let hwc = torch
        .call_method("frombuffer", (py_bytes,), Some(&kwargs))?
        .call_method1("reshape", (h as i64, w as i64, 3i64))?;
    let chw = hwc.call_method1("permute", (2i64, 0i64, 1i64))?;
    let f32 = chw.call_method1("to", (torch.getattr("float32")?,))?;
    Ok(f32.call_method1("__truediv__", (255.0f32,))?.unbind())
}

/// Convert a PyArrow RecordBatch to a Rust RecordBatch via the Arrow C Data Interface.
fn pyarrow_to_rust_batch(pyarrow_batch: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    pyarrow_to_rust_batch_inner(pyarrow_batch.py(), pyarrow_batch)
}

fn pyarrow_to_rust_batch_inner(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    use pyo3::types::PyCapsule;
    // Call __arrow_c_array__() which returns (schema_capsule, array_capsule)
    let capsules = obj.call_method0("__arrow_c_array__")?;
    let schema_capsule: Bound<'_, PyCapsule> = capsules
        .get_item(0)?
        .downcast_into()
        .map_err(|e| PyRuntimeError::new_err(format!("schema capsule downcast: {e}")))?;
    let array_capsule: Bound<'_, PyCapsule> = capsules
        .get_item(1)?
        .downcast_into()
        .map_err(|e| PyRuntimeError::new_err(format!("array capsule downcast: {e}")))?;
    let py_rb = PyRecordBatch::from_arrow_pycapsule(&schema_capsule, &array_capsule)?;
    Ok(py_rb.into_inner())
}

/// A single episode (trajectory) backed by an Arrow RecordBatch.
///
/// Obtain episodes from `Dataset.episode`, `LazyDataset.get_episode` or
/// `Episode.from_pyarrow`; there is no public constructor.
#[pyclass(module = "data_core", name = "Episode")]
#[derive(Clone)]
struct PyEpisode {
    inner: RustEpisode,
}

#[pymethods]
impl PyEpisode {
    /// Number of timesteps (rows) in this episode.
    #[getter]
    fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    /// Index of this episode in its dataset.
    #[getter]
    fn episode_index(&self) -> i64 {
        self.inner.metadata().episode_index
    }

    /// Return the episode data as a pyarrow.RecordBatch without copying.
    fn to_pyarrow(&self, py: Python) -> PyResult<PyObject> {
        let batch = self.inner.data().clone();
        let py_batch = PyRecordBatch::new(batch);
        py_batch.to_pyarrow(py)
    }

    /// Create an Episode from a pyarrow.RecordBatch without copying.
    ///
    /// The episode index is read from the first row of an `episode_index`
    /// int64 column, or 0 if there is none. Raises RuntimeError if the batch
    /// has no rows.
    #[classmethod]
    fn from_pyarrow(_cls: &Bound<'_, PyType>, pyarrow_batch: &Bound<'_, PyAny>) -> PyResult<Self> {
        let batch = pyarrow_to_rust_batch(pyarrow_batch)?;
        let episode = RustEpisode::from_record_batch(batch).map_err(to_py_err)?;
        Ok(PyEpisode { inner: episode })
    }

    /// Decode JPEG columns `observation.image.{camera}` to raw RGB pixels.
    ///
    /// Args:
    ///     camera_names: Cameras to decode, for example ["cam_high", "cam_low"].
    ///
    /// Returns a new Episode with `.rgb` (HWC uint8), `.width` and `.height`
    /// columns added for each camera, so later augmentations can work on
    /// pixels without re-encoding JPEG. Cameras without a JPEG column are
    /// skipped. Raises RuntimeError if a frame fails to decode.
    fn decode_images(&self, camera_names: Vec<String>) -> PyResult<Self> {
        let refs: Vec<&str> = camera_names.iter().map(|s| s.as_str()).collect();
        let decoded = self.inner.decode_images(&refs).map_err(to_py_err)?;
        Ok(PyEpisode { inner: decoded })
    }

    /// Return True if the episode has a column called `name`.
    fn has_column(&self, name: &str) -> bool {
        self.inner.has_column(name)
    }

    /// Return (width, height) for a camera with raw pixel columns.
    ///
    /// Raises RuntimeError if the camera has no `.width` and `.height`
    /// columns, for example before `decode_images`.
    fn image_dimensions(&self, camera_name: &str) -> PyResult<(u32, u32)> {
        self.inner.image_dimensions(camera_name).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "Episode(index={}, steps={})",
            self.inner.metadata().episode_index,
            self.inner.num_steps()
        )
    }
}

/// An in-memory robotics dataset: a list of episodes plus metadata.
///
/// Create one with `Dataset.from_local` or `LazyDataset.load_all`; there is
/// no public constructor.
#[pyclass(module = "data_core", name = "Dataset")]
#[derive(Clone)]
struct PyDataset {
    inner: RustDataset,
}

/// Thread count and chunk size used by `Pipeline.run_parallel`.
///
/// `ParallelConfig()` uses Rayon's default thread pool and a chunk size of 1.
/// The setters return a new object and leave the original unchanged. Apply a
/// config with `Dataset.with_parallel_config`.
#[pyclass(module = "data_core", name = "ParallelConfig")]
#[derive(Clone)]
struct PyParallelConfig {
    inner: ::data_core::ParallelConfig,
}

#[pymethods]
impl PyParallelConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: ::data_core::ParallelConfig::new(),
        }
    }

    /// Return a copy that uses `num_threads` worker threads.
    fn set_num_threads(&self, num_threads: usize) -> Self {
        let mut cfg = self.inner.clone();
        cfg.num_threads = Some(num_threads);
        Self { inner: cfg }
    }

    /// Return a copy with the given chunk size (values below 1 become 1).
    fn set_chunk_size(&self, chunk_size: usize) -> Self {
        let mut cfg = self.inner.clone();
        cfg.chunk_size = chunk_size.max(1);
        Self { inner: cfg }
    }

    /// Configured thread count, or None for Rayon's default pool.
    #[getter]
    fn num_threads(&self) -> Option<usize> {
        self.inner.num_threads
    }

    /// Number of episodes per parallel task.
    #[getter]
    fn chunk_size(&self) -> usize {
        self.inner.chunk_size
    }
}

#[pymethods]
impl PyDataset {
    /// Load every episode of a local LeRobot v2 or v3 dataset into memory.
    ///
    /// For v3 datasets, video frames are decoded and stored as JPEG bytes in
    /// `observation.image.{camera}` columns.
    #[classmethod]
    fn from_local(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let loader = LeRobotLoader::new(path);
        let dataset = loader.load().map_err(to_py_err)?;
        Ok(PyDataset { inner: dataset })
    }

    /// Number of episodes.
    #[getter]
    fn num_episodes(&self) -> usize {
        self.inner.num_episodes()
    }

    /// Total number of steps across all episodes.
    #[getter]
    fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    /// Dataset name (the source path for datasets loaded from disk).
    #[getter]
    fn name(&self) -> String {
        self.inner.metadata().name.clone()
    }

    /// Robot type from info.json, or None.
    #[getter]
    fn robot_type(&self) -> Option<String> {
        self.inner.metadata().robot_type.clone()
    }

    /// Frames per second from info.json, or None.
    #[getter]
    fn fps(&self) -> Option<f64> {
        self.inner.metadata().fps
    }

    /// Camera names declared in the info.json features.
    ///
    /// For LeRobot v3 datasets these cameras are stored as video files, not
    /// as columns in the Parquet data.
    fn camera_names(&self) -> Vec<String> {
        self.inner.metadata().camera_names.clone()
    }

    /// Return the episode at position `index`.
    ///
    /// Raises RuntimeError if `index` is out of range.
    fn episode(&self, index: usize) -> PyResult<PyEpisode> {
        let ep = self.inner.episode(index).map_err(to_py_err)?;
        Ok(PyEpisode { inner: ep.clone() })
    }

    /// Return all episodes as a list.
    fn episodes(&self) -> Vec<PyEpisode> {
        self.inner
            .episodes()
            .iter()
            .map(|ep| PyEpisode { inner: ep.clone() })
            .collect()
    }

    /// Write the dataset to `output_dir` in the LeRobot v2 layout.
    ///
    /// Writes meta/info.json and one Parquet file per episode under
    /// data/chunk-000/.
    fn save_lerobot(&self, output_dir: &str) -> PyResult<()> {
        let writer = LeRobotWriter::new(output_dir);
        writer.write(&self.inner).map_err(to_py_err)
    }

    /// Return a copy that uses `num_threads` threads for parallel operations.
    ///
    /// Other parallel settings, such as the chunk size, are kept.
    ///
    /// Example:
    ///     dataset = Dataset.from_local("path/to/dataset").set_num_threads(8)
    fn set_num_threads(&self, num_threads: usize) -> Self {
        let config = self
            .inner
            .parallel_config()
            .clone()
            .with_threads(num_threads);
        PyDataset {
            inner: self.inner.clone().with_parallel_config(config),
        }
    }

    /// Return a copy that groups `chunk_size` episodes per parallel task.
    ///
    /// Example:
    ///     dataset = Dataset.from_local("path/to/dataset").set_chunk_size(10)
    fn set_chunk_size(&self, chunk_size: usize) -> Self {
        let mut config = self.inner.parallel_config().clone();
        config.chunk_size = chunk_size;
        PyDataset {
            inner: self.inner.clone().with_parallel_config(config),
        }
    }

    /// Return a copy that uses the given ParallelConfig.
    fn with_parallel_config(&self, config: &PyParallelConfig) -> Self {
        PyDataset {
            inner: self
                .inner
                .clone()
                .with_parallel_config(config.inner.clone()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(name='{}', episodes={}, steps={})",
            self.inner.metadata().name,
            self.inner.num_episodes(),
            self.inner.num_steps()
        )
    }

    /// Number of episodes.
    fn __len__(&self) -> usize {
        self.inner.num_episodes()
    }
}

/// Wrapper that calls a Python class instance as a LocalAugmentation.
/// The class must have a run(episode: pa.RecordBatch) -> pa.RecordBatch method.
struct PyLocalAugmentation {
    name: String,
    instance: PyObject,
}

impl PyLocalAugmentation {
    /// Validate that the Python object has the correct interface.
    fn validate(py: Python, instance: &PyObject) -> PyResult<String> {
        // Check if object has 'run' method
        let obj = instance.bind(py);
        if !obj.hasattr("run")? {
            return Err(PyValueError::new_err(
                "LocalAugmentation class must have a 'run' method",
            ));
        }

        // Get the run method and check it's callable
        let run_method = obj.getattr("run")?;
        if !run_method.is_callable() {
            return Err(PyValueError::new_err(
                "LocalAugmentation 'run' method must be callable",
            ));
        }

        // Try to get a name from the class
        let name = if obj.hasattr("__class__")? {
            let cls = obj.getattr("__class__")?;
            if cls.hasattr("__name__")? {
                cls.getattr("__name__")?.extract::<String>()?
            } else {
                "PyLocalAugmentation".to_string()
            }
        } else {
            "PyLocalAugmentation".to_string()
        };

        Ok(name)
    }
}

impl LocalAugmentation for PyLocalAugmentation {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, episode: &RustEpisode) -> ::data_core::Result<RustEpisode> {
        Python::with_gil(|py| {
            // Convert Rust RecordBatch → PyArrow RecordBatch (zero-copy)
            let batch = episode.data().clone();
            let py_batch = PyRecordBatch::new(batch);
            let pyarrow_batch =
                py_batch
                    .to_pyarrow(py)
                    .map_err(|e| DataCoreError::Augmentation {
                        msg: format!("Failed to convert to PyArrow: {e}"),
                    })?;

            // Call the Python augmentation's run method
            let obj = self.instance.bind(py);
            let run_method = obj
                .getattr("run")
                .map_err(|e| DataCoreError::Augmentation {
                    msg: format!("Failed to get 'run' method from '{}': {e}", self.name),
                })?;

            let result =
                run_method
                    .call1((pyarrow_batch,))
                    .map_err(|e| DataCoreError::Augmentation {
                        msg: format!("Python augmentation '{}' run method failed: {e}", self.name),
                    })?;

            // Convert PyArrow RecordBatch back → Rust RecordBatch (zero-copy)
            let new_batch = pyarrow_to_rust_batch_inner(py, &result).map_err(|e| {
                DataCoreError::Augmentation {
                    msg: format!(
                        "Failed to convert result from PyArrow in '{}': {e}",
                        self.name
                    ),
                }
            })?;

            episode.clone().with_data(new_batch)
        })
    }
}

// Safety: Python GIL protects the PyObject
unsafe impl Send for PyLocalAugmentation {}
unsafe impl Sync for PyLocalAugmentation {}

/// Wrapper that calls a Python class instance as a GlobalAugmentation.
/// The class must have a run(episodes: List[pa.RecordBatch], metadata: dict) -> List[pa.RecordBatch] method.
struct PyGlobalAugmentation {
    name: String,
    instance: PyObject,
}

impl PyGlobalAugmentation {
    /// Validate that the Python object has the correct interface.
    fn validate(py: Python, instance: &PyObject) -> PyResult<String> {
        // Check if object has 'run' method
        let obj = instance.bind(py);
        if !obj.hasattr("run")? {
            return Err(PyValueError::new_err(
                "GlobalAugmentation class must have a 'run' method",
            ));
        }

        // Get the run method and check it's callable
        let run_method = obj.getattr("run")?;
        if !run_method.is_callable() {
            return Err(PyValueError::new_err(
                "GlobalAugmentation 'run' method must be callable",
            ));
        }

        // Try to get a name from the class
        let name = if obj.hasattr("__class__")? {
            let cls = obj.getattr("__class__")?;
            if cls.hasattr("__name__")? {
                cls.getattr("__name__")?.extract::<String>()?
            } else {
                "PyGlobalAugmentation".to_string()
            }
        } else {
            "PyGlobalAugmentation".to_string()
        };

        Ok(name)
    }
}

impl GlobalAugmentation for PyGlobalAugmentation {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, dataset: &RustDataset) -> ::data_core::Result<RustDataset> {
        Python::with_gil(|py| {
            // Convert episodes to list of PyArrow RecordBatches
            let mut py_episodes = Vec::new();
            for episode in dataset.episodes() {
                let batch = episode.data().clone();
                let py_batch = PyRecordBatch::new(batch);
                let pyarrow_batch =
                    py_batch
                        .to_pyarrow(py)
                        .map_err(|e| DataCoreError::Augmentation {
                            msg: format!("Failed to convert episode to PyArrow: {e}"),
                        })?;
                py_episodes.push(pyarrow_batch);
            }

            // Convert dataset metadata to Python dict
            let metadata = dataset.metadata();
            let metadata_dict = pyo3::types::PyDict::new(py);
            metadata_dict.set_item("name", &metadata.name).unwrap();
            metadata_dict
                .set_item("robot_type", &metadata.robot_type)
                .unwrap();
            metadata_dict.set_item("fps", metadata.fps).unwrap();
            metadata_dict
                .set_item("num_episodes", metadata.num_episodes)
                .unwrap();
            metadata_dict
                .set_item("num_steps", metadata.num_steps)
                .unwrap();
            metadata_dict
                .set_item("action_dim", metadata.action_dim)
                .unwrap();
            metadata_dict
                .set_item("state_dim", metadata.state_dim)
                .unwrap();
            metadata_dict
                .set_item("camera_names", &metadata.camera_names)
                .unwrap();
            metadata_dict.set_item("tasks", &metadata.tasks).unwrap();
            metadata_dict.set_item("source", &metadata.source).unwrap();

            // Call the Python augmentation's run method
            let obj = self.instance.bind(py);
            let run_method = obj
                .getattr("run")
                .map_err(|e| DataCoreError::Augmentation {
                    msg: format!("Failed to get 'run' method from '{}': {e}", self.name),
                })?;

            let result = run_method
                .call1((py_episodes, metadata_dict))
                .map_err(|e| DataCoreError::Augmentation {
                    msg: format!(
                        "Python global augmentation '{}' run method failed: {e}",
                        self.name
                    ),
                })?;

            // Convert result back to Rust episodes
            let result_list = result.downcast::<pyo3::types::PyList>().map_err(|e| {
                DataCoreError::Augmentation {
                    msg: format!(
                        "Global augmentation '{}' must return a list: {e}",
                        self.name
                    ),
                }
            })?;

            let mut new_episodes = Vec::new();
            for (idx, item) in result_list.iter().enumerate() {
                let new_batch = pyarrow_to_rust_batch_inner(py, &item).map_err(|e| {
                    DataCoreError::Augmentation {
                        msg: format!(
                            "Failed to convert result episode {} from PyArrow in '{}': {e}",
                            idx, self.name
                        ),
                    }
                })?;
                let episode = RustEpisode::from_record_batch(new_batch)?;
                new_episodes.push(episode);
            }

            // Create new dataset with augmented episodes
            // Update metadata with new counts
            let mut new_metadata = metadata.clone();
            new_metadata.num_episodes = new_episodes.len();
            new_metadata.num_steps = new_episodes.iter().map(|e| e.num_steps()).sum();

            RustDataset::new(new_episodes, new_metadata, dataset.schema().clone())
        })
    }
}

// Safety: Python GIL protects the PyObject
unsafe impl Send for PyGlobalAugmentation {}
unsafe impl Sync for PyGlobalAugmentation {}

/// Raise ValueError if a built-in step's `camera_names` names a camera the
/// dataset does not have. Image augmentations skip missing cameras, so a typo
/// would otherwise leave the images unchanged without any error. No check is
/// made when the dataset declares no cameras.
fn check_camera_names(
    step: &str,
    params: &serde_json::Value,
    available: &[String],
) -> PyResult<()> {
    if available.is_empty() {
        return Ok(());
    }
    let unknown: Vec<&str> = params["camera_names"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .filter(|c| !available.iter().any(|a| a == c))
        .collect();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "'{step}': unknown camera(s) {unknown:?}; this dataset has {available:?}"
        )))
    }
}

/// Create a built-in local augmentation from name + JSON params.
fn create_builtin_augmentation(
    name: &str,
    params: &serde_json::Value,
) -> PyResult<data_core_augmentations::BoxedLocalAug> {
    use data_core_augmentations::{action, image};

    // Image augmentations need explicit camera names: a default would not
    // match any real dataset and would silently leave the images unchanged.
    let parse_cameras = |p: &serde_json::Value| -> PyResult<Vec<String>> {
        let cameras: Vec<String> = p["camera_names"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if cameras.is_empty() {
            return Err(PyValueError::new_err(format!(
                "'{name}' needs a non-empty \"camera_names\" list, for example the \
                 names from LazyDataset.camera_names()"
            )));
        }
        Ok(cameras)
    };

    let window_size = params["window_size"].as_u64().unwrap_or(0) as usize;

    match name {
        "action_noise" => {
            let std = params["std"].as_f64().unwrap_or(0.01) as f32;
            Ok(std::sync::Arc::new(action::ActionNoise::new(std)))
        }
        "temporal_subsample" => {
            let step = params["step"].as_u64().unwrap_or(2) as usize;
            Ok(std::sync::Arc::new(action::TemporalSubsample::new(step)))
        }
        "state_scaling" => {
            let scale = params["scale"].as_f64().unwrap_or(1.0) as f32;
            let column = params["column"]
                .as_str()
                .unwrap_or("observation.state")
                .to_string();
            Ok(std::sync::Arc::new(action::StateScaling::new(
                scale, column,
            )))
        }
        "color_jitter" => {
            let brightness = params["brightness"].as_f64().unwrap_or(0.1) as f32;
            let contrast = params["contrast"].as_f64().unwrap_or(0.1) as f32;
            let saturation = params["saturation"].as_f64().unwrap_or(0.1) as f32;
            let cameras = parse_cameras(params)?;
            Ok(std::sync::Arc::new(
                image::ColorJitter::new(brightness, contrast, saturation, cameras)
                    .with_window_size(window_size),
            ))
        }
        "random_crop" => {
            let height = params["height"].as_u64().unwrap_or(224) as u32;
            let width = params["width"].as_u64().unwrap_or(224) as u32;
            let cameras = parse_cameras(params)?;
            Ok(std::sync::Arc::new(
                image::RandomCrop::new(height, width, cameras).with_window_size(window_size),
            ))
        }
        "gaussian_blur" => {
            let sigma = params["sigma"].as_f64().unwrap_or(1.0) as f32;
            let cameras = parse_cameras(params)?;
            Ok(std::sync::Arc::new(image::GaussianBlur::new(
                sigma, cameras,
            )))
        }
        "gaussian_noise" => {
            let std = params["std"].as_f64().unwrap_or(0.02) as f32;
            let cameras = parse_cameras(params)?;
            Ok(std::sync::Arc::new(
                image::GaussianNoise::new(std, cameras).with_window_size(window_size),
            ))
        }
        "horizontal_flip" => {
            let p = params["p"].as_f64().unwrap_or(0.5) as f32;
            let cameras = parse_cameras(params)?;
            let mirror_dims: Vec<usize> = params["action_mirror_dims"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            Ok(std::sync::Arc::new(
                image::HorizontalFlip::new(p, cameras)
                    .with_action_mirror_dims(mirror_dims)
                    .with_window_size(window_size),
            ))
        }
        "state_noise" => {
            let std = params["std"].as_f64().unwrap_or(0.01) as f32;
            let column = params["column"]
                .as_str()
                .unwrap_or("observation.state")
                .to_string();
            Ok(std::sync::Arc::new(action::StateNoise::new(std, column)))
        }
        "frame_dropout" => {
            let drop_prob = params["drop_prob"].as_f64().unwrap_or(0.1) as f32;
            let cameras = parse_cameras(params)?;
            Ok(std::sync::Arc::new(action::FrameDropout::new(
                drop_prob, cameras,
            )))
        }
        "fused_image" => {
            let cameras = parse_cameras(params)?;
            let mut aug = image::FusedImageAugmentation::new(cameras).with_window_size(window_size);
            if let (Some(h), Some(w)) = (
                params["crop_height"].as_u64(),
                params["crop_width"].as_u64(),
            ) {
                aug = aug.with_crop(h as u32, w as u32);
            }
            let brightness = params["brightness"].as_f64().unwrap_or(0.0) as f32;
            let contrast = params["contrast"].as_f64().unwrap_or(0.0) as f32;
            let saturation = params["saturation"].as_f64().unwrap_or(0.0) as f32;
            if brightness > 0.0 || contrast > 0.0 || saturation > 0.0 {
                aug = aug.with_jitter(brightness, contrast, saturation);
            }
            let noise_std = params["noise_std"].as_f64().unwrap_or(0.0) as f32;
            if noise_std > 0.0 {
                aug = aug.with_noise(noise_std);
            }
            let flip_p = params["flip_p"].as_f64().unwrap_or(0.0) as f32;
            if flip_p > 0.0 {
                let mirror_dims: Vec<usize> = params["action_mirror_dims"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect()
                    })
                    .unwrap_or_default();
                aug = aug.with_flip(flip_p, mirror_dims);
            }
            if params["chw_output"].as_bool().unwrap_or(false) {
                aug.chw_output = true;
            }
            Ok(std::sync::Arc::new(aug))
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown builtin augmentation: {name}"
        ))),
    }
}

/// An ordered list of augmentation steps applied to a Dataset.
///
/// `Pipeline()` creates an empty pipeline. Built-in step names accepted by
/// `add_builtin`: action_noise, temporal_subsample, state_scaling,
/// state_noise, color_jitter, random_crop, gaussian_blur, gaussian_noise,
/// horizontal_flip, frame_dropout and fused_image. Python steps are added
/// with `add_local` and `add_global`.
#[pyclass(module = "data_core", name = "Pipeline")]
struct PyPipeline {
    inner: RustPipeline,
    /// (step name, params) of each built-in step, for the camera check in `run`.
    builtin_params: Vec<(String, serde_json::Value)>,
}

impl PyPipeline {
    fn check_cameras(&self, dataset: &PyDataset) -> PyResult<()> {
        let available = &dataset.inner.metadata().camera_names;
        for (name, params) in &self.builtin_params {
            check_camera_names(name, params, available)?;
        }
        Ok(())
    }
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new() -> Self {
        PyPipeline {
            inner: RustPipeline::new(),
            builtin_params: Vec::new(),
        }
    }

    /// Append a built-in Rust augmentation.
    ///
    /// Args:
    ///     name: One of the built-in names listed on the class.
    ///     params_json: JSON object with the step's parameters, for example
    ///         '{"std": 0.05}'. Missing parameters take their defaults. Image
    ///         augmentations require a "camera_names" list (names as returned
    ///         by `camera_names()`). gaussian_noise, horizontal_flip, frame_dropout
    ///         and fused_image change only raw `.rgb` pixel columns;
    ///         gaussian_blur changes only JPEG columns.
    ///
    /// Raises ValueError for an unknown name, invalid JSON, or an image
    /// augmentation without camera names. `run` and `run_parallel` raise
    /// ValueError if a step names a camera the dataset does not have.
    fn add_builtin(&mut self, name: &str, params_json: &str) -> PyResult<()> {
        let params: serde_json::Value =
            serde_json::from_str(params_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let aug = create_builtin_augmentation(name, &params)?;
        self.inner
            .add_step(data_core_augmentations::AugmentationStep::Local(aug));
        self.builtin_params.push((name.to_string(), params));
        Ok(())
    }

    /// Append a Python per-episode augmentation.
    ///
    /// `instance` must have a method `run(episode: pa.RecordBatch) ->
    /// pa.RecordBatch`, called once per episode with a positional argument.
    /// The step name is the class name. Raises ValueError if `run` is missing
    /// or not callable.
    ///
    /// Example:
    ///     class ScaleActions:
    ///         def __init__(self, factor):
    ///             self.factor = factor
    ///
    ///         def run(self, episode):
    ///             ...  # build and return a new pa.RecordBatch
    ///
    ///     pipeline = Pipeline()
    ///     pipeline.add_local(ScaleActions(0.5))
    fn add_local(&mut self, py: Python, instance: PyObject) -> PyResult<()> {
        // Validate the instance has the correct interface
        let name = PyLocalAugmentation::validate(py, &instance)?;

        let aug = PyLocalAugmentation { name, instance };
        self.inner.add_step(AugmentationStep::Local(Arc::new(aug)));
        Ok(())
    }

    /// Append a Python whole-dataset augmentation.
    ///
    /// `instance` must have a method `run(episodes: list[pa.RecordBatch],
    /// metadata: dict) -> list[pa.RecordBatch]`, called with positional
    /// arguments. `metadata` has the keys name, robot_type, fps,
    /// num_episodes, num_steps, action_dim, state_dim, camera_names, tasks
    /// and source. The returned list may have a different number of
    /// episodes. Raises ValueError if `run` is missing or not callable.
    ///
    /// Example:
    ///     class DuplicateEpisodes:
    ///         def run(self, episodes, metadata):
    ///             return episodes + episodes
    ///
    ///     pipeline = Pipeline()
    ///     pipeline.add_global(DuplicateEpisodes())
    fn add_global(&mut self, py: Python, instance: PyObject) -> PyResult<()> {
        // Validate the instance has the correct interface
        let name = PyGlobalAugmentation::validate(py, &instance)?;

        let aug = PyGlobalAugmentation { name, instance };
        self.inner.add_step(AugmentationStep::Global(Arc::new(aug)));
        Ok(())
    }

    /// Number of steps in the pipeline.
    #[getter]
    fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    /// Names of the steps, in execution order.
    fn step_names(&self) -> Vec<String> {
        self.inner
            .step_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Apply every step in order and return a new Dataset.
    ///
    /// The input dataset is not modified. Raises ValueError if a built-in
    /// step names a camera the dataset does not have. Errors raised by a step, including
    /// Python exceptions from `add_local`/`add_global` steps, are re-raised as
    /// RuntimeError.
    fn run(&self, dataset: &PyDataset) -> PyResult<PyDataset> {
        self.check_cameras(dataset)?;
        let result = self.inner.run(dataset.inner.clone()).map_err(to_py_err)?;
        Ok(PyDataset { inner: result })
    }

    /// Like `run`, but local steps run in parallel across episodes.
    ///
    /// Uses Rayon with the dataset's ParallelConfig. Global steps still run
    /// sequentially.
    ///
    /// Example:
    ///     dataset = Dataset.from_local("path/to/dataset").set_num_threads(8)
    ///     pipeline = Pipeline()
    ///     pipeline.add_builtin("action_noise", '{"std": 0.01}')
    ///     augmented = pipeline.run_parallel(dataset)
    #[pyo3(name = "run_parallel")]
    fn run_par(&self, dataset: &PyDataset) -> PyResult<PyDataset> {
        self.check_cameras(dataset)?;
        let result = self
            .inner
            .run_par(dataset.inner.clone())
            .map_err(to_py_err)?;
        Ok(PyDataset { inner: result })
    }

    fn __repr__(&self) -> String {
        let names = self.step_names().join(", ");
        format!("Pipeline(steps=[{}])", names)
    }
}

/// A single timestep returned by `LazyDataset.get_frame` or `LazyDataset[idx]`.
#[pyclass(module = "data_core", name = "Frame")]
#[derive(Clone)]
struct PyFrame {
    inner: RustFrame,
}

#[pymethods]
impl PyFrame {
    /// Global frame index in the full dataset.
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }

    /// Index of the episode this frame belongs to.
    #[getter]
    fn episode_index(&self) -> usize {
        self.inner.episode_index
    }

    /// Frame index within the episode (0-based).
    #[getter]
    fn frame_index(&self) -> usize {
        self.inner.frame_index_in_episode
    }

    /// Timestamp in seconds.
    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp
    }

    /// Action vector as a list of floats.
    #[getter]
    fn action(&self) -> Vec<f32> {
        self.inner.action.clone()
    }

    /// `observation.state` vector, or None if the dataset has none.
    #[getter]
    fn state(&self) -> Option<Vec<f32>> {
        self.inner.state.clone()
    }

    /// Reward, or None if the dataset has no `reward` column.
    #[getter]
    fn reward(&self) -> Option<f32> {
        self.inner.reward
    }

    /// True for a terminal frame; False if the dataset has no `done` column.
    #[getter]
    fn done(&self) -> bool {
        self.inner.done
    }

    /// Task index.
    #[getter]
    fn task_index(&self) -> usize {
        self.inner.task_index
    }

    /// Task description from tasks.jsonl, or None.
    #[getter]
    fn task(&self) -> Option<String> {
        self.inner.task.clone()
    }

    /// JPEG bytes for `camera`, or None if this frame has no JPEG image for it.
    fn image(&self, camera: &str) -> Option<Vec<u8>> {
        self.inner.images.get(camera).cloned()
    }

    /// Names of the cameras that have a JPEG image in this frame.
    fn camera_names(&self) -> Vec<String> {
        self.inner.images.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame(index={}, episode={}, action_dim={})",
            self.inner.index,
            self.inner.episode_index,
            self.inner.action.len()
        )
    }
}

/// A dataset that reads metadata up front and loads episodes on demand.
///
/// Its interface mirrors LeRobot's `LeRobotDataset`. Create one with
/// `LazyDataset.open`. Loaded episodes are cached.
///
/// Frame indices passed to `get_frame`, `__getitem__`, `get_frame_windowed`
/// and `get_sample_raw` count frames across the selected episodes (see
/// `with_episodes`). `get_episode` and `preload_episodes` take dataset
/// episode indices.
///
/// `with_episodes`, `with_cache_limit` and `with_raw_pixels` return a new
/// dataset that keeps the episode selection, raw-pixel mode and cache limit
/// of the original. Call them before `add_augmentation`: augmentations cannot
/// be copied, so calling a `with_*` method after adding one raises ValueError.
#[pyclass(module = "data_core", name = "LazyDataset")]
struct PyLazyDataset {
    inner: RustLazyDataset,
}

impl PyLazyDataset {
    /// Reopen the dataset with this instance's selection, raw-pixel mode and
    /// cache limit, for the `with_*` builders.
    fn reopen_with_settings(&self) -> PyResult<RustLazyDataset> {
        if self.inner.has_augmentations() {
            return Err(PyValueError::new_err(
                "with_episodes, with_cache_limit and with_raw_pixels must be called \
                 before add_augmentation",
            ));
        }
        let mut ds = RustLazyDataset::open(self.inner.metadata().root.clone())
            .map_err(to_py_err)?
            .with_cache_limit(self.inner.cache_limit());
        if let Some(selected) = self.inner.selected_episodes() {
            ds = ds.with_episodes(selected.to_vec());
        }
        if self.inner.raw_pixels() {
            ds = ds.with_raw_pixels();
        }
        Ok(ds)
    }

    fn clone_inner_for_torch(&self) -> PyResult<RustLazyDataset> {
        let root = self.inner.metadata().root.clone();
        let selected_episodes: Vec<usize> = self.inner.episode_indices().collect();

        let mut cloned = RustLazyDataset::open(root).map_err(to_py_err)?;
        let total_episodes = cloned.metadata().total_episodes;
        let is_identity_selection = selected_episodes.len() == total_episodes
            && selected_episodes.iter().enumerate().all(|(i, ep)| *ep == i);

        if !is_identity_selection {
            cloned = cloned.with_episodes(selected_episodes);
        }

        Ok(cloned)
    }
}

#[pymethods]
impl PyLazyDataset {
    /// Open a local LeRobot v2 or v3 dataset directory.
    ///
    /// Reads the metadata files (info.json, episode and task tables) and, for
    /// v3, the `episode_index` column of the data files; no video is decoded
    /// until frames are accessed.
    #[classmethod]
    fn open(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let dataset = RustLazyDataset::open(path).map_err(to_py_err)?;
        Ok(PyLazyDataset { inner: dataset })
    }

    /// Return a new dataset restricted to the given episode indices, in that order.
    fn with_episodes(&self, episodes: Vec<usize>) -> PyResult<Self> {
        let new_inner = self.reopen_with_settings()?.with_episodes(episodes);
        Ok(PyLazyDataset { inner: new_inner })
    }

    /// Return a new dataset that caches at most `limit` episodes (0 means unlimited).
    fn with_cache_limit(&self, limit: usize) -> PyResult<Self> {
        let new_inner = self.reopen_with_settings()?.with_cache_limit(limit);
        Ok(PyLazyDataset { inner: new_inner })
    }

    /// Return a new dataset that decodes video to raw RGB instead of JPEG bytes.
    ///
    /// Raw pixels are stored as FixedSizeList<UInt8> columns of H*W*3 bytes
    /// per frame, which avoids JPEG artifacts and lets augmentations work on
    /// pixels directly, at the cost of more memory.
    fn with_raw_pixels(&self) -> PyResult<Self> {
        let new_inner = self.reopen_with_settings()?.with_raw_pixels();
        Ok(PyLazyDataset { inner: new_inner })
    }

    /// Set the episode cache limit in place (0 means unlimited).
    fn set_cache_limit(&mut self, limit: usize) {
        self.inner.set_cache_limit(limit);
    }

    /// Drop all cached episodes to free memory.
    fn clear_cache(&self) {
        self.inner.clear_cache();
    }

    /// Load episodes into a read-only cache for a multi-worker DataLoader.
    ///
    /// Call this before creating a DataLoader with num_workers > 0. Forked
    /// workers share the cached pixel data copy-on-write and read it without
    /// locks. After this call, accessing an episode that was not preloaded
    /// raises RuntimeError.
    fn preload_episodes(&mut self, episode_indices: Vec<usize>) -> PyResult<()> {
        self.inner
            .preload_episodes(&episode_indices)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Dataset name (the directory name).
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Frames per second.
    #[getter]
    fn fps(&self) -> f64 {
        self.inner.fps()
    }

    /// Number of frames in the selected episodes.
    #[getter]
    fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    /// Number of selected episodes.
    #[getter]
    fn num_episodes(&self) -> usize {
        self.inner.num_episodes()
    }

    /// Camera names declared in the info.json features.
    fn camera_names(&self) -> Vec<String> {
        self.inner.camera_names().to_vec()
    }

    /// (from_index, to_index) frame range of each selected episode, in the
    /// frame index space used by `get_frame` and `__getitem__`.
    ///
    /// Without `with_episodes` these are the dataset's global ranges. With a
    /// selection, frames are numbered across the selected episodes in order,
    /// so the ranges are contiguous from 0.
    fn episode_data_index(&self) -> Vec<(usize, usize)> {
        let meta = self.inner.metadata();
        if self.inner.selected_episodes().is_none() {
            return (0..self.inner.num_episodes())
                .filter_map(|i| {
                    meta.episode(i)
                        .map(|ep| (ep.dataset_from_index, ep.dataset_to_index))
                })
                .collect();
        }
        let mut start = 0;
        self.inner
            .episode_indices()
            .filter_map(|i| meta.episode(i))
            .map(|ep| {
                let range = (start, start + ep.length);
                start += ep.length;
                range
            })
            .collect()
    }

    /// Return frame `idx`, loading its episode if needed.
    ///
    /// Raises RuntimeError if `idx` is out of range.
    fn get_frame(&self, idx: usize) -> PyResult<PyFrame> {
        let frame = self.inner.get_frame(idx).map_err(to_py_err)?;
        Ok(PyFrame { inner: frame })
    }

    /// Return episode `idx` (a dataset episode index), loading and caching it if needed.
    fn get_episode(&self, idx: usize) -> PyResult<PyEpisode> {
        let episode = self.inner.get_episode(idx).map_err(to_py_err)?;
        Ok(PyEpisode {
            inner: (*episode).clone(),
        })
    }

    /// Return a window of frames around `idx`, with augmentations applied.
    ///
    /// Args:
    ///     idx: Frame index.
    ///     window_offsets: Frame offsets relative to `idx`, for example
    ///         [-15, -10, -5, 0] for a 4-frame window with 0.1 s spacing at
    ///         50 fps. Must not be empty.
    ///
    /// Returns an Episode with len(window_offsets) rows. Offsets are clamped
    /// to the episode that contains `idx`. Augmentations added with
    /// `add_augmentation` are applied with fresh random parameters on every
    /// call.
    fn get_frame_windowed(&self, idx: usize, window_offsets: Vec<i64>) -> PyResult<PyEpisode> {
        let episode = self
            .inner
            .get_frame_windowed(idx, &window_offsets)
            .map_err(to_py_err)?;
        Ok(PyEpisode { inner: episode })
    }

    /// Add a built-in augmentation applied by the windowed accessors.
    ///
    /// Args:
    ///     name: A built-in name, as accepted by `Pipeline.add_builtin`.
    ///     params_json: JSON object with the augmentation parameters. The
    ///         "window_size" parameter keeps random parameters the same across
    ///         the frames of one window (gaussian_noise still draws new noise
    ///         for every frame).
    ///
    /// Raises ValueError for an unknown name, invalid JSON, or an image
    /// augmentation without camera names or with a camera name that is not in
    /// `camera_names()`.
    ///
    /// Example:
    ///     ds.add_augmentation(
    ///         "random_crop",
    ///         '{"height": 256, "width": 256, "camera_names": ["cam_high"], "window_size": 4}',
    ///     )
    fn add_augmentation(&mut self, name: &str, params_json: &str) -> PyResult<()> {
        let params: serde_json::Value =
            serde_json::from_str(params_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let aug = create_builtin_augmentation(name, &params)?;
        check_camera_names(name, &params, self.inner.camera_names())?;
        self.inner.add_augmentation_boxed(aug);
        Ok(())
    }

    /// Like `get_frame_windowed`, plus float32 CHW image columns in [0, 1].
    ///
    /// Each raw `observation.image.{camera}.rgb` column gets a matching
    /// `.f32` column that can be wrapped as a tensor without conversion.
    fn get_frame_windowed_f32(&self, idx: usize, window_offsets: Vec<i64>) -> PyResult<PyEpisode> {
        let episode = self
            .inner
            .get_frame_windowed_f32(idx, &window_offsets)
            .map_err(to_py_err)?;
        Ok(PyEpisode { inner: episode })
    }

    /// Return one training sample with images as raw uint8 bytes.
    ///
    /// Args:
    ///     idx: Frame index.
    ///     image_offsets: Frame offsets of the image window, clamped to the
    ///         episode that contains `idx`. Must not be empty.
    ///     action_offsets: Frame offsets of the action window, clamped to the
    ///         episode that contains `idx`.
    ///
    /// Returns a dict with these keys:
    ///     "images": {camera: (pixels, width, height, n_frames)}, always.
    ///         `pixels` holds n_frames * height * width * 3 uint8 values in
    ///         HWC order per frame (CHW when the fused_image augmentation runs
    ///         with "chw_output": true). Raw pixel columns need
    ///         `with_raw_pixels`; JPEG columns are decoded.
    ///     "state": list of state vectors, one per image frame, if the
    ///         dataset has `observation.state`.
    ///     "action": list of action vectors, if `action_offsets` is non-empty.
    ///     "action_is_pad": list of bools, True where an action offset fell
    ///         outside the episode and the boundary action was repeated.
    ///     "episode_index", "frame_index", "timestamp": read from the window;
    ///         frame_index and timestamp come from its last row.
    fn get_sample_raw(
        &self,
        py: Python<'_>,
        idx: usize,
        image_offsets: Vec<i64>,
        action_offsets: Vec<i64>,
    ) -> PyResult<PyObject> {
        use arrow_array::{
            Array, FixedSizeListArray, Float32Array, Float64Array, Int64Array, LargeBinaryArray,
            UInt8Array,
        };
        use rayon::prelude::*;

        // Get augmented image window
        let window = self
            .inner
            .get_frame_windowed(idx, &image_offsets)
            .map_err(to_py_err)?;

        let data = window.data();
        let result = PyDict::new(py);

        // --- Images: extract raw bytes directly, skip Arrow serialization ---
        // Two cases:
        //   (a) raw_pixels mode → column is `observation.image.{cam}.rgb`
        //       FixedSizeList<UInt8>, already HWC bytes. Copy contiguous buffer.
        //   (b) JPEG-cache mode → column is `observation.image.{cam}`
        //       LargeBinaryArray of JPEG-encoded frames. Decode in parallel via
        //       rayon (the `image` crate's JPEG decoder), emit HWC bytes.
        // Output contract is identical: (PyBytes(HWC uint8), w, h, n_frames).
        let images = PyDict::new(py);
        let schema = data.schema();
        for field in schema.fields() {
            let name = field.name();

            // Case (a): raw pixels
            if name.ends_with(".rgb") {
                let cam = name
                    .strip_suffix(".rgb")
                    .and_then(|s| s.strip_prefix("observation.image."))
                    .unwrap_or("");

                let col = data
                    .column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!("Column {name} not found or wrong type"))
                    })?;

                let n_frames = col.len();
                let values = col
                    .values()
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| PyRuntimeError::new_err("Expected UInt8Array"))?;

                let raw_buf = values.values();
                let py_bytes = PyBytes::new(py, raw_buf.as_ref());

                let (w, h) = window.image_dimensions(cam).map_err(to_py_err)?;
                let cam_info = (py_bytes, w, h, n_frames);
                images.set_item(cam, cam_info)?;
                continue;
            }

            // Case (b): JPEG cache. Filter to image columns only (skip
            // `.width`, `.height`, `.f32` and the `episode_index`/`action`/etc.).
            if !name.starts_with("observation.image.") {
                continue;
            }
            if name.ends_with(".width") || name.ends_with(".height") || name.ends_with(".f32") {
                continue;
            }
            // After the .rgb branch above, anything left starting with
            // `observation.image.` and not ending in a suffix is the JPEG column.
            let col = match data
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<LargeBinaryArray>())
            {
                Some(c) => c,
                None => continue, // Not a JPEG column (e.g., bare `.rgb` was a different type)
            };

            let cam = name.strip_prefix("observation.image.").unwrap_or("");
            let n_frames = col.len();

            // Collect JPEG slices for parallel decode
            let jpegs: Vec<&[u8]> = (0..n_frames).map(|i| col.value(i)).collect();

            // Decode all JPEGs in parallel → RgbImage each
            let decoded: Result<Vec<image::RgbImage>, String> = jpegs
                .par_iter()
                .map(|j| {
                    image::load_from_memory_with_format(j, image::ImageFormat::Jpeg)
                        .map(|img| img.into_rgb8())
                        .map_err(|e| format!("JPEG decode failed: {e}"))
                })
                .collect();
            let decoded = decoded.map_err(PyRuntimeError::new_err)?;

            let (w, h) = (decoded[0].width(), decoded[0].height());
            let frame_size = (w * h * 3) as usize;
            let mut flat: Vec<u8> = Vec::with_capacity(n_frames * frame_size);
            for img in &decoded {
                flat.extend_from_slice(img.as_raw());
            }
            let py_bytes = PyBytes::new(py, &flat);
            images.set_item(cam, (py_bytes, w, h, n_frames))?;
        }
        result.set_item("images", images)?;

        // --- State: extract from Arrow ---
        if let Some(col) = data.column_by_name("observation.state") {
            if let Some(list) = col.as_any().downcast_ref::<FixedSizeListArray>() {
                let n = list.len();
                let values = list
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .map(|a| a.values().to_vec())
                    .unwrap_or_default();
                let dim = values.len().checked_div(n).unwrap_or(0);
                let state_list: Vec<Vec<f32>> = (0..n)
                    .map(|i| values[i * dim..(i + 1) * dim].to_vec())
                    .collect();
                result.set_item("state", state_list)?;
            }
        }

        // --- Action: separate lightweight access ---
        // Offsets are clamped to the episode that contains `idx`, like the
        // image window; `action_is_pad` marks offsets that fell outside it.
        if !action_offsets.is_empty() {
            let (_, frame_in_ep, ep_len) = self.inner.locate_frame(idx).map_err(to_py_err)?;
            let ep_start = idx - frame_in_ep;
            let mut actions: Vec<Vec<f32>> = Vec::with_capacity(action_offsets.len());
            let mut is_pad: Vec<bool> = Vec::with_capacity(action_offsets.len());
            for &offset in &action_offsets {
                let rel = frame_in_ep as i64 + offset;
                is_pad.push(rel < 0 || rel >= ep_len as i64);
                let target = ep_start + rel.clamp(0, ep_len as i64 - 1) as usize;
                let frame = self.inner.get_frame(target).map_err(to_py_err)?;
                actions.push(frame.action);
            }
            result.set_item("action", actions)?;
            result.set_item("action_is_pad", is_pad)?;
        }

        // --- Scalars ---
        if let Some(col) = data.column_by_name("episode_index") {
            if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                result.set_item("episode_index", arr.value(0))?;
            }
        }
        let n = data.num_rows();
        if let Some(col) = data.column_by_name("frame_index") {
            if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                result.set_item("frame_index", arr.value(n - 1))?;
            }
        }
        if let Some(col) = data.column_by_name("timestamp") {
            if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                result.set_item("timestamp", arr.value(n - 1) as f32)?;
            }
        }

        Ok(result.into_any().unbind())
    }

    /// Return True if `add_augmentation` has been called.
    fn has_augmentations(&self) -> bool {
        self.inner.has_augmentations()
    }

    /// Load the selected episodes into an in-memory Dataset.
    fn load_all(&self) -> PyResult<PyDataset> {
        let dataset = self.inner.load_all().map_err(to_py_err)?;
        Ok(PyDataset { inner: dataset })
    }

    /// Wrap this dataset as a PyTorch map-style dataset (LazyTorchDataset).
    ///
    /// Args:
    ///     mode: "random" for one frame per sample, "chunk" for `chunk_size`
    ///         contiguous frames from one episode, or "episode" for whole
    ///         episodes.
    ///     chunk_size: Frames per sample in chunk mode (default 1).
    ///     pad_to_length: Episode mode only: zero-pad shorter episodes to this
    ///         length.
    ///     truncate_to_length: Episode mode only: cut longer episodes to this
    ///         length (applied before padding).
    ///
    /// The episode selection is kept; augmentations, cache limit and raw-pixel
    /// mode are not. Raises ValueError for an unknown mode, a zero length, or
    /// a chunk size longer than every episode. torch must be installed when
    /// samples are read.
    #[pyo3(signature = (mode, chunk_size=None, pad_to_length=None, truncate_to_length=None))]
    fn to_torch_dataset(
        &self,
        mode: &str,
        chunk_size: Option<usize>,
        pad_to_length: Option<usize>,
        truncate_to_length: Option<usize>,
    ) -> PyResult<PyLazyTorchDataset> {
        let inner = self.clone_inner_for_torch()?;
        PyLazyTorchDataset::new_from_lazy(
            inner,
            mode,
            chunk_size,
            pad_to_length,
            truncate_to_length,
        )
    }

    /// Return a torch.utils.data.DataLoader over `to_torch_dataset(...)`.
    ///
    /// Takes the arguments of `to_torch_dataset` plus the DataLoader
    /// arguments batch_size, shuffle, num_workers and drop_last. The loader
    /// uses `LazyTorchDataset.collate_fn`. Requires torch.
    #[pyo3(signature = (
        mode,
        batch_size=1,
        shuffle=true,
        num_workers=0,
        drop_last=false,
        chunk_size=None,
        pad_to_length=None,
        truncate_to_length=None
    ))]
    #[allow(clippy::too_many_arguments)] // mirrors the Python keyword-argument API
    fn torch_dataloader(
        &self,
        py: Python<'_>,
        mode: &str,
        batch_size: usize,
        shuffle: bool,
        num_workers: usize,
        drop_last: bool,
        chunk_size: Option<usize>,
        pad_to_length: Option<usize>,
        truncate_to_length: Option<usize>,
    ) -> PyResult<PyObject> {
        let dataset = self.to_torch_dataset(mode, chunk_size, pad_to_length, truncate_to_length)?;
        let py_dataset = Py::new(py, dataset)?;
        let dataset_obj = py_dataset.bind(py);
        let collate_fn = dataset_obj.getattr("collate_fn")?;

        let data_utils = py.import("torch.utils.data")?;
        let data_loader = data_utils.getattr("DataLoader")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("batch_size", batch_size)?;
        kwargs.set_item("shuffle", shuffle)?;
        kwargs.set_item("num_workers", num_workers)?;
        kwargs.set_item("drop_last", drop_last)?;
        kwargs.set_item("collate_fn", collate_fn)?;
        Ok(data_loader.call((dataset_obj,), Some(&kwargs))?.unbind())
    }

    /// Return frame `idx` as a Frame; negative indices count from the end.
    ///
    /// Raises IndexError when `idx` is out of range.
    fn __getitem__(&self, idx: isize) -> PyResult<PyFrame> {
        let idx = normalize_index(idx, self.inner.num_frames(), "dataset")?;
        self.get_frame(idx)
    }

    /// Number of frames in the selected episodes.
    fn __len__(&self) -> usize {
        self.inner.num_frames()
    }

    fn __repr__(&self) -> String {
        format!(
            "LazyDataset(name='{}', frames={}, episodes={})",
            self.inner.name(),
            self.inner.num_frames(),
            self.inner.num_episodes()
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TorchSampleMode {
    Random,
    Chunk,
    Episode,
}

impl TorchSampleMode {
    fn parse(mode: &str) -> PyResult<Self> {
        match mode {
            "random" => Ok(Self::Random),
            "chunk" => Ok(Self::Chunk),
            "episode" => Ok(Self::Episode),
            _ => Err(PyValueError::new_err(format!(
                "Invalid mode '{mode}'. Expected one of: random, chunk, episode"
            ))),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Chunk => "chunk",
            Self::Episode => "episode",
        }
    }
}

/// A PyTorch map-style dataset created by `LazyDataset.to_torch_dataset`.
///
/// Each item is a dict of torch tensors. In "random" mode it has
/// episode_index, frame_index, global_index, timestamp, action, reward (NaN
/// when absent), done, images (a dict of float32 CHW tensors in [0, 1]) and
/// state when available. In "chunk" and "episode" modes the per-frame fields
/// gain a leading time dimension and the dict also has start_frame_index,
/// valid_length and valid_mask.
#[pyclass(module = "data_core", name = "LazyTorchDataset")]
struct PyLazyTorchDataset {
    inner: RustLazyDataset,
    mode: TorchSampleMode,
    chunk_size: usize,
    chunk_index_map: Vec<(usize, usize)>, // (episode_index, start_frame_in_episode)
    episode_indices: Vec<usize>,
    episode_offsets: HashMap<usize, usize>, // episode_index -> subset-relative starting frame idx
    pad_to_length: Option<usize>,
    truncate_to_length: Option<usize>,
}

impl PyLazyTorchDataset {
    fn new_from_lazy(
        inner: RustLazyDataset,
        mode: &str,
        chunk_size: Option<usize>,
        pad_to_length: Option<usize>,
        truncate_to_length: Option<usize>,
    ) -> PyResult<Self> {
        let mode = TorchSampleMode::parse(mode)?;
        let episode_indices: Vec<usize> = inner.episode_indices().collect();
        let mut episode_offsets = HashMap::new();
        let mut cumulative = 0usize;
        for &ep_idx in &episode_indices {
            episode_offsets.insert(ep_idx, cumulative);
            let ep = inner.metadata().episode(ep_idx).ok_or_else(|| {
                PyRuntimeError::new_err(format!("Episode {} metadata not found", ep_idx))
            })?;
            cumulative += ep.length;
        }

        let chunk_size = chunk_size.unwrap_or(1);
        if mode == TorchSampleMode::Chunk && chunk_size == 0 {
            return Err(PyValueError::new_err(
                "chunk_size must be >= 1 in chunk mode",
            ));
        }

        if let Some(v) = pad_to_length {
            if v == 0 {
                return Err(PyValueError::new_err("pad_to_length must be >= 1"));
            }
        }
        if let Some(v) = truncate_to_length {
            if v == 0 {
                return Err(PyValueError::new_err("truncate_to_length must be >= 1"));
            }
        }

        let chunk_index_map = if mode == TorchSampleMode::Chunk {
            let mut index_map = Vec::new();
            for &ep_idx in &episode_indices {
                let ep = inner.metadata().episode(ep_idx).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("Episode {} metadata not found", ep_idx))
                })?;
                if ep.length >= chunk_size {
                    for start in 0..=(ep.length - chunk_size) {
                        index_map.push((ep_idx, start));
                    }
                }
            }
            if index_map.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "No valid chunks: chunk_size={} is larger than all selected episode lengths",
                    chunk_size
                )));
            }
            index_map
        } else {
            Vec::new()
        };

        Ok(Self {
            inner,
            mode,
            chunk_size,
            chunk_index_map,
            episode_indices,
            episode_offsets,
            pad_to_length,
            truncate_to_length,
        })
    }

    fn episode_batch_is_fixed(&self) -> bool {
        if self.mode != TorchSampleMode::Episode {
            return true;
        }
        matches!(
            (self.pad_to_length, self.truncate_to_length),
            (Some(pad), Some(trunc)) if pad >= trunc
        )
    }

    fn collect_chunk_frames(&self, episode_index: usize, start: usize) -> PyResult<Vec<RustFrame>> {
        let ep = self
            .inner
            .metadata()
            .episode(episode_index)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("Episode {} metadata not found", episode_index))
            })?;
        if start + self.chunk_size > ep.length {
            return Err(PyValueError::new_err(format!(
                "Chunk start {} with chunk_size {} exceeds episode length {}",
                start, self.chunk_size, ep.length
            )));
        }
        let base = *self.episode_offsets.get(&episode_index).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Missing subset offset for episode {}",
                episode_index
            ))
        })?;

        let mut frames = Vec::with_capacity(self.chunk_size);
        for i in 0..self.chunk_size {
            let frame = self.inner.get_frame(base + start + i).map_err(to_py_err)?;
            frames.push(frame);
        }
        Ok(frames)
    }

    fn collect_episode_frames(&self, episode_index: usize) -> PyResult<Vec<RustFrame>> {
        let ep = self
            .inner
            .metadata()
            .episode(episode_index)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("Episode {} metadata not found", episode_index))
            })?;
        let base = *self.episode_offsets.get(&episode_index).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Missing subset offset for episode {}",
                episode_index
            ))
        })?;

        let mut frames = Vec::with_capacity(ep.length);
        for i in 0..ep.length {
            let frame = self.inner.get_frame(base + i).map_err(to_py_err)?;
            frames.push(frame);
        }
        Ok(frames)
    }

    fn frame_to_tensor_dict(&self, py: Python<'_>, frame: &RustFrame) -> PyResult<PyObject> {
        let torch = py.import("torch")?;
        let sample = PyDict::new(py);
        sample.set_item(
            "episode_index",
            torch_tensor_i64_scalar(py, &torch, frame.episode_index as i64)?,
        )?;
        sample.set_item(
            "frame_index",
            torch_tensor_i64_scalar(py, &torch, frame.frame_index_in_episode as i64)?,
        )?;
        sample.set_item(
            "global_index",
            torch_tensor_i64_scalar(py, &torch, frame.index as i64)?,
        )?;
        sample.set_item(
            "timestamp",
            torch_tensor_f64_scalar(py, &torch, frame.timestamp)?,
        )?;
        sample.set_item(
            "action",
            torch_tensor_f32_vec(py, &torch, frame.action.clone())?,
        )?;
        if let Some(state) = &frame.state {
            sample.set_item("state", torch_tensor_f32_vec(py, &torch, state.clone())?)?;
        }
        sample.set_item(
            "reward",
            torch_tensor_f32_scalar(py, &torch, frame.reward.unwrap_or(f32::NAN))?,
        )?;
        sample.set_item("done", torch_tensor_bool_scalar(py, &torch, frame.done)?)?;

        let images = PyDict::new(py);
        for camera in self.inner.metadata().camera_names.iter() {
            if let Some(bytes) = frame.images.get(camera) {
                images.set_item(camera, decode_jpeg_to_chw_tensor(py, &torch, bytes)?)?;
            }
        }
        sample.set_item("images", images)?;
        Ok(sample.into_any().unbind())
    }

    fn frames_to_sequence_dict(
        &self,
        py: Python<'_>,
        episode_index: usize,
        start_frame_index: usize,
        mut frames: Vec<RustFrame>,
        apply_episode_controls: bool,
    ) -> PyResult<PyObject> {
        if apply_episode_controls {
            if let Some(truncate_to) = self.truncate_to_length {
                if frames.len() > truncate_to {
                    frames.truncate(truncate_to);
                }
            }
        }

        let torch = py.import("torch")?;
        let metadata = self.inner.metadata();
        let action_dim = metadata.action_dim.max(1);
        let state_dim = metadata.state_dim.or_else(|| {
            frames
                .iter()
                .find_map(|f| f.state.as_ref().map(|s| s.len()))
        });

        let valid_len = frames.len();
        let target_len = if apply_episode_controls {
            self.pad_to_length
                .map(|pad_to| pad_to.max(valid_len))
                .unwrap_or(valid_len)
        } else {
            valid_len
        };
        let pad_count = target_len.saturating_sub(valid_len);

        let mut actions: Vec<Vec<f32>> = Vec::with_capacity(target_len);
        let mut states: Option<Vec<Vec<f32>>> = state_dim.map(|_| Vec::with_capacity(target_len));
        let mut timestamps: Vec<f32> = Vec::with_capacity(target_len);
        let mut rewards: Vec<f32> = Vec::with_capacity(target_len);
        let mut dones: Vec<bool> = Vec::with_capacity(target_len);
        let mut global_indices: Vec<i64> = Vec::with_capacity(target_len);
        let mut frame_indices: Vec<i64> = Vec::with_capacity(target_len);
        let mut valid_mask: Vec<bool> = vec![true; valid_len];

        let camera_names = metadata.camera_names.clone();
        let mut image_tensors_by_camera: HashMap<String, Vec<Option<PyObject>>> = camera_names
            .iter()
            .map(|cam| (cam.clone(), Vec::with_capacity(target_len)))
            .collect();

        for frame in &frames {
            let mut action = frame.action.clone();
            if action.len() < action_dim {
                action.resize(action_dim, 0.0);
            } else if action.len() > action_dim {
                action.truncate(action_dim);
            }
            actions.push(action);

            if let Some(dim) = state_dim {
                let mut state = frame.state.clone().unwrap_or_else(|| vec![0.0; dim]);
                if state.len() < dim {
                    state.resize(dim, 0.0);
                } else if state.len() > dim {
                    state.truncate(dim);
                }
                if let Some(state_rows) = states.as_mut() {
                    state_rows.push(state);
                }
            }

            timestamps.push(frame.timestamp as f32);
            rewards.push(frame.reward.unwrap_or(f32::NAN));
            dones.push(frame.done);
            global_indices.push(frame.index as i64);
            frame_indices.push(frame.frame_index_in_episode as i64);

            for camera in &camera_names {
                let per_camera = image_tensors_by_camera
                    .get_mut(camera)
                    .expect("camera key must exist");
                if let Some(bytes) = frame.images.get(camera) {
                    per_camera.push(Some(decode_jpeg_to_chw_tensor(py, &torch, bytes)?));
                } else {
                    per_camera.push(None);
                }
            }
        }

        for _ in 0..pad_count {
            actions.push(vec![0.0; action_dim]);
            if let Some(dim) = state_dim {
                if let Some(state_rows) = states.as_mut() {
                    state_rows.push(vec![0.0; dim]);
                }
            }
            timestamps.push(0.0);
            rewards.push(0.0);
            dones.push(false);
            global_indices.push(-1);
            frame_indices.push(-1);
            valid_mask.push(false);
            for per_camera in image_tensors_by_camera.values_mut() {
                per_camera.push(None);
            }
        }

        let sample = PyDict::new(py);
        sample.set_item(
            "episode_index",
            torch_tensor_i64_scalar(py, &torch, episode_index as i64)?,
        )?;
        sample.set_item(
            "start_frame_index",
            torch_tensor_i64_scalar(py, &torch, start_frame_index as i64)?,
        )?;
        sample.set_item(
            "valid_length",
            torch_tensor_i64_scalar(py, &torch, valid_len as i64)?,
        )?;
        sample.set_item("action", torch_tensor_f32_2d(py, &torch, actions)?)?;
        if let Some(state_rows) = states {
            sample.set_item("state", torch_tensor_f32_2d(py, &torch, state_rows)?)?;
        }
        sample.set_item("timestamp", torch_tensor_f32_vec(py, &torch, timestamps)?)?;
        sample.set_item("reward", torch_tensor_f32_vec(py, &torch, rewards)?)?;
        sample.set_item("done", torch_tensor_bool_vec(py, &torch, dones)?)?;
        sample.set_item(
            "global_index",
            torch_tensor_i64_vec(py, &torch, global_indices)?,
        )?;
        sample.set_item(
            "frame_index",
            torch_tensor_i64_vec(py, &torch, frame_indices)?,
        )?;
        sample.set_item("valid_mask", torch_tensor_bool_vec(py, &torch, valid_mask)?)?;

        let images = PyDict::new(py);
        for camera in &camera_names {
            let per_camera = image_tensors_by_camera
                .get(camera)
                .expect("camera key must exist");
            let first_non_null = per_camera.iter().find_map(|v| v.as_ref());
            if let Some(first) = first_non_null {
                let mut filled: Vec<PyObject> = Vec::with_capacity(target_len);
                for maybe_tensor in per_camera {
                    if let Some(t) = maybe_tensor {
                        filled.push(t.clone_ref(py));
                    } else {
                        filled.push(
                            torch
                                .call_method1("zeros_like", (first.bind(py),))?
                                .unbind(),
                        );
                    }
                }
                let stacked = torch.call_method1("stack", (PyList::new(py, &filled)?, 0i64))?;
                images.set_item(camera, stacked)?;
            }
        }
        sample.set_item("images", images)?;
        Ok(sample.into_any().unbind())
    }

    fn normalize_image_keys_in_batch(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyList>,
    ) -> PyResult<()> {
        let mut all_camera_keys = HashSet::new();
        for item in batch.iter() {
            if let Ok(sample) = item.downcast::<PyDict>() {
                if let Some(images_any) = sample.get_item("images")? {
                    if let Ok(images) = images_any.downcast::<PyDict>() {
                        for (k, _) in images.iter() {
                            all_camera_keys.insert(k.extract::<String>()?);
                        }
                    }
                }
            }
        }

        if all_camera_keys.is_empty() {
            return Ok(());
        }

        let torch = py.import("torch")?;
        for camera in all_camera_keys {
            let mut template: Option<PyObject> = None;
            for item in batch.iter() {
                if let Ok(sample) = item.downcast::<PyDict>() {
                    if let Some(images_any) = sample.get_item("images")? {
                        if let Ok(images) = images_any.downcast::<PyDict>() {
                            if let Some(value) = images.get_item(&camera)? {
                                template = Some(value.unbind());
                                break;
                            }
                        }
                    }
                }
            }

            let Some(template) = template else {
                continue;
            };

            for item in batch.iter() {
                if let Ok(sample) = item.downcast::<PyDict>() {
                    if let Some(images_any) = sample.get_item("images")? {
                        let images = images_any.downcast::<PyDict>()?;
                        if images.get_item(&camera)?.is_none() {
                            let zeros = torch
                                .call_method1("zeros_like", (template.bind(py),))?
                                .unbind();
                            images.set_item(&camera, zeros)?;
                        }
                    } else {
                        let new_images = PyDict::new(py);
                        let zeros = torch
                            .call_method1("zeros_like", (template.bind(py),))?
                            .unbind();
                        new_images.set_item(&camera, zeros)?;
                        sample.set_item("images", &new_images)?;
                    }
                }
            }
        }

        Ok(())
    }
}

#[pymethods]
impl PyLazyTorchDataset {
    /// Sampling mode: "random", "chunk" or "episode".
    #[getter]
    fn mode(&self) -> String {
        self.mode.as_str().to_string()
    }

    /// Frames per sample in chunk mode (1 if not set).
    #[getter]
    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Number of samples: frames, valid chunk starts or episodes, depending on the mode.
    fn __len__(&self) -> usize {
        match self.mode {
            TorchSampleMode::Random => self.inner.num_frames(),
            TorchSampleMode::Chunk => self.chunk_index_map.len(),
            TorchSampleMode::Episode => self.episode_indices.len(),
        }
    }

    /// Return sample `idx` as a dict of tensors.
    ///
    /// Negative indices count from the end. Raises IndexError when `idx` is
    /// out of range.
    fn __getitem__(&self, idx: isize, py: Python<'_>) -> PyResult<PyObject> {
        let len = self.__len__();
        let idx = normalize_index(idx, len, "LazyTorchDataset")?;

        match self.mode {
            TorchSampleMode::Random => {
                let frame = self.inner.get_frame(idx).map_err(to_py_err)?;
                self.frame_to_tensor_dict(py, &frame)
            }
            TorchSampleMode::Chunk => {
                let (episode_index, start) = self.chunk_index_map[idx];
                let frames = self.collect_chunk_frames(episode_index, start)?;
                self.frames_to_sequence_dict(py, episode_index, start, frames, false)
            }
            TorchSampleMode::Episode => {
                let episode_index = self.episode_indices[idx];
                let frames = self.collect_episode_frames(episode_index)?;
                self.frames_to_sequence_dict(py, episode_index, 0, frames, true)
            }
        }
    }

    /// Collate a list of samples for torch.utils.data.DataLoader.
    ///
    /// In episode mode the list is returned unchanged unless both
    /// pad_to_length and truncate_to_length are set with
    /// pad_to_length >= truncate_to_length, because sample lengths can
    /// differ. Otherwise missing camera images are filled with zeros and
    /// torch's default_collate is applied.
    fn collate_fn(&self, py: Python<'_>, batch: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if self.mode == TorchSampleMode::Episode && !self.episode_batch_is_fixed() {
            return Ok(batch.clone().unbind());
        }
        if let Ok(list_batch) = batch.downcast::<PyList>() {
            self.normalize_image_keys_in_batch(py, list_batch)?;
        }
        let collate_mod = py.import("torch.utils.data._utils.collate")?;
        let default_collate = collate_mod.getattr("default_collate")?;
        Ok(default_collate.call1((batch,))?.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "LazyTorchDataset(mode='{}', samples={})",
            self.mode.as_str(),
            self.__len__()
        )
    }
}

/// Python module for the data-core robotics data engine.
#[pymodule]
#[pyo3(name = "_native")]
fn data_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyEpisode>()?;
    m.add_class::<PyDataset>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyParallelConfig>()?;
    m.add_class::<PyLazyDataset>()?;
    m.add_class::<PyLazyTorchDataset>()?;
    m.add_class::<PyFrame>()?;
    Ok(())
}
