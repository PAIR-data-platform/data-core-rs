//! Minimal HuggingFace Hub downloader for LeRobot dataset repositories.

use std::fs;
use std::path::{Component, Path, PathBuf};

use data_core::{DataCoreError, Result};

const HF_API_BASE: &str = "https://huggingface.co/api/datasets";
const HF_RESOLVE_BASE: &str = "https://huggingface.co/datasets";

/// Downloads LeRobot datasets from the HuggingFace Hub into a local cache.
///
/// Each repository is stored in `{cache_dir}/{owner}__{name}` and downloaded
/// at most once: if that directory exists, it is returned without contacting
/// the Hub. Files are fetched one at a time from the `main` revision; there
/// is no authentication, resumption, or integrity check.
pub struct LeRobotDownloader {
    cache_dir: PathBuf,
    client: reqwest::Client,
    api_base: String,
    resolve_base: String,
}

impl LeRobotDownloader {
    /// Create a downloader that caches datasets under `cache_dir`.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            client: reqwest::Client::new(),
            api_base: HF_API_BASE.to_string(),
            resolve_base: HF_RESOLVE_BASE.to_string(),
        }
    }

    /// Create a downloader using the default cache directory:
    /// `$DATA_CORE_CACHE_DIR/datasets` if that variable is set, otherwise
    /// `$HOME/.cache/data-core/datasets`.
    pub fn with_default_cache() -> Self {
        let cache = dirs_cache().join("datasets");
        Self::new(cache)
    }

    /// Download a dataset from the HuggingFace Hub.
    ///
    /// `repo_id` is like `"lerobot/aloha_sim_insertion_human"`. Returns the
    /// path to the local dataset directory.
    ///
    /// Files are downloaded into a sibling `.partial` directory that is
    /// renamed into place only after every file succeeded, so an interrupted
    /// or failed download is not mistaken for a cached dataset later.
    ///
    /// # Errors
    ///
    /// Returns an error if the file listing or any file download fails, or if
    /// the repository lists a path that would escape the dataset directory.
    pub async fn download(&self, repo_id: &str) -> Result<PathBuf> {
        let local_dir = self.cache_dir.join(repo_id.replace('/', "__"));
        if local_dir.exists() {
            tracing::info!("Dataset already cached at {}", local_dir.display());
            return Ok(local_dir);
        }

        tracing::info!("Downloading dataset {} from HuggingFace Hub...", repo_id);
        let partial_dir = self
            .cache_dir
            .join(format!("{}.partial", repo_id.replace('/', "__")));
        if partial_dir.exists() {
            fs::remove_dir_all(&partial_dir)?;
        }
        fs::create_dir_all(&partial_dir)?;

        let result = async {
            // 1. Fetch the file listing from HF API
            let file_list = self.list_repo_files(repo_id).await?;

            // 2. Download each file
            for file_path in &file_list {
                self.download_file(repo_id, file_path, &partial_dir).await?;
            }
            Ok::<usize, DataCoreError>(file_list.len())
        }
        .await;

        let num_files = match result {
            Ok(n) => n,
            Err(e) => {
                let _ = fs::remove_dir_all(&partial_dir);
                return Err(e);
            }
        };
        fs::rename(&partial_dir, &local_dir)?;

        tracing::info!("Downloaded {} files to {}", num_files, local_dir.display());
        Ok(local_dir)
    }

    /// List all files in a HuggingFace dataset repository.
    async fn list_repo_files(&self, repo_id: &str) -> Result<Vec<String>> {
        let url = format!("{}/{}", self.api_base, repo_id);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| DataCoreError::Other(format!("HF API request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(DataCoreError::Other(format!(
                "HF API returned status {} for {}",
                resp.status(),
                repo_id
            )));
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| DataCoreError::Other(format!("HF API parse error: {e}")))?;

        // The siblings array contains {"rfilename": "path/to/file"}
        let files = body["siblings"]
            .as_array()
            .ok_or_else(|| DataCoreError::Other("No siblings in HF API response".into()))?
            .iter()
            .filter_map(|s| s["rfilename"].as_str().map(String::from))
            .collect();

        Ok(files)
    }

    /// Download a single file from HuggingFace Hub.
    async fn download_file(&self, repo_id: &str, file_path: &str, local_dir: &Path) -> Result<()> {
        let url = format!(
            "{}/{}/resolve/main/{}",
            self.resolve_base, repo_id, file_path
        );
        let relative = Path::new(file_path);
        if !relative
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(DataCoreError::Other(format!(
                "Refusing to download '{file_path}': not a plain relative path"
            )));
        }
        let local_path = local_dir.join(relative);

        // Create parent directories
        if let Some(parent) = local_path.parent() {
            fs::create_dir_all(parent)?;
        }

        tracing::debug!("Downloading {} -> {}", url, local_path.display());

        let resp =
            self.client.get(&url).send().await.map_err(|e| {
                DataCoreError::Other(format!("Download failed for {file_path}: {e}"))
            })?;

        if !resp.status().is_success() {
            return Err(DataCoreError::Other(format!(
                "Download returned status {} for {}",
                resp.status(),
                file_path
            )));
        }

        let bytes = resp.bytes().await.map_err(|e| {
            DataCoreError::Other(format!("Failed to read bytes for {file_path}: {e}"))
        })?;

        fs::write(&local_path, &bytes)?;
        Ok(())
    }
}

fn dirs_cache() -> PathBuf {
    if let Ok(dir) = std::env::var("DATA_CORE_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    dirs_home().join(".cache").join("data-core")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_failed_download_leaves_no_cache_entry() {
        // Regression: the cache directory was created before downloading,
        // so after a failure the next call returned the empty or partial
        // directory as if it were a cached dataset.
        let tmp = TempDir::new().unwrap();
        let mut downloader = LeRobotDownloader::new(tmp.path());
        // Port 1 on localhost refuses connections; no network is used.
        downloader.api_base = "http://127.0.0.1:1".to_string();
        downloader.resolve_base = "http://127.0.0.1:1".to_string();

        assert!(downloader.download("owner/name").await.is_err());
        assert!(!tmp.path().join("owner__name").exists());
        assert!(!tmp.path().join("owner__name.partial").exists());
        // A second attempt fails again instead of returning a cache hit.
        assert!(downloader.download("owner/name").await.is_err());
    }

    #[tokio::test]
    async fn test_rejects_paths_outside_dataset_dir() {
        let tmp = TempDir::new().unwrap();
        let mut downloader = LeRobotDownloader::new(tmp.path());
        downloader.resolve_base = "http://127.0.0.1:1".to_string();
        for bad in ["../escape.txt", "/etc/passwd", "a/../../b"] {
            let err = downloader
                .download_file("owner/name", bad, tmp.path())
                .await
                .unwrap_err();
            assert!(err.to_string().contains("Refusing"), "{bad}: {err}");
        }
    }
}
