// CPU-only HF safetensors -> FTW converter for the Qwen3.6-35B-A3B-NVFP4
// checkpoint. No GPU, Python or torch: the safetensors reader, the fused weight
// mapping and the FTW writer are all plain Rust.

pub mod ftw;
pub mod model;
pub mod safetensors;

pub use ftw::{Counts, FtwError, FtwMeta, FtwWriter};
pub use model::{ConvertError, ConvertOptions, ConvertReport, convert};
pub use safetensors::{SafeTensorsError, TensorInfo, TensorStore};

#[cfg(test)]
pub(crate) mod testing {
    use serde_json::{Value, json};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    pub struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn new(prefix: &str) -> Self {
            let unique = format!(
                "{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(format!("ferrum-{prefix}-{unique}"));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    pub struct SynthTensor<'a> {
        pub name: &'a str,
        pub dtype: &'a str,
        pub shape: Vec<usize>,
        pub data: Vec<u8>,
    }

    pub fn write_checkpoint(dir: &Path, tensors: &[SynthTensor<'_>]) {
        let shard = "model.safetensors";
        let mut header = serde_json::Map::new();
        let mut weight_map = serde_json::Map::new();
        let mut blob = Vec::new();
        for tensor in tensors {
            let start = blob.len();
            blob.extend_from_slice(&tensor.data);
            header.insert(
                tensor.name.to_string(),
                json!({
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [start, blob.len()],
                }),
            );
            weight_map.insert(tensor.name.to_string(), json!(shard));
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut file = Vec::new();
        file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file.extend_from_slice(&header_bytes);
        file.extend_from_slice(&blob);
        std::fs::write(dir.join(shard), file).unwrap();
        let index = json!({"metadata": {}, "weight_map": Value::Object(weight_map)});
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
    }
}
