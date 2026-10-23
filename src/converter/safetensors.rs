// Minimal CPU-only safetensors reader for the HF checkpoint snapshot: parse the
// HF index and each shard's JSON header, then positional-read one tensor's exact
// bytes. Deliberately not a full safetensors library (no mmap, no dtypes beyond
// the ones an HF text/VL quantized checkpoint stores).

use crate::domain::weights::DType;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

pub const INDEX_FILE: &str = "model.safetensors.index.json";

#[derive(Debug, thiserror::Error)]
pub enum SafeTensorsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported safetensors dtype {0:?}")]
    UnsupportedDtype(String),
    #[error("missing tensor {0:?}")]
    Missing(String),
    #[error("invalid safetensors: {0}")]
    Invalid(String),
}

/// One tensor's location inside its shard file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub shard: usize,
    pub offset: u64,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub nbytes: u64,
}

/// All tensors of one HF sharded safetensors checkpoint, indexed by their HF name.
pub struct TensorStore {
    dir: PathBuf,
    shards: Vec<String>,
    files: Vec<File>,
    tensors: HashMap<String, TensorInfo>,
}

impl TensorStore {
    pub fn open(dir: &Path) -> Result<Self, SafeTensorsError> {
        let index_doc: Value = serde_json::from_slice(&std::fs::read(dir.join(INDEX_FILE))?)?;
        let weight_map = index_doc
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| SafeTensorsError::Invalid("index has no weight_map".into()))?;

        let mut shards: Vec<String> = weight_map
            .values()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        shards.sort();
        shards.dedup();
        if shards.is_empty() {
            return Err(SafeTensorsError::Invalid("index lists no shards".into()));
        }

        let mut files = Vec::with_capacity(shards.len());
        let mut tensors = HashMap::new();
        for (shard_id, shard) in shards.iter().enumerate() {
            let file = File::open(dir.join(shard))?;
            let mut len_bytes = [0u8; 8];
            read_exact_at(&file, 0, &mut len_bytes)?;
            let header_len = u64::from_le_bytes(len_bytes);
            if header_len == 0 || header_len > 1 << 30 {
                return Err(SafeTensorsError::Invalid(format!(
                    "{shard}: implausible header length {header_len}"
                )));
            }
            let mut header_bytes = vec![0u8; header_len as usize];
            read_exact_at(&file, 8, &mut header_bytes)?;
            let header: Value = serde_json::from_slice(&header_bytes)?;
            let entries = header.as_object().ok_or_else(|| {
                SafeTensorsError::Invalid(format!("{shard}: header is not an object"))
            })?;
            let data_base = 8 + header_len;
            for (name, entry) in entries {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = dtype_from_hf(entry.get("dtype").and_then(Value::as_str).ok_or_else(
                    || SafeTensorsError::Invalid(format!("{shard}: {name} has no dtype")),
                )?)?;
                let shape = entry
                    .get("shape")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        SafeTensorsError::Invalid(format!("{shard}: {name} has no shape"))
                    })?
                    .iter()
                    .map(|d| d.as_u64().map(|u| u as usize))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        SafeTensorsError::Invalid(format!(
                            "{shard}: {name} has a non-integer shape"
                        ))
                    })?;
                let offsets = entry
                    .get("data_offsets")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        SafeTensorsError::Invalid(format!("{shard}: {name} has no data_offsets"))
                    })?;
                let start = offsets.first().and_then(Value::as_u64).ok_or_else(|| {
                    SafeTensorsError::Invalid(format!("{shard}: bad offsets for {name}"))
                })?;
                let end = offsets.get(1).and_then(Value::as_u64).ok_or_else(|| {
                    SafeTensorsError::Invalid(format!("{shard}: bad offsets for {name}"))
                })?;
                if end < start {
                    return Err(SafeTensorsError::Invalid(format!(
                        "{shard}: {name} has reversed data_offsets"
                    )));
                }
                let nbytes = end - start;
                let expected = shape
                    .iter()
                    .try_fold(dtype.element_size() as u64, |acc, &d| {
                        (acc as usize).checked_mul(d).map(|v| v as u64)
                    })
                    .ok_or_else(|| {
                        SafeTensorsError::Invalid(format!("{shard}: {name} shape overflows"))
                    })?;
                if expected != nbytes {
                    return Err(SafeTensorsError::Invalid(format!(
                        "{shard}: {name} is {nbytes} bytes but its shape/dtype imply {expected}"
                    )));
                }
                let info = TensorInfo {
                    shard: shard_id,
                    offset: data_base + start,
                    dtype,
                    shape,
                    nbytes,
                };
                if tensors.insert(name.clone(), info).is_some() {
                    return Err(SafeTensorsError::Invalid(format!(
                        "tensor {name} appears in more than one shard"
                    )));
                }
            }
            files.push(file);
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            shards,
            files,
            tensors,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn shard_files(&self) -> &[String] {
        &self.shards
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>, SafeTensorsError> {
        let info = self
            .info(name)
            .ok_or_else(|| SafeTensorsError::Missing(name.to_string()))?;
        let mut out = vec![0u8; info.nbytes as usize];
        read_exact_at(&self.files[info.shard], info.offset, &mut out)?;
        Ok(out)
    }
}

fn dtype_from_hf(s: &str) -> Result<DType, SafeTensorsError> {
    match s {
        "BF16" => Ok(DType::Bf16),
        "F16" => Ok(DType::F16),
        "F32" => Ok(DType::F32),
        "F8_E4M3" => Ok(DType::F8E4M3),
        "U8" => Ok(DType::U8),
        other => Err(SafeTensorsError::UnsupportedDtype(other.to_string())),
    }
}

fn read_exact_at(file: &File, mut offset: u64, mut dst: &mut [u8]) -> std::io::Result<()> {
    while !dst.is_empty() {
        let n = file.read_at(dst, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "safetensors shard ended early",
            ));
        }
        offset += n as u64;
        dst = &mut dst[n..];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converter::testing::{SynthTensor, TempDir, write_checkpoint};

    #[test]
    fn reads_exact_bytes_from_a_sharded_index() {
        let dir = TempDir::new("st-reader");
        let tensors = [
            SynthTensor {
                name: "a",
                dtype: "BF16",
                shape: vec![2],
                data: 0x3f80u16.to_le_bytes().repeat(2),
            },
            SynthTensor {
                name: "b",
                dtype: "F32",
                shape: vec![3],
                data: [1.0f32, 2.0, 3.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            },
            SynthTensor {
                name: "c",
                dtype: "F8_E4M3",
                shape: vec![2, 2],
                data: vec![7, 8, 9, 10],
            },
        ];
        write_checkpoint(dir.path(), &tensors);

        let store = TensorStore::open(dir.path()).unwrap();
        assert_eq!(store.shard_files(), ["model.safetensors"]);
        assert_eq!(store.info("b").unwrap().dtype, DType::F32);
        assert_eq!(store.info("b").unwrap().shape, vec![3]);
        assert_eq!(store.read("a").unwrap(), tensors[0].data);
        assert_eq!(store.read("b").unwrap(), tensors[1].data);
        assert_eq!(store.read("c").unwrap(), tensors[2].data);
        assert!(matches!(
            store.read("missing"),
            Err(SafeTensorsError::Missing(_))
        ));
    }

    #[test]
    fn rejects_an_unsupported_dtype() {
        let dir = TempDir::new("st-dtype");
        write_checkpoint(
            dir.path(),
            &[SynthTensor {
                name: "x",
                dtype: "I64",
                shape: vec![1],
                data: 0u64.to_le_bytes().to_vec(),
            }],
        );
        assert!(matches!(
            TensorStore::open(dir.path()),
            Err(SafeTensorsError::UnsupportedDtype(_))
        ));
    }
}
