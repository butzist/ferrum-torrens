// Weight catalog types: how tensors are named, typed and sized engine-wide.

use std::fmt;

/// Torch dtype names exactly as the FTW index spells them (``torch.bfloat16``
/// -> ``bfloat16`` after the FreeToken writer strips the ``torch.`` prefix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bf16,
    F16,
    F32,
    F64,
    F8E4M3,
    F8E4M3Fnuz,
    I64,
    I32,
    I8,
    U8,
}

impl DType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bfloat16",
            Self::F16 => "float16",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::F8E4M3 => "float8_e4m3fn",
            Self::F8E4M3Fnuz => "float8_e4m3fnuz",
            Self::I64 => "int64",
            Self::I32 => "int32",
            Self::I8 => "int8",
            Self::U8 => "uint8",
        }
    }

    pub const fn element_size(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::F8E4M3 | Self::F8E4M3Fnuz | Self::I8 | Self::U8 => 1,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::str::FromStr for DType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bfloat16" => Ok(Self::Bf16),
            "float16" => Ok(Self::F16),
            "float32" => Ok(Self::F32),
            "float64" => Ok(Self::F64),
            "float8_e4m3fn" => Ok(Self::F8E4M3),
            "float8_e4m3fnuz" => Ok(Self::F8E4M3Fnuz),
            "int64" => Ok(Self::I64),
            "int32" => Ok(Self::I32),
            "int8" => Ok(Self::I8),
            "uint8" => Ok(Self::U8),
            other => Err(format!("unknown torch dtype {other:?}")),
        }
    }
}

/// On-disk expert-bank layout family. The FreeToken index ``quant_format`` meta
/// records the actual repacked layout (e.g. ``nvfp4_marlin``, ``nvfp4_b12x``);
/// the engine maps any member of a family to its slot format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantFormat {
    Nvfp4,
    Fp8,
    Bf16,
    Other(String),
}

impl core::str::FromStr for QuantFormat {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        if lower.contains("nvfp4") {
            Ok(Self::Nvfp4)
        } else if lower.contains("fp8") {
            Ok(Self::Fp8)
        } else if lower.contains("bf16") || lower.contains("bfloat16") {
            Ok(Self::Bf16)
        } else {
            Ok(Self::Other(s.to_string()))
        }
    }
}

/// One of the two FTW tensor kinds; unknown kinds are kept so future formats
/// do not break parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorKind {
    Weight,
    ExpertsBank,
    Other(String),
}

impl core::str::FromStr for TensorKind {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "weight" => Ok(Self::Weight),
            "experts_bank" => Ok(Self::ExpertsBank),
            other => Ok(Self::Other(other.to_string())),
        }
    }
}

impl fmt::Display for TensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weight => f.write_str("weight"),
            Self::ExpertsBank => f.write_str("experts_bank"),
            Self::Other(s) => f.write_str(s),
        }
    }
}

/// Catalog entry for one tensor in a checkpoint: where it lives in the logical
/// byte region, how to type it, and what it stores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    pub name: String,
    pub kind: TensorKind,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub global_off: u64,
    pub nbytes: u64,
}

impl TensorDesc {
    /// Number of elements described by ``shape`` (1 for a 0-d scalar).
    pub fn numel(&self) -> u64 {
        self.shape
            .iter()
            .fold(1, |acc, &d| acc.saturating_mul(d as u64))
    }

    /// Element count scaled by dtype width; equals ``nbytes`` for contiguous data.
    pub fn byte_len(&self) -> u64 {
        self.numel()
            .saturating_mul(self.dtype.element_size() as u64)
    }
}

/// Domain port: copy a byte window of the checkpoint into a caller buffer.
/// ``global_off``/``nbytes`` come from ``FtwIndex::pieces`` and may span shards.
/// Adapters (mmap shards) implement this; the core never touches disk.
#[cfg_attr(test, mockall::automock)]
pub trait WeightsReader: Send + Sync {
    fn read_range(&self, global_off: u64, nbytes: u64, dst: &mut [u8]) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_dtype_names_roundtrip() {
        for dt in [
            DType::Bf16,
            DType::F16,
            DType::F32,
            DType::F64,
            DType::F8E4M3,
            DType::F8E4M3Fnuz,
            DType::I64,
            DType::I32,
            DType::I8,
            DType::U8,
        ] {
            assert_eq!(dt.to_string().parse::<DType>().unwrap(), dt);
        }
    }

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::Bf16.element_size(), 2);
        assert_eq!(DType::F32.element_size(), 4);
        assert_eq!(DType::F8E4M3.element_size(), 1);
        assert_eq!(DType::U8.element_size(), 1);
    }

    #[test]
    fn unknown_dtype_errors() {
        assert!("strange".parse::<DType>().is_err());
    }

    #[test]
    fn quant_format_maps_to_family() {
        assert_eq!(
            "moe.nvfp4=b12x".parse::<QuantFormat>().unwrap(),
            QuantFormat::Nvfp4
        );
        assert_eq!(
            "nvfp4_marlin".parse::<QuantFormat>().unwrap(),
            QuantFormat::Nvfp4
        );
        assert_eq!(
            "nvfp4_b12x".parse::<QuantFormat>().unwrap(),
            QuantFormat::Nvfp4
        );
        assert_eq!(
            "fp8_mixed_precision".parse::<QuantFormat>().unwrap(),
            QuantFormat::Fp8
        );
        assert_eq!("bf16".parse::<QuantFormat>().unwrap(), QuantFormat::Bf16);
    }

    #[test]
    fn tensor_kind_parse() {
        assert_eq!("weight".parse::<TensorKind>().unwrap(), TensorKind::Weight);
        assert_eq!(
            "experts_bank".parse::<TensorKind>().unwrap(),
            TensorKind::ExpertsBank
        );
        assert_eq!(
            "vision".parse::<TensorKind>().unwrap(),
            TensorKind::Other("vision".into())
        );
    }

    #[test]
    fn tensor_desc_byte_len_matches_nbytes() {
        let t = TensorDesc {
            name: "x".into(),
            kind: TensorKind::Weight,
            dtype: DType::Bf16,
            shape: vec![4, 8],
            global_off: 0,
            nbytes: 64,
        };
        assert_eq!(t.numel(), 32);
        assert_eq!(t.byte_len(), t.nbytes);
    }
}
