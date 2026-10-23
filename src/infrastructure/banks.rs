// The `ExpertSlabSource` over the mmap'd FTW checkpoint: one expert's three
// NVFP4 fields (packed bytes, e4m3 block scales, f16 globals) copied out of the
// per-layer bank entries `"{gate_up,down}_{packed,scale,global}#L{layer:05}"`.
//
// Bank tensors are `[experts, ...]` row-major, so expert `e` is one contiguous
// window: one `read_range` per field, no whole-layer map.

#[cfg(any(feature = "cudarc", test))]
use crate::domain::expert_kernel::BANKS;
use crate::domain::expert_kernel::{
    Bank, ExpertShapes, ExpertSlabSource, Field, MoEError, Part, Slab,
};
#[cfg(feature = "cudarc")]
use crate::domain::expert_kernel::{PinnedBank, PinnedBankSource};
use crate::domain::weights::{TensorDesc, TensorKind, WeightsReader};
use std::sync::Arc;

use crate::infrastructure::ftw::FtwIndex;

/// The three NVFP4 fields, as the index spells them.
fn field_name(field: Field) -> &'static str {
    match field {
        Field::Packed => "packed",
        Field::Scale => "scale",
        Field::Global => "global",
    }
}

/// The two bank families, as the index spells them.
fn prefix(part: Part) -> &'static str {
    match part {
        Part::GateUp => "gate_up",
        Part::Down => "down",
    }
}

fn entry(part: Part, layer: usize, field: &str) -> String {
    format!("{}_{field}#L{layer:05}", prefix(part))
}

fn bank_entry(bank: Bank, layer: usize) -> String {
    entry(bank.part(), layer, field_name(bank.field()))
}

/// Read one layer's whole bank entry into `out`, which must be exactly
/// `experts * row_bytes` wide. The pitch comes from `shapes`, so a checkpoint
/// whose entry disagrees fails here, before any pinned memory is handed to the
/// copy kernel. Pure IO over the reader port: no CUDA, unit-testable.
pub fn read_bank_layer(
    reader: &dyn WeightsReader,
    index: &FtwIndex,
    shapes: ExpertShapes,
    experts: usize,
    layer: usize,
    bank: Bank,
    out: &mut [u8],
) -> Result<(), MoEError> {
    let name = bank_entry(bank, layer);
    let t = index
        .entry(&name)
        .filter(|t| t.kind == TensorKind::ExpertsBank)
        .ok_or_else(|| MoEError::Missing {
            entry: name.clone(),
        })?;
    let want = experts * shapes.bank_row_bytes(bank);
    if t.nbytes as usize != want || out.len() != want {
        return Err(MoEError::Bad(format!(
            "{name} is {} bytes, want {want} for {experts} experts",
            t.nbytes
        )));
    }
    reader
        .read_range(t.global_off, want as u64, out)
        .map_err(|why| MoEError::Read { entry: name, why })
}

/// Checkpoint-backed slab source. Holds no bytes itself: every call is one
/// windowed read into a fresh buffer, which the adapter then stages H2D. Owned
/// handles, so the engine can share one source with every step.
pub struct FtwBanks {
    reader: Arc<dyn WeightsReader>,
    index: Arc<FtwIndex>,
}

impl FtwBanks {
    pub fn new(reader: Arc<dyn WeightsReader>, index: Arc<FtwIndex>) -> Self {
        Self { reader, index }
    }

    fn desc(&self, name: &str) -> Result<&TensorDesc, MoEError> {
        self.index
            .entry(name)
            .filter(|t| t.kind == TensorKind::ExpertsBank)
            .ok_or_else(|| MoEError::Missing {
                entry: name.to_string(),
            })
    }

    /// `(experts, rows, cols)` from the packed entry, the shape the scales and
    /// globals are derived from (decision 23).
    fn packed_shape(&self, part: Part, layer: usize) -> Result<(usize, usize, usize), MoEError> {
        let name = entry(part, layer, "packed");
        let t = self.desc(&name)?;
        match t.shape.as_slice() {
            [experts, rows, packed_cols] => Ok((*experts, *rows, *packed_cols * 2)),
            other => Err(MoEError::Bad(format!("{name} shape {other:?}"))),
        }
    }

    /// One expert's window of one bank entry. The per-expert byte count is
    /// checked against the entry's own shape so a truncated or repacked
    /// checkpoint fails here instead of staging garbage into a slot.
    fn expert_rows(
        &self,
        part: Part,
        layer: usize,
        expert: u32,
        field: &str,
    ) -> Result<Vec<u8>, MoEError> {
        let name = entry(part, layer, field);
        let t = self.desc(&name)?;
        let experts = *t
            .shape
            .first()
            .ok_or_else(|| MoEError::Bad(format!("{name} has no expert axis")))?;
        if expert as usize >= experts {
            return Err(MoEError::Bad(format!(
                "expert {expert} past {experts} in {name}"
            )));
        }
        let per = t.nbytes as usize / experts;
        if per * experts != t.nbytes as usize {
            return Err(MoEError::Bad(format!(
                "{name} is {} bytes for {experts} experts",
                t.nbytes
            )));
        }
        let off = t.global_off + expert as u64 * per as u64;
        let mut buf = vec![0u8; per];
        self.reader
            .read_range(off, per as u64, &mut buf)
            .map_err(|why| MoEError::Read { entry: name, why })?;
        Ok(buf)
    }
}

impl ExpertSlabSource for FtwBanks {
    fn slab(&self, layer: usize, expert: u32, part: Part) -> Result<Slab, MoEError> {
        let (_experts, rows, cols) = self.packed_shape(part, layer)?;
        let packed = self.expert_rows(part, layer, expert, "packed")?;
        let scale = self.expert_rows(part, layer, expert, "scale")?;
        let global = self.expert_rows(part, layer, expert, "global")?;
        if packed.len() != rows * cols / 2
            || scale.len() != rows * cols / 16
            || global.len() != rows * 2
        {
            return Err(MoEError::Bad(format!(
                "{}#L{layer:05} expert {expert}: {} packed, {} scale, {} global bytes for {rows}x{cols}",
                prefix(part),
                packed.len(),
                scale.len(),
                global.len()
            )));
        }
        Ok(Slab {
            packed,
            scale,
            global,
        })
    }
}

/// The pinned host side of the offload cache: one page-locked, device-mapped
/// region per bank, holding every expert of every layer, filled once from the
/// FTW checkpoint. The copy kernel's `src_ptrs` are these regions' device
/// aliases, so the buffers outlive every launch and every captured graph.
#[cfg(feature = "cudarc")]
pub struct FtwPinnedBanks {
    banks: [PinnedBank; BANKS.len()],
    _buffers: Vec<crate::infrastructure::cuda_driver::PinnedBuf>,
}

#[cfg(feature = "cudarc")]
impl FtwPinnedBanks {
    /// Allocate and fill the six banks, `layers * experts` rows each. The copy
    /// kernel's `src_idx` is the flat `layer * experts + expert`.
    pub fn load(
        driver: &Arc<crate::infrastructure::cuda_driver::Driver>,
        reader: &Arc<dyn WeightsReader>,
        index: &Arc<FtwIndex>,
        shapes: ExpertShapes,
        layers: usize,
        experts: usize,
    ) -> Result<Self, MoEError> {
        let mut buffers = Vec::with_capacity(BANKS.len());
        for bank in BANKS {
            let bytes = layers * experts * shapes.bank_row_bytes(bank);
            let buf = driver
                .alloc_pinned(bytes)
                .map_err(|e| MoEError::Bad(format!("pinned {}: {e}", bank_entry(bank, 0))))?;
            buffers.push(buf);
        }
        for (buf, bank) in buffers.iter_mut().zip(BANKS) {
            let window = experts * shapes.bank_row_bytes(bank);
            for layer in 0..layers {
                let lo = layer * window;
                read_bank_layer(
                    reader.as_ref(),
                    index.as_ref(),
                    shapes,
                    experts,
                    layer,
                    bank,
                    &mut buf.as_mut_slice()[lo..lo + window],
                )?;
            }
        }
        let banks = std::array::from_fn(|i| {
            let bank = BANKS[i];
            PinnedBank {
                addr: buffers[i].ptr,
                row_bytes: shapes.bank_row_bytes(bank),
                rows: layers * experts,
            }
        });
        Ok(Self {
            banks,
            _buffers: buffers,
        })
    }
}

#[cfg(feature = "cudarc")]
impl PinnedBankSource for FtwPinnedBanks {
    fn banks(&self) -> Result<[PinnedBank; BANKS.len()], MoEError> {
        Ok(self.banks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mockall::mock! {
        Reader {}
        impl WeightsReader for Reader {
            fn read_range(&self, global_off: u64, nbytes: u64, dst: &mut [u8]) -> Result<(), String>;
        }
    }

    const EXPERTS: usize = 2;
    const ROWS: usize = 32;
    const COLS: usize = 32;

    fn desc(name: &str, shape: Vec<usize>, global_off: u64, nbytes: u64) -> TensorDesc {
        TensorDesc {
            name: name.to_string(),
            kind: TensorKind::ExpertsBank,
            dtype: crate::domain::weights::DType::U8,
            shape,
            global_off,
            nbytes,
        }
    }

    /// One layer of banks: three fields, two experts, contiguous offsets.
    fn banks(layer: usize) -> Vec<TensorDesc> {
        let mut out = Vec::new();
        let mut off = 1024u64;
        for (prefix, rows, cols) in [("gate_up", ROWS, COLS), ("down", COLS, ROWS)] {
            for (field, unit) in [("packed", cols / 2), ("scale", cols / 16), ("global", 2)] {
                let nbytes = (EXPERTS * rows * unit) as u64;
                out.push(desc(
                    &format!("{prefix}_{field}#L{layer:05}"),
                    vec![EXPERTS, rows, unit],
                    off,
                    nbytes,
                ));
                off += nbytes;
            }
        }
        out
    }

    fn index(tensors: Vec<TensorDesc>) -> FtwIndex {
        FtwIndex {
            shard_limit: 1 << 30,
            total_bytes: 1 << 30,
            tensors,
            shards: Vec::new(),
            quant_format: None,
            expert_bank_num_layers: Some(1),
            meta: Default::default(),
        }
    }

    /// A source whose reader echoes each read's own offset, so a test can see
    /// exactly which window was asked for.
    fn echoing(ix: &FtwIndex) -> FtwBanks {
        let mut reader = MockReader::new();
        reader
            .expect_read_range()
            .returning(|off: u64, nbytes: u64, dst: &mut [u8]| {
                assert_eq!(dst.len(), nbytes as usize);
                for (i, byte) in dst.iter_mut().enumerate() {
                    *byte = (off as usize + i) as u8;
                }
                Ok(())
            });
        FtwBanks::new(Arc::new(reader), Arc::new(ix.clone()))
    }

    /// The same, but every read fails.
    fn failing(ix: &FtwIndex) -> FtwBanks {
        let mut reader = MockReader::new();
        reader
            .expect_read_range()
            .returning(|_, _, _| Err("short shard".to_string()));
        FtwBanks::new(Arc::new(reader), Arc::new(ix.clone()))
    }

    #[test]
    fn reads_one_experts_three_fields_from_the_gate_up_bank() {
        let ix = index(banks(7));
        let src = echoing(&ix);
        let slab = src.slab(7, 1, Part::GateUp).unwrap();
        assert_eq!(slab.packed.len(), ROWS * COLS / 2);
        assert_eq!(slab.scale.len(), ROWS * COLS / 16);
        assert_eq!(slab.global.len(), ROWS * 2);
        // The second expert's packed window starts one expert into the entry.
        let packed = ix.entry("gate_up_packed#L00007").unwrap();
        let second = packed.global_off + (ROWS * COLS / 2) as u64;
        assert_eq!(slab.packed[0], second as u8);
    }

    #[test]
    fn down_reads_the_down_bank() {
        let ix = index(banks(7));
        let src = echoing(&ix);
        let slab = src.slab(7, 0, Part::Down).unwrap();
        let packed = ix.entry("down_packed#L00007").unwrap();
        assert_eq!(packed.shape, vec![EXPERTS, COLS, ROWS / 2]);
        assert_eq!(slab.packed.len(), COLS * ROWS / 2);
        assert_eq!(slab.packed[0], packed.global_off as u8);
    }

    #[test]
    fn rejects_an_expert_past_the_bank() {
        let ix = index(banks(0));
        let src = echoing(&ix);
        assert!(matches!(
            src.slab(0, EXPERTS as u32, Part::GateUp),
            Err(MoEError::Bad(_))
        ));
    }

    #[test]
    fn rejects_a_missing_layer() {
        let ix = index(banks(0));
        let src = echoing(&ix);
        assert!(matches!(
            src.slab(9, 0, Part::GateUp),
            Err(MoEError::Missing { .. })
        ));
    }

    #[test]
    fn rejects_a_scale_that_does_not_match_the_packed_shape() {
        let mut ix = index(banks(0));
        let scale = ix
            .tensors
            .iter_mut()
            .find(|t| t.name == "gate_up_scale#L00000")
            .unwrap();
        // One scale short per expert: the per-expert split no longer adds up.
        scale.nbytes -= (COLS / 16) as u64;
        let src = echoing(&ix);
        assert!(matches!(
            src.slab(0, 0, Part::GateUp),
            Err(MoEError::Bad(_))
        ));
    }

    #[test]
    fn propagates_a_read_failure_with_the_entry_name() {
        let ix = index(banks(0));
        let src = failing(&ix);
        match src.slab(0, 0, Part::GateUp) {
            Err(MoEError::Read { entry: name, why }) => {
                assert_eq!(name, "gate_up_packed#L00000");
                assert_eq!(why, "short shard");
            }
            other => panic!("want a read error, got {other:?}"),
        }
    }

    /// A whole-bank index whose entry sizes come straight from `shapes`, so
    /// `read_bank_layer`'s pitch check is exercised rather than bypassed.
    fn full_index(layers: usize, experts: usize, shapes: ExpertShapes) -> FtwIndex {
        let mut tensors = Vec::new();
        let mut off = 1024u64;
        for layer in 0..layers {
            for bank in BANKS {
                let nbytes = (experts * shapes.bank_row_bytes(bank)) as u64;
                tensors.push(TensorDesc {
                    name: bank_entry(bank, layer),
                    kind: TensorKind::ExpertsBank,
                    dtype: crate::domain::weights::DType::U8,
                    shape: vec![experts],
                    global_off: off,
                    nbytes,
                });
                off += nbytes;
            }
        }
        FtwIndex {
            shard_limit: 1 << 30,
            total_bytes: 1 << 30,
            tensors,
            shards: Vec::new(),
            quant_format: None,
            expert_bank_num_layers: Some(layers as u64),
            meta: Default::default(),
        }
    }

    #[test]
    fn read_bank_layer_reads_the_whole_entry_into_the_window() {
        let shapes = ExpertShapes::new(32, 32).unwrap();
        let ix = full_index(2, EXPERTS, shapes);
        let mut reader = MockReader::new();
        reader
            .expect_read_range()
            .returning(|off: u64, nbytes: u64, dst: &mut [u8]| {
                assert_eq!(dst.len(), nbytes as usize);
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = (off as usize + i) as u8;
                }
                Ok(())
            });
        let mut out = vec![0u8; EXPERTS * shapes.bank_row_bytes(Bank::GateUpPacked)];
        read_bank_layer(
            &reader,
            &ix,
            shapes,
            EXPERTS,
            1,
            Bank::GateUpPacked,
            &mut out,
        )
        .unwrap();
        let t = ix.entry("gate_up_packed#L00001").unwrap();
        assert_eq!(out.len() as u64, t.nbytes);
        assert_eq!(out[0], t.global_off as u8);
    }

    #[test]
    fn read_bank_layer_rejects_a_truncated_entry() {
        let shapes = ExpertShapes::new(32, 32).unwrap();
        let mut ix = full_index(1, EXPERTS, shapes);
        ix.tensors
            .iter_mut()
            .find(|t| t.name == "down_scale#L00000")
            .unwrap()
            .nbytes -= 1;
        let mut reader = MockReader::new();
        reader.expect_read_range().returning(|_, _, _| Ok(()));
        let mut out = vec![0u8; EXPERTS * shapes.bank_row_bytes(Bank::DownScale)];
        let err = read_bank_layer(&reader, &ix, shapes, EXPERTS, 0, Bank::DownScale, &mut out)
            .unwrap_err();
        assert!(matches!(err, MoEError::Bad(_)), "{err}");
    }

    #[test]
    fn read_bank_layer_reports_a_missing_layer() {
        let shapes = ExpertShapes::new(32, 32).unwrap();
        let ix = full_index(1, EXPERTS, shapes);
        let reader = MockReader::new();
        let mut out = vec![0u8; EXPERTS * shapes.bank_row_bytes(Bank::DownGlobal)];
        let err = read_bank_layer(&reader, &ix, shapes, EXPERTS, 9, Bank::DownGlobal, &mut out)
            .unwrap_err();
        assert!(matches!(err, MoEError::Missing { .. }), "{err}");
    }

    /// The whole pinned path on a device: allocate six banks, fill every layer
    /// from the index, and hand back six distinct non-zero device aliases with
    /// the pitch the copy kernel expects. Skips (loudly) without a device.
    #[cfg(feature = "cudarc")]
    #[test]
    fn pinned_banks_load_fills_every_bank_from_the_index() {
        use crate::infrastructure::cuda_driver::Driver;

        let Ok(driver) = Driver::open() else {
            eprintln!("no libcuda on this box, skipping");
            return;
        };
        if driver.init().is_err() || driver.device_count().unwrap_or(0) == 0 {
            eprintln!("no CUDA device, skipping");
            return;
        }
        driver.attach_primary_context(0).expect("primary context");
        let driver = Arc::new(driver);

        let shapes = ExpertShapes::new(32, 32).unwrap();
        let experts = 2;
        let layers = 2;
        let ix = Arc::new(full_index(layers, experts, shapes));
        let mut reader = MockReader::new();
        reader
            .expect_read_range()
            .times(layers * BANKS.len())
            .returning(|_, _, dst: &mut [u8]| {
                dst.fill(0xa5);
                Ok(())
            });
        let reader: Arc<dyn WeightsReader> = Arc::new(reader);

        let pinned = FtwPinnedBanks::load(&driver, &reader, &ix, shapes, layers, experts).unwrap();
        let banks = pinned.banks().unwrap();
        for (i, bank) in banks.iter().enumerate() {
            assert_eq!(bank.row_bytes, shapes.bank_row_bytes(BANKS[i]));
            assert_eq!(bank.rows, layers * experts);
            assert_ne!(bank.addr, 0, "bank {i} has no device alias");
            for other in &banks[i + 1..] {
                assert_ne!(bank.addr, other.addr, "banks {i} share an alias");
            }
        }
    }
}
