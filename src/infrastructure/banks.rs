// The `ExpertSlabSource` over the mmap'd FTW checkpoint: one expert's three
// NVFP4 fields (packed bytes, e4m3 block scales, f16 globals) copied out of the
// per-layer bank entries `"{gate_up,down}_{packed,scale,global}#L{layer:05}"`.
//
// Bank tensors are `[experts, ...]` row-major, so expert `e` is one contiguous
// window: one `read_range` per field, no whole-layer map.

use crate::domain::expert_kernel::{ExpertSlabSource, MoEError, Part, Slab};
use crate::domain::weights::{TensorDesc, TensorKind, WeightsReader};
use crate::infrastructure::ftw::FtwIndex;

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

/// Checkpoint-backed slab source. Holds no bytes itself: every call is one
/// windowed read into a fresh buffer, which the adapter then stages H2D.
pub struct FtwBanks<'a> {
    reader: &'a dyn WeightsReader,
    index: &'a FtwIndex,
}

impl<'a> FtwBanks<'a> {
    pub fn new(reader: &'a dyn WeightsReader, index: &'a FtwIndex) -> Self {
        Self { reader, index }
    }

    fn desc(&self, name: &str) -> Result<&'a TensorDesc, MoEError> {
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

impl ExpertSlabSource for FtwBanks<'_> {
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

    /// Fill the reader from an index: every read echoes its own offset, so a
    /// test can see which window was asked for.
    fn echoing(index: &FtwIndex) -> MockReader {
        let tensors = index.tensors.clone();
        let mut reader = MockReader::new();
        reader
            .expect_read_range()
            .returning(move |off: u64, nbytes: u64, dst: &mut [u8]| {
                assert_eq!(dst.len(), nbytes as usize);
                for (i, byte) in dst.iter_mut().enumerate() {
                    *byte = (off as usize + i) as u8;
                }
                // Keep the tensors alive so the closure cannot drift from them.
                let _ = &tensors;
                Ok(())
            });
        reader
    }

    #[test]
    fn reads_one_experts_three_fields_from_the_gate_up_bank() {
        let ix = index(banks(7));
        let reader = echoing(&ix);
        let src = FtwBanks::new(&reader, &ix);
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
        let reader = echoing(&ix);
        let src = FtwBanks::new(&reader, &ix);
        let slab = src.slab(7, 0, Part::Down).unwrap();
        let packed = ix.entry("down_packed#L00007").unwrap();
        assert_eq!(packed.shape, vec![EXPERTS, COLS, ROWS / 2]);
        assert_eq!(slab.packed.len(), COLS * ROWS / 2);
        assert_eq!(slab.packed[0], packed.global_off as u8);
    }

    #[test]
    fn rejects_an_expert_past_the_bank() {
        let ix = index(banks(0));
        let reader = echoing(&ix);
        let src = FtwBanks::new(&reader, &ix);
        assert!(matches!(
            src.slab(0, EXPERTS as u32, Part::GateUp),
            Err(MoEError::Bad(_))
        ));
    }

    #[test]
    fn rejects_a_missing_layer() {
        let ix = index(banks(0));
        let reader = echoing(&ix);
        let src = FtwBanks::new(&reader, &ix);
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
        let reader = echoing(&ix);
        let src = FtwBanks::new(&reader, &ix);
        assert!(matches!(
            src.slab(0, 0, Part::GateUp),
            Err(MoEError::Bad(_))
        ));
    }

    #[test]
    fn propagates_a_read_failure_with_the_entry_name() {
        let ix = index(banks(0));
        let mut reader = MockReader::new();
        reader
            .expect_read_range()
            .returning(|_, _, _| Err("short shard".to_string()));
        let src = FtwBanks::new(&reader, &ix);
        match src.slab(0, 0, Part::GateUp) {
            Err(MoEError::Read { entry: name, why }) => {
                assert_eq!(name, "gate_up_packed#L00000");
                assert_eq!(why, "short shard");
            }
            other => panic!("want a read error, got {other:?}"),
        }
    }
}
