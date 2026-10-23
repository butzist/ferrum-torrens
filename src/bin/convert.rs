// Ferrum-Torrens HF -> FTW converter (CPU only): no Python, no torch, no GPU.

use ferrum_torrens::converter::{ConvertOptions, convert};
use ferrum_torrens::infrastructure::ftw::ALIGN;

const USAGE: &str = "\
usage: ferrum-torrens-convert --model <hf_dir> --out <ftw_dir> [--shard-gib <f>]
                              [--layers <N>] [--limit-tensors <names-file>]

  --model <dir>          HF safetensors snapshot (with model.safetensors.index.json)
  --out <dir>            FTW output directory (must not already hold freetoken_weight.json)
  --shard-gib <f>        shard limit in GiB (default 8; must be a multiple of 4096 bytes)
  --layers <N>           convert only the first N transformer layers (0-based count)
  --limit-tensors <file> only output names matching a glob per line ('*' wildcard)
";

fn main() {
    if let Err(e) = run() {
        eprintln!("ferrum-torrens-convert: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut model = None;
    let mut out = None;
    let mut shard_gib = 8.0f64;
    let mut layers = None;
    let mut limit_file = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(args.next().ok_or("--model needs a path")?),
            "--out" => out = Some(args.next().ok_or("--out needs a path")?),
            "--shard-gib" => {
                let value = args.next().ok_or("--shard-gib needs a number")?;
                shard_gib = value
                    .parse()
                    .map_err(|_| format!("bad --shard-gib value {value:?}"))?;
            }
            "--layers" => {
                let value = args.next().ok_or("--layers needs a count")?;
                layers = Some(
                    value
                        .parse()
                        .map_err(|_| format!("bad --layers value {value:?}"))?,
                );
            }
            "--limit-tensors" => {
                limit_file = Some(args.next().ok_or("--limit-tensors needs a path")?)
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    let model = model.ok_or("--model is required")?;
    let out = out.ok_or("--out is required")?;
    if !shard_gib.is_finite() || shard_gib <= 0.0 {
        return Err(format!("--shard-gib must be positive, got {shard_gib}"));
    }
    let shard_limit = (shard_gib * (1u64 << 30) as f64) as u64;
    if !shard_limit.is_multiple_of(ALIGN) {
        return Err(format!(
            "--shard-gib {shard_gib} is {shard_limit} bytes, not a multiple of {ALIGN}"
        ));
    }
    let limit = match limit_file {
        None => None,
        Some(path) => {
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
            Some(
                text.lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty() && !line.starts_with('#'))
                    .map(str::to_string)
                    .collect(),
            )
        }
    };

    let mut opts = ConvertOptions::new(&model, &out);
    opts.shard_limit = shard_limit;
    opts.layers = layers;
    opts.limit = limit;
    let report = convert(&opts).map_err(|e| e.to_string())?;

    eprintln!(
        "wrote {} weights + {} expert-bank tensors to {} ({} bytes, {} shards) in {:.1}s",
        report.weight_count,
        report.experts_bank_count,
        out,
        report.total_bytes,
        report.shard_count,
        report.elapsed.as_secs_f64()
    );
    if !report.copied_metadata.is_empty() {
        eprintln!("copied {} metadata files", report.copied_metadata.len());
    }
    Ok(())
}
