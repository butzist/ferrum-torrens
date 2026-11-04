# The served checkpoints. Each entry is data only: a Hugging Face repo, the
# recursive NAR hash of its downloaded snapshot (see `nix/model.nix`), and the
# name the API advertises. `flake.nix` exposes one package per entry and
# `nix/module.nix` selects one by `model.variant`; because the hash is inert
# until a derivation is referenced, evaluating this table builds nothing.
#
# Refresh a hash with `scripts/fetch-hf-hash.sh <repo> <dir>`: it fetches the
# snapshot the way `nix/model.nix` does (resuming, verifying each file against
# the Hub API) and prints the `nix hash path` value to paste here.
{
  "qwen3.6-35b-a3b-nvfp4" = {
    repo = "nvidia/Qwen3.6-35B-A3B-NVFP4";
    hash = "sha256-JTcnVrnwsMBjPzBouRPPQKya4CGjHMkZJYFmdvnzg4U=";
    name = "Qwen3.6-35B-A3B-NVFP4";
  };

  "ornith-1.5-35b-a3b-nvfp4" = {
    repo = "ornith-ai/Ornith-1.5-35B-A3B-NVFP4";
    hash = "__ORNITH_HF_HASH__";
    name = "Ornith-1.5-35B-A3B-NVFP4";
  };
}
