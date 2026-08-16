"""Extract ESM-2 mean-pooled embeddings + zero-shot mutation scores for the Full set.

Self-contained: builds HXB2 references from work/hivdb_full/refs fasta (no repo dep),
reads reconstructed sequences from work/prepared/*.csv, deduplicates per gene, runs
ESM-2 650M (frozen, CPU) once per unique sequence, saves:

  work/emb_full/{gene}_mean.npy      (n_unique, 1280) float32
  work/emb_full/{gene}_mutscore.npy  (n_unique,) float32
  work/emb_full/{gene}_seqs.json     {"sequences": [...]}  index = row in mean.npy
  work/emb_full/meta.json

Downstream maps each prepared row to its embedding via the sequence string.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parent.parent.parent
PREP = PROJ / "work" / "prepared"
REF_DIR = PROJ / "notebooks" / "data" / "refs"
OUT_DIR = PROJ / "work" / "emb_full"
GENE_OF_CLASS = {"PI": "PR", "NRTI": "RT", "NNRTI": "RT", "INI": "IN", "CAI": "CA"}
REPR_LAYER = 33


def read_fasta_seq(path: Path) -> str:
    return "".join(l.strip() for l in path.read_text().splitlines()
                   if l and not l.startswith(">"))


def build_references() -> dict[str, str]:
    pol = read_fasta_seq(REF_DIR / "P04585.fasta")
    gag = read_fasta_seq(REF_DIR / "P04591.fasta")
    spec = {
        "PR": (pol, 489, 587, {3: "I"}),
        "RT": (pol, 588, 1187, {214: "F", 570: "E"}),
        "IN": (pol, 1148, 1435, {10: "E", 72: "I", 123: "S", 124: "T", 127: "K", 232: "D"}),
        "CA": (gag, 133, 363, {}),
    }
    refs = {}
    for g, (src, s, e, patch) in spec.items():
        seq = list(src[s - 1:e])
        for p, aa in patch.items():
            seq[p - 1] = aa
        refs[g] = "".join(seq)
    return refs


def unique_seqs_per_gene() -> dict[str, list[str]]:
    by_gene: dict[str, set] = {}
    for cls, g in GENE_OF_CLASS.items():
        p = PREP / f"{cls}.csv"
        if not p.exists():
            continue
        for s in pd.read_csv(p)["sequence"].tolist():
            if isinstance(s, str) and s:
                by_gene.setdefault(g, set()).add(s)
    return {g: sorted(s) for g, s in by_gene.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", nargs="+", default=["CA", "PR", "IN", "RT"])
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs = build_references()
    all_uniq = unique_seqs_per_gene()
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()}", flush=True)

    import esm
    print("Loading ESM-2 650M (frozen, CPU)...", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    bc = alphabet.get_batch_converter()

    meta = {}
    if (OUT_DIR / "meta.json").exists():
        meta = json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))

    for gene in args.genes:
        seqs = all_uniq.get(gene, [])
        if not seqs:
            print(f"[skip] {gene}: no sequences", flush=True); continue
        ref = refs[gene]
        n = len(seqs)
        print(f"\n=== {gene}: {n} unique seqs (ref len {len(ref)}) ===", flush=True)
        mean_emb = np.zeros((n, 1280), dtype=np.float32)
        mut = np.zeros((n,), dtype=np.float32)
        t0 = time.time()
        for start in range(0, n, args.batch):
            chunk = seqs[start:start + args.batch]
            _, _, tokens = bc([(f"s{start+j}", s) for j, s in enumerate(chunk)])
            with torch.no_grad():
                out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
            reps = out["representations"][REPR_LAYER]
            lp = torch.log_softmax(out["logits"], dim=-1)
            for j, seq in enumerate(chunk):
                L = len(seq)
                mean_emb[start + j] = reps[j, 1:L + 1, :].mean(dim=0).numpy()
                score = 0.0
                for pos in range(min(L, len(ref))):
                    wt, mt = ref[pos], seq[pos]
                    if mt == wt:
                        continue
                    score += float(lp[j, pos + 1, alphabet.get_idx(mt)]
                                   - lp[j, pos + 1, alphabet.get_idx(wt)])
                mut[start + j] = score
            done = min(start + args.batch, n)
            if done % 40 == 0 or done == n:
                el = time.time() - t0
                eta = (n - done) / (done / el) if done else 0
                print(f"  {gene}: {done}/{n} ({el:.0f}s, ETA {eta:.0f}s)", flush=True)
        np.save(OUT_DIR / f"{gene}_mean.npy", mean_emb)
        np.save(OUT_DIR / f"{gene}_mutscore.npy", mut)
        (OUT_DIR / f"{gene}_seqs.json").write_text(
            json.dumps({"sequences": seqs}, ensure_ascii=False), encoding="utf-8")
        meta[gene] = {"n_unique": n, "ref_len": len(ref), "seconds": round(time.time() - t0, 1)}
        (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  saved {gene}: {meta[gene]['seconds']}s", flush=True)
    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
