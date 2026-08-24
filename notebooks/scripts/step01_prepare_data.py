"""Prepare HIVDB Full genotype-phenotype tables for all downstream scripts.

Canonical raw inputs live under notebooks/data/hivdb_full/:
  {PI,NRTI,NNRTI,INSTI,CAI}_DataSet.Full.txt
notebooks/data/refs/P04585.fasta
  notebooks/data/refs/P04591.fasta

Optional download from Stanford HIVDB (tables may update over time):
  mkdir -p notebooks/data/hivdb_full
  cd notebooks/data/hivdb_full
  for f in PI NRTI NNRTI INSTI CAI; do
    curl -L -O "https://hivdb.stanford.edu/download/GenoPhenoDatasets/${f}_DataSet.Full.txt"
  done

Outputs:
  work/prepared/{PI,NRTI,NNRTI,INSTI,CAI}.csv
  work/prepared/summary.json
  results/notebooks/00_data/*.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJ / "notebooks" / "data" / "hivdb_full"
REF_DIR = PROJ / "notebooks" / "data" / "refs"
PREP_DIR = PROJ / "work" / "prepared"
OUT_DIR = PROJ / "results" / "notebooks" / "00_data"

PAPER_DRUGS = {
    "PI": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
    "NRTI": ["3TC", "ABC", "AZT", "D4T", "DDI", "TDF"],
    "NNRTI": ["EFV", "ETR", "NVP", "RPV"],
    "INSTI": ["RAL", "EVG", "DTG", "BIC", "CAB"],
    "CAI": ["LEN"],
}
GENE_OF_CLASS = {"PI": "PR", "NRTI": "RT", "NNRTI": "RT", "INSTI": "IN", "CAI": "CA"}
CLASS_ORDER = ["PI", "NRTI", "NNRTI", "INSTI", "CAI"]
FC_RESISTANT = 3.0
GAP_CHARS = {"-", ".", ""}
DEL_CHARS = {"~"}
INS_CHARS = {"#"}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta_seq(path: Path) -> str:
    return "".join(line.strip() for line in path.read_text().splitlines() if line and not line.startswith(">"))


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
    for gene, (src, start, end, patch) in spec.items():
        seq = list(src[start - 1 : end])
        for pos, aa in patch.items():
            seq[pos - 1] = aa
        refs[gene] = "".join(seq)
    return refs


def get_position_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def reconstruct_one(row: pd.Series, position_cols: list[str], reference: str) -> str:
    residues = []
    for i, col in enumerate(position_cols):
        aa = row[col]
        ref_aa = reference[i] if i < len(reference) else "X"
        if pd.isna(aa):
            residues.append(ref_aa)
            continue
        aa = str(aa).strip()
        if aa in GAP_CHARS:
            residues.append(ref_aa)
        elif aa in DEL_CHARS or aa in INS_CHARS:
            continue
        elif aa and aa[0].isalpha():
            residues.append(aa[0].upper())
        else:
            residues.append(ref_aa)
    return "".join(residues)


def prepare_class(drug_class: str, refs: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    path = RAW_DIR / f"{drug_class}_DataSet.Full.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\n"
            "Place Full tables under notebooks/data/hivdb_full/ (see module docstring)."
        )
    df = pd.read_csv(path, sep="\t", dtype=str)
    gene = GENE_OF_CLASS[drug_class]
    pos_cols = get_position_columns(df)
    if not pos_cols:
        raise ValueError(f"{drug_class}: no P1..Pn columns")
    ref = refs[gene]

    out = pd.DataFrame({"SeqID": df["SeqID"].values})
    for meta in ["PtID", "Subtype", "SeqType"]:
        out[meta] = df[meta].values if meta in df.columns else pd.NA
    out["gene"] = gene
    out["sequence"] = [reconstruct_one(row, pos_cols, ref) for _, row in df.iterrows()]
    out["seq_len"] = out["sequence"].str.len()

    label_stats = {}
    for drug in PAPER_DRUGS[drug_class]:
        if drug not in df.columns:
            print(f"  [warn] {drug} not in {drug_class}; skip")
            continue
        fc = pd.to_numeric(df[drug], errors="coerce")
        label = fc.apply(lambda v: np.nan if pd.isna(v) else float(v >= FC_RESISTANT))
        out[f"{drug}_label"] = label.values
        n_valid = int(label.notna().sum())
        n_res = int((label == 1).sum())
        label_stats[drug] = {
            "n_valid": n_valid,
            "n_resistant": n_res,
            "n_susceptible": n_valid - n_res,
            "prevalence": round(n_res / n_valid, 4) if n_valid else None,
        }

    n_nonB = int((out["Subtype"].notna() & (out["Subtype"] != "B") & (out["Subtype"] != "Unknown") & (out["Subtype"] != "U")).sum())
    stats = {
        "drug_class": drug_class,
        "gene": gene,
        "n_records": int(len(out)),
        "n_isolates": int(out["SeqID"].nunique()),
        "n_patients": int(out["PtID"].nunique()) if out["PtID"].notna().any() else None,
        "n_unique_sequences": int(out["sequence"].nunique()),
        "n_nonB": n_nonB,
        "n_subtypes": int(out["Subtype"].nunique()),
        "seq_len_mode": int(out["seq_len"].mode().iloc[0]),
        "drugs": label_stats,
    }
    return out, stats


def collapse_subtype(value) -> str:
    if pd.isna(value) or value in ("Unknown", "U", ""):
        return "Unknown"
    return "B" if value == "B" else "non-B"


def write_fig1_tables(frames: dict[str, pd.DataFrame], all_stats: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cal_rows, qc_rows, drug_rows, comp_rows, detail_rows = [], [], [], [], []
    for cls in CLASS_ORDER:
        df = frames[cls]
        dedup = df.drop_duplicates(subset=["SeqID"])
        seqs = dedup["sequence"].astype(str)
        lengths = seqs.str.len()
        nonstd = seqs.apply(lambda s: sum(ch not in STANDARD_AA for ch in s))
        cal_rows.append(
            {
                "drug_class": cls,
                "records": len(df),
                "patients": int(dedup["PtID"].nunique()) if dedup["PtID"].notna().any() else None,
                "isolates": int(dedup["SeqID"].nunique()),
                "unique_sequences": int(seqs.nunique()),
                "records_per_unique_seq": round(len(df) / max(seqs.nunique(), 1), 3),
            }
        )
        qc_rows.append(
            {
                "drug_class": cls,
                "gene": GENE_OF_CLASS[cls],
                "len_mode": int(lengths.mode().iloc[0]),
                "n_len_outliers": int((lengths != lengths.mode().iloc[0]).sum()),
                "n_seq_nonstd_aa": int((nonstd > 0).sum()),
                "pct_seq_nonstd_aa": round(100 * (nonstd > 0).mean(), 2),
                "n_exact_duplicate_seq": int(seqs.duplicated().sum()),
            }
        )
        for drug, stats in all_stats[cls]["drugs"].items():
            drug_rows.append({"drug_class": cls, "drug": drug, "gene": GENE_OF_CLASS[cls], **stats})
        st = dedup["Subtype"]
        grp = st.map(collapse_subtype).value_counts()
        b, nb, unk = int(grp.get("B", 0)), int(grp.get("non-B", 0)), int(grp.get("Unknown", 0))
        tot = b + nb + unk
        comp_rows.append(
            {
                "drug_class": cls,
                "B": b,
                "non_B": nb,
                "Unknown": unk,
                "total": tot,
                "non_B_pct": round(100 * nb / tot, 1) if tot else 0,
            }
        )
        for sub, n in st[st.map(collapse_subtype) == "non-B"].value_counts().items():
            detail_rows.append({"drug_class": cls, "subtype": sub, "n": int(n)})

    def add_all(dfx: pd.DataFrame, sum_cols: list[str]) -> pd.DataFrame:
        row = {c: (dfx[c].sum() if c in sum_cols else "ALL") for c in dfx.columns}
        return pd.concat([dfx, pd.DataFrame([row])], ignore_index=True)

    caliber = add_all(pd.DataFrame(cal_rows), ["records", "patients", "isolates", "unique_sequences"])
    caliber.loc[caliber.index[-1], "records_per_unique_seq"] = round(
        caliber["records"][:-1].sum() / max(caliber["unique_sequences"][:-1].sum(), 1), 3
    )
    comp = add_all(pd.DataFrame(comp_rows), ["B", "non_B", "Unknown", "total"])
    comp.loc[comp.index[-1], "non_B_pct"] = round(
        100 * comp["non_B"][:-1].sum() / max(comp["total"][:-1].sum(), 1), 1
    )
    caliber.to_csv(OUT_DIR / "sample_caliber.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(OUT_DIR / "sequence_qc.csv", index=False)
    pd.DataFrame(drug_rows).to_csv(OUT_DIR / "data_summary.csv", index=False)
    comp.to_csv(OUT_DIR / "subtype_composition.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(OUT_DIR / "subtype_detail.csv", index=False)

    # global baseline used by fig1
    gb = pd.DataFrame(
        [
            ("C", 50.4, 50.2, 50.7),
            ("A", 12.4, 12.2, 12.6),
            ("B", 11.3, 11.1, 11.5),
            ("G", 2.9, 2.9, 3.0),
            ("D", 2.6, 2.5, 2.7),
            ("F", 0.9, 0.8, 0.9),
            ("CRFs", 15.1, 14.9, 15.3),
            ("URFs", 2.0, 1.9, 2.1),
        ],
        columns=["subtype", "global_pct", "ci_low", "ci_high"],
    )
    gb["source"] = "Hemelaar_LancetMicrobe_2024"
    gb["window"] = "2016-2021"
    gb.to_csv(OUT_DIR / "global_subtype_baseline.csv", index=False)
    all_comp = comp[comp["drug_class"] == "ALL"].iloc[0]
    train_b = 100 * all_comp["B"] / max(all_comp["B"] + all_comp["non_B"], 1)
    cmp2 = pd.DataFrame(
        {
            "group": ["B", "non-B"],
            "hivdb_train_pct": [round(train_b, 1), round(100 - train_b, 1)],
            "global_2016_21_pct": [11.3, round(100 - 11.3, 1)],
        }
    )
    cmp2.to_csv(OUT_DIR / "train_vs_global_subtype.csv", index=False)
    print(f"saved fig1 tables → {OUT_DIR}")


def main() -> None:
    if not (REF_DIR / "P04585.fasta").exists() or not (REF_DIR / "P04591.fasta").exists():
        raise FileNotFoundError(f"Missing reference fasta under {REF_DIR}")
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    refs = build_references()
    all_stats: dict = {}
    frames: dict[str, pd.DataFrame] = {}
    for drug_class in CLASS_ORDER:
        print(f"=== {drug_class} ===")
        table, stats = prepare_class(drug_class, refs)
        frames[drug_class] = table
        all_stats[drug_class] = stats
        table.to_csv(PREP_DIR / f"{drug_class}.csv", index=False)
        print(
            f"  records={stats['n_records']} isolates={stats['n_isolates']} "
            f"patients={stats['n_patients']} nonB={stats['n_nonB']}"
        )
        for drug, ds in stats["drugs"].items():
            print(f"    {drug}: n={ds['n_valid']} R={ds['n_resistant']} ({ds['prevalence']})")
    (PREP_DIR / "summary.json").write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_fig1_tables(frames, all_stats)
    print(f"saved prepared tables → {PREP_DIR}")


if __name__ == "__main__":
    main()
