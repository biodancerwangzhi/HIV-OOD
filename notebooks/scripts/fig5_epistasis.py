"""Q5 探路：线性分类器是否抹杀上位性(epistasis)？ESM 是否隐式编码了它？

审稿人质疑：HIV 耐药常是突变 A+B 协同(非线性上位性)，线性分类器抓不住;
若 ESM 表示配线性分类器(我们的主方法)，是否浪费了突变间的协同信息?

实验设计(严格控制变量，簇外 OOD 协议，与 Q1/Q2/Q4 一致):
  三个线性模型**共用同一分类器**(LR C=0.01)，只变特征集，隔离"特征"这一个变量:
    A. binary          : 二元突变编码(仅主效应)         —— 线性，无交互
    B. binary+int      : 二元编码 + top-K 高频位点两两交互 —— 线性，显式交互
    C. esm_lr          : ESM mean pooling                —— 线性，隐式(表示内含?)
  再加一个非线性参照:
    D. esm_xgb         : ESM mean + XGBoost              —— 非线性

判读:
  gain_int = B − A  : 显式交互对传统编码是否有增益(上位性是否可利用)
  esm_lr vs binary+int:
    esm_lr ≳ binary+int → ESM 隐式编码了 epistasis，线性足够 ✅(强机制解释)
    binary+int > esm_lr → epistasis 是关键、ESM 没抓住，需调整结论
  esm_lr vs esm_xgb:
    esm_lr ≳ esm_xgb → 在 ESM 上叠加非线性无增益 → 印证"表示已含高阶信息"

案例研究(→图7): PR 已知耐药位点中挑一对占据均衡的突变对，把 ESM 表示 PCA 到 2D，
  看 {双野生/仅A/仅B/双突变} 四组能否分开 → ESM 表示是否"看得见"突变组合。

交互位点在每个训练 fold 内按频率选取(防泄漏)。
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
_scripts=_Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
from _common import SEED as COMMON_SEED, pin_compute_threads
pin_compute_threads()
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import xgboost as xgb

PROJ = Path(__file__).resolve().parent.parent.parent
EMB = PROJ / "work" / "emb_full"  # was work/emb (missing)
PREP = PROJ / "work" / "prepared"

GENE_OF = {"PI": "PR", "NRTI": "RT", "NNRTI": "RT"}
DRUGS = {
    "PI": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
    "NRTI": ["3TC", "ABC", "AZT", "D4T", "DDI", "TDF"],
    "NNRTI": ["EFV", "ETR", "NVP", "RPV"],
}
HXB2_PR = ("PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD"
           "QILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF")
HXB2_RT = ("PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV"
           "FAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPL"
           "DEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVI"
           "YQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWT"
           "VQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCKLLRGTKALTEVIPLTEEAE"
           "LELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGA"
           "HTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETWETWWTEYWQATWIPEWEFVNTP"
           "PLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTNRGRQKVVTLTDTTNQKTELQ")
SEED = 42
K_CLUSTERS = 5
C_LR = 0.01          # 公平搭档(Q1得出 esm 最优 LR 强度)，三个线性模型统一使用
TOP_K = 25           # 交互项使用的高频突变位点数 → K*(K-1)/2 个两两交互
SCARCE = {"RPV", "DDI", "TDF", "ETR", "D4T", "AZT"}
# PR 主要 PI 耐药位点(HXB2 蛋白酶 1-based) → 0-based 索引
PR_DRM_1BASED = [30, 32, 33, 46, 47, 48, 50, 53, 54, 73, 76, 82, 84, 88, 90]

def binary_encode(seqs, ref):
    """相对 HXB2 的二元突变编码(每位点是否突变)。"""
    L = len(ref)
    X = np.zeros((len(seqs), L), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j in range(min(len(s), L)):
            if s[j] != ref[j]:
                X[i, j] = 1.0
    return X


def pairwise_interactions(Xb, positions):
    """给定二元主效应矩阵和一组位点，构造这些位点两两 AND 交互项。"""
    cols = []
    P = len(positions)
    for a in range(P):
        ca = Xb[:, positions[a]]
        for b in range(a + 1, P):
            cols.append(ca * Xb[:, positions[b]])
    if not cols:
        return np.zeros((Xb.shape[0], 0), dtype=np.float32)
    return np.stack(cols, axis=1).astype(np.float32)


def top_mut_positions(Xb_train, k):
    """按训练集突变频率选 top-k 位点(防泄漏: 只看训练 fold)。稳定排序。"""
    freq = Xb_train.mean(axis=0)
    return np.argsort(-freq, kind="stable")[:k].tolist()


def load_gene(gene):
    seqs = json.loads((EMB / f"{gene}_seqs.json").read_text())["sequences"]
    mean = np.load(EMB / f"{gene}_mean.npy")
    return {s: i for i, s in enumerate(seqs)}, mean


def make_lr():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C_LR,
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
            solver="lbfgs",
            tol=1e-4,
        ),
    )


def make_xgb():
    return xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, random_state=SEED,
                             eval_metric="logloss", n_jobs=1, tree_method="hist")


def cluster_oof(build_train_test, make_clf, y, clu):
    """通用簇外 OOF: build_train_test(tr_mask, te_mask) -> (Xtr, Xte)。"""
    oof = np.zeros(len(y), dtype=np.float32)
    for c in np.unique(clu):
        te = clu == c; tr = ~te
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean() if tr.sum() else 0.5
            continue
        Xtr, Xte = build_train_test(tr, te)
        m = make_clf(); m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


def eval_drug(Xb, Xesm, y, clu):
    """对单药跑 4 个方法，返回 AUC 字典。"""
    def bt_binary(tr, te):
        return Xb[tr], Xb[te]

    def bt_binint(tr, te):
        pos = top_mut_positions(Xb[tr], TOP_K)
        Itr = pairwise_interactions(Xb[tr], pos)
        Ite = pairwise_interactions(Xb[te], pos)
        return np.hstack([Xb[tr], Itr]), np.hstack([Xb[te], Ite])

    def bt_esm(tr, te):
        return Xesm[tr], Xesm[te]

    return {
        "binary": roc_auc_score(y, cluster_oof(bt_binary, make_lr, y, clu)),
        "binary_int": roc_auc_score(y, cluster_oof(bt_binint, make_lr, y, clu)),
        "esm_lr": roc_auc_score(y, cluster_oof(bt_esm, make_lr, y, clu)),
        "esm_xgb": roc_auc_score(y, cluster_oof(bt_esm, make_xgb, y, clu)),
    }


def case_study_pr(cache, clu_map):
    """PR 上挑一对占据均衡的已知耐药突变对，ESM 表示 PCA 到 2D 供图7。"""
    idx, mean = cache["PR"]
    seqs = list(idx.keys())
    Xb = binary_encode(seqs, HXB2_PR)
    drm = [p - 1 for p in PR_DRM_1BASED if p - 1 < Xb.shape[1]]
    # 在 DRM 位点两两组合里，挑 min(四格计数) 最大的一对(2x2 最均衡)
    best = None
    for i in range(len(drm)):
        for j in range(i + 1, len(drm)):
            a, b = Xb[:, drm[i]], Xb[:, drm[j]]
            cells = [((a == 0) & (b == 0)).sum(), ((a == 1) & (b == 0)).sum(),
                     ((a == 0) & (b == 1)).sum(), ((a == 1) & (b == 1)).sum()]
            score = min(cells)
            if best is None or score > best[0]:
                best = (score, drm[i], drm[j], cells)
    _, pa, pb, cells = best
    a, b = Xb[:, pa], Xb[:, pb]
    grp = np.where((a == 0) & (b == 0), "WT",
          np.where((a == 1) & (b == 0), f"only_{pa+1}",
          np.where((a == 0) & (b == 1), f"only_{pb+1}", "both")))
    coords = PCA(n_components=2, random_state=SEED, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(mean[[idx[s] for s in seqs]]))
    out = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "group": grp})
    dst = PROJ / "work" / "results" / "probe_q5_casestudy_pr.csv"
    out.to_csv(dst, index=False)
    print(f"\n案例(PR): 位点对 P{pa+1} × P{pb+1}  "
          f"四格[WT,only{pa+1},only{pb+1},both]={cells}")
    print(f"  ESM 2D 坐标 + 分组 -> {dst}")
    return pa + 1, pb + 1, cells


def main():
    pin_compute_threads()
    cache = {g: load_gene(g) for g in ["PR", "RT"]}
    clu_map = {}
    for gene, ref in [("PR", HXB2_PR), ("RT", HXB2_RT)]:
        idx, _ = cache[gene]
        seqs = list(idx.keys())
        km = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10, algorithm="lloyd")
        clu_map[gene] = dict(zip(seqs, km.fit_predict(binary_encode(seqs, ref))))
    print("setup done", flush=True)

    rows = []
    for cls, drugs in DRUGS.items():
        gene = GENE_OF[cls]
        idx, mean = cache[gene]
        cl = clu_map[gene]
        ref = HXB2_PR if gene == "PR" else HXB2_RT
        df = pd.read_csv(PREP / f"{cls}.csv", dtype={"SeqID": str})
        seqs_all = df["sequence"].tolist()
        emb_row = np.array([idx.get(s, -1) for s in seqs_all])
        clu_all = np.array([cl.get(s, -1) for s in seqs_all])
        Xb_all = binary_encode(seqs_all, ref)
        for drug in drugs:
            y = pd.to_numeric(df[f"{drug}_label"], errors="coerce").values
            valid = (~np.isnan(y)) & (emb_row >= 0) & (clu_all >= 0)
            yv = y[valid].astype(int)
            if len(np.unique(yv)) < 2:
                continue
            r = emb_row[valid]; clu = clu_all[valid]
            Xb = Xb_all[valid]; Xesm = mean[r]
            res = eval_drug(Xb, Xesm, yv, clu)
            rec = {"drug": drug, "class": cls, "n": len(yv),
                   "scarce": drug in SCARCE, **res}
            rec["gain_int"] = res["binary_int"] - res["binary"]
            rec["esm_vs_int"] = res["esm_lr"] - res["binary_int"]
            rec["esm_lr_vs_xgb"] = res["esm_lr"] - res["esm_xgb"]
            rows.append(rec)
            print(f"  {cls}/{drug:4s} n={len(yv):4d} "
                  f"bin={res['binary']:.3f} bin+int={res['binary_int']:.3f}"
                  f"(Δ{rec['gain_int']:+.3f}) esm_lr={res['esm_lr']:.3f} "
                  f"esm_xgb={res['esm_xgb']:.3f}", flush=True)

    R = pd.DataFrame(rows)
    out = PROJ / "work" / "results" / "probe_q5_epistasis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(out, index=False)

    print("\n===== 分组均值 (簇外OOD) =====")
    cols = ["binary", "binary_int", "esm_lr", "esm_xgb"]
    hdr = " ".join(f"{c:>10s}" for c in cols)
    print(f"  {'组':8s} " + hdr + f" {'gain_int':>10s}")
    for grp, sub in [("稀缺药", R[R.scarce]), ("充足药", R[~R.scarce]), ("全部", R)]:
        vals = " ".join(f"{sub[c].mean():10.4f}" for c in cols)
        print(f"  {grp:8s} " + vals + f" {sub['gain_int'].mean():+10.4f}")

    print("\n  判读:")
    gi = R["gain_int"].mean()
    ev = R["esm_vs_int"].mean()
    ex = R["esm_lr_vs_xgb"].mean()
    print(f"  - 显式交互对 binary 的平均增益 gain_int = {gi:+.4f} "
          f"({'上位性可利用' if gi > 0.003 else '增益微弱/交互难在OOD泛化'})")
    print(f"  - esm_lr − binary+int = {ev:+.4f} "
          f"({'ESM≳显式交互,隐式编码epistasis,线性足够✅' if ev > -0.003 else '显式交互反超,需调整结论'})")
    print(f"  - esm_lr − esm_xgb    = {ex:+.4f} "
          f"({'ESM上叠非线性无增益,印证表示已含高阶信息' if ex > -0.003 else 'ESM上非线性仍有增益'})")

    case_study_pr(cache, clu_map)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

