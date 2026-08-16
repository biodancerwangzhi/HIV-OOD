# Notebooks 运行说明

`fig1.ipynb`–`fig5.ipynb` / `figS1.ipynb` 是**绘图 notebook**（读 CSV → 出图）。
全部计算逻辑在 `scripts/`。

---

## 目录结构

```
notebooks/
├── 00_data_preparation.ipynb   # 交互式数据准备（与 step01_prepare_data.py 同口径）
├── fig1.ipynb ~ fig5.ipynb     # 主图绘图
├── figS1.ipynb                 # 补充图（cluster k 可靠性）
├── scripts/
│   ├── _common.py                      # 共享路径 / 模型（确定性 LGB）
│   ├── step01_prepare_data.py            # Full HIVDB → work/prepared + 00_data 统计
│   ├── step02_extract_embeddings.py      # ESM-2 mean embeddings → work/emb_full
│   ├── step03_protocol_benchmark.py      # ★ 唯一主结果表（3方法×4协议）
│   ├── step04_fig4_compute.py                 # Fig4 全流程（预测/中间表/机制/DRM）
│   ├── step05_fig5_compute.py                 # Fig5 全流程（消融/上位性/矩阵+Brier）
│   ├── step06_cross_split_similarity.py  # 跨 split Hamming 相似度
│   ├── step07_protocol_summary.py        # fig3 中间统计表
│   └── run_all.sh                      # 一键按依赖顺序运行
└── CURRENT_STATE.md
```

---

## 正确依赖顺序（唯一真相源）

```
step01_prepare_data.py
        │  work/prepared/*.csv
        │  results/notebooks/00_data/*
        ▼
step02_extract_embeddings.py          （可缓存跳过）
        │  work/emb_full/*
        ▼
        │  CLUSTER_K=12（_common.py 写死；figS1 可选）
        ▼
step03_protocol_benchmark.py          ★ 唯一主 AUROC 表
        │  work/results/probe_protocol_full_selected_k.csv
        │  (+ 同步 probe_ood_full_selected_k.csv)
        ├─► step04_fig4_compute.py
        ├─► step05_fig5_compute.py
        ├─► step06_cross_split_similarity.py
        └─► step07_protocol_summary.py
                ▼
        fig1–fig5 / figS1 绘图 notebook
```

**不要**再维护两套互相独立的 `esm_lgb` 数字。  
`probe_ood_full_selected_k.csv` 只是 protocol 的兼容投影。

---

## 一键运行

```bash
cd <project_root>

# 全量（含数据准备 + embedding + fig2 Hamming）
bash notebooks/scripts/run_all.sh

# 已有 work/prepared 与 work/emb_full 时（推荐日常复现）
bash notebooks/scripts/run_all.sh --skip-heavy
```

Python 默认使用 `venv/bin/python`（若存在）。

---

## 分步运行

### Step 0 — 数据与 embedding（一次性）

```bash
python notebooks/scripts/step01_prepare_data.py
python notebooks/scripts/step02_extract_embeddings.py   # 需较长时间 / 建议有 GPU
```

数据源：`work/hivdb_full/{CLASS}_DataSet.Full.txt` + `refs/*.fasta`  
（**不是** `repo/data/raw` 过滤集）

### Step 1 — Cluster OOD k（已写死）

Cluster OOD 使用固定 **`CLUSTER_K = 12`**（`notebooks/scripts/_common.py` → `selected_k()`）。  
不再依赖运行时生成 `selected_cluster_k.csv`。  
`figS1.ipynb` 仍可做 k 敏感性/可靠性补充图，产出仅作说明，不参与主链路。

### Step 2 — 主协议评测（核心）

```bash
python notebooks/scripts/step03_protocol_benchmark.py
```

### Step 3 — Fig4

```bash
python notebooks/scripts/step04_fig4_compute.py
```

内部顺序：subtype OOD 预测 → scarcity/FC/barrier 中间表 → mechanism 汇总 → DRM burden。

> 静态结构资产 `fig4_ram_to_ligand_3d_distance.csv`（PDB 距离）需已存在，不由 ML 脚本重算。

### Step 4 — Fig5

```bash
python notebooks/scripts/step05_fig5_compute.py
```

内部顺序：feature ablation → epistasis → benchmark matrix + Brier。

### Step 5 — Fig2 / Fig3 汇总

```bash
python notebooks/scripts/step06_cross_split_similarity.py
python notebooks/scripts/step07_protocol_summary.py
```

### Step 6 — 绘图

```bash
for nb in notebooks/fig1.ipynb notebooks/fig2.ipynb notebooks/fig3.ipynb \
          notebooks/fig4.ipynb notebooks/fig5.ipynb notebooks/figS1.ipynb; do
  venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

---

## 可选 figS1 扫描（补充图）

主结果不依赖此步。若要重画 Fig S1 / 刷新 `cluster_validation/*` 敏感性表：

```bash
venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/figS1.ipynb
```

---


---

## 复现进度

串行复现检查（备份旧产物 → 重跑 → 字节/数值比对；通过后标记，无需再测）。

| 脚本 | 状态 | 备注 |
|---|---|---|
| `step01_prepare_data.py` | ✅ 已复现 | 输出与备份字节一致 |
| `step02_extract_embeddings.py` | ✅ 已复现 | CA 全量重提 npy/seqs 字节一致；全基因 seq 与 prepared 对齐；meta.seconds 为运行时字段可不同 |
| `step03_protocol_benchmark.py` | ✅ 已接受 baseline | 2026-07-27 04:56 `probe_protocol_full_selected_k.csv`（+08:37 ood 投影）；binary/esm_lr 跨跑锁死；esm_lgb 当前环境可复现；已删除 7/19 `.previous.csv` |
| `step04_fig4_compute.py` | ✅ 已复现 | 2026-07-27 23:41 baseline；二次重跑 00:01 15/15 字节一致（SHA256 全匹配） |
| `step05_fig5_compute.py` | ✅ 已复现 | 2026-07-28 strict baseline `_repro_backup_fig5_strict_20260728_071328`；Full prepared 口径；thread-pin + LR/LGB/XGB/PCA 确定性修复后 run1/run2 6/6 SHA256 全匹配 |
| `step06_cross_split_similarity.py` | ✅ 已复现 | 2026-07-28 `_repro_backup_fig2_20260728_092411`；CSV×2 与旧备份字节一致；run1/2/3 全文件含 npz 字节锁死；仅旧 npz 的 PI_cluster 明细与当前不同（汇总 4 位小数仍同） |
| `step07_protocol_summary.py` | ✅ 已复现 | 2026-07-28 `_repro_backup_fig3_summary_20260728_095443`；6/6 与旧备份字节一致；run1/run2 全匹配 |

## 复现性约定

1. **随机种子**统一 `SEED=42`（`_common.py`）。
2. **LightGBM** 统一：`n_jobs=1, deterministic=True, force_col_wise=True` + bagging/feature seeds。
3. **Cluster OOD k** 固定 `CLUSTER_K=12`（`_common.selected_k()`），不在运行时重选。
4. **主结果只认** `work/results/probe_protocol_full_selected_k.csv`。
5. `fig2` 会把旧 CSV 备份为 `*.prev.csv` 再对比数值差。
6. 标签阈值：fold-change ≥ 3.0 → resistant。

---

## 已知非脚本再生资产

| 文件 | 说明 |
|---|---|
| `work/emb_full/*` | ESM-2 前向；可脚本重提但耗时 |
| `results/notebooks/fig4/fig4_ram_to_ligand_3d_distance.csv` | PDB 结构距离，静态资产 |
| `results/notebooks/fig4/pdb/*.pdb` | 结构文件 |
| `results/notebooks/cluster_validation/selected_cluster_k.csv` | 可选历史/S1 产物；主链路已改读 `CLUSTER_K=12` |

### 参考序列
参考序列已移到 `notebooks/data/refs/P04585.fasta` 和 `P04591.fasta`（体积很小，2.2KB），所以上传到 GitHub 后仍能复现。
