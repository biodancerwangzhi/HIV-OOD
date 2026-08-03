# notebooks/scripts

计算脚本目录。绘图 notebook 只读这里（及 `work/results`）产出的 CSV。

## 快速复现

```bash
# 已有 prepared + emb_full
bash notebooks/scripts/run_all.sh --skip-heavy

# 从原始 Full 数据重跑（含 embedding 提取）
bash notebooks/scripts/run_all.sh
```

## 关键脚本

| 脚本 | 作用 | 产出 |
|---|---|---|
| `_common.py` | 共享路径 / 确定性模型 | — |
| `fig0_prepare_data.py` | Full HIVDB 重建序列+标签 | `work/prepared/*.csv` |
| `fig0_extract_embeddings.py` | ESM-2 mean pool | `work/emb_full/*` |
| `fig3_protocol_benchmark.py` | **唯一主 AUROC 表** | `probe_protocol_full_selected_k.csv` |
| `fig4_compute.py` | Fig4 全流程（预测/中间表/机制/DRM） | `results/notebooks/fig4/*` |
| `fig5_compute.py` | Fig5 全流程（消融/上位性/矩阵） | `work/results/probe_fig5.csv` 等 + `results/notebooks/fig5/*` |
| `fig2_cross_split_similarity.py` | 泄漏/相似度 | `results/notebooks/fig2/*` |
| `fig3_protocol_summary.py` | fig3 汇总 | `results/notebooks/fig3/*` |

## 依赖顺序

见上级 `notebooks/README.md`。

## Cluster OOD k

主链路固定 `CLUSTER_K = 12`（`_common.selected_k()`）。  
不再要求先跑 `figS1.ipynb` 生成 `selected_cluster_k.csv`。

