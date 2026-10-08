# LinkedIn 岗位申请数据分析：原始数据重建与报告修正

[English](README.md) · [修正版报告 PDF](reports/corrected_report.pdf) · [可编辑报告](reports/corrected_report.md) · [已执行 Notebook](notebooks/reanalysis.ipynb) · [简历要点](docs/RESUME_BULLETS.md)

这是对 2025 年课程项目的重新分析，2026 年 10 月完成代码重构与核验。原项目使用过 R、Python 和 Notebook；本仓库将维护中的分析流程统一重建为 **Python**。新结果来自找回的原始数据，不是对历史错误数字的照抄。

## 找回了什么

数据来源为 Arsh Koneru 的 [LinkedIn Job Postings (2023–2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)，固定为 Kaggle 第 13 版。原始 ZIP 内的 `postings.csv` 完整可读：

- **123,849 条岗位记录、31 列、123,849 个唯一岗位 ID**。
- 23,320 条记录有申请数，23,319 条同时有申请数和浏览量，占原始岗位的 **18.8%**。
- 有记录的申请数全部为正数；100,529 条申请数缺失。缺失不能一律当作 0。
- 原始 CSV 的大小与 Kaggle v13 文件清单一致；本地 SHA-256 已固定。上游清单没有提供远程文件哈希，因此不把大小匹配夸大为远程密码学验证。

另一个同名工作 CSV 出现串列、重复行和时间戳精度损失，因此重建从 ZIP 中的完整原件开始。旧报告、PPT、R 文件等保留在本地归档中，没有被覆盖。公开仓库不包含完整原始数据、完整岗位描述、个人电脑路径或账号凭据。

## 重跑得到什么

所有主模型使用完全相同的五折样本外划分；类别处理、薪酬分位数、TF-IDF、SVD 和调参均在相应训练数据内完成。

| 模型 | 样本数 | MAE | RMSE | 平均 Poisson deviance |
| --- | ---: | ---: | ---: | ---: |
| Poisson GLM | 23,319 | 4.987 | 23.004 | 4.018 |
| L1 Poisson | 23,319 | 4.974 | 22.906 | 3.993 |
| XGBoost | 23,319 | **4.106** | **15.530** | **3.026** |

XGBoost 的 MAE 比本次重跑的 Poisson 模型低 **17.7%**，比训练集均值基线低 **66.6%**。这些是预测误差改善，不是实际申请量提升；也不能与旧报告的错误指标直接比较。

描述文本的 SVD 特征使 Poisson MAE 从 5.110 降为 4.987，改善 **2.40%**。这个对照仅移除描述 SVD，仍然保留标题关键词和文本长度，因此不能宣称已经衡量了“所有文本”的价值。

![五折评估结果](results/figures/model_comparison.png)

## 原报告和简历需要怎样修正

**曝光定义：** 主模型不再把 `days_open × (views + 1)` 当成已验证的曝光量，而是估计 `log(views + 1)` 的系数。数据缺少逐岗位抓取时点；到期日不是已观察的开放时长。旧公式保留为敏感性检验，其 MAE 为 6.083，高于相同样本上的主 Poisson 模型。

**申请流程：** 在控制已纳入特征后，两种站内申请流程的记录申请数相对 OffsiteApply 约为 1.32 倍。但流程之间的记录率和事件定义可能不同，不能解释成“切换流程使完成申请增加 32%”。

**全职岗位：** Contract、Part-time、Internship 和 Temporary 相对 Full-time 的区间均包含 1，不能概括为“全职岗位普遍表现更好”。

**技术关键词：** SQL、Java 标题关键词呈正向条件关联；Python 的区间包含 1。因此应明确具体关键词，并避免把关键词关联写成因果效果或已验证的技能需求。这里的 SQL 是岗位标题特征，不代表项目已使用 SQL 编程。

**评估口径：** 使用统一的样本外预测计算 MAE、RMSE 和平均 deviance，另外保存 deviance 总和。旧报告混用指标、图表版本及聚合方式的结论已撤回。详见[纠错清单](docs/CORRECTIONS.md)。

## 如何复现

推荐 Python 3.11。克隆仓库并创建、激活虚拟环境后：

```bash
python -m pip install -r requirements.lock.txt
python -m pip install -e ".[download,notebook,report,test]"
python scripts/download_data.py
python -m linkedin_analytics --data data/raw/postings.csv --config config.json --output results
python scripts/build_report.py
python scripts/build_notebook.py --execute
python -m pytest -q
```

如果已经有原始 ZIP，也可直接将 `--data` 指向该文件。下载器会校验固定文件大小和 SHA-256；具体版本、授权及 Kaggle 认证说明见[数据说明](docs/DATA_SOURCE.md)。首次运行可能需要安装依赖或按 Kaggle 正常流程登录，本仓库不保存凭据。

本次四线程 CPU 完整运行约 **10.4 分钟**。Notebook 默认审阅已保存的结果，顶到底执行不会重复耗时训练；独立重跑模型的命令已明确列出。报告生成时会重新核对当前 OOF 指标、来源指纹和配置，避免引用过期验证记录。

## 结论边界

本项目描述有记录岗位中的申请响应，浏览量也是同期记录。因此它不是发帖前预测，也不代表全体 LinkedIn 岗位。随机岗位划分可能让同一公司或相似描述跨越训练和测试集；尚未完成未来时间或新公司外推验证。公司聚类稳健区间服务于描述性关联，不能修复样本选择偏差或证明因果关系。

新代码使用 [MIT 许可](LICENSE)；数据及派生分析成果按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 保留来源署名和相应义务。详见[许可边界](docs/DATA_LICENSE.md)。
