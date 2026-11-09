# 合同条款智能问答 Agent

一个能指出答案出处的合同审查助手。核心不是「能答」，而是「答得可被核对」：
每条回答都必须给出它出自哪份文件、哪一章、哪一条、第几页。

## 当前状态

**D3：检索与最小问答闭环完成。** 569 条条款已建 BM25 + 向量双路索引，命令行可提问并返回
带条款号与页码的引用。冒烟结果见 `docs/d3-smoke.md`，解析验收见 `docs/parse-acceptance.md`。

进度：D1 骨架 ✓ · D2 解析与切分 ✓ · D3 索引与 CLI 问答 ✓

## MVP

- PDF 入库与条款级切分：已完成
- BM25 + 向量混合检索、RRF 融合：已完成
- 单文档问答与强制条款引用：已完成
- 重排（rerank）与引用校验、最多 2 轮回退：进行中
- 60 条离线评测集与指标报告
- Streamlit 溯源界面

明确不做：扫描件 OCR 全流程、法律意见生成、多人权限、移动端、自建向量数据库集群。
边界理由见 `docs/mvp-scope.md`。

## 快速开始

要求 Python 3.11+ 与 uv。

```bash
uv sync
cp .env.example .env      # 填入 LLM 与 Embedding 密钥
uv run contract-qa check  # 环境与数据检查
uv run contract-qa parse  # PDF -> 条款（输出 data/processed/*.jsonl）
uv run contract-qa index  # 建 BM25 与向量索引
uv run contract-qa ask "宁国市垃圾中转站项目的合同总价是多少？"
uv run pytest
uv run ruff check .
```

未安装 uv 时，纯标准库路径同样可跑通（除向量检索外）：

```bash
PYTHONPATH=src python -m contract_qa check
PYTHONPATH=src python -m contract_qa parse
python -m unittest discover -s tests -v
```

`ask` 支持 `--top-k`（送入模型的条款数，默认 8）与 `--show-hits`（打印双路召回排名与融合分）。

## 模型配置

DeepSeek 只提供对话模型，没有 Embedding 接口，因此需要两家服务：

| 用途 | 服务 | 默认模型 |
| --- | --- | --- |
| 问答生成 | DeepSeek | `deepseek-chat` |
| 向量化 | 硅基流动（OpenAI 兼容） | `BAAI/bge-m3`（1024 维） |

密钥写入 `.env`，该文件已被 `.gitignore` 排除，不会进入版本库。

## 数据

15 份公开政府采购文件（10 份合同 + 5 份招标文件，共 635 页），来源为中国政府采购网、
北京市政府采购网、河南省政府采购网及各地公共资源交易中心。元数据见 `data/document-register.csv`，
下载脚本 `scripts/fetch_public_documents.py`。

其中 5 份为纯扫描件（无可抽取文本），刻意保留为解析失败样本，不参与切分。
PDF 原文不进版本库。

## 目录

```text
src/contract_qa/
  text.py           PDF 取文、页眉页脚与目录清洗、标题分级
  splitter.py       条款切分（article / outline / chunk 三种模式）
  parse.py          解析编排与 JSONL 输出
  index.py          BM25 与 Chroma 向量索引
  retrieval.py      双路召回 + 加权 RRF 融合
  qa.py             问答与引用构造
  llm.py            OpenAI 兼容客户端（向量化 + 对话）
  config.py         .env 加载
  cli.py            命令行入口
scripts/            数据获取、验收与冒烟测试
tests/              自动化测试
docs/               范围、设计与验收记录
data/raw/           原始 PDF（不入库）
data/processed/     条款、索引与产物（不入库）
```

## 切分规则

条款单元按文档自身的编号体系选择，不强行统一：

- **article**：`第X条` 出现 8 次以上，按条切，章/节写入 `chapter_path`
- **outline**：靠 `一、` / `（一）` / `1、` / `1.1` 分层，从最粗的编号层往下试切，
  取第一条「中位长度落在 60–700 字符」的层级
- **chunk**：没有稳定编号，按 700 字符窗口切，强制带页码

三条硬约束：目录页整页丢弃（否则会留下「第一章」三个字的残行污染章节路径）；
只有标题没有正文的条目并入下一条；超过 1200 字符的条款按标点切开，片段继承
父条款编号并写入 `parent_clause_id`。

## 检索策略

双路召回后用加权 RRF 融合，权重可用环境变量 `BM25_WEIGHT` / `DENSE_WEIGHT` 调。

BM25 权重高于向量，是本项目语料上的实测结论：bge-m3 对「投标保证金」这类精确术语
会召回语义相近但无用的条款（如「反商业贿赂承诺书」），关键词召回反而精确。
向量负责语义泛化，关键词负责术语命中，根治靠 rerank。

## 指标纪律

Recall@5、引用正确率、不可答硬答率、P95 延迟与成本只记录实测结果。
计划书中的数字仅是示例，不得直接写入简历或 README。

## 下一步：D4–D6

D4 加 route / rewrite / generate 节点与严格 JSON 输出；D5 接 rerank；D6 加 verify 节点
做引用校验与最多 2 轮回退。当前基线：569 条条款、冒烟 6 题中 4 题可答、2 题正确拒答、
平均 1.84 秒、单题约 3.3k tokens。

## License

MIT
