# 实际运行记录

## 待运行的开发样本

[workflow-qa-cases.md](workflow-qa-cases.md) 是澄岳八份长篇手册的 **40 题清单与参考答案**，四类各 10 题。对应机器样本为 [workflow_qa_cases.json](../tests/fixtures/workflow_qa_cases.json)。全部为 `not_run`，不属于下列实际输出或 Trace，也不能作为知识库语料上传。

## 已保存的运行材料

- `test-results.xml`：完整 pytest **241 项通过，0 失败、0 错误、0 跳过**，使用独立测试数据库、真实 PostgreSQL/pgvector 和 LangGraph、测试模型适配器。
- [真实模型考试验收](live/exam-20261102-final/summary.json)：9 条原始用例和 1 条文档替换变体，覆盖六类考试验收点，选定集原文核验 10/10 通过。每题 JSON 保存输入、初次/最终输出、Trace、已读原文和真实模型调用摘要；[review.json](live/exam-20261102-final/review.json) 为 AI 逐条事实审阅，[verification.json](live/exam-20261102-final/verification.json) 保存引用与冻结来源独立核验，不冒称考生人工验收。7 条已回答、2 条资料不足、1 条正确报告执行未完成，不能把10/10解读为回答成功率。
- [首轮失败记录](live/exam-20261102/review.json)：保留冲突中追加15天建议、资格未知却直接回答的两处错误，以及其余原始输出；修正提示后从新环境完整复测，没有覆盖旧记录。
- `live/exam-20261102-final/sources/`：原20题对应的八份冻结制度，以及500→600的D04副本。工作区D07的并发修订保持原状，本次验收不代表新D07已经验证。资料来源SHA、模型/框架版本、预算和源码哈希见同目录 `manifest.json`。
- `thinking-smoke.json`：真实 DeepSeek 思考模式 + BGE 的两轮协议冒烟结果。使用独立合成住宿文档验证工具往返、上下文追问与引用；共 9 次 API 请求。仅含公开结果/Trace 与请求计数，不含推理正文；临时知识库已清理。这不是 20 题评估结果。
- `scripted/*.json`：脚本决策适配器 + 实际检索/读取/持久化导出的5组记录，全部包含 `scripted_fixture_not_model_evaluation`。输入在 `run.question`，输出在 `run.result`，实际事件在 `trace`。
- `ui-smoke.png`：本地浏览器查看引用原文的界面检查。

重跑脚本夹具：保持数据库启动、停止后台 Worker，执行 `.venv/bin/python scripts/demo_cases.py`。脚本创建独立演示库并保留数据库记录，原 `data/policies` 不改写。

复现真实验收：按 README 配好真实模型后执行 `.venv/bin/python scripts/exam_acceptance.py --output examples/live/exam-rerun --source-revision a3e478af065c3542c80fc199750d8bef309fe3c0`。它创建独立数据库，不写业务库；新输出仍需独立事实审阅。原20题其余11题及长篇40题尚未执行，不宣称整体业务正确率。
