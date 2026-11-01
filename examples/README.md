# 实际运行记录

- `test-results.xml`：pytest 153 项工程测试结果，使用真实 PostgreSQL/pgvector 和 LangGraph、测试模型适配器。
- `thinking-smoke.json`：真实 DeepSeek 思考模式 + BGE 的两轮协议冒烟结果。使用独立合成住宿文档验证工具往返、上下文追问与引用；共 9 次 API 请求。仅含公开结果/Trace 与请求计数，不含推理正文；临时知识库已清理。这不是 20 题评估结果。
- `scripted/*.json`：脚本决策适配器 + 实际检索/读取/持久化导出的5组记录，全部包含 `scripted_fixture_not_model_evaluation`。输入在 `run.question`，输出在 `run.result`，实际事件在 `trace`。
- `ui-smoke.png`：本地浏览器查看引用原文的界面检查。

重跑脚本夹具：保持数据库启动、停止后台 Worker，执行 `.venv/bin/python scripts/demo_cases.py`。脚本创建独立演示库并保留数据库记录，原 `data/policies` 不改写。

`live/` 尚无真实模型结果。完成密钥与真实索引配置后执行 `scripts/evaluate.py`；人工审阅事实和引用后才能填写业务通过率。
