# Changelog

## 2026-11-01

### feat: 初始化企业知识问答 Agent

- 新增 FastAPI、LangGraph、PostgreSQL/pgvector 后端，接入 DeepSeek、BGE 向量与重排、MinerU PDF 解析。
- 支持多格式文档上传、替换、分块重建、知识库版本管理、混合检索、原文引用和补充后恢复问答。
- 新增 React/TypeScript 网页，提供对话、SSE Trace、引用查看和 PDF 预览。
- 实现持久化队列、Worker 租约、checkpoint、幂等、超时与调用预算，以及统一启动和 live/mock/测试环境隔离。
- 纳入合成制度、20 题评估样本、工程测试、执行记录和项目文档；20 题真实模型业务评估仍待运行。
- 建立更新记录约定：Feature.md 保留本地记录并忽略跟踪，Changelog.md 随代码提交。
- 验证：168 项测试通过，Python F/E9/B023 静态检查通过，前端 TypeScript/Vite 生产构建通过。
