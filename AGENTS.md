## 开发遵循
### 更新记录准则
每次更改写入 Feature.md，记录当前更新的功能，Feature.md不进入git追踪。当commit的时候把Feature.md总结并写入 Changelog.md。
Changelog.md进入git追踪。在填写 commit 的 message的时候，从写入的Changelog.md的总结中提取并写入。

### 项目结构：
README.md # 环境、安装、运行、评审入口
src/ # 项目代码与模型/工具接入
tests/ # 可执行测试
examples/ # 输入、实际输出、Trace
data/ # 合成资料和规则（按题目需要）

docs/
01-requirements.md # 用户、目标、范围和成功标准
02-solution.md # 调用流程、动态决策、技术取舍
03-evaluation.md # 测试预期、实际结果、失败和限制
04-ai-usage.md # AI 辅助内容、个人修改及验证
05-retrospective.md # 最难问题、学习收获、下一步

依赖文件 # 如 requirements.txt / package.json
.env.example # 所需变量；无变量时注明无需配置
