# PAI-Loop vs. 顶尖 Harness —— 能力对标与差距分析

**Date:** 2026-07-12
**Status:** For review（战略对标，非单一 feature 设计）
**Scope:** PAI-Loop agent core（`backend/agent/`, `backend/app/`）与 2025–2026 前沿 harness 生态的横向对比
**Audience:** agent core 团队 · 技术决策

---

## TL;DR

- 2026 年行业共识：**Agent = Model + Harness，harness 已取代模型成为质量的第一杠杆。**
- PAI-Loop 的定位是**服务端、可自托管、多租户、OpenAI-Responses 兼容的 agent 平台**（不是本地编码 CLI）。因此前沿里偏本地编码的特性（speculative edits / worktree / 本地 checkpoint / 浏览器工具）优先级低；而**上下文管理 / 子 agent / MCP / 验证 / 调度 / 治理**这类对企业运营型平台反而更刚需。
- **强项**（第一梯队水平，勿丢）：Skills 渐进式披露 + 安装/挂载、OpenAI-Responses 协议 + 可断线恢复的 background run、多租户隔离 + NAS/沙箱 + STS、ES 混合 RAG。
- **三个最扎眼的空白**：~~① 通用子 agent（上下文防火墙）~~ ✅ **已落地（2026-07-12）** ② MCP live client ③ 多层上下文管理。子 agent 补齐后,剩下两个仍是前沿地基,且都能在现有干净架构上低成本落地。（同批把主循环"同 turn 多 tool_call 串行 dispatch"升级为**并发 dispatch**——本就该是默认行为。）
- 紧接着补 **plan/验证回路** 与 **调度器**（后者顺带兑现 "Loop" 的产品承诺）。

---

## 1. 背景：为什么"harness 就是质量本身"

到 2026 年初，benchmark 证据已让从业者确信 harness 是质量的主要杠杆：

- SWE-bench Verified 上六个前沿模型分差 **<1 分**，但同一个 Claude Opus 4.5 换三套 harness 从 **50.2% 摆到 55.4%**；SWE-bench Pro 上仅 scaffolding 变化即产生 **22+ 分**摆动。
- Terminal-Bench 2.0 上 GPT-5.2-Codex 通用 harness 57.5% vs 自家 Codex CLI 64.7%——**仅 harness 就 7 分差**。
- "Harness engineering" 已成为被命名的独立学科（Mitchell Hashimoto 2026-02 提出，OpenAI、Martin Fowler、LangChain 相继formalize）。

**顶尖产品真正的工程投入集中在 4 个地方**（不是模型，不是 UI）：

1. **上下文工程** —— 对抗 "context rot"。Chroma Research（2025-07）测 18 个前沿模型，**每一个**都随输入变长而退化——"1M token 窗口并不能可靠地在 1M token 上推理"，叠加 "lost in the middle"。Anthropic 把根因归为有限的 **attention budget**（n² 两两关系），因此 "context engineering —— 策展最小的高信号 token 集"是 prompt engineering 的继任者。**这是投入最重的地方。**
2. **子 agent 作为"上下文防火墙"** —— 不是角色扮演，而是让子 agent 在独立窗口做脏活（读 20+ 文件），只回传 1–2K token 摘要，防止主上下文被污染。
3. **验证回路** —— generator/evaluator 分离、LLM-as-judge、测试驱动自检。"三轮带严格自检 > 十轮不检查"。
4. **可移植的扩展面** —— Skills（渐进式披露）、MCP（连接性）、Hooks（确定性控制）、Plugins；且都在往开放标准收敛（agentskills.io / MCP / AGENTS.md，2025-12 归 Linux 基金会 Agentic AI Foundation）。

---

## 2. 顶尖 harness 的核心 feature（前沿速览）

按能力域归纳（详见 §4 差距表逐项对照）：

### 2.1 Agent loop core
- 规范定义（Simon Willison，已成行业标准）："An LLM agent runs tools in a loop to achieve a goal。"
- 现代最佳实践：**turn/预算上限**（max-turns + max-budget-USD）、**per-turn reasoning effort 档位**（Claude `effort` low→max、Codex `none`→`xhigh`、Amp low/medium/high/ultra）、**只读工具并行 / 写工具串行**、**KV-cache 感知设计**（稳定前缀、append-only、masking 而非删除工具）。

### 2.2 上下文管理（工程投入最重）
- **多层 compaction**：Claude Code 三层——microcompact（无模型调用丢弃陈旧 tool result，落盘可回取）/ full compact（模型调用摘要）/ session-memory compact。
- **模型原生 compaction（2026 前沿）**：GPT-5.1-Codex-Max 首个原生跨多窗口 compaction，跨百万 token；GPT-5.2-Codex 支撑 **7+ 小时**会话。
- **server 端 context editing**：Anthropic `clear_tool_uses_20250919` 按时序清老 tool result（换占位符），100 轮 eval 省 ~84% token；context editing + memory tool = 比 baseline **+39%**。
- **外部文件记忆 / just-in-time 检索**：Manus "filesystem 即终极 context"；只持有轻量标识（路径/查询/链接），运行时再加载（progressive disclosure）。
- **recitation**：持续重写的 `todo.md` 把目标推回近端注意力，对抗 lost-in-the-middle。

### 2.3 子 agent / 多 agent
- 2025 年"是否要多 agent"之争，2026 收敛为**综合**：读多、可并行、广度优先的活（研究）用多 agent；写多、依赖重的活（编码）用单线程（否则冲突决策污染共享状态）。
- 统一模式是 **context firewall**：子 agent 不是扮演角色，而是防上下文污染的技术手段——脏活在新窗口做，只回传 1–2K token 摘要。
- **验证 agent**：Anthropic 发现"把干活的 agent 和评判的 agent 分开是强杠杆"（自评会自夸）。

### 2.4 扩展面（四件套）
- **Agent Skills**（2025-10）：`SKILL.md` + 三级渐进式披露（L1 名称+描述常驻 / L2 触发时读正文 / L3 按需加载资源与脚本，"code beats tokens"）。2025-12 成开放标准，~40 客户端采用。
- **MCP**："AI 的 USB-C"，JSON-RPC，primitives = tools/resources/prompts + sampling/roots/elicitation；transports = stdio + Streamable HTTP；2025-11-25 spec 加 async tasks、server 端 agent loop、OAuth 2.1 强化。Registry 2026-04 已 **9400+ server**。
- **Hooks**：harness（非模型）在生命周期事件触发的确定性回调；PreToolUse 是 guardrail 甜点位。
- **Plugins / 指令文件**：打包 命令+子 agent+hooks+MCP；指令文件收敛到 **AGENTS.md** 跨工具标准。

### 2.5 规划 / 验证
- **Plan mode**（近乎普及）：只读权限态 + 批准门。
- **TODO 锚定**："Research → Plan → Implement + 频繁刻意 compaction"，目标把 context 利用率控在 **40–60%**（>60% 模型没空间推理）。
- **验证优先回路**：TDD 复兴（agent 消除写测试成本，测试给客观成功信号）；无测试时驱动 headless 浏览器（Playwright MCP）自检；LLM-as-judge / 对抗检查。

### 2.6 自主 / 长时程
- **checkpoint/resume + worktree 隔离**：即时回滚 + 并行分支不冲突。
- **异步云 + 定时 agent**（已是 table stakes）：Cursor Cloud Agents（8 路并行 + Multi-Agent Judging）、Copilot coding agent（Mission Control）、Codex cloud、Jules、Devin "Manage Devins"；**Claude Routines**（cron/HTTP/GitHub 事件触发，laptop-off）、**Managed Agents**（server 托管有状态 agent）。

### 2.7 权限 / 安全
- 收敛架构：**capability（sandbox）与 consent（approval）两个正交拨盘**。
- Claude Code 六种权限模式含 **auto**（分类器模型逐动作裁决）；Codex `sandbox_mode × approval_policy`；protected paths + hard deny 在所有模式生效。
- 沙箱：OS 原生（Seatbelt / bubblewrap+netns / Landlock+seccomp，凭据外置）+ 云端 microVM；执行期常关/代理网络出口。

---

## 3. PAI-Loop 现状定位

一句话：**一个可运营的服务端 agent 平台**——`Agent.run` 是干净的无状态每请求工具循环（build messages → stream model → dispatch tools → 循环至 max_steps），OpenAI-Responses 协议 + 可断线恢复的 background run，多租户 + JWT/RBAC，AgentRun 沙箱 + NAS/skill 挂载，ES 混合 RAG，Skills 渐进式披露 + 安装/挂载，rolling summary + user memory。

参照物在团队 spec 里写得很清楚：claude-code、nanobot、pydantic-ai、OpenClaw/Hermes、Manus context engineering——说明团队已在主动对标前沿。

---

## 4. 差距表：13 项顶尖能力 × PAI-Loop 现状

| # | 顶尖能力 | PAI-Loop 现状 | 判定 |
|---|---|---|---|
| 1 | **缓存感知 agent loop**（稳定前缀、append-only、effort 档位、turn/预算上限、只读工具并行） | Context v2 已做分层稳定前缀 + volatile 尾块；有 `MAX_RECURSION_STEPS`、idle timeout、HITL pause；**同 turn 多 tool_call 已并发 dispatch**（`_dispatch_parallel`，2026-07-12） | ✅ **基本齐平**——但缓存是"结构性"的（无显式 `cache_control`）；无 effort 档位；无 USD 预算上限 |
| 2 | **多层上下文管理**（microcompact + 模型 compact + context editing + 文件记忆） | **单层**：一条 LLM rolling summary（`app/summarizer.py`）+ `agent/budgeting.py` `fit` 截断；工具结果只 `cap_tool_result` 截断，不落盘不可回取 | 🔴 **明显落后** |
| 3 | **子 agent / 上下文防火墙** | ✅ **已落地**（2026-07-12）：`spawn_subagent` 工具 + `app/subagent.py` runner + 内置只读 `explore` worker（KB+代码+web）；隔离子 context、只回传摘要；并行 fan-out 由主循环并发 dispatch 承担 | ✅ **补齐**（曾是最大架构缺口）——待补:子事件实时透传、验证 agent、usage 入账 |
| 4 | **工作流形工具 + 可恢复错误** | `ToolBox.dispatch` 有 tenacity retry-3、错误捕获、artifact/notice 捕获；工具集精简合理 | 🟡 **尚可**——错误信息未做"可恢复化" prompt 工程 |
| 5 | **工具层上下文经济**（tool-search / tools-as-code） | 工具每轮全量加载 schema | 🟡 当前工具少不痛，**接 MCP 后会痛** |
| 6 | **验证回路**（evaluator 分离 / LLM-judge / TDD 自检） | **无** | 🔴 **缺失，质量杠杆** |
| 7 | **Plan mode + TODO 锚定**（recitation） | **无** plan mode、无 todo 重注入 | 🔴 **缺失** |
| 8 | **四面扩展**（Skills / MCP / Hooks / 命令） | Skills ✅ 很强（L1/L2/L3 + install/mount + SKILL.md 兼容）；**MCP 仅 `agent/tools/mcp.py` adapter，无 live client**；**无 hooks** | 🟡 **一强两缺** |
| 9 | **双档权限**（sandbox×approval + 分类器自动批准 + 保护路径） | RBAC + 工具级 admin gate + HITL pause | 🟡 **有基座，缺细粒度/自动批准** |
| 10 | **OS 级 + microVM 沙箱**（网络隔离、凭据外置） | AgentRun 网关沙箱 + 按用户/skill 挂载 + STS 注入，沙箱无浏览器可达网络 | ✅ **齐平**（企业向） |
| 11 | **checkpoint/resume + worktree 隔离** | 有 background run 的 resume/cancel；无 agent 动作回滚/rewind | 🟡 定位下**优先级低** |
| 12 | **异步云 + 定时/事件 agent**（Routines） | **无调度器**——"Loop" 目前=工具循环；`app/jobs.py` 有 `run_after`、KB `sync_schedule` 字段但注 "no scheduler in MVP" | 🔴 **缺失，且与产品名直接相关** |
| 13 | **Eval + 可观测** | OTel 存在但 lean 模式下 inert；无 eval 套件 | 🟡 **缺** |

### 强项（第一梯队水平，勿丢）
- **Skills 渐进式披露 + 安装/挂载 + SKILL.md 兼容**——业界领先水准。
- **OpenAI-Responses 协议 + 可断线恢复的 background run**（resume/cancel）——很好的运营基座。
- **多租户隔离 + NAS/沙箱 + STS 注入**——企业向硬功夫。
- **ES 混合 RAG（BM25 + kNN）+ 本地降级**——你们的血统优势。

---

## 5. 急需完成的（优先级 roadmap）

排序依据：**impact（质量/能力提升）× 定位契合度 × 是否解锁其它能力（地基性）**。

### 🥇 第一梯队 —— 地基级，最高优先

**① 通用子 agent 框架 + 上下文防火墙**（第一优先，无之一）✅ **已落地（2026-07-12）**
- **为什么**：前沿几乎所有能力的前提——验证 agent、并行研究、长任务、复杂 KB 检索都建立在"独立窗口子 agent 只回传摘要"之上。~~现在完全空白。~~
- **落地**：`Agent.run` 是无状态循环，加一个 `spawn_subagent` 工具（隔离 context、继承 soul + 工具子集、只回传 final message ~1–2K token）代价可控。读多的活（KB 深挖、多源检索）走并行 fan-out；写多的活走单线程——直接采用行业已收敛结论。
- **实际实现**：`app/subagent.py`（`SubagentRunner` + 内置只读 `explore` worker，专为代码 explore/知识库搜索调优）+ `agent/tools/builtin/spawn_subagent.py`（单工具）+ `app/builder.build_subagent_context`（空 history 防火墙）。深度上限 1、超时/步数/并发有界、能力门控 + 前端可视化。**并行 fan-out 不做批量工具**——主循环并发 dispatch 同 turn 多调用（见 ⑪）。详见 `docs/superpowers/specs/2026-07-12-spawn-subagent-design.md`。**后续项**：子事件实时透传、验证 agent（②，几乎免费）、usage 入账聚合。

**② MCP live client（把 adapter 接通）**（ROI 最高）
- **为什么**：对企业平台，MCP 是接入组织内部工具/数据的标准通路，比对编码 CLI 更刚需。9400+ server 生态、已归 Linux 基金会。
- **落地**：`agent/tools/mcp.py` 的描述符→`Tool` 映射**已写好**，缺的只是 stdio + Streamable HTTP 连接层 + OAuth 2.1。工作量小、解锁面大。

**③ 多层上下文管理**（可靠性 + 成本双赢）
- **为什么**：现在单层 rolling summary + 截断，长任务会"丢中段"；直接改善缓存命中率（Manus："缓存命中率是生产 agent 最重要的单一指标"，10x 成本差）。
- **落地**：补三件——(a) **microcompact**：无模型调用丢弃陈旧 tool result；(b) **工具结果落 store + 可回取 handle**（而非直接截断）；(c) **server 端 context editing**（`clear_tool_uses` 式，老 tool result 换占位符）。

### 🥈 第二梯队 —— 高价值，紧随其后

**④ Plan mode + 验证回路**（质量杠杆）
- Plan mode（只读规划态 + 操作员批准门）既提质量又给运营方控制感；generator/evaluator 分离 / LLM-judge（自评会自夸，必须分离评审者）。**子 agent 框架（①）落地后，验证 agent 几乎免费搭建。**

**⑤ 调度器：定时/事件触发的自主 run**（兑现 "Loop" 之名）
- 产品叫 PAI-**Loop**，但目前无自主/循环能力，叙事与实现有落差。
- **地基已在**：JobQueue 是持久化 worker + 有 `run_after`/`sync_schedule` 字段，补一个 cron/事件生产者即可对标 Claude Routines。对"可运营平台"是天然差异化点。

**⑥ Hooks（生命周期确定性回调）**（契合 "operate the deployment" 叙事）
- PreToolUse 作为策略/审计/合规门，是企业平台的治理刚需。已有边缘 guardrail，抽象成通用 hook 系统即可。

### 🥉 第三梯队 —— 定位相关，可后置

- ⑦ 双档权限 + 分类器自动批准（细粒度 sandbox×approval）
- ⑧ tool-search / tools-as-code（接 MCP、工具膨胀后再做）
- ⑨ 记忆的相关性检索（Context v2 设计里 ② 已显式 defer）
- ⑩ eval 套件 + 打通 OTel 可观测
- ~~⑪ 只读工具并行执行（廉价延迟优化）~~ ✅ **已落地**（2026-07-12）——主循环对同 turn 内多个无依赖 tool_call 并发 dispatch（`agent.agent._dispatch_parallel`，`TOOL_DISPATCH_CONCURRENCY=8`）；单发 / `return_direct` 仍串行。本就该是 agent loop 的默认行为。

---

## 6. 一句话总结

> PAI-Loop 在 **Skills、协议/流式基座、多租户沙箱、RAG** 上已站在第一梯队；最扎眼的三个空白里**子 agent（上下文防火墙）已于 2026-07-12 补齐**，剩 **MCP 接通、多层上下文管理**——都是前沿地基，且都能在现有干净架构上低成本落地。紧接着补 **plan/验证** 与 **调度器**（后者还顺带兑现 "Loop" 的产品承诺）。基座已经很扎实，差的是把"单请求工具循环"升级为"能自主编排、能长时程、能被治理"的完整 harness。

---

## 附：主要来源

- Anthropic, *Building Effective Agents* — https://www.anthropic.com/engineering/building-effective-agents
- Anthropic, *Effective Context Engineering for AI Agents* — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic, *Context Management* (context editing + memory tool) — https://www.anthropic.com/news/context-management
- Anthropic, *How We Built Our Multi-Agent Research System* — https://www.anthropic.com/engineering/built-multi-agent-research-system
- Anthropic, *Harness Design for Long-Running Apps* — https://www.anthropic.com/engineering/harness-design-long-running-apps
- Anthropic, *Writing Effective Tools for Agents* — https://www.anthropic.com/engineering/writing-tools-for-agents
- Anthropic, *Advanced Tool Use* (tool search) — https://www.anthropic.com/engineering/advanced-tool-use
- Anthropic, *Code Execution with MCP* — https://www.anthropic.com/engineering/code-execution-with-mcp
- Anthropic, *Agent Skills* — https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- Cognition, *Don't Build Multi-Agents* (Walden Yan) — https://cognition.com/blog/dont-build-multi-agents
- Manus, *Context Engineering for AI Agents* — https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
- Simon Willison, *Tools in a Loop* / *Designing Agentic Loops* — https://simonwillison.net/2025/May/22/tools-in-a-loop/ · https://simonwillison.net/2025/Sep/30/designing-agentic-loops/
- Chroma Research, *Context Rot* — https://particula.tech/blog/chroma-context-rot-long-context-degradation
- Cursor, *Secure Codebase Indexing* — https://cursor.com/blog/secure-codebase-indexing
- Aider, *Repository Map* — https://aider.chat/docs/repomap.html
- Amp Manual (Oracle / Task) — https://ampcode.com/manual
- Model Context Protocol — https://modelcontextprotocol.io/docs/learn/architecture
- OpenAI, *GPT-5.1-Codex-Max* — https://openai.com/index/gpt-5-1-codex-max/
- Claude Code Agent Loop / Permission Modes — https://code.claude.com/docs/en/agent-sdk/agent-loop · https://code.claude.com/docs/en/permission-modes
- Martin Fowler, *Harness Engineering* — https://martinfowler.com/articles/harness-engineering.html
- LangChain, *The Anatomy of an Agent Harness* — https://www.langchain.com/blog/the-anatomy-of-an-agent-harness
- Benchmark 对比：https://www.digitalapplied.com/blog/swe-bench-verified-june-2026-benchmark-vs-scaffolding-analysis · https://explainx.ai/blog/agent-harness-engineering-terminal-bench-langchain-2026

> *说明：部分 Claude Code 内部数值阈值（如 ~83.5% auto-compact 触发点）来自第三方逆向，随版本变化；官方文档仅定性描述。*
