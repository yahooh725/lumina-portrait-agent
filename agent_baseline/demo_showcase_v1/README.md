# Lumina 拾光人像画室（本地可视化 Web Demo）

本目录提供零外部 Web 框架依赖的本地可视化交互界面（纯 Python 标准库 HTTP 服务 + 零 CDN 原生前端），直接接入 `src/` 统一树状 Agent 内核与 `src/engine/` 原生人像精修引擎。

## 核心交互功能

- **画室式界面**：亮色液态玻璃质感，支持「晨光 / 暖瓷」双光影氛围切换、当前人像实时四色色卡提取与可编辑作品署名。
- **精细效果对比**：原图/结果滑块对比、并排对比、纯净结果视图、按住 `Space` 瞬时看原图、2.5× 双镜局部放大镜与一键导出对比卡。
- **多轮树状分支修图**：底部胶片时间线支持点击任意历史轮次原地分叉，派生多套修图方案并一键切换对比，撤销全程无损。
- **助手式对话面板**：统一右侧对话助手，自动展示六工具实时状态与本轮 Diff，支持展开查看原始 JSON 计划与执行审计。

## 快速启动

在仓库根目录下运行：

### 1. 预检环境（只读、不占 GPU）

```bash
python3 agent_baseline/demo_showcase_v1/web_demo.py --preflight
```

### 2. 启动六工具 LoRA 完整版（需要 GPU）

```bash
PORTRAIT_GPU_ID=0 PORTRAIT_DEMO_PORT=8787 bash agent_baseline/demo_showcase_v1/launch_web_demo.sh
```

如需指定自定义基座模型路径或 Hugging Face 模型 ID：

```bash
PORTRAIT_BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct bash agent_baseline/demo_showcase_v1/launch_web_demo.sh
```

### 3. 无 GPU 启动（规则规划器 + 原生像素引擎实时精修）

```bash
PORTRAIT_DEMO_PLANNER=rules bash agent_baseline/demo_showcase_v1/launch_web_demo.sh
```

### 4. 纯 CPU 冒烟模式（快速体验前端交互与分支管理）

```bash
PORTRAIT_DEMO_PLANNER=mock bash agent_baseline/demo_showcase_v1/launch_web_demo.sh
```

启动后在浏览器访问 `http://127.0.0.1:8787`（远程服务器可通过 `ssh -L 8787:127.0.0.1:8787 <用户>@<服务器>` 转发到本地打开）。
