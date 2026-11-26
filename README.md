# Lumina 拾光人像画室

基于 **Qwen3-VL-2B + 语言侧 LoRA**、**树状会话 Agent 内核** 与 **原生高速人像精修引擎** 构建的中文单人人像对话修图系统。

上传任意单人人像，用自然中文连续下达修图意图（如「帮我磨皮」「瘦脸调到 0.4」「背景虚化轻一点」「回到第 1 轮试试大眼」），系统将自动完成多模态语义规划、结构校验、原图重渲染、多方案分支对比与无损撤销，严格保持未提及参数不被误改。

---

## 核心特性

### 1. 原生六工具高速人像精修引擎（`src/engine/`）
完全自研封装的进程内人像精修管线，内置持久化 MediaPipe 478 点人脸网格与多类人像分割，**按原图 SHA256 自动缓存解析结果**，多轮交互零子进程冷启动开销（单轮缓存命中渲染约 **0.18s–0.51s**）：

| 工具标识 | 中文名称 | 原生算法设计要点 |
| --- | --- | --- |
| `slim_face` | 瘦脸 | 6 对渐进式面颊/下颌轮廓点向中轴牵引，内置下巴、双耳、嘴角三重保护场与雅可比防撕裂约束 |
| `enlarge_eyes` | 大眼 | 沿内外眼角主轴构造旋转椭圆放大位移场，对眉弓施加柔和衰减保护，告别生硬圆球感 |
| `remove_acne` | 祛痘 | Lab 空间红度/暗度稳健 z-score 异常检测、非线性敏感度与显著度上限防误报，局部 ROI Telea 修补 + 微纹理回填 |
| `smooth_skin` | 磨皮 | CIE Lab 双层高低频分离，亮度与色度通道分别保边平滑并自适应回填毛孔细节，严格保护五官清晰度 |
| `blur_bg` | 背景虚化 | 原图灰度引导细化人像 Alpha 掩码（发丝贴合）+ 归一化卷积散景模糊，彻底消除边缘溢色光晕 |
| `whiten_teeth` | 牙齿美白 | 严格露齿可见性门控（开口比 + 嘴内异物遮挡筛查，闭嘴自动跳过）+ Lab 去黄温和提亮 |

> **原图重渲染机制**：每一轮调整均基于原始无损图片 + 当前完整激活参数一次性重新渲染，避免反复重采样或多次磨皮导致的画质累积劣化。

### 2. 树状 Agent 内核（`src/`）
参考轻量 Agent Harness 架构思想实现，兼顾可审计性、可控性与多方案探索：

- **追加式树状会话（`TreeSessionStore`）**：以 `id` / `parent_id` 节点记录在 `events.jsonl` 中，支持从任意历史节点原地分叉派生方案 A / B / C，一键切换对比，撤销全程无损。
- **三通道工具协议（`ToolOutcome`）**：将发给模型的极简反馈、写回程序的状态快照、展示给前端的完整审计明细彻底分流，避免底层耗时与路径信息污染大模型上下文。
- **上下文投影降噪（`convert_to_llm`）**：仅沿当前活跃分支投影干净对话历史，并自动折叠卡死的连续重复澄清，不依赖硬编码中文关键词规则替代模型理解。
- **钩子屏障与幂等保护**：支持 `before_tool_call` / `after_tool_call` 拦截扩展与 `request_id` 幂等重放。

### 3. 「拾光人像画室」可视化 Web Demo
零外部 Web 框架依赖（纯 Python 标准库 HTTP 服务 + 零 CDN 原生前端）：

- **画室式浅色液态玻璃界面**：支持「晨光 / 暖瓷」双光影氛围切换、实时自动提取当前人像 4 枚主色调色卡、可编辑作品署名。
- **专业对比工具**：原图/结果水晶滑块对比、并排对比、纯净结果视图、按住 `Space` 瞬时看原图、`Ctrl+Z` 快捷撤销、2.5× 双镜局部放大镜与一键导出高清水印对比卡。
- **多分支胶片时间线 & 统一助手面板**：点击底部任意历史缩略图即可原地分叉探索不同修图风格；右侧融合式对话助手实时呈现六工具状态 Diff、原始 JSON 计划与执行/跳过审计。

---

## 架构概览

```text
用户中文指令 ──▶ PortraitAgent (src/agent/harness.py)
                   ├── convert_to_llm() 树状分支上下文投影 (src/session/tree_store.py)
                   ├── SixToolLoraPlanner / RulePlanner (src/agent/planners.py)
                   ├── 纯结构化 Schema 校验与状态合并 (src/tools/protocol.py)
                   ├── before / after 工具钩子屏障 & 露齿可见性门控
                   └── RetouchAdapter ──▶ LuminaRetouchEngine (src/engine/)
                                             ├── analyzer.py (持久化检测 + SHA256 缓存)
                                             ├── skin.py     (防伪影祛痘 + Lab 高低频磨皮)
                                             ├── optics.py   (无光晕散景虚化 + 自然牙齿美白)
                                             └── geometry.py (连续瘦脸 + 旋转椭圆大眼)
```

更多架构演进与工程设计细节见 [Pi 借鉴与架构演进](docs/pi借鉴与架构演进.md) 与 [工具接口与状态规范](docs/工具接口与状态.md)。

---

## 实验评测摘要

仓库内置已训练好的 **3.1 MB 六工具语言侧 LoRA 权重**（基于 `Qwen3-VL-2B-Instruct`，冻结视觉编码器，语言侧 `q_proj`/`v_proj` rank 4，80.3 万可训练参数，1800 条多模态图像+状态+历史→JSON 计划样本微调 1 epoch）：

| 评测项 | 评测口径 | 实测结果 |
| --- | --- | --- |
| **开发集规划准确率（`dev280`）** | 280 条六工具计划诊断开发集 | 原生底座 **60.7%（170/280）** → 四工具 LoRA **88.6%（248/280）** → 六工具 LoRA **90.7%（254/280）** |
| **端到端计划到出图** | 24 张新图 / 48 轮完整多轮会话 | **21/24 会话成功**，**48/48 纯 JSON 输出**，闭嘴美白门控 1/1 准确跳过 |
| **模型规划延迟（暖机）** | 单张 GPU 推理 | P50 约 **1.14 秒** |
| **原生引擎渲染耗时（CPU）** | 缓存命中多轮重渲染 | 单/双工具约 **0.18 秒**，五工具联合渲染约 **0.51 秒**（相比外部子进程冷启动提速 30× 以上） |

> 注：仓库同时保留了 v2 受控定向补训脚本（`agent_baseline/tool_expand_v1/six_tool_data_v2/`），该对照实验在 `dev280` 达 255/280 但因 `new80` 略低于门禁（74/80）而未替换正式权重，当前默认启用综合表现最稳健的 v1 六工具 LoRA 权重。

---

## 快速开始

### 1. 环境准备

建议使用 Python 3.10–3.12：

```bash
git clone https://github.com/yahooh725/lumina-portrait-agent.git
cd lumina-portrait-agent
pip install -r requirements.txt
```

### 2. 一键体验（无需 GPU）

- **规则规划 + 原生像素引擎实时精修**（首次运行将自动下载轻量 MediaPipe 模型至本地缓存，或通过 `LUMINA_VISION_MODEL_DIR` 指定离线目录）：

```bash
PORTRAIT_DEMO_PLANNER=rules bash agent_baseline/demo_showcase_v1/launch_web_demo.sh
```

- **纯 CPU 交互冒烟模式**（零模型依赖，立即体验界面、滑块对比与分支树）：

```bash
PORTRAIT_DEMO_PLANNER=mock bash agent_baseline/demo_showcase_v1/launch_web_demo.sh
```

启动后在浏览器打开 `http://127.0.0.1:8787`，直接点击左侧「上传本地单人人像」即可开始修图。

### 3. 启动完整多模态 LoRA 版本（需要 GPU）

将 `Qwen/Qwen3-VL-2B-Instruct` 下载至 `agent_baseline/models/Qwen3-VL-2B-Instruct`（或直接通过环境变量指定 Hugging Face 模型 ID / 本地路径），仓库已内置对应的六工具 LoRA 权重：

```bash
PORTRAIT_BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
PORTRAIT_GPU_ID=0 \
PORTRAIT_DEMO_PORT=8787 \
bash agent_baseline/demo_showcase_v1/launch_web_demo.sh
```

### 4. 独立调用原生修图引擎 CLI

```bash
# 分析人像关键点与露齿状态
python3 -m src.engine analyze /path/to/portrait.jpg

# 从原图按参数直接渲染
python3 -m src.engine edit /path/to/portrait.jpg -o out.png \
  --slim-face 0.4 --enlarge-eyes 0.3 --smooth-skin 0.4 --blur-bg 0.5
```

### 5. 运行单元测试与接口冒烟测试

```bash
# 树状会话、分支切换、三通道返回、钩子屏障、原生引擎算子单测
python3 src/tests/test_harness.py

# Web Demo HTTP 接口与多轮分支流程冒烟测试
python3 agent_baseline/demo_showcase_v1/smoke_test_demo.py
```

---

## 目录结构

```text
portrait-agent/
├── README.md                          # 项目说明（纯中文）
├── LICENSE                            # MIT 开源协议
├── requirements.txt                   # Python 依赖清单
├── configs/                           # 工具注册表、计划 JSON Schema、规划提示词与环境变量示例
├── docs/                              # 架构演进、工具接口规范、数据训练与评测文档
├── examples/                          # 结构化操作计划设计样例
├── weights/
│   └── six_tool_lora_v1/              # 内置 3.1MB 六工具 LoRA 权重（开箱即用）
├── src/
│   ├── engine/                        # Lumina 原生六工具高速人像精修引擎
│   │   ├── analyzer.py                # 持久化 MediaPipe 检测与按图缓存
│   │   ├── geometry.py                # 连续轮廓瘦脸 + 旋转椭圆大眼 + 防折叠保护
│   │   ├── skin.py                    # 防伪影自适应祛痘 + Lab 高低频磨皮
│   │   ├── optics.py                  # 引导细化散景虚化 + 露齿门控牙齿美白
│   │   └── pipeline.py                # 统一进程内管线与 CLI 入口
│   ├── agent/                         # PortraitAgent 调度内核与可插拔规划器
│   ├── session/                       # 追加式树状 JSONL 会话存储与上下文投影
│   ├── tools/                         # 协议校验与原生引擎三通道适配层
│   └── tests/                         # 核心单元测试
└── agent_baseline/
    ├── app.py                         # 规则规划与基础校验模块
    ├── demo_showcase_v1/              # 「拾光人像画室」可视化 Web Demo 与启动脚本
    └── tool_expand_v1/six_tool_data_v2/ # LoRA 微调与门禁评测参考脚本
```

---

## 致谢与许可

- 多模态基座模型：[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- 人脸关键点与语义分割视觉基础模型：[MediaPipe Tasks](https://developers.google.com/mediapipe)
- 树状会话与轻量 Agent Harness 架构启发：[pi](https://github.com/earendil-works/pi)

本项目代码基于 [MIT License](LICENSE) 开源。使用第三方基座模型或公开人像数据集进行二次训练与部署时，请遵守各自对应的许可证要求。
