# 🚁 **AerialVLN: 基于视觉和语言导航的无人机项目**

[![GitHub stars](https://img.shields.io/github/stars/AirVLN/AirVLN?style=social)](https://github.com/AirVLN/AirVLN) 
[![License](https://img.shields.io/github/license/AirVLN/AirVLN)](LICENSE) 
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/AirVLN/AirVLN/actions)

---

## 📖 **目录**
1. [简介](#简介)
2. [项目特色](#项目特色)
3. [快速开始](#快速开始)
4. [云端大模型测评基准](#云端大模型测评基准)
5. [使用示例](#使用示例)
6. [常见问题](#常见问题)
7. [引用](#引用)
8. [联系方式](#联系方式)
9. [致谢](#致谢)
10. [终极版本设想](#终极版本设想)

You may refer to the [English version of this page](https://github.com/AirVLN/AirVLN/blob/main/README.md).


---

## 🌟 **简介**

**摘要：**
Recently emerged Vision-and-Language Navigation (VLN) tasks have drawn significant attention in both computer vision and natural language processing communities. Existing VLN tasks are built for agents that navigate on the ground, either indoors or outdoors. However, many tasks require intelligent agents to carry out in the sky, such as UAV-based goods delivery, traffic/security patrol, and scenery tour, to name a few. Navigating in the sky is more complicated than on the ground because agents need to consider the flying height and more complex spatial relationship reasoning. To fill this gap and facilitate research in this field, we propose a new task named AerialVLN, which is UAV-based and towards outdoor environments. We develop a 3D simulator rendered by near-realistic pictures of 25 city-level scenarios. Our simulator supports continuous navigation, environment extension and configuration. We also proposed an extended baseline model based on the widely-used cross-modal-alignment (CMA) navigation methods. We find that there is still a significant gap between the baseline model and human performance, which suggests AerialVLN is a new challenging task.

近年来，视觉与语言导航（Vision-and-Language Navigation，简称 VLN）任务在计算机视觉和自然语言处理领域引起了广泛关注。然而，现有的 VLN 任务主要面向地面导航代理，无论是在室内还是室外。然而，许多实际任务需要智能代理在空中执行操作，例如基于无人机（UAV）的货物配送、交通/安全巡逻以及风景巡游等。相比地面导航，空中导航更加复杂，因为代理需要考虑飞行高度以及更复杂的空间关系推理。为填补这一空白并促进该领域的研究，我们提出了一项新任务，名为 AerialVLN，专注于基于无人机的户外导航。我们开发了一个3D模拟器，该模拟器使用接近真实的图像渲染了 25 个城市级场景。我们的模拟器支持连续导航、环境扩展和配置功能。此外，我们基于广泛使用的CMA方法，提出了一个扩展的基线模型。研究表明，基线模型与人类性能之间仍存在显著差距，这表明 AerialVLN 是一项具有挑战性的全新任务。

---

## 🚀 **项目特色**

- **真实感3D模拟器**：提供 25 个城市级场景，图像逼真。
- **跨模态对齐模型**：通过视觉和语言信息实现高级导航。
- **可扩展框架**：支持添加新的环境和配置。
- **综合数据集**：包括 AerialVLN 和 AerialVLN-S，用于模型训练和评估。

![Instruction: Take off, fly through the tower of cable bridge and down to the end of the road. Turn left, fly over the five-floor building with a yellow shop sign and down to the intersection on the left. Head to the park and turn right, fly along the edge of the park. March forward, at the intersection turn right, and finally land in front of the building with a red billboard on its rooftop.](./files/instruction_graph.jpg)
Instruction: Take off, fly through the tower of cable bridge and down to the end of the road. Turn left, fly over the five-floor building with a yellow shop sign and down to the intersection on the left. Head to the park and turn right, fly along the edge of the park. March forward, at the intersection turn right, and finally land in front of the building with a red billboard on its rooftop.

---

## 🛠️ **快速开始**

### 前置条件
- Ubuntu 操作系统
- Nvidia GPU(s)
- Python 3.8+
- Conda


### 安装依赖

#### 第1步: 创建并进入工作区文件夹
```bash
mkdir AirVLN_ws
cd AirVLN_ws
```

#### 第2步: 克隆仓库

```bash
git clone https://github.com/AirVLN/AirVLN.git
cd AirVLN
```

#### 第3步: 创建并激活虚拟环境

```bash
conda create -n AirVLN python=3.8
conda activate AirVLN
```

#### 第4步: 安装 pip 依赖

```bash
pip install pip==24.0 setuptools==63.2.0
pip install -r requirements.txt
pip install airsim==1.7.0
```

#### 第5步: 安装 PyTorch 和 PyTorch Transformers

在[ PyTorch 官网](https://pytorch.org/get-started/locally/)选择正确 CUDA 版本的 PyTorch 。
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cuxxx
```

然后安装依赖于 PyTorch 的 pytorch-transformers。
```bash
pip install pytorch-transformers==1.2.0
```

### 模型 & 模拟器 & 数据集

#### 第6步: 为后续步骤创建目录

```bash
cd ..
mkdir -p ENVs\
  DATA/data/aerialvln\
  DATA/data/aerialvln-s\
  DATA/models/ddppo-models
```

#### 第7步: 下载预训练模型

从 [这里](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7/habitat_baselines/rl/ddppo) 下载 **gibson-2plus-resnet50.pth** 把它放到 `./DATA/models/ddppo-models` 目录下.

#### 第8步: 下载模拟器

AerialVLN 模拟器（约 35GB） 可通过 Kaggle 网站 下载，也可使用以下 cURL 命令：
```bash
#!/bin/bash
curl -L -o ~/Downloads/aerialvln-simulators.zip\
  https://www.kaggle.com/api/v1/datasets/download/shuboliu/aerialvln-simulators
```

您还可以通过 kagglehub 下载，并将其放置到 `./ENVs` 目录下：
```bash
import kagglehub

# 下载最新版本
path = kagglehub.dataset_download("shuboliu/aerialvln")

print("数据集文件路径:", path)
```

其它下载链接: [百度网盘 (提取码=vbv9)](https://pan.baidu.com/s/1IB9OjWXG2nDDdjwCdjVxBw?pwd=vby9)

#### 第9步: 下载数据集

AerialVLN 和 AerialVLN-S 注释数据集（均小于 100MB） 可通过以下方法获取：

- AerialVLN 数据集: [https://www.kaggle.com/datasets/shuboliu/aerialvln](https://www.kaggle.com/datasets/shuboliu/aerialvln)
- AerialVLN-S 数据集: [https://www.kaggle.com/datasets/shuboliu/aerialvln-s](https://www.kaggle.com/datasets/shuboliu/aerialvln-s)

或者使用以下命令下载：
```bash
#!/bin/bash
curl -L -o ~/Downloads/aerialvln.zip\
  https://www.kaggle.com/api/v1/datasets/download/shuboliu/aerialvln
```
以及
```bash
#!/bin/bash
curl -L -o ~/Downloads/aerialvln.zip\
  https://www.kaggle.com/api/v1/datasets/download/shuboliu/aerialvln-s
```

其它下载链接: [百度网盘 (提取码=cgwh)](https://pan.baidu.com/s/1mhNeqDjipXULMa2PfTaZKQ?pwd=cgwh)

### 目录结构

最终，你的项目目录应该类似于以下结构：

```bash
- Project workspace
    - AirVLN
    - DATA
        - data
            - aerialvln
            - aerialvln-s
        - models
            - ddppo-models
    - ENVs
      - env_1
      - env_2
      - ...
```

## 云端大模型测评基准

本仓库已经在原始 AirVLN 本地模型评测流程的基础上，新增了一套云端大模型测评基准。它的目标是：完全抛弃本地大模型推理，把动作决策交给 OpenAI-compatible 云端多模态模型，同时尽量保持原项目的仿真环境、测试集遍历、动作执行、视频生成和指标计算逻辑不变。

### 设计目标

- **只替换动作决策器**：原本由本地 checkpoint policy 输出动作，现在由云端视觉语言模型输出离散动作。
- **保持 AirVLN 评测逻辑一致**：仍然使用 `AirVLNENV`、AirSim 仿真、`makeActions`、`get_obs`、原始 `success/nDTW/sDTW/path_length` 等指标。
- **支持多模态输入**：每步可以发送 RGB 图、伪彩色深度图、深度统计摘要、pose、历史动作和语义记忆。
- **支持实验复现**：保存配置快照、prompt hash、episode list、运行 manifest、逐步轨迹和失败分析。
- **支持模型比较**：固定同一批 episode 后，可以比较不同模型、prompt、深度策略或 memory 设置的结果差异。

### 关键文件

| 文件 | 作用 |
| --- | --- |
| `configs/cloud_eval.yaml` | 云端测评主配置文件，常用参数都在这里修改。 |
| `configs/prompts/cloud_vln_depth_v2_system.txt` | 云端模型 system prompt 模板。 |
| `configs/prompts/cloud_vln_depth_v2_user.txt` | 云端模型 user prompt 模板。 |
| `scripts/cloud_eval.sh` | 正式云端评测入口。 |
| `scripts/cloud_smoke_test.sh` | 快速体检入口，先测云模型能力，再跑极小规模仿真评测。 |
| `src/vlnce_src/cloud_eval.py` | 云端评测主流程。 |
| `src/vlnce_src/cloud_model.py` | OpenAI-compatible 云端模型客户端、prompt 构造、图像编码、动作解析。 |
| `src/vlnce_src/check_cloud_model.py` | 云模型能力自检脚本。 |
| `src/vlnce_src/audit_cloud_alignment.py` | 云端评测和原本本地评测的对齐审计脚本。 |
| `src/vlnce_src/compare_cloud_runs.py` | 两次云端评测结果对比脚本。 |

### 运行前准备

本项目当前约定的 conda 环境名是 `cxj`。在交互式终端中可以手动进入环境：

```bash
conda activate cxj
```

如果直接运行脚本，`scripts/cloud_eval.sh` 和 `scripts/cloud_smoke_test.sh` 会尝试自动执行 conda shell hook 并激活 `cxj`。

运行云端评测前，还需要确保 AirSim 仿真服务已经启动。通常在另一个终端运行：

```bash
cd /data/lyj/cxj/AirVLN_ws/AirVLN
python -u ./airsim_plugin/AirVLNSimulatorServerTool.py
```

如果你已经长期保持仿真服务运行，则不需要重复启动。

### 云端 API 配置

云端评测使用 OpenAI-compatible Chat Completions API。主要配置在 `configs/cloud_eval.yaml` 中：

```yaml
cloud_model: Ali-dashscope/Qwen3.5-Plus
cloud_base_url: https://xplt.sdu.edu.cn:4000
cloud_api_key: ""
cloud_api_key_env: DASHSCOPE_API_KEY
cloud_disable_proxy: true
cloud_verify_ssl: false
```

说明：

- `cloud_model`：云端模型名。
- `cloud_base_url`：OpenAI-compatible API 地址。
- `cloud_api_key`：API key。也可以留空，改用环境变量。
- `cloud_api_key_env`：当 `cloud_api_key` 为空时，从该环境变量读取 key。
- `cloud_disable_proxy`：是否忽略系统代理。某些内网或校园代理服务需要直连时应设为 `true`。
- `cloud_verify_ssl`：是否校验 HTTPS 证书。部分自签名服务需要设为 `false`。

如果不想把 key 写进被 git 跟踪的配置文件，可以复制一份本地配置：

```bash
cp configs/cloud_eval.yaml configs/cloud_eval.local.yaml
```

然后只在 `configs/cloud_eval.local.yaml` 里填写 key。`cloud_eval.sh` 会优先读取 `configs/cloud_eval.local.yaml`，该文件默认不应提交到公开仓库。

### 常用评测参数

| 参数 | 说明 |
| --- | --- |
| `name` | 实验名称，输出目录为 `../DATA/output/{name}/eval/`。 |
| `EVAL_DATASET` | 评测 split，可选 `train`、`val_seen`、`val_unseen`、`test`。 |
| `EVAL_NUM` | 评测 episode 数量，`-1` 表示全量。 |
| `maxAction` | 单个 episode 最大动作步数。 |
| `EVAL_GENERATE_VIDEO` | 是否生成视频。 |
| `batchSize` | 云端评测建议保持 `1`。 |
| `cloud_temperature` | 采样温度，评测建议为 `0.0`。 |
| `cloud_max_tokens` | 单次模型回复最大 token 数。启用 memory 后建议不低于 `128`。 |
| `cloud_timeout` | 单次 API 请求超时时间，单位秒。 |
| `cloud_max_retries` | 单步动作请求失败或解析失败后的重试次数。 |
| `cloud_fallback_action` | 多次失败后的兜底动作，如 `STOP`。 |
| `cloud_max_api_calls` | 限制单次评测最多发起多少次云端动作请求，`-1` 表示不限。 |
| `simulator_tool_port` | AirSim 仿真服务端口，默认 `30000`。 |

### 数据集规模

当前本地 `DATA/data/aerialvln/` 中各 split 的 episode 数量如下：

| split | episode 数量 |
| --- | ---: |
| `train` | 16386 |
| `val_seen` | 1818 |
| `val_unseen` | 2310 |
| `test` | 4830 |

当前默认配置使用：

```yaml
EVAL_DATASET: train
EVAL_NUM: -1
```

因此默认会对当前数据量最大的 `train` split 的 16386 条 episode 做全量评测。如果要跑 `val_unseen` 全量，则改为：

```yaml
EVAL_DATASET: val_unseen
EVAL_NUM: -1
```

### 模型输入内容

每一步云端模型会看到一个离散动作决策问题。根据配置不同，输入可能包括：

- **自然语言导航指令**：来自数据集，是从起点到终点的全局导航描述。
- **当前 RGB 图像**：无人机相机当前观测。
- **伪彩色深度图**：由深度数组渲染为 PNG 图像，红/黄表示近障碍，蓝色表示较远空间。
- **深度摘要**：包括中心、左右、上下区域深度均值，前方中心最小深度，近障比例，`grid_mean` 和 `grid_min`。
- **当前 pose**：`[x, y, z, qw, qx, qy, qz]`。
- **最近动作历史**：最近若干步的动作和 memory。
- **语义记忆 memory**：模型每步返回的短文本进展记忆，会在下一步带回 prompt。

模型必须返回一个 JSON 对象，例如：

```json
{"action_id": 1, "action_name": "MOVE_FORWARD", "memory": "short navigation progress note"}
```

支持的动作如下：

| action_id | action_name | 含义 |
| --- | --- | --- |
| 0 | `STOP` | 停止并结束 episode。 |
| 1 | `MOVE_FORWARD` | 前进 5 米。 |
| 2 | `TURN_LEFT` | 左转 15 度。 |
| 3 | `TURN_RIGHT` | 右转 15 度。 |
| 4 | `GO_UP` | 上升 2 米。 |
| 5 | `GO_DOWN` | 下降 2 米。 |
| 6 | `MOVE_LEFT` | 左移 5 米。 |
| 7 | `MOVE_RIGHT` | 右移 5 米。 |

### 深度图配置

深度图本质是数值数组。云端视觉语言模型一般不能直接理解完整的大矩阵，因此当前实现使用两种互补方式：

1. 把深度数组渲染成伪彩色 PNG，作为第二张图发送给多模态模型。
2. 把深度数组压缩成结构化文本摘要，作为 prompt 的一部分发送。

相关配置：

```yaml
cloud_depth_mode: both
cloud_use_depth_summary: true
cloud_depth_near_percentile: 2.0
cloud_depth_far_percentile: 98.0
cloud_depth_grid_size: 3
cloud_depth_near_threshold: 0.05
```

`cloud_depth_mode` 可选：

- `none`：不发送深度信息。
- `summary`：只发送深度文本摘要。
- `image`：只发送伪彩色深度图。
- `both`：同时发送伪彩色深度图和文本摘要，推荐用于正式评测。

### Prompt 模板

Prompt 已经从代码中外置：

```yaml
cloud_prompt_system_path: configs/prompts/cloud_vln_depth_v2_system.txt
cloud_prompt_user_path: configs/prompts/cloud_vln_depth_v2_user.txt
```

每次运行都会保存 prompt 路径和 SHA256 hash，便于复现实验。模板支持的动态字段包括：

- `{instruction}`
- `{step}`
- `{action_history_json}`
- `{memory_block}`
- `{pose_block}`
- `{rgb_block}`
- `{depth_image_block}`
- `{depth_summary_block}`
- `{response_schema}`

### 正式评测

从工作区根目录运行：

```bash
cd /data/lyj/cxj/AirVLN_ws
bash ./AirVLN/scripts/cloud_eval.sh
```

脚本会：

1. 激活 `cxj` 环境。
2. 进入 `AirVLN` 目录。
3. 优先读取 `configs/cloud_eval.local.yaml`，否则读取 `configs/cloud_eval.yaml`。
4. 执行 `src/vlnce_src/cloud_eval.py`。

当前默认配置已经调整为适合全量 benchmark：

```yaml
EVAL_NUM: -1
maxAction: 500
EVAL_GENERATE_VIDEO: False
cloud_save_input_images: false
cloud_save_request_json: false
cloud_resume: true
cloud_save_episode_list: true
```

这会保留必要指标、运行报告、失败分析、轨迹 JSON、云端日志和 episode list，但不保存视频和每步输入图像，避免全量评测时磁盘快速增长。

### 快速体检 Smoke Test

如果只想确认当前配置、云模型和仿真链路是否能跑通，可以执行：

```bash
cd /data/lyj/cxj/AirVLN_ws
bash ./AirVLN/scripts/cloud_smoke_test.sh
```

它会先运行云模型能力自检，再跑一个极小规模的仿真评测。默认：

```yaml
cloud_smoke_eval_num: 1
cloud_smoke_max_action: 5
```

Smoke test 会临时开启输入保存，便于检查 RGB、深度图、prompt 和请求 JSON。

### 云模型能力自检

不启动 AirSim，只检查云端模型能力：

```bash
cd /data/lyj/cxj/AirVLN_ws/AirVLN
conda run -n cxj python src/vlnce_src/check_cloud_model.py --cloud_config configs/cloud_eval.yaml
```

检查项包括：

- 文本调用是否成功。
- 是否支持多图输入。
- 是否能稳定输出动作 JSON。
- 是否能根据合成深度图做基本避障判断。

输出位置：

```text
../DATA/output/{name}/eval/model_checks/{time}/cloud_model_check.json
```

### 评测对齐审计

用于检查云端评测是否仍和原本本地评测在关键流程上保持一致：

```bash
cd /data/lyj/cxj/AirVLN_ws/AirVLN
conda run -n cxj python src/vlnce_src/audit_cloud_alignment.py
```

它会检查：

- 数据集 split 映射。
- `EVAL_NUM` 控制逻辑。
- 环境 reset。
- `makeActions` 和 `get_obs`。
- 视频生成。
- 指标聚合。

输出位置：

```text
../DATA/output/cloud_alignment_audit.json
```

### 固定 episode 子集

为了公平比较不同模型或 prompt，系统会保存本次实际使用的 episode list：

```text
../DATA/output/{name}/eval/results/{time}/episode_list_selected.json
../DATA/output/{name}/eval/results/{time}/episode_list_evaluated.json
```

后续可以把某次 `episode_list_evaluated.json` 填到配置中：

```yaml
cloud_episode_list_path: ../DATA/output/Cloud-Qwen35-Plus/eval/results/某次运行/episode_list_evaluated.json
```

这样不同模型会评测完全同一批 episode。

### 评测输出目录

所有输出默认位于：

```text
../DATA/output/{name}/eval/
```

主要目录如下：

| 目录 | 内容 |
| --- | --- |
| `logs/` | 主日志文件。 |
| `results/{time}/` | 聚合指标、配置快照、manifest、报告、失败分析。 |
| `videos/{time}/` | 每个 episode 的评测视频。 |
| `trajectories/{time}/cloud_{model}/` | 每个 episode 的逐步轨迹 JSON。 |
| `cloud_logs/{time}/` | 每个 episode 的逐步云模型回复日志。 |
| `cloud_inputs/{time}/` | 可选保存的 prompt、RGB、深度图、请求 JSON。 |
| `intermediate_results/{time}/` | 所有 episode 原始 info 汇总。 |
| `intermediate_results_every/{time}/` | 单 episode 结果，支持中断后续跑。 |
| `model_checks/{time}/` | 云模型能力自检结果。 |
| `TensorBoard/{time}/` | TensorBoard 指标。 |

### 数据保存与开关

云端评测会保存两类数据：

1. **必要评测数据**：用于指标统计、复现、续跑和结果追踪，默认保存，目前不建议关闭。
2. **调试复盘数据**：用于查看每一步输入、图像和请求内容，默认关闭或可配置，开启后会明显增加磁盘占用。

#### 始终保存或基本始终保存的数据

| 数据 | 位置 | 内容 | 是否可关闭 |
| --- | --- | --- | --- |
| 主日志 | `DATA/output/{name}/eval/logs/` | 配置、场景切换、每步动作、episode 结果、warning/error。 | 暂无配置开关。 |
| 聚合结果 | `DATA/output/{name}/eval/results/{time}/` | 指标、配置快照、manifest、失败分析、Markdown 报告、episode list。 | 暂无配置开关。 |
| 单 episode 结果 | `DATA/output/{name}/eval/intermediate_results_every/{time}/cloud_{model}/` | 每个 episode 的最终 `info`，也用于中断后续跑。 | 暂无配置开关。 |
| 全量中间结果 | `DATA/output/{name}/eval/intermediate_results/{time}/` | 所有 episode 原始 `info` 汇总。 | 暂无配置开关。 |
| 逐步轨迹 | `DATA/output/{name}/eval/trajectories/{time}/cloud_{model}/` | 每个 episode 的逐步动作、pose、深度摘要、memory、fallback/error。 | 暂无配置开关。 |
| 云端逐步日志 | `DATA/output/{name}/eval/cloud_logs/{time}/` | 每步 raw response、动作解析、延迟、重试、fallback、错误。 | 暂无配置开关。 |
| TensorBoard | `DATA/output/{name}/eval/TensorBoard/{time}/` | 聚合指标写入 TensorBoard。 | 暂无配置开关。 |
| LMDB eval 缓存 | `DATA/img_features/eval/{name}/{split}_{time}/` | 原 AirVLN 环境在 eval 模式下创建的 LMDB 缓存。 | 暂无云端配置开关。 |

这些数据是当前评测可追溯性的基础。即使关闭视频和调试输入，以上大部分文件仍会生成。

#### 可配置保存的数据

| 数据 | 位置 | 控制参数 | 说明 |
| --- | --- | --- | --- |
| 视频 | `DATA/output/{name}/eval/videos/{time}/` | `EVAL_GENERATE_VIDEO` | 每个 episode 一个 mp4。开启后便于人工复盘，但会增加耗时和磁盘占用。 |
| 每步 prompt/RGB/深度图 | `DATA/output/{name}/eval/cloud_inputs/{time}/{episode_id}/` | `cloud_save_input_images` | 保存 `step_XXXX_prompt.txt`、`step_XXXX_rgb.jpg`、`step_XXXX_depth.png`。 |
| 每步 request JSON | `DATA/output/{name}/eval/cloud_inputs/{time}/{episode_id}/` | `cloud_save_request_json` | 保存去掉 base64 图像内容后的 `step_XXXX_request.json`。通常需要配合 `cloud_save_input_images: true` 使用。 |
| cloud log 额外调试字段 | `DATA/output/{name}/eval/cloud_logs/{time}/` | `cloud_save_prompts` | 当前主要用于额外写入 pose 等调试字段。注意：保存 prompt 文件的主开关是 `cloud_save_input_images`。 |
| 续跑跳过 | `intermediate_results_every/` | `cloud_resume` | 不是保存开关，而是控制是否利用已有单 episode 结果跳过已完成样本。 |
| 固定 episode list | `results/{time}/episode_list_*.json` | `cloud_save_episode_list` | 保存本次选中和实际完成的 episode id，方便后续公平比较。 |

#### 推荐保存策略

正式大规模 benchmark 建议减少调试文件：

```yaml
EVAL_GENERATE_VIDEO: False
cloud_save_input_images: false
cloud_save_request_json: false
cloud_save_prompts: false
cloud_resume: true
cloud_save_episode_list: true
```

这种设置仍会保存必要指标、配置快照、manifest、失败分析、轨迹 JSON 和云端日志，但不会保存视频、每步图片和请求 JSON。

调试少量失败样本时建议开启完整复盘材料：

```yaml
EVAL_NUM: 1
maxAction: 20
EVAL_GENERATE_VIDEO: True
cloud_save_input_images: true
cloud_save_request_json: true
cloud_save_prompts: true
```

这种设置会额外保存视频、每步 prompt、RGB、伪彩色深度图和 request JSON，适合分析模型到底看到了什么、回复了什么，以及为什么选择某个动作。

#### 只在单独工具运行时保存的数据

| 工具 | 输出 | 说明 |
| --- | --- | --- |
| `check_cloud_model.py` | `DATA/output/{name}/eval/model_checks/{time}/cloud_model_check.json` | 只在运行模型能力自检或 smoke test 时生成。 |
| `audit_cloud_alignment.py` | `DATA/output/cloud_alignment_audit.json` | 只在运行评测对齐审计脚本时生成。 |
| `compare_cloud_runs.py` | 默认 `run_b/comparisons/` | 只在运行结果对比脚本时生成 JSON、CSV 和 Markdown 对比报告。 |

### results 目录中的关键文件

一次正式评测结束后，`results/{time}/` 下通常包含：

| 文件 | 说明 |
| --- | --- |
| `stats_cloud_{model}_{split}.json` | 聚合后的官方指标和云端调用统计。 |
| `summary_cloud_{model}_{split}.csv` | 指标 CSV，方便表格处理。 |
| `cloud_call_stats_{model}_{split}.json` | 请求次数、重试次数、错误步数、fallback 步数、平均延迟等。 |
| `failure_analysis_cloud_{model}_{split}.json` | 失败 episode、碰撞、低 nDTW、行为标签等分析。 |
| `run_manifest.json` | 本次运行的模型、配置、prompt hash、git commit、输出路径。 |
| `config_snapshot.json` | 运行时配置快照，API key 会被掩码。 |
| `report_cloud_{model}_{split}.md` | 可读 Markdown 运行报告。 |
| `episode_list_selected.json` | 初始化后选中的 episode 列表。 |
| `episode_list_evaluated.json` | 实际完成评测的 episode 列表。 |

### 逐步轨迹日志

每个 episode 会保存一个轨迹 JSON：

```text
../DATA/output/{name}/eval/trajectories/{time}/cloud_{model}/{episode_id}.json
```

内容包括：

- episode id、trajectory id、自然语言指令。
- 最终指标。
- 每一步动作。
- 执行动作前后的 pose。
- 深度摘要。
- 模型返回的 memory。
- 请求耗时、重试次数、fallback、错误信息。
- 动作后是否 done。

这类文件用于复盘“模型为什么失败”，比只看最终指标更有用。

### 失败诊断

`failure_analysis_cloud_*.json` 会记录每个 episode 的失败原因和行为标签。当前支持的行为标签包括：

| 标签 | 含义 |
| --- | --- |
| `not_success` | 最终未成功。 |
| `collision` | 发生碰撞。 |
| `max_action_reached` | 达到最大步数，疑似超时或卡住。 |
| `cloud_fallback_used` | 至少一步使用了 fallback 动作。 |
| `very_low_ndtw` | nDTW 很低，轨迹严重偏离。 |
| `early_stop` | 很早 STOP 且未成功。 |
| `collision_after_forward` | 最后一步前进后碰撞。 |
| `repeated_same_action` | 连续重复同一动作。 |
| `turning_loop` | 大量转向，疑似原地转圈。 |
| `no_forward_progress` | 长时间没有前进行为。 |

### 结果对比

比较两次云端评测：

```bash
cd /data/lyj/cxj/AirVLN_ws/AirVLN

python src/vlnce_src/compare_cloud_runs.py \
  --run_a ../DATA/output/模型A/eval/results/运行时间A \
  --run_b ../DATA/output/模型B/eval/results/运行时间B
```

默认输出到 `run_b/comparisons/`：

```text
cloud_run_comparison.json
cloud_run_metric_deltas.csv
cloud_run_comparison.md
```

对比内容包括：

- 两次运行的配置差异。
- success、nDTW、sDTW、碰撞、步数、延迟等指标差异。
- 哪些 episode 变好，哪些变差。
- 行为标签增减。
- 两次运行的 episode 覆盖是否一致。

### 中断续跑

配置项：

```yaml
cloud_resume: true
```

开启后，程序会扫描：

```text
../DATA/output/{name}/eval/intermediate_results_every/
```

如果某个 episode 已经有结果，则跳过该 episode。适合云端请求中断、仿真服务中断后继续评测。

### 调试输入保存

默认不保存每步完整输入，避免磁盘快速膨胀。如需复盘 prompt、RGB、深度图和请求 JSON，可开启：

```yaml
cloud_save_input_images: true
cloud_save_request_json: true
cloud_save_prompts: true
```

输出位于：

```text
../DATA/output/{name}/eval/cloud_inputs/{time}/{episode_id}/
```

注意：开启后会明显增加磁盘占用。

### 配置校验

评测启动时会自动检查配置，包括：

- `cloud_model`、`cloud_base_url` 是否为空。
- API key 是否来自配置或环境变量。
- prompt 文件是否存在。
- `cloud_depth_mode` 是否合法。
- `cloud_depth_grid_size` 是否大于等于 1。
- 深度近远百分位是否合理。
- `EVAL_NUM` 和 `maxAction` 是否合理。
- `cloud_verify_ssl=false` 时给出 warning。
- `batchSize != 1` 时给出 warning。

如果出现严重配置错误，程序会在连接仿真和调用 API 前直接报错。

## 🔧 **使用示例**

导航脚本示例，请参考 [scripts 文件夹](https://github.com/AirVLN/AirVLN/tree/main/scripts)下的文件。

### 本地 checkpoint 评测

当前工作空间中的本地 seq2seq 评测脚本已经适配 `cxj` 环境和 `PYTHONPATH`：

```bash
cd /data/lyj/cxj/AirVLN_ws
bash ./AirVLN/scripts/eval.sh
```

默认使用：

```text
/data/lyj/cxj/AirVLN_ws/DATA/output/AirVLN-seq2seq/train/checkpoint/20251123-101135-680949/ckpt.LAST.pth
```

默认测试配置为：

```text
EVAL_DATASET=val_unseen
EVAL_NUM=-1
LOCAL_EVAL_MAX_ACTION=100
LOCAL_EVAL_BATCH_SIZE=1
LOCAL_EVAL_GPU_DEVICE=2
```

其中 `EVAL_NUM=-1` 表示跑完整个 split；当前默认是全量 `val_unseen`。`LOCAL_EVAL_MAX_ACTION=100` 是为了控制本地 checkpoint 全量评测耗时；如果需要严格恢复原始 AirVLN 默认步长，可以临时设置 `LOCAL_EVAL_MAX_ACTION=500`。

可以通过环境变量临时覆盖：

```bash
LOCAL_EVAL_CKPT_PATH=/path/to/ckpt.499.pth \
LOCAL_EVAL_DATASET=val_unseen \
LOCAL_EVAL_NUM=50 \
LOCAL_EVAL_MAX_ACTION=100 \
LOCAL_EVAL_BATCH_SIZE=1 \
LOCAL_EVAL_GPU_DEVICE=2 \
bash ./AirVLN/scripts/eval.sh
```

脚本不会自动启动 `AirVLNSimulatorServerTool.py`。运行前请确保仿真服务已经在另一个终端启动。

*提示：如果您是第一次使用AirVLN代码，请先通过可视化确认在[AirVLNSimulatorClientTool.py](https://github.com/AirVLN/AirVLN/blob/main/airsim_plugin/AirVLNSimulatorClientTool.py)中函数`_getImages`获取的图像的通道顺序符合预期！*

## 📚 **常见问题**

1. 错误:
    ```
    [Errno 98] Address already in use
    Traceback (most recent call last):
      File "./airsim_plugin/AirVLNSimulatorServerTool.py", line 535, in <module>
        addr, server, thread = serve()
    TypeError: cannot unpack non-iterable NoneType object
    ```
    可能的解决方案：终结端口（默认30000）正在使用的进程或更改端口。

2. 错误:
    ```
    - INFO - _run_command:139 - Failed to open scenes, machine 0: 127.0.0.1:30000
    - ERROR - run:34 - Request timed out
    - ERROR - _changeEnv:397 - Failed to open scenes Failed to open scenes
    ```
    可能的解决方案：
      * 确保可以单独打开`./ENVs`文件夹中的Airsim场景。如果服务器不支持GUI，您可以采用无头模式或虚拟显示。
      * 确保使用了GPU。
      * 尝试减少 batchsize（例如，设置 `--batchSize 1`）。

如果上述方案都无效，您可以[提一个issue](https://github.com/AirVLN/AirVLN/issues)或[通过邮件联系我们](#联系方式).

## 📜 **引用**

如果您在研究中使用了 AerialVLN，请引用以下文献：

```
@inproceedings{liu_2023_AerialVLN,
  title={AerialVLN: Vision-and-language Navigation for UAVs},
  author={Shubo Liu and Hongsheng Zhang and Yuankai Qi and Peng Wang and Yanning Zhang and Qi Wu},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

此外，我们注意到有些学者希望将AerialVLN数据集及其仿真器应用于除VLN以外的其他研究领域，我们欢迎这样的做法！我们同样欢迎您与我们联络告知[我们](#contact)您的拟应用领域。

## ✉️ **联系方式**
如果您有任何问题，请联络： [Shubo LIU](mailto:shubo.liu@mail.nwpu.edu.cn)

## 🥰 **致谢**
* 我们使用了[Habitat](https://github.com/facebookresearch/habitat-lab)的预训练模型. 衷心感谢。

## 终极版本设想

当前云端测评已经具备完整的基础能力：云端多模态模型调用、深度图输入、prompt 外置、模型自检、评测对齐审计、固定 episode 子集、轨迹日志、失败诊断、运行报告和结果对比。后续如果继续向“终极版本”推进，我设想的方向如下。

### 1. 批量消融实验

增加批量实验脚本，自动比较不同输入组合：

- RGB only。
- RGB + pose。
- RGB + depth summary。
- RGB + depth image。
- RGB + depth image + summary。
- RGB + depth + pose。
- RGB + depth + memory。
- RGB + depth + pose + memory。

所有组合使用同一个 `episode_list_evaluated.json`，并自动调用 `compare_cloud_runs.py` 生成总表。这样可以系统回答：深度图、深度摘要、pose、memory 到底分别贡献了多少。

### 2. 空间轨迹诊断

当前失败诊断主要依赖动作序列。终极版本应进一步利用 pose 和每步指标，增加空间行为诊断：

- 净位移距离。
- 实际路径长度 / 净位移比。
- 是否原地绕圈。
- 是否越走越远。
- 每步 `distance_to_goal` 是否下降。
- 碰撞前最近几步动作和深度摘要。
- 是否出现“前方深度很近但仍 MOVE_FORWARD”的 unsafe violation。

这能更准确解释模型失败原因。

### 3. 模型回复质量统计

进一步统计云端模型是否遵守协议：

- 非 JSON 回复率。
- JSON 缺字段率。
- `action_id` 和 `action_name` 不一致率。
- fallback 率。
- memory 为空或过长比例。
- 平均回复长度。
- 平均延迟和 P95 延迟。
- 超时率和重试率。

这样可以区分“模型格式控制差”和“模型导航能力差”这两类问题。

### 4. Prompt 版本实验管理

Prompt 虽然已经外置，但终极版本应进一步管理 prompt 版本：

```text
configs/prompts/
  cloud_vln_depth_v2_system.txt
  cloud_vln_depth_v2_user.txt
  cloud_vln_depth_v3_system.txt
  cloud_vln_depth_v3_user.txt
```

每次运行保存 prompt name、路径、hash 和内容快照。结果对比时直接显示 prompt 差异。

### 5. 单 episode 复盘工具

增加类似下面的工具：

```bash
python src/vlnce_src/inspect_episode.py \
  --trajectory ../DATA/output/.../trajectories/.../{episode_id}.json
```

自动生成单 episode Markdown 报告，展示：

- 自然语言指令。
- 最终指标。
- 每步动作。
- 每步 memory。
- 每步 pose 和距离变化。
- 每步深度风险。
- raw response 摘要。
- 对应 RGB 和深度图路径。

这会让分析典型失败样本更高效。

### 6. 运行索引

每次运行后写入累计索引：

```text
../DATA/output/{name}/eval/runs_index.jsonl
```

记录模型、prompt、dataset、episode 数、success、nDTW、报告路径、manifest 路径、视频目录、轨迹目录。这样长期实验后不用手动翻目录。

### 7. 配置生成器

保留一个 base config，批量实验时自动生成临时配置：

```text
../DATA/output/{name}/eval/generated_configs/
```

避免反复手改 `configs/cloud_eval.yaml`，减少配置污染和误操作。

### 8. 安全约束的 log-only 模式

终极版本可以增加一个只记录、不干预动作的安全监控：

```yaml
cloud_safety_guard_mode: log_only
```

当深度摘要显示前方近障碍，但模型仍输出 `MOVE_FORWARD` 时，不修改动作，只记录：

```text
unsafe_forward_with_near_obstacle
```

这样既能分析安全问题，又不会改变 benchmark 本身的纯粹性。
