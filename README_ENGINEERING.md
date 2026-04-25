# SafeMed-VQA Pro++ 实现执行文档（Implementation Only）

> 本文档只保留项目实现所需的核心信息。
> 不包含可选讨论、背景性论证、模糊建议或无关流程说明。
> Codex 必须严格按本文档执行。

---

## 1. 项目目标

构建一个医学多模态安全系统，基于 `Qwen/Qwen3-VL-8B` 进行 LoRA 微调，并在少量高风险样本上进行 DPO 对齐，使模型能够：

1. 读取医学图像和问题；
2. 先做图像证据审查；
3. 再输出结构化 JSON；
4. 在证据不足时触发安全拒答；
5. 在正常样本上尽量保留回答能力；
6. 输出可用于后处理校准的原始置信度。

---

## 2. 固定技术选型

### 2.1 Student 模型

- `Qwen/Qwen3-VL-8B`

### 2.2 Teacher 模型

- `Qwen/Qwen3-VL-235B-A22B-Instruct`
- 通过 **SiliconFlow OpenAI-compatible API** 调用
- Teacher 仅用于数据蒸馏，不在本地加载

### 2.3 训练方式

- SFT：LoRA 微调
- DPO：只在筛选出的部分高风险样本上进行
- 不做全参数微调

### 2.4 运行环境

- 远程平台：AutoDL
- 训练卡：**单张 RTX PRO 6000 96GB**
- 非训练阶段优先使用无卡模式

---

## 3. 阶段与算力分配

### 无卡阶段

这些阶段不得占用 GPU：

1. 项目目录初始化
2. Python 环境安装
3. 数据下载与整理
4. 图像增强与反事实样本生成
5. SiliconFlow Teacher API 蒸馏
6. 训练数据 JSONL 落盘
7. DPO 候选样本筛选与组织
8. 评估结果统计、画图、汇总

### GPU 阶段（单张 RTX PRO 6000）

这些阶段必须使用 GPU：

1. SFT 冒烟测试
2. 正式 SFT LoRA 微调
3. 使用 SFT 模型生成 DPO rejected 输出
4. 小规模 DPO 训练
5. 必要时的批量测试集推理

---

## 4. 项目目录结构

```text
SafeMed-VQA-ProPP/
├── README_ENGINEERING.md
├── .env
├── .gitignore
├── requirements.txt
├── configs/
│   ├── datagen.yaml
│   ├── sft.yaml
│   ├── dpo.yaml
│   └── eval.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── stress_test/
│   └── dpo_pairs/
├── scripts/
│   ├── 00_prepare_dataset.py
│   ├── 01_datagen_pro_plus.py
│   ├── 02_train_sft.py
│   ├── 03_build_dpo_pairs.py
│   ├── 04_train_dpo.py
│   ├── 05_calibrate_confidence.py
│   ├── 06_eval_safety.py
│   └── 07_run_inference.py
├── src/
│   ├── siliconflow_client.py
│   ├── dataset_utils.py
│   ├── image_augment.py
│   ├── prompt_templates.py
│   ├── json_utils.py
│   ├── metrics.py
│   ├── calibration.py
│   └── model_utils.py
├── outputs/
│   ├── logs/
│   ├── predictions/
│   ├── reports/
│   └── figures/
└── checkpoints/
```

---

## 5. 安全与密钥规则

### 5.1 必须遵守

- API Key 不得硬编码进源码
- API Key 不得写进 Markdown
- API Key 不得提交到 GitHub
- 所有运行时代码必须从项目根目录 `.env` 读取 `SILICONFLOW_API_KEY`

### 5.2 `.env` 文件格式

```env
SILICONFLOW_API_KEY=YOUR_REAL_KEY
```

### 5.3 Python 读取方式

```python
from dotenv import load_dotenv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise RuntimeError("Missing SILICONFLOW_API_KEY in .env")
```

---

## 6. 输出 JSON 协议

模型最终输出必须严格满足下述结构：

```json
{
  "explanation": "先说明图像质量、关键结构是否可见、是否有伪影或证据缺失，再说明判断依据。",
  "raw_confidence": 0.73,
  "decision": "answer",
  "abstain_type": "none",
  "risk_level": "low",
  "answer": "normal chest x-ray"
}
```

### 字段约束

- `explanation`: 字符串，必须先做证据审查，再给出推理依据
- `raw_confidence`: 0 到 1 的浮点数
- `decision`: `answer` 或 `abstain`
- `abstain_type`: `none | visual_insufficiency | region_missing | high_risk_uncertainty`
- `risk_level`: `low | medium | high`
- `answer`: 若 `decision=abstain`，必须输出与 explanation 一致的拒答说明

### 一致性规则

- 如果 `decision=answer`，则 `abstain_type` 必须为 `none`
- 如果 `decision=abstain`，则 `abstain_type` 不能为 `none`
- 不允许 explanation 说“图像看不清”，answer 却给出明确医学结论

---

## 7. 数据构建规则

### 7.1 原始数据集

- 使用 VQA-RAD
- 保留原始 train/test 划分
- 不修改测试集标签定义

### 7.2 训练数据组成

训练集由以下部分构成：

1. 正常样本（原始图像 + 原始问题）
2. 视觉退化样本
3. 关键区域缺失样本
4. 高风险不确定样本

### 7.3 允许的增强方式

#### A. 视觉退化

- 高斯模糊
- 斑点噪声
- 分辨率降低
- 对比度/亮度异常

#### B. 区域缺失

- 随机裁剪
- 局部遮挡
- 中心区域屏蔽
- 边界截断

#### C. 高风险不确定

- 轻中度证据不足
- 不一定强制拒答，由 Teacher 判定

### 7.4 元数据字段

每条样本至少包含：

```json
{
  "sample_id": "...",
  "image_path": "...",
  "question": "...",
  "split": "train",
  "is_counterfactual": true,
  "degradation_type": "gaussian_blur",
  "severity": "high",
  "source_sample_id": "..."
}
```

---

## 8. Teacher API 调用规范

### 8.1 接口地址

- `POST https://api.siliconflow.cn/v1/chat/completions`

### 8.2 固定 Teacher 模型

- `Qwen/Qwen3-VL-235B-A22B-Instruct`

### 8.3 请求格式

使用 OpenAI-compatible chat completions。

```json
{
  "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are a cautious medical vision-language teacher. Always output valid JSON only."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze the image and answer the medical question in strict JSON."},
        {
          "type": "image_url",
          "image_url": {"url": "file:///abs/path/or/http_url"}
        }
      ]
    }
  ],
  "temperature": 0.2,
  "max_tokens": 800,
  "response_format": {"type": "text"}
}
```

### 8.4 Teacher 输出要求

Teacher 必须：

1. 只输出 JSON
2. 不输出 markdown 代码块
3. 不输出额外解释文字
4. explanation 必须先写图像证据审查
5. 当证据不足时优先选择 `abstain`
6. 轻度退化但证据仍足够时允许回答

### 8.5 失败重试规则

若出现以下任一情况，必须自动重试：

1. HTTP 请求失败
2. 响应非 200
3. 返回内容无法解析为 JSON
4. 缺少必要字段
5. explanation / decision / answer 逻辑冲突

建议：

- 最大重试次数：3
- 指数退避：2s / 4s / 8s
- 每次失败记录日志到 `outputs/logs/datagen_errors.jsonl`

---

## 9. 脚本执行顺序

Codex 必须按以下顺序实现和运行：

1. `scripts/00_prepare_dataset.py`
2. `src/dataset_utils.py`
3. `src/image_augment.py`
4. `src/json_utils.py`
5. `src/prompt_templates.py`
6. `src/siliconflow_client.py`
7. `scripts/01_datagen_pro_plus.py`
8. `src/model_utils.py`
9. `scripts/02_train_sft.py`
10. `scripts/03_build_dpo_pairs.py`
11. `scripts/04_train_dpo.py`
12. `src/calibration.py`
13. `src/metrics.py`
14. `scripts/05_calibrate_confidence.py`
15. `scripts/06_eval_safety.py`
16. `scripts/07_run_inference.py`

不得跳过前置模块直接实现后续脚本。

---

## 10. 每个脚本的职责

### 10.1 `scripts/00_prepare_dataset.py`

职责：

- 下载或读取 VQA-RAD
- 整理图像路径和问答字段
- 生成统一中间索引文件
- 保存 train/val/test 清单

产物：

- `data/interim/train_index.jsonl`
- `data/interim/test_index.jsonl`

### 10.2 `scripts/01_datagen_pro_plus.py`

职责：

- 读取原始 train 样本
- 构造反事实增强样本
- 调 Teacher API 生成结构化监督数据
- 做 JSON 校验
- 落盘训练 JSONL

产物：

- `data/processed/train_safemed_pro.jsonl`
- `outputs/logs/datagen_errors.jsonl`

### 10.3 `scripts/02_train_sft.py`

职责：

- 加载 Qwen3-VL-8B
- 构建 LoRA 配置
- 使用 SFTTrainer 训练
- 记录 wandb 日志
- 保存 checkpoint

产物：

- `checkpoints/sft/...`

### 10.4 `scripts/03_build_dpo_pairs.py`

职责：

- 加载 SFT 模型
- 在压力样本上生成输出
- 找出高风险 hallucination 样本
- 构造 chosen / rejected 对

产物：

- `data/dpo_pairs/dpo_pairs.jsonl`

### 10.5 `scripts/04_train_dpo.py`

职责：

- 加载 SFT checkpoint
- 读取 DPO pairs
- 做小规模 DPO 训练
- 保存 DPO checkpoint

产物：

- `checkpoints/dpo/...`

### 10.6 `scripts/05_calibrate_confidence.py`

职责：

- 在验证集上拟合置信度映射
- 保存 calibration 参数

产物：

- `outputs/reports/calibration.json`

### 10.7 `scripts/06_eval_safety.py`

职责：

- 读取模型推理结果
- 计算核心指标
- 输出报告与图表

产物：

- `outputs/reports/eval_metrics.json`
- `outputs/figures/*.png`

### 10.8 `scripts/07_run_inference.py`

职责：

- 加载最终模型
- 对输入图像和问题进行推理
- 输出统一 JSON
- 可选应用 calibration

---

## 11. `src/` 模块要求

### `src/siliconflow_client.py`
必须实现：

- 读取 `.env`
- 构造 SiliconFlow 请求
- 支持 image_url 多模态消息
- 支持重试
- 返回解析后的文本或 JSON

### `src/dataset_utils.py`
必须实现：

- VQA-RAD 读取
- 索引构建
- 路径标准化
- train/test 划分加载

### `src/image_augment.py`
必须实现：

- blur/noise/resize/contrast augmentation
- crop/occlusion/masking augmentation
- 记录增强元数据

### `src/prompt_templates.py`
必须实现：

- 正常样本 Teacher prompt
- 退化样本 Teacher prompt
- 图问错配样本 Teacher prompt
- 高风险不确定样本 Teacher prompt

### `src/json_utils.py`
必须实现：

- JSON 提取
- 字段校验
- 类型校验
- 一致性校验
- 非法样本过滤

### `src/model_utils.py`
必须实现：

- 模型加载
- processor/tokenizer 加载
- LoRA 配置创建
- 推理辅助函数

### `src/calibration.py`
必须实现：

- raw confidence 收集
- calibration 拟合
- confidence 映射

### `src/metrics.py`
必须实现：

- Accuracy@Answer
- ASR
- FDR
- ECE
- Coverage
- Selective Risk
- AURC
- JSON Validity
- Consistency Error

---

## 12. 配置文件要求

### `configs/datagen.yaml`
至少包含：

- 数据路径
- 增强比例
- 增强类型开关
- Teacher model name
- API timeout
- retry count
- output path

### `configs/sft.yaml`
至少包含：

- model name
- LoRA rank / alpha / dropout
- batch size
- grad accumulation
- learning rate
- epochs
- max length
- save steps

### `configs/dpo.yaml`
至少包含：

- checkpoint path
- pair path
- batch size
- beta
- lr
- epochs

### `configs/eval.yaml`
至少包含：

- test file path
- prediction path
- calibration file path
- metric output path

---

## 13. 评估指标

必须实现以下指标：

1. `Accuracy@Answer`
2. `ASR`
3. `FDR`
4. `ECE`
5. `Coverage`
6. `Selective Risk`
7. `AURC`
8. `JSON Validity`
9. `Consistency Error`

### 定义要求

- `ASR`: 在应拒答样本中成功拒答的比例
- `FDR`: 在正常可答样本中错误拒答的比例
- `Coverage`: 模型选择回答的比例
- `JSON Validity`: 返回合法 JSON 的比例
- `Consistency Error`: explanation/decision/answer 冲突比例

---

## 14. Codex 执行约束

Codex 必须遵守：

1. 先实现基础模块，再实现训练脚本
2. 不得硬编码 API Key
3. 不得修改输出 JSON 协议
4. 不得删除核心指标
5. 不得跳过数据校验
6. 不得把所有异常样本一律标成 abstain，必须保留部分轻度退化可答样本
7. 代码必须模块化
8. 每个脚本必须可单独运行
9. 所有路径必须可通过配置文件调整
10. 所有中间产物必须落盘，不能只存在内存中

---

## 15. 第一阶段执行目标

在交给 Codex 后，第一阶段只要求完成以下内容：

1. 项目目录与配置文件骨架
2. `.env` 读取
3. VQA-RAD 数据准备
4. 图像增强模块
5. SiliconFlow Teacher 客户端
6. Teacher 蒸馏脚本
7. 训练数据 JSONL 成功生成

只有在这部分完成后，才进入 SFT。

---

## 16. 第二阶段执行目标

1. LoRA-SFT 成功启动并保存 checkpoint
2. 能用 SFT 模型跑验证样本
3. 能生成 DPO pairs
4. 能完成小规模 DPO
5. 能输出评估结果

---

## 17. 最终验收标准

项目完成时必须满足：

1. 能从 `.env` 成功读取 API Key
2. 能生成 `train_safemed_pro.jsonl`
3. 能跑通 LoRA SFT
4. 能生成 DPO pair
5. 能完成小规模 DPO
6. 能输出统一 JSON 推理结果
7. 能计算所有核心安全指标
8. 整个流程可在 AutoDL 上复现

---

## 18. 启动前人工已完成事项

在 Codex 开始执行前，人工已经完成：

1. AutoDL 实例创建
2. Python 环境初始化
3. `.env` 已放入项目根目录
4. Git 仓库已初始化
5. 本项目文档已放在根目录，文件名为 `README_ENGINEERING.md`

Codex 不需要再解释这些前置步骤，只需要开始实现项目代码。
