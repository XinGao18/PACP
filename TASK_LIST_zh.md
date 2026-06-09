# PCPL-VAD 代码任务列表

> 项目：Prompt-aligned Class Prototype Learning for Weakly Supervised Fine-grained Video Anomaly Detection  
> 创建日期：2026-04-23  
> 修订依据：`method.md`（2026-06-08）

---

## 实现阶段与依赖关系

```
阶段一（基础骨架）：      模块一 → 模块二
阶段二（核心模型）：      模块二 → 模块三 → 模块四
阶段三（训练与评估）：    模块四 → 模块五 → 模块六
阶段四（可选分析）：      模块六 → 模块七
全程并行：                模块八（实验管理与可复现性）
```

---

## 模块一：项目结构与配置

**目标**：建立清晰的项目骨架和统一配置体系，使配置能够覆盖弱监督细粒度类别、CLIP 冻结编码器、prompt bank、class prototype、top-k MIL 与损失权重。

### 子任务

- [&#10004;] 初始化项目目录：`configs/`、`data/`、`models/`、`losses/`、`trainers/`、`evaluators/`、`utils/`、`scripts/`、`experiments/`
- [&#10004;] 创建 `README.md`、`.gitignore`、`requirements.txt`、`setup.py`
- [&#10004;] 更新 YAML 配置：数据路径、官方 split、类别集合、CLIP 版本、`num_segments=T`、prompt context length、`top_k`、`alpha`、`tau_p/tau_e`、`lambda_align/lambda_sep`
- [&#10004;] 实现配置加载器（argparse + YAML merge，支持配置继承）
- [&#10004;] 更新 `requirements.txt`：保留 torch、torchvision、clip、einops、scikit-learn、tensorboard、wandb、decord、h5py；移除主线不需要的 diffusers 依赖

### 建议文件

```
configs/
  base.yaml
  ucf_crime.yaml
  xd_violence.yaml
utils/
  config.py        # class Config, def load_config()
  logger.py
```

---

## 模块二：数据处理模块

**目标**：实现弱监督细粒度视频异常检测的数据加载，只使用视频级类别标签 `y ∈ {0, 1, ..., C}`，其中 `0` 表示 normal，`1...C` 表示细粒度异常类别；训练阶段不使用 frame-level / segment-level temporal annotation。

### 子任务

- [&#10004;] 定义 `BaseVideoDataset` 抽象基类，规范 `__getitem__` 返回格式
- [&#10004;] 统一数据返回格式：`frames_or_features`、`video_label`、`class_name`、`video_id`、`num_frames`，并确保 normal 类索引为 `0`
- [&#10004;] 实现或调整 `UCFCrimeDataset`：读取官方训练/测试划分，支持视频级 normal/abnormal 与细粒度类别标签
- [&#10004;] 实现或调整 `XDViolenceDataset`：读取官方训练/测试划分，支持多类异常类别映射
- [&#10004;] 实现 `T` 个代表帧或 snippet 的均匀采样逻辑，保证训练和推理使用一致采样协议
- [&#10004;] 实现数据增强与 CLIP 标准预处理 pipeline（224×224，ImageNet 归一化）
- [&#10004;] 调整离线特征提取脚本：可输出冻结 CLIP appearance features；运动特征由模块三在 CLIP feature space 中计算，不再在数据层计算 raw frame TD

### 建议文件

```
data/
  datasets/
    base_dataset.py      # class BaseVideoDataset
    ucf_crime.py         # class UCFCrimeDataset
    xd_violence.py       # class XDViolenceDataset
  transforms.py          # def get_clip_transform()
  data_factory.py        # def build_dataloader()
scripts/
  extract_features.py    # 冻结 CLIP appearance feature 提取入口
```

---

## 模块三：外观-运动特征编码与时序增强模块

**目标**：使用冻结 CLIP image encoder 提取外观特征，在 CLIP feature space 中通过相邻特征差分构造运动线索，再通过投影层与轻量时序头得到 segment-level enhanced visual representation `z_t`。

**输入**：均匀采样得到的 `T` 个 frame/snippet 或预提取 CLIP appearance features。  
**输出**：外观特征 `a_t`、feature-level motion features `m_t`、融合特征 `u_t`、时序增强表示 `z_t`。

### 子任务

- [&#10004;] 封装冻结的 `CLIPImageEncoder`，输出 L2 归一化 appearance feature：`a_t = Norm(E_I(x_t))`
- [&#10004;] 实现 feature-level temporal difference：`m_t = Norm(a_t - a_{t-1})`，首段 `m_1` 使用零向量或首个有效差分
- [&#10004;] 实现外观-运动拼接与投影：`u_t = phi_f([a_t; m_t])`
- [&#10004;] 实现轻量 temporal enhancement head：`z_{1:T} = H_theta(u_{1:T})`，默认使用 temporal convolutional head，保留 temporal Transformer 作为可选配置
- [&#10004;] 保证输出维度与 prompt embedding / class prototype 所在空间一致，并对相似度计算前的向量执行 L2 归一化

### 建议文件

```
models/
  encoders/
    clip_encoder.py          # class CLIPImageEncoder
    appearance_motion.py     # class AppearanceMotionEncoder
  temporal/
    temporal_head.py         # class TemporalEnhancementHead
  pcpl_model.py              # class PCPLModel
```

---

## 模块四：可学习语义提示库与 Prompt-Aligned Class Prototype 模块

**目标**：为 normal 与每个细粒度异常类别构造一个可学习 text prompt，并学习一个对应的 visual class prototype；通过 prompt branch 与 prototype branch 的相似度共同产生 segment-level class logits。

**输入**：模块三输出的 `z_t`、类别词表 `{normal, class_1, ..., class_C}`。  
**输出**：prompt embeddings `e_c`、visual prototypes `p_c`、segment-level class logits `s_{t,c}`、segment-level class probabilities `P_{t,c}`。

### 子任务

- [&#10004;] 建立类别词表，包含 normal 类与所有细粒度异常类别，并保证类别索引与数据标签一致
- [&#10004;] 实现 learnable semantic prompt bank：`P_c = [v_1]...[v_M] a surveillance video of [CLASS_c]`
- [&#10004;] 使用冻结 CLIP text encoder 编码 prompt：`e_c = Norm(E_T(P_c))`，只训练 shared context tokens
- [&#10004;] 为每个类别学习一个 visual class prototype `p_c ∈ R^d`，包括 normal 类
- [&#10004;] 实现 prototype-based similarity：`s^p_{t,c} = tau_p · cos(z_t, p_c)`
- [&#10004;] 实现 prompt-based similarity：`s^e_{t,c} = tau_e · cos(z_t, e_c)`
- [&#10004;] 实现最终 logits：`s_{t,c} = s^p_{t,c} + alpha · s^e_{t,c}`，并输出 softmax 概率 `P_{t,c}`

### 建议文件

```
models/
  prompts/
    prompt_bank.py           # class LearnablePromptBank
  prototypes/
    class_prototype.py       # class PromptAlignedClassPrototype
    class_logits.py          # similarity and logit fusion
```

---

## 模块五：Top-k MIL 与 Prompt-Prototype 损失模块

**目标**：在只有视频级类别标签的弱监督条件下，将 segment-level class logits 通过 per-class top-k pooling 聚合为 video-level class scores，并联合优化分类、prompt-prototype alignment 与 prototype separation。

**输入**：segment-level logits `s_{t,c}`、prompt embeddings `e_c`、visual prototypes `p_c`、视频级类别标签 `y`。  
**输出**：video-level scores `S_c`、video-level probabilities `P_c^video`、总损失 `L`。

### 子任务

- [&#10004;] 实现 per-class top-k MIL：对每个类别 `c` 选取 `{s_{t,c}}` 的 top-k segment logits
- [&#10004;] 实现 video-level score：`S_c = (1/k) * sum_{t in TopK_c} s_{t,c}`
- [&#10004;] 实现细粒度视频级分类损失：`L_cls = -log P_y^video`
- [&#10004;] 实现 prompt-prototype alignment loss：`L_align = mean_c(1 - cos(p_c, e_c))`
- [&#10004;] 实现 prototype separation loss：惩罚不同类别 prototype 的余弦相似度超过 margin `delta`
- [&#10004;] 实现总损失：`L = L_cls + lambda_align * L_align + lambda_sep * L_sep`
- [&#10004;] 训练阶段只更新 projection layer、temporal enhancement head、learnable context tokens 和 class prototypes；CLIP image/text encoder 均保持冻结

### 建议文件

```
losses/
  topk_mil_loss.py       # top-k video-level classification loss
  prototype_loss.py      # alignment and separation losses
  total_loss.py          # total objective
scripts/
  train.py               # 训练入口
```

---

## 模块六：推理、定位与评估模块

**目标**：使用同一模型同时输出视频级细粒度异常类别和 segment-level anomaly score，不额外引入 detection head，并按官方协议评估分类与定位性能。

**输入**：segment-level class probabilities `P_{t,c}`、video-level scores `S_c`、视频帧数、GT 标注。  
**输出**：预测类别 `y_hat`、segment/frame anomaly score、AUC、AP、accuracy、macro-F1 或 mAP。

### 子任务

- [&#10004;] 实现视频级类别预测：`y_hat = argmax_c P_c^video`
- [&#10004;] 实现 anomaly score 方案 A：`A_t = 1 - P_{t,0}`
- [&#10004;] 实现 anomaly score 方案 B：`A_t = max_{c=1...C} P_{t,c}`，作为可配置备选
- [&#10004;] 将 segment-level anomaly score 插值到 frame level，并支持可选 temporal smoothing
- [&#10004;] UCF-Crime 默认输出 AUC，XD-Violence 默认输出 AP
- [&#10004;] 细粒度类别评估输出 accuracy、macro-F1 或 mAP，并保存每个视频的预测类别与概率

### 建议文件

```
evaluators/
  evaluator.py
  metrics.py
  visualizer.py
scripts/
  evaluate.py
```

---

## 模块七：原型诊断与消融分析模块（可选）

**目标**：围绕 prompt-aligned class prototype 主线进行可解释诊断与消融实验，分析 prompt branch、prototype branch、top-k MIL 与 temporal head 对性能的影响；该模块不改变主线训练/推理定义。

### 子任务

- [ ] 可视化 prompt-prototype cosine matrix，检查每个 visual prototype 是否贴近对应 semantic prompt embedding
- [ ] 可视化 prototype separation matrix，检查不同类别 prototype 是否满足 margin 约束
- [ ] 输出每个视频 top-k segments 的 class probabilities，用于分析弱监督定位是否聚焦异常片段
- [ ] 设计消融实验：关闭 prompt branch（`alpha=0`）、关闭 prototype branch、不同 `top_k`、不同 temporal head、不同 `lambda_align/lambda_sep`
- [ ] 导出实验汇总 CSV，并保留主要指标、类别指标和 prototype 诊断指标
- [ ] 不将 VLM、反事实生成、扩散模型或证据线索库作为当前 `method.md` 主线模块

### 建议文件

```
evaluators/
  prototype_analysis.py
scripts/
  analyze_prototypes.py
  run_ablation.sh
```

---

## 模块八：实验管理与可复现性模块（全程并行）

**目标**：统一管理训练日志、实验目录、配置快照、prompt bank、class prototypes、checkpoint、指标文件和随机种子，保证弱监督细粒度分类与定位结果可复现。

### 子任务

- [ ] 实现 `Logger`（TensorBoard / WandB 可选）
  - 记录：`L_cls`、`L_align`、`L_sep`、total loss、AUC/AP、accuracy、macro-F1/mAP、LR、top-k segment score、prompt-prototype cosine、prototype separation statistics
- [ ] 实现实验目录自动管理：`experiments/{dataset}_{timestamp}_{run_name}/`
  - 自动保存：配置快照、类别词表、prompt context tokens、class prototypes、best/latest checkpoint、评估结果 JSON/CSV
- [ ] 实现可复现性工具：`set_seed()`（torch + numpy + random 三端固定）
- [ ] 实现 `count_parameters()`、`print_model_summary()`，区分 frozen parameters 与 trainable parameters
- [ ] 记录代码版本信息；若当前目录不是 git repository，则在日志中标记 commit hash 为 unavailable

### 建议文件

```
utils/
  logger.py
  model_utils.py         # def count_parameters(), def set_seed()
  checkpoint.py          # class CheckpointManager
scripts/
  run_ablation.sh
```

---

## 关键设计决策

| 决策点 | 推荐方案 | 备选方案 |
|--------|----------|----------|
| 监督形式 | 仅视频级类别标签 `y ∈ {0,...,C}` | 使用 frame/segment 标注（不符合当前弱监督设定） |
| normal 类索引 | `0` | 其他索引（不推荐） |
| CLIP image encoder | 冻结 | 微调（不作为主线） |
| CLIP text encoder | 冻结 | 微调（不作为主线） |
| 运动线索 | CLIP feature-level temporal difference | optical flow / raw frame TD encoder（不作为主线） |
| 外观-运动融合 | `[a_t; m_t]` concat + trainable projection `phi_f` | learnable weighted sum |
| 时序增强 | lightweight temporal convolutional head | temporal Transformer head |
| prompt 设计 | shared learnable context tokens + class name | 为每类手写多模板 |
| class prototype | 每个类别一个 visual prototype，包含 normal 类 | 固定数量编号原型（不使用） |
| segment logits | `s^p_{t,c} + alpha · s^e_{t,c}` | 单独 prototype 或单独 prompt 分支 |
| MIL 聚合 | per-class top-k pooling | 全视频平均池化 |
| 训练目标 | `L_cls + lambda_align·L_align + lambda_sep·L_sep` | MIL ranking / SupCon / evidence / counterfactual loss（不作为主线） |
| anomaly score | `1 - P_{t,0}` | `max_{c=1...C} P_{t,c}` |
| 可选分析 | prototype/prompt 诊断与消融 | VLM、扩散、反事实生成（不作为当前主线） |

---

## 总体项目结构

```
PCPL/
├── configs/
│   ├── base.yaml
│   ├── ucf_crime.yaml
│   └── xd_violence.yaml
├── data/
│   ├── datasets/
│   │   ├── base_dataset.py
│   │   ├── ucf_crime.py
│   │   └── xd_violence.py
│   ├── transforms.py
│   └── data_factory.py
├── models/
│   ├── encoders/
│   │   ├── clip_encoder.py
│   │   └── appearance_motion.py
│   ├── temporal/
│   │   └── temporal_head.py
│   ├── prompts/
│   │   └── prompt_bank.py
│   ├── prototypes/
│   │   ├── class_prototype.py
│   │   └── class_logits.py
│   └── pcpl_model.py
├── losses/
│   ├── topk_mil_loss.py
│   ├── prototype_loss.py
│   └── total_loss.py
├── trainers/
│   └── trainer.py
├── evaluators/
│   ├── evaluator.py
│   ├── metrics.py
│   ├── visualizer.py
│   └── prototype_analysis.py
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── model_utils.py
│   └── checkpoint.py
├── scripts/
│   ├── extract_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── analyze_prototypes.py
│   └── run_ablation.sh
├── experiments/               # 运行时自动生成，不纳入版本控制
├── method.md
├── Project Description.txt
├── TASK_LIST_zh.md            # 本文件
├── requirements.txt
├── setup.py
└── README.md
```

---

## 方法主线实现优先级

- **P0**：统一官方 split、类别词表、normal 类索引、`num_segments` 与视频级标签读取方式。
- **P1**：实现冻结 CLIP image encoder、feature-level temporal difference、concat projection 与 temporal enhancement head。
- **P2**：实现 learnable semantic prompt bank，并确保 CLIP text encoder 冻结、shared context tokens 可训练。
- **P3**：实现每类一个 visual class prototype，并完成 prototype similarity、prompt similarity 与 logits 融合。
- **P4**：实现 per-class top-k MIL、`L_cls`、`L_align`、`L_sep` 与总损失。
- **P5**：实现视频级细粒度分类、segment/frame anomaly score、AUC/AP 与类别指标。
- **P6**：实现日志、checkpoint、prototype 诊断和消融实验。

---

## 验收标准

1. 训练阶段只使用视频级类别标签，不使用 frame-level 或 segment-level temporal annotation。
2. CLIP image encoder 与 CLIP text encoder 均保持冻结。
3. 运动线索来自 CLIP feature-level temporal difference，不引入 optical flow、额外 motion encoder 或 raw frame TD 主线。
4. 每个类别包含一个 learnable text prompt embedding 和一个 learnable visual class prototype，normal 类也包含在内。
5. Segment-level logits 同时来自 prototype-based similarity 与 prompt-based similarity。
6. Video-level scores 由 per-class top-k segment logits 聚合得到。
7. 总损失仅包含 `L_cls`、`L_align`、`L_sep` 及其权重组合；不包含主线 MIL ranking、SupCon、evidence、counterfactual 或 diffusion loss。
8. 推理阶段通过 `P_{t,c}` 得到视频级类别和 anomaly score，不额外引入 detection head。
9. 评估同时报告细粒度分类指标和异常定位指标：accuracy/macro-F1/mAP、AUC/AP。
10. 模块七只作为诊断与消融分析，不改变主线训练、推理和评估定义。

---

## 旧方案处理结果

- 已删除主线外的 evidence、diffusion、counterfactual、CAP、contrastive、pseudo-label、dual-stream 和 raw-TD 相关旧代码。
- 当前训练和评估脚本只围绕 fine-grained class logits、top-k MIL、prompt-prototype alignment 与 prototype separation 组织。
- 当前模块一至模块六不维护旧 checkpoint、旧特征字段或旧模型入口兼容。
