# 深度伪造图像检测（多模型集成）

基于多架构卷积网络的多模型集成深伪图像检测系统，内置软投票、硬投票与加权投票三种策略，支持 AMP 混合精度与 Captum 可解释性分析，可在本地（Windows 单 GPU）与 Kaggle（双 T4 GPU）稳定运行。

## 功能特性
- 多模型：EfficientNet、ResNet、ConvNeXt 等，可灵活扩展
- 三种集成：Soft/Hard/Weighted Voting，权重可按验证集性能自动计算
- 可解释性：集成 Captum（Integrated Gradients、Grad-CAM）可视化关键区域
- 性能优化：AMP 混合精度、早停、可选学习率调度、显存清理
- 跨环境：本地脚本/Notebook 与 Kaggle Notebook 双方案

## 目录结构（简要）
- deepfake-image-learning-ensemble_Laptop.ipynb（本地 Notebook）
- deepfake-image-learning-ensemble_kaggle.ipynb（Kaggle Notebook）
- deepfake-image_learning.ipynb（单模型示例 Notebook）
- deepfake_image_learning_ensemble_local.py（本地脚本入口）
- works/models/（训练产出模型存放）
- 历史训练数据/（历史实验与模型归档）

## 环境要求
- Python ≥ 3.8，PyTorch ≥ 1.10，torchvision ≥ 0.11
- 依赖：opencv-python、numpy、pandas、scikit-learn、matplotlib、seaborn、tqdm、captum（可选）
- 硬件：建议 8GB+ 显存（Kaggle 双 T4 更佳）

## 数据集组织（示例）
Dataset/
├─ Train/
│  ├─ Real/
│  └─ Fake/
└─ Validation/
   ├─ Real/
   └─ Fake/

支持 .jpg/.jpeg/.png/.bmp；常用尺寸 256×256，代码会自动 Resize。

## 快速开始
1) 本地脚本（推荐在 Windows 使用脚本以便稳定启用多进程 DataLoader）
- 配置数据路径、训练参数后运行：
  python deepfake_image_learning_ensemble_local.py
- 产物（权重、曲线、图表）默认保存到 works/ 下。

2) Notebook 模式
- 本地：打开 deepfake-image-learning-ensemble_Laptop.ipynb，逐 Cell 运行
- Kaggle：打开 deepfake-image-learning-ensemble_kaggle.ipynb，按提示设置 BASE_PATH 与参数后运行

## 关键参数（常用）
- IMG_SIZE：输入图像尺寸（如 256）
- BATCH_SIZE/VAL_BATCH_SIZE：批大小（按显存调整）
- EPOCHS：训练轮数
- NUM_WORKERS：DataLoader 进程数
  - Windows+Notebook 环境建议设 0；若需 >0，推荐改用 .py 脚本运行
  - Kaggle/类 Linux 环境可设 4~8 以提速
- USE_AMP：是否启用混合精度（GPU 推荐开启）

## 训练与评估
- 单模型训练：记录训练/验证损失与准确率，支持早停与最佳权重保存
- 集成推理：软投票、硬投票、加权投票（基于各模型验证性能计算权重）
- 可视化：训练曲线、混淆矩阵、分类报告、F1 对比、预测置信度分布、Captum 热力图

## 常见问题与建议
- Windows 下 Notebook + num_workers>0 可能与 CUDA/多进程冲突导致不稳定；若遇到卡死/显存异常：
  1) 将 NUM_WORKERS 设为 0；或
  2) 使用 deepfake_image_learning_ensemble_local.py 以脚本方式运行再启用多进程
- 显存不足：下调 BATCH_SIZE 或关闭部分数据增强；确保每个 epoch/训练结束已执行显存清理
- 训练慢：开启 USE_AMP；在 Kaggle 将 NUM_WORKERS 提升至 4~8；适度增大 BATCH_SIZE

## 结果示例（示意）
- 最佳单模型准确率：~92%+
- 集成（加权投票）准确率：~94% 左右（具体以数据与配置为准）

## 许可证
本项目采用 MIT 许可证。