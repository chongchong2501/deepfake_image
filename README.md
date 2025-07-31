# 深度伪造图像检测项目

基于EfficientNet-B0的深度伪造图像检测系统，支持Kaggle双T4 GPU训练。

## 📋 项目概述

本项目使用深度学习技术来检测图像是否为深度伪造（Deepfake）图像。通过EfficientNet-B0预训练模型进行迁移学习，实现对真实图像和伪造图像的二分类。

## 🚀 主要特性

- ✅ **多GPU支持**: 自动检测并支持Kaggle双T4 GPU训练
- ✅ **高效模型**: 基于EfficientNet-B0的轻量级高性能模型
- ✅ **数据增强**: 包含多种数据增强技术提升模型泛化能力
- ✅ **早停机制**: 防止过拟合的早停策略
- ✅ **实时监控**: GPU使用情况和训练进度实时监控
- ✅ **可视化**: 训练曲线和混淆矩阵可视化

## 📁 项目结构

```
deepfake_image/
├── deepfake_image.py      # 主训练脚本
├── deepfake-image.ipynb   # Jupyter Notebook版本
└── README.md              # 项目说明文档
```

## 🛠️ 环境要求

### Python依赖
```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

### 硬件要求
- **推荐**: Kaggle双T4 GPU (32GB总显存)
- **最低**: 单GPU (8GB+ 显存)
- **CPU**: 多核处理器
- **内存**: 16GB+ RAM

## 📊 数据集格式

项目期望的数据集结构：
```
/kaggle/input/deepfake-and-real-images/Dataset/Train/
├── Real/          # 真实图像文件夹
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Fake/          # 伪造图像文件夹
    ├── fake1.jpg
    ├── fake2.jpg
    └── ...
```

## 🔧 配置参数

### 训练参数
- **图像尺寸**: 256×256
- **学习率**: 1e-4
- **训练轮数**: 30 epochs
- **早停耐心**: 5 epochs

### 多GPU配置
- **单GPU**: Batch Size = 32, Workers = 4
- **双GPU**: Batch Size = 64, Workers = 8

## 🚀 使用方法

### 1. 在Kaggle环境中运行

```python
# 直接运行主脚本
python deepfake_image.py
```

### 2. 自定义数据路径

修改脚本中的 `BASE_PATH` 变量：
```python
BASE_PATH = '/path/to/your/dataset/Train/'
```

### 3. 调整训练参数

根据需要修改以下参数：
```python
IMG_SIZE = 256          # 图像尺寸
LEARNING_RATE = 1e-4    # 学习率
EPOCHS = 30             # 训练轮数
```

## 📈 训练过程

### 自动GPU检测
程序会自动检测可用GPU数量并进行相应配置：
```
可用GPU数量: 2
使用多GPU训练: ['Tesla T4', 'Tesla T4']
Batch Size: 64
Num Workers: 8
```

### 训练监控
- 实时显示训练和验证损失
- GPU内存使用情况监控
- 每5个epoch显示详细GPU状态

### 模型保存
- 自动保存验证损失最低的模型为 `best_model.pth`
- 支持多GPU模型的正确保存和加载

## 📊 结果可视化

训练完成后自动生成：

1. **训练曲线图**
   - 训练损失 vs 验证损失
   - 验证准确率变化

2. **混淆矩阵**
   - 真实 vs 预测标签的混淆矩阵热图

3. **分类报告**
   - 精确率、召回率、F1分数等详细指标

## 🔍 性能优化

### 数据加载优化
- `pin_memory=True`: 加速CPU到GPU数据传输
- `non_blocking=True`: 异步数据传输
- 多线程数据加载

### 多GPU优化
- `nn.DataParallel`: 自动数据并行
- 动态batch size调整
- 负载均衡


## 📝 更新日志

### v2.0 (最新)
- ✅ 添加完整的多GPU支持
- ✅ 优化数据加载性能
- ✅ 添加GPU使用监控
- ✅ 改进模型保存机制

### v1.0
- ✅ 基础的深度伪造检测功能
- ✅ EfficientNet-B0模型
- ✅ 数据增强和早停

## 📄 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 本项目仅用于学术研究和教育目的，请勿用于恶意用途。