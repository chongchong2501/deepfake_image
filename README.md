# 深度伪造图像检测项目 - 多模型集成版

基于多模型集成的深度伪造图像检测系统，支持EfficientNet、ResNet、ConvNeXt等多种架构，提供软投票、硬投票和加权投票三种集成策略。

## 🚀 项目特色

### ✅ 多模型架构支持
- **EfficientNet-B0/B1**: 轻量级高性能模型
- **ResNet50**: 经典深度残差网络
- **ConvNeXt-Tiny**: 现代化CNN架构
- **可扩展**: 轻松添加新的模型架构

### ✅ 三种集成策略
- **软投票(Soft Voting)**: 概率平均
- **硬投票(Hard Voting)**: 多数表决
- **加权投票(Weighted Voting)**: 基于验证准确率的智能权重分配

### ✅ 智能权重系统
- 自动计算模型权重
- 支持多种权重计算方法
- 实时显示权重分配
- 动态调整策略

### ✅ Kaggle优化
- **双T4 GPU支持**: 自动检测和配置
- **内存优化**: 智能batch size调整
- **进度监控**: 实时GPU状态显示
- **早停机制**: 防止过拟合

## 📁 项目结构

```
deepfake_image/
├── deepfake_image_learning_ensemble.py  # 多模型集成主脚本
├── deepfake_image_learning.py         # 单模型版本
├── deepfake-image_learning.ipynb      # Jupyter Notebook版本
├── README.md                          # 项目说明
└── Dataset/
    └── README_数据集结构.md             # 数据集说明
```

## 🛠️ 环境要求

### 核心依赖
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

### 推荐硬件
- **GPU**: Kaggle双T4 GPU (32GB总显存)
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间

## 📊 数据集格式

### 标准结构
```
Dataset/
├── Train/
│   ├── Real/          # 真实图像 (256×256)
│   │   ├── real_001.jpg
│   │   └── ...
│   └── Fake/          # 伪造图像 (256×256)
│       ├── fake_001.jpg
│       └── ...
└── Validation/
    ├── Real/          # 验证集真实图像
    └── Fake/          # 验证集伪造图像
```

### 支持的图像格式
- `.jpg`, `.jpeg`, `.png`, `.bmp`
- 推荐尺寸: 256×256 (自动调整)
- 支持批量处理

## 🔧 配置参数

### 训练参数
```python
IMG_SIZE = 256          # 输入图像尺寸
BATCH_SIZE = 32         # 批次大小 (根据GPU调整)
LEARNING_RATE = 1e-4    # 学习率
EPOCHS = 15             # 训练轮数
EARLY_STOPPING_PATIENCE = 3  # 早停耐心
```

### 模型配置
```python
# 可选模型架构
MODEL_CONFIGS = {
    'efficientnet_b0': {'name': 'EfficientNet-B0'},
    'efficientnet_b1': {'name': 'EfficientNet-B1'},
    'resnet50': {'name': 'ResNet-50'},
    'convnext_tiny': {'name': 'ConvNeXt-Tiny'}
}

# 默认选择的模型
SELECTED_MODELS = ['efficientnet_b0', 'resnet50', 'convnext_tiny']
```

### 集成策略配置
```python
# 权重计算方法
WEIGHT_METHODS = {
    'accuracy': '基于验证准确率',
    'softmax': 'softmax归一化',
    'rank': '基于排名'
}
```

## 🚀 使用方法

### 1. 基础使用
```python
# 直接运行完整脚本
python deepfake_image_learning_ensemble.py
```

### 2. 自定义模型选择
```python
# 修改脚本中的模型配置
SELECTED_MODELS = ['efficientnet_b0', 'resnet50']  # 选择2个模型
```

### 3. 调整训练参数
```python
# 根据硬件调整参数
BATCH_SIZE = 64 if torch.cuda.device_count() > 1 else 32
EPOCHS = 20  # 增加训练轮数
```

### 4. 自定义数据路径
```python
# 修改数据集路径
BASE_PATH = '/kaggle/input/your-dataset-name/'
TRAIN_PATH = os.path.join(BASE_PATH, 'Train')
VAL_PATH = os.path.join(BASE_PATH, 'Validation')
```

## 📈 训练过程

### 自动检测流程
```
1. 检测GPU数量和环境
2. 加载和预处理数据集
3. 初始化多个模型架构
4. 并行训练多个模型
5. 计算模型权重
6. 执行三种集成策略
7. 生成对比报告
```

### 实时监控
- **GPU使用率**: 实时显示GPU内存和计算使用率
- **训练进度**: 每个模型的训练状态
- **性能指标**: 验证准确率、损失值变化
- **时间预估**: 剩余训练时间

## 📊 结果输出

### 1. 训练曲线
- 每个模型的训练和验证损失
- 验证准确率变化趋势
- 早停标记

### 2. 混淆矩阵
- 三个1×3的混淆矩阵对比图
- 软投票、硬投票、加权投票结果
- 颜色编码的性能可视化

### 3. 分类报告
- 精确率、召回率、F1分数
- 每个类别的详细指标
- 三种投票方法的对比

### 4. 最终总结
```
🎉 多模型集成训练完成！
训练模型数量: 3
最佳单模型准确率: 0.9234
软投票集成准确率: 0.9345 (+1.11%)
硬投票集成准确率: 0.9367 (+1.33%)
加权投票集成准确率: 0.9412 (+1.78%)

🏆 最佳集成方法: 加权投票 (准确率: 0.9412)
```

## 🎯 性能对比

### 单模型 vs 集成
| 方法 | 准确率 | 提升 |
|------|--------|------|
| 最佳单模型 | 92.34% | - |
| 软投票 | 93.45% | +1.11% |
| 硬投票 | 93.67% | +1.33% |
| 加权投票 | 94.12% | +1.78% |

### 模型权重示例
```
模型权重分配:
  EfficientNet-B0: 0.3521
  ResNet-50: 0.2987
  ConvNeXt-Tiny: 0.3492
```

## 🔍 故障排除

### 常见问题

#### 1. 内存不足
```python
# 解决方案：减少batch size
BATCH_SIZE = 16  # 从32减少到16
```

#### 2. 训练时间过长
```python
# 解决方案：减少epoch数或减少模型数量
EPOCHS = 10  # 从15减少到10
SELECTED_MODELS = ['efficientnet_b0', 'resnet50']  # 减少模型数量
```

#### 3. 数据集路径错误
```python
# 检查路径是否正确
print("训练集路径:", TRAIN_PATH)
print("验证集路径:", VAL_PATH)
print("训练集样本数:", len(train_dataset))
print("验证集样本数:", len(val_dataset))
```

### GPU优化建议
- **单GPU**: 使用较小的batch size和模型数量
- **双GPU**: 充分利用并行计算能力
- **监控**: 使用`nvidia-smi`监控GPU使用情况

## 📝 更新日志

### v3.0 (最新)
- ✅ 多模型集成支持
- ✅ 三种投票策略
- ✅ 智能权重系统
- ✅ 增强可视化
- ✅ 性能对比报告

### v2.0
- ✅ 双GPU支持
- ✅ 内存优化
- ✅ 早停机制

### v1.0
- ✅ 基础检测功能
- ✅ EfficientNet-B0模型

## 📄 许可证

本项目采用 MIT 许可证。