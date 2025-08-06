# =====================
# Kaggle深度伪造图像检测 - 多模型集成投票版本
# =====================

# Cell 1: 导入依赖和环境设置
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as patches
from tqdm.auto import tqdm
import warnings
import time
from PIL import Image
import gc
warnings.filterwarnings('ignore')

# 解释工具导入
try:
    from captum.attr import LayerGradCam, IntegratedGradients
    from captum.attr import visualization as viz
    CAPTUM_AVAILABLE = True
except ImportError:
    print("⚠️ Captum not available. Install with: pip install captum")
    CAPTUM_AVAILABLE = False

# 设置matplotlib使用英文字体和高DPI
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

print("🚀 Kaggle Multi-Model Ensemble Deepfake Detection")
print(f"PyTorch Version: {torch.__version__}")
print(f"Captum Available: {CAPTUM_AVAILABLE}")

# Cell 2: 参数配置
BASE_PATH = '/kaggle/input/deepfake-and-real-images/Dataset'
TRAIN_PATH = os.path.join(BASE_PATH, 'Train')
VAL_PATH = os.path.join(BASE_PATH, 'Validation')

# 训练参数
# 图像大小
IMG_SIZE = 256

# 训练批次大小
BATCH_SIZE = 32

# 学习率
LEARNING_RATE = 1e-4

# 训练轮数
EPOCHS = 30

# 权重衰减系数
WEIGHT_DECAY = 1e-4

# 早停轮数
PATIENCE = 5

# 数据加载器的工作进程数量
NUM_WORKERS = 4

# 多GPU设置
NUM_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    if NUM_GPUS > 1:
        print(f"Multi-GPU Training: {[torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]}")
        print(f"GPU Count: {NUM_GPUS}")
        NUM_WORKERS = 4  # 多GPU时增加数据加载线程
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        NUM_WORKERS = 2  # 单GPU时减少数据加载线程
    
    for i in range(NUM_GPUS):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
else:
    NUM_WORKERS = 0
    print("Using CPU Training")

# 创建输出目录
PLOTS_DIR = './works/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots will be saved to: {PLOTS_DIR}")



# Cell 3: 数据加载函数
classes = ['Real', 'Fake']

def create_dataframe(data_path, dataset_type):
    """创建数据集DataFrame"""
    filepaths, labels = [], []
    
    for label_idx, cls in enumerate(classes):
        folder = os.path.join(data_path, cls)
        if os.path.exists(folder):
            for img_name in os.listdir(folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(folder, img_name))
                    labels.append(label_idx)
    
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    print(f"{dataset_type}集图片数: {len(df)}")
    if len(df) > 0:
        print(f"{dataset_type}集类别分布:")
        for idx, cls in enumerate(classes):
            count = len(df[df['label'] == idx])
            print(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")
    return df

# Cell 4: 数据预处理和增强
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DeepfakeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        label = self.df.iloc[idx]['label']
        return img, label

# Cell 5: 模型定义
def create_efficientnet_b0():
    """创建EfficientNet-B0模型"""
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(1280, 2)
    return model

def create_resnet18():
    """创建ResNet18模型"""
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, 2)
    return model

def create_convnext_tiny():
    """创建ConvNeXt-Tiny模型"""
    model = models.convnext_tiny(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(768, 2)
    return model

# 模型配置字典
MODEL_CONFIGS = {
    'efficientnet_b0': {
        'create_fn': create_efficientnet_b0,
        'name': 'EfficientNet-B0'
    },
    'resnet18': {
        'create_fn': create_resnet18,
        'name': 'ResNet18'
    },
    'convnext_tiny': {
        'create_fn': create_convnext_tiny,
        'name': 'ConvNeXt-Tiny'
    }
}

# Cell 6: 单模型训练函数
def train_single_model(model_key, train_loader, val_loader, save_path):
    """训练单个模型"""
    print(f"\n🔥 Starting Training {MODEL_CONFIGS[model_key]['name']}")
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 创建模型
    model = MODEL_CONFIGS[model_key]['create_fn']()
    model = model.to(DEVICE)
    
    # 多GPU支持
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
        print(f"✅ Model configured for multi-GPU training with {NUM_GPUS} GPUs")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 训练记录
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_losses, val_accuracies, learning_rates = [], [], [], []
    val_f1_scores = []
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存模型时处理多GPU情况
            if NUM_GPUS > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved, validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⛔ Early stopping triggered")
                break
    
    # 计算训练时间
    training_time = time.time() - start_time
    print(f"⏱️ Training completed in {training_time:.2f} seconds")
    
    return {
         'best_acc': best_val_acc,
         'train_losses': train_losses,
         'val_losses': val_losses,
         'val_accuracies': val_accuracies,
         'val_f1_scores': val_f1_scores,
         'learning_rates': learning_rates,
         'training_time': training_time
     }

# Cell 7: 可视化函数
def plot_training_history(model_results, save_dir=PLOTS_DIR):
    """绘制训练历史可视化"""
    print("📊 Generating training history visualizations...")
    
    # 1. 单模型训练历史 (2x2 子图)
    for model_key, results in model_results.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{MODEL_CONFIGS[model_key]["name"]} Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(results['train_losses']) + 1)
        
        # Loss曲线
        axes[0, 0].plot(epochs, results['train_losses'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, results['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy曲线
        axes[0, 1].plot(epochs, results['val_accuracies'], 'g-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Validation Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate曲线
        axes[1, 0].plot(epochs, results['learning_rates'], 'purple', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation Accuracy分布
        axes[1, 1].hist(results['val_accuracies'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(results['best_acc'], color='red', linestyle='--', linewidth=2, label=f'Best: {results["best_acc"]:.4f}')
        axes[1, 1].set_title('Validation Accuracy Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{model_key}_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
    
    # 2. 多模型对比图 (四线对比)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Model Training Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 验证Loss对比
    for i, (model_key, results) in enumerate(model_results.items()):
        epochs = range(1, len(results['val_losses']) + 1)
        axes[0, 0].plot(epochs, results['val_losses'], color=colors[i % len(colors)], 
                       label=MODEL_CONFIGS[model_key]['name'], linewidth=2)
    axes[0, 0].set_title('Validation Loss Comparison', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 验证Accuracy对比
    for i, (model_key, results) in enumerate(model_results.items()):
        epochs = range(1, len(results['val_accuracies']) + 1)
        axes[0, 1].plot(epochs, results['val_accuracies'], color=colors[i % len(colors)], 
                       label=MODEL_CONFIGS[model_key]['name'], linewidth=2)
    axes[0, 1].set_title('Validation Accuracy Comparison', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 训练时长对比
    model_names = [MODEL_CONFIGS[key]['name'] for key in model_results.keys()]
    training_times = [results['training_time'] for results in model_results.values()]
    bars = axes[1, 0].bar(model_names, training_times, color=colors[:len(model_names)], alpha=0.7)
    axes[1, 0].set_title('Training Time Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score对比
    for i, (model_key, results) in enumerate(model_results.items()):
        epochs = range(1, len(results['val_f1_scores']) + 1)
        axes[1, 1].plot(epochs, results['val_f1_scores'], color=colors[i % len(colors)], 
                       label=MODEL_CONFIGS[model_key]['name'], linewidth=2)
    axes[1, 1].set_title('F1-Score Comparison', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'multi_model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, title, save_name, save_dir=PLOTS_DIR):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    
    # 添加准确率信息
    accuracy = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f}', 
             transform=plt.gca().transAxes, ha='center', fontweight='bold')
    
    save_path = os.path.join(save_dir, f'{save_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_ensemble_analysis(models, val_loader, device, save_dir=PLOTS_DIR):
    """绘制集成分析可视化"""
    print("📊 Generating ensemble analysis visualizations...")
    
    # 收集所有模型的预测概率
    all_probs = []
    all_preds = []
    y_true = []
    
    for model in models:
        model.eval()
        probs = []
        preds = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                prob = F.softmax(outputs, dim=1)
                probs.extend(prob.cpu().numpy())
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                
                if len(y_true) == 0:  # 只在第一个模型时收集真实标签
                    y_true.extend(labels.cpu().numpy())
        
        all_probs.append(np.array(probs))
        all_preds.append(np.array(preds))
    
    all_probs = np.array(all_probs)  # shape: (n_models, n_samples, n_classes)
    all_preds = np.array(all_preds)  # shape: (n_models, n_samples)
    y_true = np.array(y_true)
    
    # 计算集成预测
    ensemble_probs = np.mean(all_probs, axis=0)  # 平均概率
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_confidence = np.max(ensemble_probs, axis=1)
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ensemble Analysis', fontsize=16, fontweight='bold')
    
    # 1. 预测概率直方图
    for i, model_name in enumerate(['EfficientNet-B0', 'ResNet18', 'ConvNeXt-Tiny']):
        if i < len(all_probs):
            fake_probs = all_probs[i][:, 1]  # 假图片的概率
            axes[0, 0].hist(fake_probs, bins=30, alpha=0.6, label=model_name, density=True)
    
    axes[0, 0].set_title('Prediction Probability Distribution (Fake Class)', fontweight='bold')
    axes[0, 0].set_xlabel('Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 模型一致性热图
    n_models = len(all_preds)
    consistency_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            consistency_matrix[i, j] = np.mean(all_preds[i] == all_preds[j])
    
    model_names = ['EfficientNet-B0', 'ResNet18', 'ConvNeXt-Tiny'][:n_models]
    sns.heatmap(consistency_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=model_names, yticklabels=model_names, ax=axes[0, 1])
    axes[0, 1].set_title('Model Prediction Consistency', fontweight='bold')
    
    # 3. 集成置信度对比（正确vs错误预测）
    correct_mask = ensemble_preds == y_true
    correct_confidence = ensemble_confidence[correct_mask]
    incorrect_confidence = ensemble_confidence[~correct_mask]
    
    axes[1, 0].hist(correct_confidence, bins=30, alpha=0.7, label='Correct Predictions', 
                   color='green', density=True)
    axes[1, 0].hist(incorrect_confidence, bins=30, alpha=0.7, label='Incorrect Predictions', 
                   color='red', density=True)
    axes[1, 0].set_title('Ensemble Prediction Confidence', fontweight='bold')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 各类别F1-score柱状图
    class_names = ['Real', 'Fake']
    # 计算每个类别的F1分数
    f1_scores = f1_score(y_true, ensemble_preds, average=None)  # 返回每个类别的F1分数
    
    bars = axes[1, 1].bar(class_names, f1_scores, color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[1, 1].set_title('Per-Class F1-Score', fontweight='bold')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_ylim(0, 1)
    
    # 添加数值标签
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ensemble_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def plot_interpretability_analysis(models, val_loader, device, save_dir=PLOTS_DIR, num_samples=4):
    """绘制模型解释性分析（Grad-CAM + Integrated Gradients）"""
    print("📊 Generating interpretability analysis...")
    
    if not CAPTUM_AVAILABLE:
        print("⚠️ Captum not available, skipping interpretability analysis")
        return
    
    # 获取一些样本进行分析
    sample_images = []
    sample_labels = []
    sample_preds = []
    
    models[0].eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = models[0](inputs)
            preds = torch.argmax(outputs, dim=1)
            
            # 选择一些有趣的样本（预测正确和错误的）
            for i in range(min(num_samples, len(inputs))):
                sample_images.append(inputs[i])
                sample_labels.append(labels[i].item())
                sample_preds.append(preds[i].item())
            
            if len(sample_images) >= num_samples:
                break
    
    # 为每个模型生成解释
    for model_idx, model in enumerate(models):
        model_name = ['EfficientNet-B0', 'ResNet18', 'ConvNeXt-Tiny'][model_idx]
        
        # 创建子图
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{model_name} - Interpretability Analysis', fontsize=16, fontweight='bold')
        
        for sample_idx in range(num_samples):
            input_tensor = sample_images[sample_idx].unsqueeze(0)
            true_label = sample_labels[sample_idx]
            pred_label = sample_preds[sample_idx]
            
            # 原始图像
            img_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # 归一化到[0,1]
            axes[sample_idx, 0].imshow(img_np)
            axes[sample_idx, 0].set_title(f'Original\nTrue: {true_label}, Pred: {pred_label}')
            axes[sample_idx, 0].axis('off')
            
            try:
                # Grad-CAM
                if hasattr(model, 'features'):  # EfficientNet/ResNet
                    target_layer = model.features[-1]
                elif hasattr(model, 'stages'):  # ConvNeXt
                    target_layer = model.stages[-1]
                else:
                    # 尝试找到最后一个卷积层
                    target_layer = None
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            target_layer = module
                
                if target_layer is not None:
                    grad_cam = LayerGradCam(model, target_layer)
                    attribution = grad_cam.attribute(input_tensor, target=pred_label)
                    
                    # 显示Grad-CAM
                    grad_cam_np = attribution.squeeze().cpu().numpy()
                    axes[sample_idx, 1].imshow(grad_cam_np, cmap='jet', alpha=0.7)
                    axes[sample_idx, 1].imshow(img_np, alpha=0.3)
                    axes[sample_idx, 1].set_title('Grad-CAM')
                    axes[sample_idx, 1].axis('off')
                else:
                    axes[sample_idx, 1].text(0.5, 0.5, 'Grad-CAM\nNot Available', 
                                           ha='center', va='center', transform=axes[sample_idx, 1].transAxes)
                    axes[sample_idx, 1].axis('off')
                
                # Integrated Gradients
                ig = IntegratedGradients(model)
                attribution = ig.attribute(input_tensor, target=pred_label, n_steps=50)
                
                # 显示Integrated Gradients
                ig_np = attribution.squeeze().cpu().numpy()
                ig_np = np.transpose(ig_np, (1, 2, 0))
                ig_np = np.abs(ig_np).sum(axis=2)  # 取绝对值并求和
                axes[sample_idx, 2].imshow(ig_np, cmap='hot')
                axes[sample_idx, 2].set_title('Integrated Gradients')
                axes[sample_idx, 2].axis('off')
                
                # 叠加显示
                axes[sample_idx, 3].imshow(img_np, alpha=0.7)
                axes[sample_idx, 3].imshow(ig_np, cmap='hot', alpha=0.3)
                axes[sample_idx, 3].set_title('Overlay')
                axes[sample_idx, 3].axis('off')
                
            except Exception as e:
                print(f"⚠️ Error generating interpretability for sample {sample_idx}: {e}")
                for col in range(1, 4):
                    axes[sample_idx, col].text(0.5, 0.5, f'Error:\n{str(e)[:50]}...', 
                                             ha='center', va='center', transform=axes[sample_idx, col].transAxes)
                    axes[sample_idx, col].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{model_name.lower().replace("-", "_")}_interpretability.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")

# Cell 8: 集成预测函数
def load_trained_models(model_paths):
    """加载训练好的模型"""
    models_dict = {}
    for model_key, path in model_paths.items():
        if os.path.exists(path):
            model = MODEL_CONFIGS[model_key]['create_fn']()
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model = model.to(DEVICE)
            
            # 多GPU支持
            if NUM_GPUS > 1:
                model = nn.DataParallel(model)
                print(f"✅ 已加载 {MODEL_CONFIGS[model_key]['name']} (多GPU模式)")
            else:
                print(f"✅ 已加载 {MODEL_CONFIGS[model_key]['name']}")
            
            model.eval()
            models_dict[model_key] = model
        else:
            print(f"❌ 模型文件不存在: {path}")
    return models_dict

def ensemble_predict(models_dict, data_loader, voting_type='soft', weights=None):
    """集成预测"""
    all_predictions = []
    all_labels = []
    model_outputs = {key: [] for key in models_dict.keys()}
    
    # 如果是加权投票但没有提供权重，则使用等权重
    if voting_type == 'weighted' and weights is None:
        weights = {key: 1.0 for key in models_dict.keys()}
    
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="集成预测"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # 收集每个模型的预测
            batch_predictions = []
            for model_key, model in models_dict.items():
                outputs = model(imgs)
                if voting_type in ['soft', 'weighted']:
                    probs = torch.softmax(outputs, dim=1)
                    batch_predictions.append(probs.cpu().numpy())
                else:  # hard voting
                    _, predicted = torch.max(outputs, 1)
                    batch_predictions.append(predicted.cpu().numpy())
                
                model_outputs[model_key].extend(outputs.cpu().numpy())
            
            # 集成预测
            if voting_type == 'soft':
                # 软投票：平均概率
                ensemble_probs = np.mean(batch_predictions, axis=0)
                ensemble_pred = np.argmax(ensemble_probs, axis=1)
            elif voting_type == 'weighted':
                # 加权投票：根据权重加权平均概率
                weighted_probs = np.zeros_like(batch_predictions[0])
                total_weight = 0
                for i, (model_key, probs) in enumerate(zip(models_dict.keys(), batch_predictions)):
                    weight = weights[model_key]
                    weighted_probs += probs * weight
                    total_weight += weight
                ensemble_probs = weighted_probs / total_weight
                ensemble_pred = np.argmax(ensemble_probs, axis=1)
            else:
                # 硬投票：多数投票
                batch_predictions = np.array(batch_predictions)
                ensemble_pred = []
                for i in range(batch_predictions.shape[1]):
                    votes = batch_predictions[:, i]
                    ensemble_pred.append(np.bincount(votes).argmax())
                ensemble_pred = np.array(ensemble_pred)
            
            all_predictions.extend(ensemble_pred)
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), model_outputs

def calculate_model_weights(model_results, weight_method='accuracy'):
    """计算模型权重"""
    weights = {}
    
    if weight_method == 'accuracy':
        # 基于验证准确率计算权重
        accuracies = {key: results['best_acc'] for key, results in model_results.items()}
        total_acc = sum(accuracies.values())
        
        for key, acc in accuracies.items():
            weights[key] = acc / total_acc
            
    elif weight_method == 'softmax':
        # 使用softmax归一化准确率作为权重
        accuracies = np.array([results['best_acc'] for results in model_results.values()])
        softmax_weights = np.exp(accuracies * 10) / np.sum(np.exp(accuracies * 10))  # 乘以10增强差异
        
        for i, key in enumerate(model_results.keys()):
            weights[key] = softmax_weights[i]
            
    elif weight_method == 'rank':
        # 基于排名的权重分配
        sorted_models = sorted(model_results.items(), key=lambda x: x[1]['best_acc'], reverse=True)
        n_models = len(sorted_models)
        
        for i, (key, _) in enumerate(sorted_models):
            weights[key] = (n_models - i) / sum(range(1, n_models + 1))
    
    return weights

# Cell 9: 加载数据
print("📂 加载数据集...")
train_df = create_dataframe(TRAIN_PATH, "训练")
val_df = create_dataframe(VAL_PATH, "验证")

# 限制验证集大小为6400以减少内存使用
MAX_VAL_SAMPLES = 6400
if len(val_df) > MAX_VAL_SAMPLES:
    print(f"⚠️ 验证集过大 ({len(val_df)} 张)，随机采样 {MAX_VAL_SAMPLES} 张图片")
    # 保持类别平衡的随机采样
    val_df = val_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), MAX_VAL_SAMPLES//2), random_state=42)
    ).reset_index(drop=True)
    print(f"✅ 验证集采样完成，当前大小: {len(val_df)}")
    print(f"验证集类别分布:")
    for idx, cls in enumerate(classes):
        count = len(val_df[val_df['label'] == idx])
        print(f"  {cls}: {count} ({count/len(val_df)*100:.1f}%)")

print(f"\n📊 数据集总览:")
print(f"训练集总数: {len(train_df)}")
print(f"验证集总数: {len(val_df)}")
print(f"验证批次数: {len(val_df) // BATCH_SIZE + (1 if len(val_df) % BATCH_SIZE > 0 else 0)}")

# 创建数据集和数据加载器
train_dataset = DeepfakeDataset(train_df, transform=train_transform)
val_dataset = DeepfakeDataset(val_df, transform=val_transform)

# 使用动态配置的num_workers和pin_memory
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Cell 10: 训练所有模型
print("\n🚀 开始训练多个模型...")

# 选择要训练的模型（可以根据需要调整）
selected_models = ['efficientnet_b0', 'resnet18', 'convnext_tiny']  # 减少模型数量以适应Kaggle环境
model_paths = {}
model_results = {}

for model_key in selected_models:
    save_path = f"best_{model_key}_model.pth"
    model_paths[model_key] = save_path
    
    # 使用新的训练函数返回格式
    model_results[model_key] = train_single_model(
        model_key, train_loader, val_loader, save_path
    )
    
    print(f"✅ {MODEL_CONFIGS[model_key]['name']} 训练完成，最佳验证准确率: {model_results[model_key]['best_acc']:.4f}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Cell 11: 增强训练历史可视化
print("\n📊 生成训练历史可视化...")
plot_training_history(model_results)

# Cell 12: 集成预测和评估
print("\n🔮 开始集成预测...")

# 加载训练好的模型
trained_models = load_trained_models(model_paths)

# 计算模型权重
print("\n⚖️ 计算模型权重...")
model_weights = calculate_model_weights(model_results, weight_method='accuracy')
print("模型权重分配:")
for model_key, weight in model_weights.items():
    print(f"  {MODEL_CONFIGS[model_key]['name']:15}: {weight:.4f}")

# 软投票预测
print("\n📊 软投票集成预测:")
soft_predictions, true_labels, _ = ensemble_predict(trained_models, val_loader, voting_type='soft')
soft_accuracy = accuracy_score(true_labels, soft_predictions)
print(f"软投票准确率: {soft_accuracy:.4f}")

# 硬投票预测
print("\n📊 硬投票集成预测:")
hard_predictions, _, _ = ensemble_predict(trained_models, val_loader, voting_type='hard')
hard_accuracy = accuracy_score(true_labels, hard_predictions)
print(f"硬投票准确率: {hard_accuracy:.4f}")

# 加权投票预测
print("\n📊 加权投票集成预测:")
weighted_predictions, _, _ = ensemble_predict(trained_models, val_loader, voting_type='weighted', weights=model_weights)
weighted_accuracy = accuracy_score(true_labels, weighted_predictions)
print(f"加权投票准确率: {weighted_accuracy:.4f}")

# Cell 13: 结果对比和可视化
# 单模型结果对比
print("\n📈 模型性能对比:")
print("="*50)
for model_key in selected_models:
    best_acc = model_results[model_key]['best_acc']
    print(f"{MODEL_CONFIGS[model_key]['name']:15}: {best_acc:.4f}")

print(f"{'软投票集成':15}: {soft_accuracy:.4f}")
print(f"{'硬投票集成':15}: {hard_accuracy:.4f}")
print(f"{'加权投票集成':15}: {weighted_accuracy:.4f}")

# 增强混淆矩阵可视化
print("\n📊 生成混淆矩阵可视化...")
plot_confusion_matrix(true_labels, soft_predictions, "Soft Voting Ensemble", "soft_voting")
plot_confusion_matrix(true_labels, hard_predictions, "Hard Voting Ensemble", "hard_voting")
plot_confusion_matrix(true_labels, weighted_predictions, "Weighted Voting Ensemble", "weighted_voting")

# Cell 14: 详细分类报告
print("\n📋 软投票详细分类报告:")
print("="*50)
print(classification_report(true_labels, soft_predictions, target_names=classes))

print("\n📋 硬投票详细分类报告:")
print("="*50)
print(classification_report(true_labels, hard_predictions, target_names=classes))

print("\n📋 加权投票详细分类报告:")
print("="*50)
print(classification_report(true_labels, weighted_predictions, target_names=classes))

# Cell 15: 集成分析和解释性可视化
print("\n📊 生成集成分析可视化...")
plot_ensemble_analysis(trained_models, val_loader, device)

print("\n📊 生成模型解释性分析...")
plot_interpretability_analysis(trained_models, val_loader, device)

# Cell 16: 最终总结
print("\n" + "="*60)
print("🎉 多模型集成训练完成！")
print("="*60)
print(f"训练的模型数量: {len(selected_models)}")
print(f"最佳单模型准确率: {max([results['best_acc'] for results in model_results.values()]):.4f}")
print(f"软投票集成准确率: {soft_accuracy:.4f}")
print(f"硬投票集成准确率: {hard_accuracy:.4f}")
print(f"加权投票集成准确率: {weighted_accuracy:.4f}")

# 计算提升幅度
best_single = max([results['best_acc'] for results in model_results.values()])
soft_improvement = (soft_accuracy - best_single) * 100
hard_improvement = (hard_accuracy - best_single) * 100
weighted_improvement = (weighted_accuracy - best_single) * 100

print(f"软投票相对提升: {soft_improvement:+.2f}%")
print(f"硬投票相对提升: {hard_improvement:+.2f}%")
print(f"加权投票相对提升: {weighted_improvement:+.2f}%")

# 找出最佳集成方法
ensemble_results = {
    '软投票': soft_accuracy,
    '硬投票': hard_accuracy,
    '加权投票': weighted_accuracy
}
best_ensemble = max(ensemble_results, key=ensemble_results.get)
print(f"\n🏆 最佳集成方法: {best_ensemble} (准确率: {ensemble_results[best_ensemble]:.4f})")

print(f"\n💾 保存的模型文件:")
for model_key, path in model_paths.items():
    print(f"  {MODEL_CONFIGS[model_key]['name']}: {path}")

print(f"\n⚖️ 模型权重分配:")
for model_key, weight in model_weights.items():
    print(f"  {MODEL_CONFIGS[model_key]['name']}: {weight:.4f}")