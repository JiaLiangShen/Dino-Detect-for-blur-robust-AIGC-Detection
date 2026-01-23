import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from torch.utils.data import Dataset, DataLoader
import random
from threading import Lock
from transformers import AutoImageProcessor, AutoModel

# ===============================
# 配置参数
# ===============================
# DinoV3配置 - 使用代码2的路径
BASE_MODEL_PATH = "/home/work/xueyunqi/dinov3-vit7b16-pretrain-lvd1689m"
STUDENT_CHECKPOINT = "/home/work/xueyunqi/11ar/checkpoints/student/0917_dinov3_teacher_student/448_best_student_model.pth"
TEACHER_CHECKPOINT = "/home/work/xueyunqi/11ar/checkpoints/teacher/0917_dinov3_teacher_student/448_best_teacher_model.pth"

# 数据配置 - 修改为单张测试图片
TEST_IMAGE_PATH = "/home/work/xueyunqi/11ar/1013_cosine_sim/0.2orib_results_blur_analysis_dinov3/1014_own_imagedataset/train/aigc_motionblur/0/009.jpeg"  # 当前目录下的测试图片
OUTPUT_DIR = "dinov3_aigc_experiments_single_image"

# 实验配置
BLUR_STRENGTH_RANGE = (0.02, 0.10, 0.02)  # (最小值, 最大值, 步长)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1  # 单张图片测试
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

# 配置matplotlib
plt.switch_backend('Agg')
matplotlib_lock = Lock()

print(f"Using device: {DEVICE}")

# ===============================
# 模型加载函数（来自代码2）
# ===============================

def load_model_from_checkpoint(checkpoint_path, base_model_path):
    """
    Load model from checkpoint file (from code 2)
    """
    print(f"  Loading model from checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"  Error: Checkpoint file not found at {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  ✓ Checkpoint loaded from disk")
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        return None
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Found model_state_dict in checkpoint")
    else:
        print(f"  Error: No model_state_dict found in checkpoint")
        return None
    
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
    
    state_dict = remove_module_prefix(state_dict)
    
    is_teacher_student = any(key.startswith(('teacher.', 'student_')) for key in state_dict.keys())
    
    if is_teacher_student:
        print(f"  Detected teacher-student model checkpoint")
        
        backbone_weights = {}
        prefix = 'teacher.backbone.'
        prefix_len = len(prefix)
        
        for key, value in state_dict.items():
            if key.startswith(prefix):
                backbone_key = key[prefix_len:]
                if len(backbone_weights) < 3:
                    print(f"    Debug: '{key}' -> '{backbone_key}'")
                backbone_weights[backbone_key] = value
        
        if not backbone_weights:
            print(f"  Warning: No backbone weights found in checkpoint")
            model = AutoModel.from_pretrained(base_model_path)
            model.eval()
            model = model.to(DEVICE)
            print(f"  ✓ Model moved to {DEVICE}")
            return model
        
        print(f"  Found {len(backbone_weights)} backbone parameters")
        
        print(f"  Loading base model...")
        model = AutoModel.from_pretrained(base_model_path)
        print(f"  ✓ Base model loaded")
        
        model_keys = list(model.state_dict().keys())[:3]
        print(f"    Model expects keys like: {model_keys}")
        
        print(f"  Loading backbone weights...")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(backbone_weights, strict=False)
            print(f"  ✓ Backbone weights loaded")
            print(f"    Missing keys: {len(missing_keys)}")
            print(f"    Unexpected keys: {len(unexpected_keys)}")
            
            if len(missing_keys) > 0:
                print(f"    First 3 missing keys: {missing_keys[:3]}")
            if len(unexpected_keys) > 0:
                print(f"    First 3 unexpected keys: {unexpected_keys[:3]}")
                
        except Exception as e:
            print(f"  Error loading backbone weights: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Using base pretrained model instead...")
        
        del checkpoint
        del state_dict
        del backbone_weights
        torch.cuda.empty_cache()
        
        print(f"  Moving model to {DEVICE}...")
        model = model.to(DEVICE)
        print(f"  ✓ Model moved to {DEVICE}")
    
    else:
        print(f"  Standard model checkpoint")
        model = AutoModel.from_pretrained(base_model_path)
        
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"  ✓ Model weights loaded")
        except Exception as e:
            print(f"  Warning: Error loading weights: {e}")
        
        model = model.to(DEVICE)
        print(f"  ✓ Model moved to {DEVICE}")
    
    model.eval()
    print(f"  ✓ Model set to eval mode")
    
    return model

# ===============================
# DinoV3 + MLP 分类器（修改为使用transformers模型）
# ===============================

class DinoV3AIGCClassifier(nn.Module):
    """基于DinoV3的AIGC分类器 - 冻结backbone + MLP头"""
    
    def __init__(self, base_model_path, checkpoint_path=None, num_classes=2, mlp_hidden_dim=512, dropout=0.2):
        super().__init__()
        
        # 加载预训练的DinoV3模型（使用代码2的方法）
        print("Loading DinoV3 backbone...")
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.backbone = load_model_from_checkpoint(checkpoint_path, base_model_path)
        else:
            print("Loading pretrained model...")
            self.backbone = AutoModel.from_pretrained(base_model_path)
            self.backbone = self.backbone.to(DEVICE)
            print(f"✓ Pretrained model loaded and moved to {DEVICE}")
        
        # 冻结backbone参数
        print("Freezing DinoV3 parameters...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 获取DinoV3的特征维度
        self.feature_dim = self.backbone.config.hidden_size  # 通常是1024
        self.num_register_tokens = self.backbone.config.num_register_tokens
        
        # 添加MLP分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, num_classes)
        )
        
        print(f"DinoV3 feature dim: {self.feature_dim}")
        print(f"MLP classifier: {self.feature_dim} -> {mlp_hidden_dim} -> {mlp_hidden_dim//2} -> {num_classes}")
    
    def forward(self, x):
        """前向传播"""
        with torch.no_grad():  # backbone冻结，不需要计算梯度
            # 提取DinoV3特征
            outputs = self.backbone(pixel_values=x)
            
            # 使用CLS token作为全局特征 (第一个token)
            global_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, feature_dim]
            
            # 获取patch features (去除CLS和register tokens)
            patch_features = outputs.last_hidden_state[:, 1 + self.num_register_tokens:, :]
        
        # MLP分类
        logits = self.classifier(global_features)
        
        return {
            'logits': logits,
            'features': global_features.detach(),  # 用于分析
            'patch_features': patch_features.detach()  # 用于attention分析
        }

# ===============================
# 运动模糊函数（来自代码2）
# ===============================

def motion_blur(img, kernel_size=15, angle=0):
    """
    对输入图片添加运动模糊效果
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    M = cv2.getRotationMatrix2D((kernel_size/2-0.5, kernel_size/2-0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred

def add_motion_blur_pil(pil_image, intensity_range=(0.01, 0.10), prob=1.0):
    """
    为PIL图像添加运动模糊
    """
    if random.random() > prob:
        return pil_image
    
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    intensity = random.uniform(*intensity_range)
    kernel_size = int(15 * intensity / 0.1)
    kernel_size = max(3, min(kernel_size, 15))
    angle = random.uniform(0, 360)
    
    blurred_cv = motion_blur(img_cv, kernel_size=kernel_size, angle=angle)
    blurred_pil = Image.fromarray(cv2.cvtColor(blurred_cv, cv2.COLOR_BGR2RGB))
    
    return blurred_pil

# ===============================
# 数据变换（使用AutoImageProcessor）
# ===============================

def get_transforms(processor):
    """获取数据变换 - 使用AutoImageProcessor"""
    def transform_fn(image):
        # 使用processor处理图像
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)  # 返回tensor [C, H, W]
    
    return transform_fn, transform_fn  # train和test使用相同的transform

# ===============================
# 单图片测试数据集
# ===============================

class SingleImageDataset(Dataset):
    """单张图片测试数据集"""
    
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # 为不同的模糊强度创建数据列表
        self.data = [(image_path, 0.0)]  # 原始图片
        
        print(f"Single image dataset loaded: {image_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, blur_strength = self.data[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用模糊（如果需要）
        if blur_strength > 0:
            image = add_motion_blur_pil(image, intensity_range=(blur_strength, blur_strength), prob=1.0)
        
        # 数据变换
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        return image_tensor, 0, f"blur_{blur_strength:.2f}"  # label=0 (placeholder)

# ===============================
# 运动模糊鲁棒性测试（修改为单图片）
# ===============================

def test_blur_robustness_single_image(model, image_path, processor, output_dir):
    """测试单张图片的运动模糊鲁棒性"""
    print(f"Testing motion blur robustness on single image: {image_path}")
    
    model.eval()
    model = model.to(DEVICE)
    
    # 生成模糊强度序列
    min_strength, max_strength, step = BLUR_STRENGTH_RANGE
    blur_strengths = [0.0] + [round(s, 2) for s in np.arange(min_strength, max_strength + step, step)]
    
    results = {
        'blur_strengths': blur_strengths,
        'predictions': [],
        'confidences': [],
        'feature_norms': []
    }
    
    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')
    
    for strength in blur_strengths:
        print(f"Testing with blur strength: {strength}")
        
        # 应用运动模糊
        if strength > 0:
            test_image = add_motion_blur_pil(original_image, 
                                            intensity_range=(strength, strength), 
                                            prob=1.0)
        else:
            test_image = original_image
        
        # 预处理
        inputs = processor(images=test_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(DEVICE)
        
        # 预测
        with torch.no_grad():
            outputs = model(pixel_values)
            probs = F.softmax(outputs['logits'], dim=1)
            prediction = torch.argmax(outputs['logits'], dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
            feature_norm = torch.norm(outputs['features']).item()
        
        results['predictions'].append(prediction)
        results['confidences'].append(confidence)
        results['feature_norms'].append(feature_norm)
        
        pred_label = "AIGC" if prediction == 1 else "Real"
        print(f"  Prediction: {pred_label}, Confidence: {confidence:.3f}, Feature Norm: {feature_norm:.3f}")
    
    # 保存结果
    with open(os.path.join(output_dir, 'single_image_blur_robustness.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘制结果
    plot_single_image_robustness(results, image_path, output_dir)
    
    return results

def plot_single_image_robustness(results, image_path, output_dir):
    """绘制单张图片的鲁棒性结果"""
    blur_strengths = results['blur_strengths']
    confidences = results['confidences']
    feature_norms = results['feature_norms']
    predictions = results['predictions']
    
    with matplotlib_lock:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 预测标签随模糊变化
        pred_colors = ['blue' if p == 0 else 'red' for p in predictions]
        ax1.scatter(blur_strengths, predictions, c=pred_colors, s=100, alpha=0.6)
        ax1.set_xlabel('Motion Blur Strength')
        ax1.set_ylabel('Prediction (0=Real, 1=AIGC)')
        ax1.set_title(f'Prediction vs Blur Strength\n{os.path.basename(image_path)}')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Real', 'AIGC'])
        ax1.grid(True, alpha=0.3)
        
        # 置信度随模糊变化
        ax2.plot(blur_strengths, confidences, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Motion Blur Strength')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Prediction Confidence vs Blur Strength')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # 特征范数随模糊变化
        ax3.plot(blur_strengths, feature_norms, 'purple', marker='d', linewidth=2)
        ax3.set_xlabel('Motion Blur Strength')
        ax3.set_ylabel('Feature L2 Norm')
        ax3.set_title('Feature Magnitude vs Blur Strength')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'single_image_robustness.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

# ===============================
# 特征分析（修改为单图片）
# ===============================

def analyze_features_single_image(model, image_path, processor, output_dir):
    """分析单张图片在不同模糊条件下的特征变化"""
    print(f"\nAnalyzing features for single image: {image_path}")
    
    model.eval()
    model = model.to(DEVICE)
    
    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')
    
    blur_strengths = [0.0, 0.02, 0.04, 0.06, 0.08]
    
    feature_analysis_results = {
        'blur_strengths': blur_strengths,
        'similarities': [],
        'feature_norms': []
    }
    
    # 获取原始图像特征
    inputs = processor(images=original_image, return_tensors="pt")
    original_tensor = inputs['pixel_values'].to(DEVICE)
    
    with torch.no_grad():
        original_outputs = model(original_tensor)
        original_features = original_outputs['features']
    
    # 分析每个模糊强度
    for strength in blur_strengths:
        print(f"Analyzing blur strength: {strength}")
        
        if strength == 0.0:
            similarity = 1.0
            feature_norm = torch.norm(original_features).item()
        else:
            # 应用模糊
            blurred_image = add_motion_blur_pil(original_image, 
                                               intensity_range=(strength, strength), 
                                               prob=1.0)
            
            inputs = processor(images=blurred_image, return_tensors="pt")
            blurred_tensor = inputs['pixel_values'].to(DEVICE)
            
            with torch.no_grad():
                blurred_outputs = model(blurred_tensor)
                blurred_features = blurred_outputs['features']
            
            # 计算余弦相似度
            similarity = F.cosine_similarity(original_features, blurred_features, dim=1).item()
            feature_norm = torch.norm(blurred_features).item()
        
        feature_analysis_results['similarities'].append(similarity)
        feature_analysis_results['feature_norms'].append(feature_norm)
        
        print(f"  Similarity: {similarity:.4f}, Feature Norm: {feature_norm:.4f}")
    
    # 保存结果
    with open(os.path.join(output_dir, 'single_image_feature_analysis.json'), 'w') as f:
        json.dump(feature_analysis_results, f, indent=2)
    
    # 绘制特征分析结果
    plot_single_image_feature_analysis(feature_analysis_results, image_path, output_dir)
    
    return feature_analysis_results

def plot_single_image_feature_analysis(results, image_path, output_dir):
    """绘制单张图片的特征分析结果"""
    blur_strengths = results['blur_strengths']
    similarities = results['similarities']
    feature_norms = results['feature_norms']
    
    with matplotlib_lock:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 特征相似度
        ax1.plot(blur_strengths, similarities, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Motion Blur Strength')
        ax1.set_ylabel('Feature Cosine Similarity')
        ax1.set_title(f'Feature Similarity vs Motion Blur {os.path.basename(image_path)}')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 特征范数
        ax2.plot(blur_strengths, feature_norms, 'r-s', linewidth=2, markersize=8)
        ax2.set_xlabel('Motion Blur Strength')
        ax2.set_ylabel('Feature L2 Norm')
        ax2.set_title('Feature Magnitude vs Motion Blur')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'single_image_feature_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

# ===============================
# 注意力可视化（修改为单图片）
# ===============================

def visualize_attention_single_image(model, image_path, processor, output_dir):
    """可视化单张图片在不同模糊条件下的注意力图"""
    print(f"Generating attention visualizations for: {image_path}")
    
    model.eval()
    model = model.to(DEVICE)
    
    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size
    
    blur_strengths = [0.0, 0.02, 0.04, 0.06,0.08,0.10]
    
    with matplotlib_lock:
        fig, axes = plt.subplots(3, len(blur_strengths), figsize=(4*len(blur_strengths), 12))
        
        for i, strength in enumerate(blur_strengths):
            # 准备图像
            if strength == 0.0:
                test_image = original_image
                title_prefix = "Original"
            else:
                test_image = add_motion_blur_pil(original_image, 
                                                intensity_range=(strength, strength), 
                                                prob=1.0)
                title_prefix = f"Blur: {strength}"
            
            # 预处理
            inputs = processor(images=test_image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(DEVICE)
            
            with torch.no_grad():
                outputs = model(pixel_values)
                logits = outputs['logits']
                patch_features = outputs['patch_features']
                
                # 预测结果
                probs = F.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1).item()
                confidence = torch.max(probs, dim=1)[0].item()
                
                # 计算注意力图
                batch_size, num_patches, feature_dim = patch_features.shape
                patch_size = int(np.sqrt(num_patches))
                
                features_2d = patch_features.reshape(1, patch_size, patch_size, feature_dim)
                features_2d = features_2d.permute(0, 3, 1, 2)
                
                feature_norms = torch.norm(features_2d.squeeze(0), dim=0)
                attention_map = (feature_norms - feature_norms.min()) / (feature_norms.max() - feature_norms.min())
                attention_map = attention_map.cpu().numpy()
            
            pred_label = "AIGC" if prediction == 1 else "Real"
            
            # 第一行: 原始/模糊图像
            axes[0, i].imshow(test_image)
            axes[0, i].set_title(f'{title_prefix} {pred_label} ({confidence:.3f})', fontsize=12)
            axes[0, i].axis('off')
            
            # 第二行: 注意力热力图
            attention_resized = cv2.resize(attention_map, original_size, interpolation=cv2.INTER_CUBIC)
            im1 = axes[1, i].imshow(attention_resized, cmap='hot')
            axes[1, i].set_title('Attention Map', fontsize=12)
            axes[1, i].axis('off')
            plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # 第三行: 叠加可视化
            overlay = cv2.addWeighted(
                np.array(test_image), 0.6,
                cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0
            )
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            axes[2, i].imshow(overlay)
            axes[2, i].set_title('Attention Overlay', fontsize=12)
            axes[2, i].axis('off')
        
        plt.suptitle(f'Attention Analysis: {os.path.basename(image_path)}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'single_image_attention.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"  Saved: {os.path.join(output_dir, 'single_image_attention.png')}")

# ===============================
# 主实验函数
# ===============================

def main():
    """主实验函数 - 单图片测试版本"""
    print("DinoV3 + MLP for AIGC Detection - Single Image Test")
    print("=" * 70)
    
    # 检查环境
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查测试图片
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        return
    
    print(f"Test image: {TEST_IMAGE_PATH}")
    
    # 加载processor
    print("Loading image processor...")
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL_PATH)
    
    # 初始化模型
    print("\nInitializing DinoV3 + MLP model...")
    model = DinoV3AIGCClassifier(
        base_model_path=BASE_MODEL_PATH,
        checkpoint_path=STUDENT_CHECKPOINT,  # 使用student checkpoint
        num_classes=2,
        mlp_hidden_dim=512,
        dropout=0.2
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    try:
        # 实验1: 测试运动模糊鲁棒性
        print("" + "="*70)
        print("EXPERIMENT 1: Motion Blur Robustness Testing on Single Image")
        print("="*70)
        
        blur_results = test_blur_robustness_single_image(
            model, TEST_IMAGE_PATH, processor, OUTPUT_DIR
        )
        
        # 实验2: 特征稳定性分析
        print("" + "="*70)
        print("EXPERIMENT 2: Feature Stability Analysis")
        print("="*70)
        
        feature_results = analyze_features_single_image(
            model, TEST_IMAGE_PATH, processor, OUTPUT_DIR
        )
        
        # 实验3: 注意力可视化
        print("\n" + "="*70)
        print("EXPERIMENT 3: Attention Map Visualization")
        print("="*70)
        
        visualize_attention_single_image(
            model, TEST_IMAGE_PATH, processor, OUTPUT_DIR
        )
        
        # 生成综合报告
        generate_single_image_report(blur_results, feature_results, OUTPUT_DIR)
        
        print("" + "="*70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"Results saved in: {OUTPUT_DIR}")
        print("="*70)
        
    except Exception as e:
        print(f"Error during experiments: {e}")
        import traceback
        traceback.print_exc()

# ===============================
# 综合报告生成
# ===============================

def generate_single_image_report(blur_results, feature_results, output_dir):
    """生成单张图片的综合实验报告"""
    print("\nGenerating comprehensive experiment report...")
    
    report_path = os.path.join(output_dir, 'experiment_report.md')
    
    # 计算关键指标
    baseline_confidence = blur_results['confidences'][0]
    final_confidence = blur_results['confidences'][-1]
    confidence_drop = baseline_confidence - final_confidence
    
    baseline_similarity = feature_results['similarities'][0]
    final_similarity = feature_results['similarities'][-1]
    similarity_drop = baseline_similarity - final_similarity
    
    with open(report_path, 'w') as f:
        f.write("# DinoV3 + MLP AIGC Detection - Single Image Analysis")
        
        f.write("## Test Image")
        f.write(f"- **Image Path**: `{TEST_IMAGE_PATH}`\n")
        f.write(f"- **Blur Range Tested**: {BLUR_STRENGTH_RANGE[0]} - {BLUR_STRENGTH_RANGE[1]}")
        
        f.write("## Model Configuration")
        f.write(f"- **Base Model**: {BASE_MODEL_PATH}")
        f.write(f"- **Checkpoint**: {STUDENT_CHECKPOINT}")
        f.write(f"- **Device**: {DEVICE}")
        
        f.write("## Blur Robustness Results")
        f.write(f"- **Baseline Confidence** (no blur): {baseline_confidence:.3f}")
        f.write(f"- **Final Confidence** (max blur): {final_confidence:.3f}")
        f.write(f"- **Confidence Drop**: {confidence_drop:.3f}")
        f.write(f"- **Prediction Changes**: {len(set(blur_results['predictions']))} different predictions")
        
        f.write("## Feature Stability Results")
        f.write(f"- **Baseline Similarity**: {baseline_similarity:.3f}")
        f.write(f"- **Final Similarity**: {final_similarity:.3f}")
        f.write(f"- **Similarity Drop**: {similarity_drop:.3f}")
        
        f.write("## Key Findings")
        if confidence_drop < 0.2:
            f.write("1. **High Robustness**: Model maintains stable predictions under motion blur")
        elif confidence_drop < 0.5:
            f.write("1. **Moderate Robustness**: Model shows some sensitivity to motion blur")
        else:
            f.write("1. **Low Robustness**: Model is significantly affected by motion blur")
        
        if similarity_drop < 0.1:
            f.write("2. **Stable Features**: DinoV3 features remain consistent under blur")
        else:
            f.write("2. **Feature Degradation**: DinoV3 features show noticeable changes under blur")
        
        f.write("## Generated Files")
        f.write("- `single_image_blur_robustness.json`: Robustness test data")
        f.write("- `single_image_robustness.png`: Robustness visualization")
        f.write("- `single_image_feature_analysis.json`: Feature analysis data")
        f.write("- `single_image_feature_analysis.png`: Feature visualization")
        f.write("- `single_image_attention.png`: Attention map visualization")
        f.write("- `experiment_report.md`: This report")
    
    print(f"Report saved: {report_path}")
    
    # 打印关键结果摘要
    print("" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Test Image: {TEST_IMAGE_PATH}")
    print(f"Confidence Drop: {confidence_drop:.3f}")
    print(f"Feature Similarity Drop: {similarity_drop:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()
