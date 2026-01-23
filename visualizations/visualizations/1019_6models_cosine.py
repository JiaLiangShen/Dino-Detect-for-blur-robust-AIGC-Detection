import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading
import random
from transformers import AutoImageProcessor, AutoModel
import open_clip
import torch.nn as nn
import json # 新增: 用于保存JSON
import seaborn as sns # 新增: final绘图用到
# 新增: AIDE 模型导入
# import models.AIDE as AIDE  # 需要确保AIDE模块在路径中
import timm  # ConvNeXt模型需要timm库
# 新增: AIDE 模型路径配置
AIDE_CHECKPOINT = "/home/work/xueyunqi/results/progan_train/progan_train.pth"  # 根据你的实际路径调整

MOTION_BLUR_ANALYSIS = True  # 是否启用运动模糊分析
# 修改: BLUR_STRENGTH_RANGE 变为每0.025步长
BLUR_STRENGTH_RANGE = (0.00, 0.20, 0.025)  # (最小值, 最大值, 步长)
BLUR_ANGLE = 0  # 运动模糊角度
# 修改: 输出文件夹名字
MOTION_BLUR_OUTPUT_DIR = "attention_similarity_analysis"  # 运动模糊分析输出文件夹

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                       dim_feedforward=int(embed_dim * mlp_ratio),
                                       batch_first=True)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x


# ============================================================================
# 新增：Custom ViT 模型路径配置
# ============================================================================
CUSTOM_VIT_PATH = "/home/work/xueyunqi/11ar/comu_foren_dp/pretrained_weights/model_v11_ViT_224_base_ckpt.pt"

# 新增: UnivFD 和 DRCT 模型路径
UNIVFD_CHECKPOINT = "/home/work/xueyunqi/UniversalFakeDetect/pretrained_weights/fc_weights.pth"
DRCT_CONVNEXT_CHECKPOINT = "/home/work/xueyunqi/pretrained/DRCT-2M/sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth"

# 设备配置
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# Model Configuration
# ============================================================================
BASE_MODEL_PATH = "/home/work/xueyunqi/dinov3-vit7b16-pretrain-lvd1689m"
STUDENT_CHECKPOINT = "/home/work/xueyunqi/11ar/checkpoints/student/0917_dinov3_teacher_student/448_best_student_model.pth"
TEACHER_CHECKPOINT = "/home/work/xueyunqi/11ar/checkpoints/teacher/0917_dinov3_teacher_student/448_best_teacher_model.pth"


DATASET_BASE_DIR = "/home/work/xueyunqi/11ar_datasets/test/progan"
REAL_ORIGINAL_DIR = os.path.join(DATASET_BASE_DIR, "sampled_images")
FAKE_ORIGINAL_DIR = os.path.join(DATASET_BASE_DIR, "1_fake")

OUTPUT_BASE_DIR = "cosine_similarity_analysis" # 输出文件夹



# 配置matplotlib为线程安全模式
plt.switch_backend('Agg')
matplotlib_lock = Lock()

# Set matplotlib parameters for high quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif'],
    'font.size': 12,
    'axes.labelsize': 13, # 修改
    'axes.titlesize': 16, # 修改
    'xtick.labelsize': 11, # 修改
    'ytick.labelsize': 11, # 修改
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# final绘图需要的常量
COL_DINO = "#E69F00"  # orange
COL_CLIP = "#56B4E9"  # blue
COL_CUSTOM_VIT = "#009E73"  # green (Community Forensics)
COL_UNIVFD = "#F0E442"  # yellow (UnivFD) - 替换原来的 Effort
COL_DRCT_CONVB = "#CC79A7"  # pink/purple (DRCT) - 替换原来的 AIDE
COL_AIDE = "#D55E00"  # red-orange (AIDE) - 新颜色
# pink/purple (AIDE)

LINEWIDTH   = 2.2
MARKERSIZE  = 7
GRID_COLOR  = "#b0b0b0"   
GRID_ALPHA  = 0.35
GRID_LS     = "--"       
GRID_LW     = 0.8


def load_univfd_model():
    """
    加载 UnivFD 模型 (基于 CLIP ViT-L/14)
    """
    print(f"  Loading UnivFD model from: {UNIVFD_CHECKPOINT}")
    
    if not os.path.exists(UNIVFD_CHECKPOINT):
        print(f"  Error: Model file not found at {UNIVFD_CHECKPOINT}")
        return None, None
    
    try:
        # 使用 open_clip 创建 CLIP ViT-L/14 架构
        print(f"  Creating CLIP ViT-L/14 architecture for UnivFD...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", 
            pretrained="openai"  # 使用 OpenAI 的预训练权重作为基础
        )
        print(f"  ✓ Base model architecture created")
        
        # 加载 fc 层权重
        print(f"  Loading UnivFD fc weights...")
        fc_state_dict = torch.load(UNIVFD_CHECKPOINT, map_location='cpu')
        
        # UnivFD 只训练了最后的 fc 层,需要添加这个分类头
        # 根据 validate.py 的代码,fc 层是一个简单的线性层
        import torch.nn as nn
        
        # 获取 CLIP 的输出维度 (ViT-L/14 是 768)
        clip_output_dim = model.visual.output_dim
        
        # 创建 fc 层 (二分类)
        model.fc = nn.Linear(clip_output_dim, 1)
        
        # 加载 fc 权重
        model.fc.load_state_dict(fc_state_dict)
        print(f"  ✓ FC layer weights loaded")
        
        model = model.to(DEVICE)
        model.eval()
        print(f"  ✓ UnivFD model loaded and moved to {DEVICE}")
        
        return model, preprocess
        
    except Exception as e:
        print(f"  Error loading UnivFD model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


class ConvNeXtFeatureExtractor:
    """使用hook提取ConvNeXt的中间特征"""
    def __init__(self, model):
        self.model = model
        self.features = None
        self.hook = None
        
    def _hook_fn(self, module, input, output):
        """Hook函数，保存特征"""
        self.features = output
        
    def register_hook(self, layer_name='stages.3'):
        """注册hook到指定层"""
        # ConvNeXt的最后一个stage通常是'stages.3'
        for name, module in self.model.base_model.named_modules():
            if layer_name in name and isinstance(module, nn.Sequential):
                self.hook = module.register_forward_hook(self._hook_fn)
                print(f"    Hook registered at: {name}")
                break
                
    def remove_hook(self):
        """移除hook"""
        if self.hook is not None:
            self.hook.remove()
            
    def extract_features(self, x):
        """提取特征"""
        _ = self.model.base_model(x)
        return self.features

def load_drct_convnext_model():
    """
    加载 DRCT ConvNeXt-Base 模型
    基于官方训练脚本的配置
    """
    print(f"  Loading DRCT ConvNeXt model from: {DRCT_CONVNEXT_CHECKPOINT}")
    
    if not os.path.exists(DRCT_CONVNEXT_CHECKPOINT):
        print(f"  Error: Model file not found at {DRCT_CONVNEXT_CHECKPOINT}")
        return None, None
    
    try:
        # 创建ConvNeXt-Base模型（根据训练脚本的配置）
        print(f"  Creating ConvNeXt-Base architecture...")
        
        # 使用timm创建convnext_base模型
        # 注意：根据训练脚本，使用的是 convnext_base_in22k
        base_model = timm.create_model(
            'convnext_base.fb_in22k',  # 或 'convnext_base_in22k'
            pretrained=False,  # 不加载ImageNet预训练权重，因为我们要加载DRCT的权重
            num_classes=0  # 移除分类头，只要特征提取器
        )
        
        print(f"  ✓ ConvNeXt-Base architecture created")
        
        # 添加分类头（根据训练配置：embedding_size=1024, num_classes=2）
        embedding_size = 1024
        num_classes = 2
        
        # ConvNeXt的特征维度通常是1024（for base variant）
        feature_dim = base_model.num_features
        print(f"  ConvNeXt feature dimension: {feature_dim}")
        
        # 创建完整模型（特征提取器 + 投影层 + 分类头）
        class DRCTConvNeXt(nn.Module):
            def __init__(self, base_model, feature_dim, embedding_size, num_classes):
                super(DRCTConvNeXt, self).__init__()
                self.base_model = base_model
                
                # 如果需要embedding层
                if embedding_size is not None and embedding_size != feature_dim:
                    self.embedding = nn.Linear(feature_dim, embedding_size)
                    self.fc = nn.Linear(embedding_size, num_classes)
                else:
                    self.embedding = None
                    self.fc = nn.Linear(feature_dim, num_classes)
            
            def forward(self, x):
                features = self.base_model(x)
                if self.embedding is not None:
                    features = self.embedding(features)
                return self.fc(features)
            
            def forward_features(self, x):
                """提取特征（不经过分类头）"""
                features = self.base_model(x)
                if self.embedding is not None:
                    features = self.embedding(features)
                return features
        
        model = DRCTConvNeXt(base_model, feature_dim, embedding_size, num_classes)
        print(f"  ✓ DRCT ConvNeXt model structure created")
        
        # 加载checkpoint
        checkpoint = torch.load(DRCT_CONVNEXT_CHECKPOINT, map_location='cpu')
        print(f"  ✓ Checkpoint loaded from disk")
        
        # 打印checkpoint的keys以调试
        if isinstance(checkpoint, dict):
            print(f"  Checkpoint keys: {list(checkpoint.keys())[:10]}")
            
            # 尝试找到state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 处理可能的'module.'前缀（多GPU训练产生的）
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 移除 'module.' 前缀
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        print(f"  Sample keys from state_dict: {list(new_state_dict.keys())[:5]}")
        
        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"  ✓ Model weights loaded")
        print(f"    Missing keys: {len(missing_keys)}")
        print(f"    Unexpected keys: {len(unexpected_keys)}")
        
        if len(missing_keys) > 0:
            print(f"    First 5 missing keys: {missing_keys[:5]}")
        if len(unexpected_keys) > 0:
            print(f"    First 5 unexpected keys: {unexpected_keys[:5]}")
        
        model = model.to(DEVICE)
        model.eval()
        print(f"  ✓ DRCT ConvNeXt model loaded and moved to {DEVICE}")
        
        # ConvNeXt使用标准的ImageNet预处理
        return model, "drct_convnext"
        
    except Exception as e:
        print(f"  Error loading DRCT ConvNeXt model: {e}")
        import traceback
        traceback.print_exc()
        return None, None




def load_aide_model():
    """加载AIDE模型 - 基于官方实现"""
    print(f"  Loading AIDE model from: {AIDE_CHECKPOINT}")
    
    if not os.path.exists(AIDE_CHECKPOINT):
        print(f"  Error: Model file not found at {AIDE_CHECKPOINT}")
        return None, None
    
    try:
        # 使用AIDE官方的模型创建方式
        model = AIDE.__dict__['AIDE'](
            resnet_path=None,  # 如果需要可以指定
            convnext_path=None  # 如果需要可以指定
        )
        
        print(f"  ✓ AIDE model architecture created")
        
        # 加载checkpoint
        checkpoint = torch.load(AIDE_CHECKPOINT, map_location='cpu')
        print(f"  ✓ Checkpoint loaded from disk")
        
        # 处理checkpoint格式 - 根据官方代码的加载方式
        if isinstance(checkpoint, dict):
            # 检查不同的键名
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_without_ddp' in checkpoint:
                state_dict = checkpoint['model_without_ddp']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # 如果没有找到标准键，尝试直接使用checkpoint
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 移除可能的 'module.' 前缀（分布式训练产生的）
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 移除 'module.' 前缀
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"  ✓ Model weights loaded")
        print(f"    Missing keys: {len(missing_keys)}")
        print(f"    Unexpected keys: {len(unexpected_keys)}")
        
        if len(missing_keys) > 0:
            print(f"    First 5 missing keys: {missing_keys[:5]}")
        if len(unexpected_keys) > 0:
            print(f"    First 5 unexpected keys: {unexpected_keys[:5]}")
        
        model = model.to(DEVICE)
        model.eval()
        print(f"  ✓ AIDE model loaded and moved to {DEVICE}")
        
        # AIDE使用自定义预处理（需要根据AIDE的数据预处理方式调整）
        return model, "aide"
        
    except Exception as e:
        print(f"  Error loading AIDE model: {e}")
        import traceback
        traceback.print_exc()
        return None, None




# ============================================================================
# Model Loading Functions - 来自代码一
# ============================================================================
def load_model_from_checkpoint(checkpoint_path, base_model_path):
    """
    Load model from checkpoint file
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


def load_custom_vit_model():
    """加载Custom ViT模型"""
    print(f"  Loading Custom ViT from: {CUSTOM_VIT_PATH}")
    
    if not os.path.exists(CUSTOM_VIT_PATH):
        print(f"  Error: Model file not found at {CUSTOM_VIT_PATH}")
        return None, None
    
    try:
        checkpoint = torch.load(CUSTOM_VIT_PATH, map_location='cpu')
        print(f"  ✓ Checkpoint loaded from disk")
        
        model = VisionTransformer(
            img_size=224, 
            patch_size=16, 
            embed_dim=768, 
            depth=12, 
            num_heads=12
        )
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model = model.to(DEVICE)
        model.eval()
        print(f"  ✓ Custom ViT model loaded and moved to {DEVICE}")
        
        # Custom ViT使用自己的预处理
        return model, "custom"
        
    except Exception as e:
        print(f"  Error loading Custom ViT model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def load_clip_model():
    """加载CLIP模型"""
    print(f"  Loading CLIP ViT-H/14 (LAION-2B)...")
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        model = model.to(DEVICE)
        model.eval()
        print(f"  ✓ CLIP model loaded and moved to {DEVICE}")
        
        return model, preprocess
        
    except Exception as e:
        print(f"  Error loading CLIP model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# Image Collection Functions - 修改为读取所有图片
# ============================================================================
def get_image_files_from_directory(folder_path, num_images=None):
    """
    获取指定文件夹中的图片文件
    
    Args:
        folder_path: 文件夹路径
        num_images: 要读取的图片数量，如果为None则读取所有图片
    
    Returns:
        list: 图片文件路径列表
    """
    image_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder not found at {folder_path}")
        return []
    
    print(f"Collecting images from {folder_path}...")
    
    # 支持的图片扩展名
    valid_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP']
    
    # 收集所有图片文件
    all_images = []
    for ext in valid_extensions:
        all_images.extend(list(folder.glob(f'*{ext}')))
    
    # 按文件名排序
    all_images = sorted(all_images, key=lambda x: x.name)
    
    # 如果指定了数量限制
    if num_images is not None:
        all_images = all_images[:num_images]
        print(f"  Limiting to first {num_images} images")
    
    for img_path in all_images:
        image_files.append(img_path)
        if len(image_files) <= 5 or len(image_files) == len(all_images):  # 只打印前5张和最后一张
            print(f"  ✓ Found image: {img_path.name}")
        elif len(image_files) == 6:
            print(f"  ... (showing first 5 and last)")
    
    print(f"✓ Found {len(image_files)} images in {folder_path}")
    return image_files


# ============================================================================
# Motion Blur Functions - 来自代码一
# ============================================================================
def motion_blur(img, kernel_size=15, angle=0):
    """
    对输入图片添加运动模糊效果
    
    Args:
        img (np.ndarray): 输入的图片数组（BGR格式）
        kernel_size (int): 模糊核的大小，决定模糊强度
        angle (float): 模糊方向的角度（度），0为水平
    
    Returns:
        np.ndarray: 添加运动模糊后的图片数组
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    M = cv2.getRotationMatrix2D((kernel_size/2-0.5, kernel_size/2-0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


def strength_to_kernel_size(strength, base_size=15):
    """将模糊强度转换为kernel_size"""
    # strength范围0.01-0.10，映射到kernel_size
    min_kernel = 3
    max_kernel = 15
    kernel_size = int(min_kernel + (strength - 0.01) * (max_kernel - min_kernel) / 0.09)
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def apply_motion_blur_to_pil_image(image_pil, strength, angle=0):
    """对PIL图像应用运动模糊"""
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    kernel_size = strength_to_kernel_size(strength)
    blurred_cv = motion_blur(image_cv, kernel_size=kernel_size, angle=angle)
    blurred_pil = Image.fromarray(cv2.cvtColor(blurred_cv, cv2.COLOR_BGR2RGB))
    return blurred_pil

# ============================================================================
# 新增: 子集分析辅助函数
# ============================================================================
def analyze_subset(model, processor, image_files, model_name, model_type, 
                   clip_preprocess, blur_strengths, subset_name="subset"):
    """
    对图片子集进行运动模糊分析 - 使用Cosine Similarity
    """
    print(f"{'='*80}")
    print(f"Analyzing {subset_name} subset - {len(image_files)} images")
    print('='*80)
    
    # 存储所有图片的相似度数据
    all_image_similarities = {}
    
    for img_idx, image_path in enumerate(image_files):
        image_name = image_path.stem
        
        if (img_idx + 1) % 10 == 0 or img_idx == 0 or img_idx == len(image_files) - 1:
            print(f"[{img_idx+1}/{len(image_files)}] Processing: {image_name}")
        
        try:
            original_image = Image.open(image_path).convert('RGB')
            original_size = original_image.size
            
            # 计算原始图片的cosine similarity矩阵
            try:
                original_image_tensor, _, _ = preprocess_image_for_blur_analysis(
                    original_image, model_type, processor, clip_preprocess
                )
                
                if original_image_tensor is None:
                    print(f"    ⚠ Warning: Failed to preprocess {image_name}, skipping.")
                    continue
                
                # ⭐ 修改：计算cosine similarity矩阵
                original_cosine_matrix = compute_cosine_similarity_matrix(
                    model, original_image_tensor, model_type
                )
                
                if original_cosine_matrix is None or original_cosine_matrix.numel() == 0:
                    print(f"    ⚠ Warning: Failed to compute cosine similarity for {image_name}, skipping.")
                    continue
                
            except Exception as e:
                print(f"    ✗ Error in preprocessing/computing {image_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            image_similarities = {}
            
            for strength in blur_strengths:
                if strength == 0.0:
                    test_image = original_image
                    cosine_matrix = original_cosine_matrix
                else:
                    test_image = apply_motion_blur_to_pil_image(original_image, strength, BLUR_ANGLE)
                    image_tensor, _, _ = preprocess_image_for_blur_analysis(
                        test_image, model_type, processor, clip_preprocess
                    )
                    
                    if image_tensor is None:
                        print(f"    ⚠ Warning: Failed to preprocess blurred image at strength {strength}")
                        continue
                    
                    # ⭐ 修改：计算模糊图片的cosine similarity矩阵
                    cosine_matrix = compute_cosine_similarity_matrix(model, image_tensor, model_type)
                    
                    if cosine_matrix is None or cosine_matrix.numel() == 0:
                        print(f"    ⚠ Warning: Failed to compute cosine similarity for blurred image at strength {strength}")
                        continue
                
                #  修改：计算两个cosine similarity矩阵之间的相似度
                # 方法1: 使用矩阵的均值相似度
                original_mean = original_cosine_matrix.mean().item()
                blurred_mean = cosine_matrix.mean().item()
                
                # 方法2: 使用矩阵的相关系数（推荐）
                if isinstance(original_cosine_matrix, torch.Tensor):
                    orig_flat = original_cosine_matrix.cpu().numpy().flatten()
                    blur_flat = cosine_matrix.cpu().numpy().flatten()
                else:
                    orig_flat = original_cosine_matrix.flatten()
                    blur_flat = cosine_matrix.flatten()
                
                correlation = np.corrcoef(orig_flat, blur_flat)[0, 1]
                similarity = correlation if not np.isnan(correlation) else 0.0
                
                image_similarities[f'{strength:.3f}'] = float(similarity)
            
            all_image_similarities[image_name] = image_similarities
            
        except Exception as e:
            print(f"  ✗ Error analyzing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算平均相似度
    subset_avg_similarities = {}
    for strength in blur_strengths:
        strength_str = f'{strength:.3f}'
        similarities_at_strength = []
        for img_name, img_data in all_image_similarities.items():
            if strength_str in img_data:
                similarities_at_strength.append(img_data[strength_str])
        
        if similarities_at_strength:
            avg_similarity = np.mean(similarities_at_strength)
            subset_avg_similarities[strength_str] = float(avg_similarity)
        else:
            subset_avg_similarities[strength_str] = 0.0
    
    print(f"✓ Completed analysis for {subset_name} subset")
    print(f"  Processed {len(all_image_similarities)}/{len(image_files)} images successfully")
    
    return subset_avg_similarities, all_image_similarities


# ============================================================================
# Image Preprocessing Functions
# ============================================================================
def make_transform_for_dinov3(processor):
    """创建使用AutoImageProcessor的transform"""
    def transform(image):
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)
    return transform

def preprocess_images_batch(image_paths, model_type, processor=None, clip_preprocess=None):
    """批量预处理图像 - 支持多种模型类型"""
    print(f"Preprocessing {len(image_paths)} images for model type: {model_type}...")
    
    batch_tensors = []
    original_images = []
    original_sizes = []
    image_names = []
    
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 根据模型类型选择预处理方法
            if model_type == "dinov3":
                inputs = processor(images=image, return_tensors="pt")
                processed_image = inputs['pixel_values'].squeeze(0)
            elif model_type == "custom":
                processed_image = preprocess_custom_vit(image).squeeze(0)
            elif model_type == "clip":
                processed_image = clip_preprocess(image)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            batch_tensors.append(processed_image)
            original_images.append(image)
            original_sizes.append(original_size)
            image_names.append(image_path.stem)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    if batch_tensors:
        batch_tensor = torch.stack(batch_tensors)
        print(f"Created batch tensor: {batch_tensor.shape}")
        return batch_tensor, original_images, original_sizes, image_names
    else:
        return None, [], [], []


def preprocess_drct_convnext(image):
    """DRCT ConvNeXt预处理 - 使用ImageNet标准"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor


def preprocess_custom_vit(image):
    """Custom ViT预处理"""
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor

def preprocess_image_for_blur_analysis(image_pil, model_type, processor=None, clip_preprocess=None):
    """为运动模糊分析预处理图像 - 支持多种模型"""
    try:
        if model_type == "dinov3":
            inputs = processor(images=image_pil, return_tensors="pt")
            processed_image = inputs['pixel_values']
            
        elif model_type == "custom":
            processed_image = preprocess_custom_vit(image_pil)
            
        elif model_type == "aide":
            processed_image = preprocess_aide(image_pil)
            
        elif model_type == "drct_convnext":  # 新增
            processed_image = preprocess_drct_convnext(image_pil)
            
        elif model_type == "clip" or model_type == "univfd" or model_type == "drct":
            if clip_preprocess is None:
                print(f"    ⚠ Warning: clip_preprocess is None for {model_type}")
                return None, image_pil, image_pil.size
            processed_image = clip_preprocess(image_pil).unsqueeze(0)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 检查处理后的图像
        if processed_image is None or processed_image.numel() == 0:
            print(f"    ⚠ Warning: Preprocessing resulted in empty tensor for {model_type}")
            return None, image_pil, image_pil.size
        
        return processed_image, image_pil, image_pil.size
        
    except Exception as e:
        print(f"    ✗ Error in preprocessing for {model_type}: {e}")
        import traceback
        traceback.print_exc()
        return None, image_pil, image_pil.size




def extract_patch_features(model, image_tensor, model_type):
    """
    提取patch-level特征用于cosine similarity计算
    采用原始cosine的方式：先切片，再normalize，最后squeeze
    Returns: [num_patches, feature_dim] 的特征张量
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        if model_type == "dinov3":
            # DINOv3模型
            outputs = model(pixel_values=image_tensor)
            num_register = model.config.num_register_tokens
            
            # 原始cosine方式：先切片 [1, 196, 768]
            patch_features = outputs.last_hidden_state[:, 1 + num_register:, :]
            # 再normalize
            patch_features = F.normalize(patch_features, dim=-1)
            # 最后squeeze
            patch_features = patch_features.squeeze(0)  # [196, 768]
            
        elif model_type == "custom" or model_type == "aide":
            # Custom ViT 和 AIDE 模型
            outputs = model(image_tensor)  # [1, 197, 768]
            
            # 原始cosine方式：先切片 [1, 196, 768]
            patch_features = outputs[:, 1:, :]
            # 再normalize
            patch_features = F.normalize(patch_features, dim=-1)
            # 最后squeeze
            patch_features = patch_features.squeeze(0)  # [196, 768]

        elif model_type == "drct_convnext":
            try:
                # 使用hook提取中间特征
                extractor = ConvNeXtFeatureExtractor(model)
                extractor.register_hook(layer_name='stages.3')  # 最后一个stage
                
                features = extractor.extract_features(image_tensor)  # [B, C, H, W]
                extractor.remove_hook()
                
                if features.dim() == 4:
                    B, C, H, W = features.shape
                    patch_features = features.reshape(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
                    patch_features = F.normalize(patch_features, dim=-1)
                    patch_features = patch_features.squeeze(0)  # [H*W, C]
                else:
                    print(f"    ⚠ Unexpected feature shape: {features.shape}")
                    return torch.zeros((196, 1024)).to(device)
                    
            except Exception as e:
                print(f"    ✗ Error extracting ConvNeXt features: {e}")
                return torch.zeros((196, 1024)).to(device)
            
        elif model_type == "clip" or model_type == "univfd" or model_type == "drct":
            # CLIP、UnivFD 和 DRCT 模型
            # 提取transformer特征
            x = model.visual.conv1(image_tensor)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            
            cls_token = model.visual.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + model.visual.positional_embedding
            x = model.visual.ln_pre(x)
            
            x = x.permute(1, 0, 2)
            x = model.visual.transformer(x)
            x = x.permute(1, 0, 2)
            
            # 原始cosine方式：先切片 [1, N, C]
            patch_features = x[:, 1:, :]
            
            # 检查是否为空
            if patch_features.numel() == 0:
                print(f"    ⚠ Warning: Empty patch features for {model_type}")
                return torch.zeros((256, 1024)).to(device)
            
            # 再normalize
            patch_features = F.normalize(patch_features, dim=-1)
            # 最后squeeze
            patch_features = patch_features.squeeze(0)  # [N, C]
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return patch_features  # [num_patches, feature_dim]



def extract_dense_features_batch(model, batch_tensor, model_type, processor=None):
    """批量提取dense features - 支持多种模型"""
    print(f"Extracting dense features for batch of {batch_tensor.shape[0]} images...")
    
    device = next(model.parameters()).device
    batch_tensor = batch_tensor.to(device)
    
    with torch.no_grad():
        if model_type == "dinov3":
            outputs = model(pixel_values=batch_tensor)
            num_register = model.config.num_register_tokens
            patch_features = outputs.last_hidden_state[:, 1 + num_register:, :]
            patch_features = F.normalize(patch_features, dim=-1)
            
        elif model_type == "custom":
            outputs = model(batch_tensor)
            patch_features = outputs[:, 1:, :]
            patch_features = F.normalize(patch_features, dim=-1)
            
        elif model_type == "clip":
            x = model.visual.conv1(batch_tensor)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            
            cls_token = model.visual.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + model.visual.positional_embedding
            x = model.visual.ln_pre(x)
            
            x = x.permute(1, 0, 2)
            x = model.visual.transformer(x)
            x = x.permute(1, 0, 2)
            
            patch_features = x[:, 1:, :]
            patch_features = F.normalize(patch_features, dim=-1)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}") # 补全这一行
        
        batch_size, num_patches, feature_dim = patch_features.shape
        patch_size = int(np.sqrt(num_patches))
        
        print(f"Batch feature shape: {patch_features.shape}")
        print(f"Patch grid size: {patch_size}x{patch_size}")
        print(f"Feature dimension: {feature_dim}")
        
        dense_features = patch_features.reshape(batch_size, patch_size, patch_size, feature_dim)
        dense_features = dense_features.permute(0, 3, 1, 2)
        
    return dense_features, patch_size



def compute_cosine_similarity_matrix(model, image_tensor, model_type):
    """
    计算patch之间的cosine similarity矩阵 - 采用原始cosine的直接方式
    Returns: [num_patches, num_patches] 的相似度矩阵
    """
    try:
        # 提取并normalize特征
        feat_map = extract_patch_features(model, image_tensor, model_type)
        
        if feat_map is None or feat_map.numel() == 0:
            print(f"    ⚠ Warning: Empty feature map for {model_type}")
            patch_size = 16 if model_type in ["clip", "univfd", "drct"] else 14
            return torch.zeros((patch_size*patch_size, patch_size*patch_size))
        
        # 直接计算cosine similarity矩阵（原cosine的方式）
        cosine_sim = torch.matmul(feat_map, feat_map.T)
        
        return cosine_sim
        
    except Exception as e:
        print(f"    ✗ Error computing cosine similarity for {model_type}: {e}")
        patch_size = 16 if model_type in ["clip", "univfd", "drct"] else 14
        return torch.zeros((patch_size*patch_size, patch_size*patch_size))


# ============================================================================
# Attention Map Functions - 支持多种模型类型
# ============================================================================
def get_gradcam_attention_map(model, image_tensor, model_type, target_layer_idx=-1):
    """
    使用GradCAM方法计算注意力图 - 适配transformers、Custom ViT和CLIP模型
    """
    model.eval()
    
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    features = []
    gradients = []
    
    def forward_hook(module, input, output):
        features.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    try:
        if model_type == "dinov3":
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                target_layer = model.encoder.layer[target_layer_idx]
            elif hasattr(model, 'blocks'):
                target_layer = model.blocks[target_layer_idx]
            else:
                print(f"  Warning: Cannot find standard transformer layers in DINOv3, using feature_norm method")
                return compute_attention_map(model, image_tensor, model_type, method='feature_norm')
                
        elif model_type == "custom" or model_type == "aide":
            target_layer = model.blocks[target_layer_idx]
            
        elif model_type == "clip" or model_type == "effort":
            target_layer = model.visual.transformer.resblocks[target_layer_idx]
        else:
            raise ValueError(f"Unknown model type for GradCAM: {model_type}")
            
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
    except Exception as e:
        print(f"  Warning: Could not register hooks for {model_type} GradCAM ({e}), using feature_norm method")
        return compute_attention_map(model, image_tensor, model_type, method='feature_norm')
    
    model.zero_grad()
    
    # Forward pass
    if model_type == "dinov3":
        outputs = model(pixel_values=image_tensor)
        cls_token = outputs.last_hidden_state[:, 0, :]
    elif model_type == "custom":
        outputs = model(image_tensor)
        cls_token = outputs[:, 0, :]
    elif model_type == "clip":
        x = model.visual.conv1(image_tensor)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        cls_token_embed = model.visual.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token_embed, x], dim=1)
        x = x + model.visual.positional_embedding
        x = model.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        cls_token = x[:, 0, :] # CLIP的CLS token
    
    target = cls_token.sum()
    target.backward()
    
    forward_handle.remove()
    backward_handle.remove()
    
    if len(features) == 0 or len(gradients) == 0:
        print(f"Warning: No gradients captured for {model_type} GradCAM, using feature magnitude method")
        return compute_attention_map(model, image_tensor, model_type, method='feature_norm')
    
    feature = features[0]
    gradient = gradients[0]
    
    # 处理特征和梯度，去除CLS token和可能的register tokens
    if model_type == "dinov3":
        num_register = model.config.num_register_tokens
        feature = feature[:, 1 + num_register:, :]
        gradient = gradient[:, 1 + num_register:, :]
    elif model_type == "custom":
        feature = feature[:, 1:, :] # Custom ViT
        gradient = gradient[:, 1:, :]
    elif model_type == "clip":
        # CLIP的transformer输出可能是[seq_len, batch, dim]
        if feature.dim() == 3 and feature.shape[0] == feature.shape[1]: # 假设是[seq_len, batch, dim]
             feature = feature.permute(1, 0, 2)  # [batch, seq_len, dim]
             gradient = gradient.permute(1, 0, 2)
        
        feature = feature[:, 1:, :]  # CLIP
        gradient = gradient[:, 1:, :]
    
    weights = gradient.mean(dim=1, keepdim=True)
    cam = (weights * feature).sum(dim=2)
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    num_patches = cam.shape[1]
    patch_size = int(np.sqrt(num_patches))
    attention_map = cam.reshape(patch_size, patch_size)
    
    return attention_map.cpu().detach().numpy()

def compute_attention_map(model, image_tensor, model_type, method='gradcam'):
    """
    计算注意力图 - 支持多种方法和模型类型
    """
    if method == 'gradcam':
        return get_gradcam_attention_map(model, image_tensor, model_type)
    
    elif method == 'feature_norm':
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            if model_type == "dinov3":
                outputs = model(pixel_values=image_tensor)
                num_register = model.config.num_register_tokens
                patch_features = outputs.last_hidden_state[:, 1 + num_register:, :]
                
            elif model_type == "custom" or model_type == "aide":
                outputs = model(image_tensor)
                patch_features = outputs[:, 1:, :]
                
            elif model_type == "clip" or model_type == "univfd" or model_type == "drct":  # 修改这里
                x = model.visual.conv1(image_tensor)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                cls_token = model.visual.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + model.visual.positional_embedding
                x = model.visual.ln_pre(x)
                x = x.permute(1, 0, 2)
                x = model.visual.transformer(x)
                x = x.permute(1, 0, 2)
                patch_features = x[:, 1:, :]
                
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # 检查 patch_features
            if patch_features.numel() == 0:
                print(f"    ⚠ Warning: Empty patch features for {model_type}")
                patch_size = 16 if model_type in ["clip", "effort"] else 14
                return np.zeros((patch_size, patch_size))
            
            # 计算特征范数
            feature_norms = torch.norm(patch_features, dim=-1)
            
            # 重塑为2D
            batch_size, num_patches = feature_norms.shape
            patch_size = int(np.sqrt(num_patches))
            
            if patch_size * patch_size != num_patches:
                print(f"    ⚠ Warning: Cannot reshape to square. num_patches={num_patches}")
                patch_size = int(np.sqrt(num_patches))
                feature_norms = feature_norms[:, :patch_size*patch_size]
            
            attention_map = feature_norms.reshape(patch_size, patch_size)
            
            # 归一化
            att_min = attention_map.min()
            att_max = attention_map.max()
            
            if att_max == att_min:
                attention_map = torch.ones_like(attention_map) * 0.5
            else:
                attention_map = (attention_map - att_min) / (att_max - att_min + 1e-8)
            
            return attention_map.cpu().numpy()
    
    else:
        raise ValueError(f"Unknown method: {method}")


def get_attention_map_alternative(model, image_tensor, model_type):
    """
    备用方法：使用最后一层的注意力权重或CLS token与patch的相似度（如果可用）
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        if model_type == "dinov3":
            return compute_attention_map(model, image_tensor, model_type, method='feature_norm')
            
        elif model_type == "custom" or model_type == "aide":
            try:
                outputs = model(image_tensor)
                patch_features = outputs[:, 1:, :]
                
                if patch_features.numel() == 0:
                    print(f"    ⚠ Warning: Empty patch features in alternative method for {model_type}")
                    return np.zeros((14, 14))
                
                # 计算patch之间的平均相似度
                patch_features_norm = F.normalize(patch_features, dim=-1)
                similarity = torch.matmul(patch_features_norm, patch_features_norm.transpose(1, 2))
                attention_weights = similarity.mean(dim=1)  # 对所有patch求平均
                
            except Exception as e:
                print(f"    ✗ Error in alternative method for {model_type}: {e}")
                return np.zeros((14, 14))
            
        elif model_type == "clip" or model_type == "univfd" or model_type == "drct": 
            # CLIP、UnivFD、DRCT - 使用CLS token对各patch的注意力
            try:
                x = model.visual.conv1(image_tensor)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                
                cls_token = model.visual.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + model.visual.positional_embedding
                x = model.visual.ln_pre(x)
                
                x = x.permute(1, 0, 2)
                x = model.visual.transformer(x)
                x = x.permute(1, 0, 2)
            
                if x.numel() == 0:
                    print(f"    ⚠ Warning: Empty transformer output for {model_type}")
                    return np.zeros((16, 16))  # CLIP ViT-L/14 是 16x16
                
                # 计算CLS token与patch的相似度
                cls_feature = x[:, 0:1, :]  # [B, 1, C]
                patch_features = x[:, 1:, :]  # [B, N, C]
                
                if patch_features.numel() == 0:
                    print(f"    ⚠ Warning: Empty patch features for {model_type}")
                    return np.zeros((16, 16))
                
                cls_norm = F.normalize(cls_feature, dim=-1)
                patch_norm = F.normalize(patch_features, dim=-1)
                
                attention_weights = torch.matmul(cls_norm, patch_norm.transpose(1, 2)).squeeze(1)  # [B, N]
                
            except Exception as e:
                print(f"    ✗ Error in alternative method for {model_type}: {e}")
                import traceback
                traceback.print_exc()
                return np.zeros((16, 16))
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
        if attention_weights.numel() == 0:
            print(f"    ⚠ Warning: Empty attention weights for {model_type}")
            patch_size = 16 if model_type in ["clip", "effort"] else 14
            return np.zeros((patch_size, patch_size))
        
        batch_size, num_patches = attention_weights.shape
        patch_size = int(np.sqrt(num_patches))
        attention_map = attention_weights.reshape(patch_size, patch_size)
        
        att_min = attention_map.min()
        att_max = attention_map.max()
        
        if att_max == att_min:
            print(f"    ⚠ Warning: Uniform attention map for {model_type}, using 0.5")
            attention_map = torch.ones_like(attention_map) * 0.5
        else:
            attention_map = (attention_map - att_min) / (att_max - att_min + 1e-8)
        
        return attention_map.cpu().numpy()

def compute_multi_scale_attention(model, image_tensor, model_type):
    """
    多尺度注意力：结合特征范数和相似度 (或GradCAM作为主要)
    """
    # 🔥 添加输入检查
    if image_tensor is None or image_tensor.numel() == 0:
        print(f"    ⚠ Warning: Empty input tensor for {model_type}")
        patch_size = 16 if model_type in ["clip", "effort"] else 14
        return np.zeros((patch_size, patch_size))
    
    # 对于 DINOv3、CLIP、UnivFD 和 DRCT，直接使用 feature_norm
    if model_type in ["clip", "univfd", "drct"]:  # 修改这里,删除 "effort"
        print(f"    Using feature_norm for {model_type} attention map (GradCAM disabled).")
        try:
            attention_norm = compute_attention_map(model, image_tensor, model_type, method='feature_norm')
            
            if attention_norm is None or (isinstance(attention_norm, np.ndarray) and attention_norm.size == 0):
                print(f"    ⚠ Warning: Empty attention map from feature_norm")
                patch_size = 16 if model_type in ["clip", "effort"] else 14
                return np.zeros((patch_size, patch_size))
            
            attention_sim = get_attention_map_alternative(model, image_tensor, model_type)
            
            if attention_sim is None or (isinstance(attention_sim, np.ndarray) and attention_sim.size == 0):
                attention_sim = attention_norm
            
            attention_combined = 0.6 * attention_norm + 0.4 * attention_sim
            
            # 🔥 归一化前检查
            if attention_combined.max() == attention_combined.min():
                attention_combined = np.ones_like(attention_combined) * 0.5
            else:
                attention_combined = (attention_combined - attention_combined.min()) / (attention_combined.max() - attention_combined.min() + 1e-8)
            
            return attention_combined
            
        except Exception as e:
            print(f"    ✗ Error in attention computation for {model_type}: {e}")
            patch_size = 16 if model_type in ["clip", "effort"] else 14
            return np.zeros((patch_size, patch_size))
    
    # 对于其他模型，优先尝试GradCAM
    try:
        attention_map_gradcam = get_gradcam_attention_map(model, image_tensor, model_type)
        
        if attention_map_gradcam is not None and attention_map_gradcam.max() > 0:
            print(f"    Using GradCAM for {model_type} attention map.")
            return attention_map_gradcam
    except Exception as e:
        print(f"    ⚠ GradCAM failed for {model_type}: {e}")

    print(f"    GradCAM failed for {model_type}, falling back to feature_norm + similarity.")
    
    try:
        # 方法1: 特征范数
        attention_norm = compute_attention_map(model, image_tensor, model_type, method='feature_norm')
        
        if attention_norm is None or (isinstance(attention_norm, np.ndarray) and attention_norm.size == 0):
            print(f"    ⚠ Warning: Empty attention_norm")
            patch_size = 16 if model_type in ["clip", "effort"] else 14
            return np.zeros((patch_size, patch_size))
        
        # 方法2: 相似度
        attention_sim = get_attention_map_alternative(model, image_tensor, model_type)
        
        if attention_sim is None or (isinstance(attention_sim, np.ndarray) and attention_sim.size == 0):
            print(f"    ⚠ Warning: Empty attention_sim, using attention_norm only")
            attention_sim = attention_norm
        
        # 融合两种方法
        attention_combined = 0.6 * attention_norm + 0.4 * attention_sim
        
        # 🔥 再次归一化前检查
        if attention_combined.max() == attention_combined.min():
            attention_combined = np.ones_like(attention_combined) * 0.5
        else:
            attention_combined = (attention_combined - attention_combined.min()) / (attention_combined.max() - attention_combined.min() + 1e-8)
        
        return attention_combined
        
    except Exception as e:
        print(f"    ✗ Error in fallback attention computation for {model_type}: {e}")
        import traceback
        traceback.print_exc()
        patch_size = 16 if model_type in ["clip", "effort"] else 14
        return np.zeros((patch_size, patch_size))





def compute_pca_features(dense_features):
    """计算PCA特征"""
    batch_size, feature_dim, h, w = dense_features.shape
    features_flat = dense_features.squeeze(0).permute(1, 2, 0).reshape(-1, feature_dim).cpu().numpy()
    
    try:
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features_flat)
    except:
        feature_variance = np.var(features_flat, axis=0)
        top_3_indices = np.argsort(feature_variance)[-3:]
        pca_features = features_flat[:, top_3_indices]
    
    pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
    pca_image = (pca_features * 255).astype(np.uint8).reshape(h, w, 3)
    
    return pca_image


# ============================================================================
# Motion Blur Analysis Functions - 主要修改这里
# ============================================================================
def analyze_motion_blur_effects(model, processor, image_files, output_dir, model_name, model_type, clip_preprocess=None):
    print(f"{'='*80}")
    print(f"MOTION BLUR ANALYSIS - [{model_name}] Attention Map Similarity Collection")
    print('='*80)
    
    min_strength, max_strength, step = BLUR_STRENGTH_RANGE
    blur_strengths = np.arange(min_strength, max_strength + step, step)
    blur_strengths = [round(s, 3) for s in blur_strengths]
    
    print(f"Blur strengths to test: {blur_strengths}")
    print(f"Processing {len(image_files)} images...")
    print(f"Using attention computation method: multi-scale (GradCAM + feature_norm + similarity)")
    
    # 存储所有图片的相似度数据
    # 结构: {image_name: {blur_strength: similarity, ...}}
    all_image_similarities = {} 
    
    for img_idx, image_path in enumerate(image_files):
        image_name = image_path.stem
        print(f"[{img_idx+1}/{len(image_files)}] Analyzing motion blur effects for: {image_name}")
        
        try:
            original_image = Image.open(image_path).convert('RGB')
            original_size = original_image.size
            
            # 计算原始图片的注意力图
            original_image_tensor, _, _ = preprocess_image_for_blur_analysis(original_image, model_type, processor, clip_preprocess)
            original_attention_map = compute_multi_scale_attention(model, original_image_tensor, model_type)
            
            if original_attention_map.max() == 0:
                print(f"    ⚠ Warning: Original attention map for {image_name} is all zeros! Skipping this image.")
                continue

            image_similarities = {}
            
            for strength in blur_strengths: # 迭代所有的模糊强度
                print(f"  Processing blur strength: {strength}")
                
                if strength == 0.0:
                    test_image = original_image
                    attention_map = original_attention_map
                else:
                    test_image = apply_motion_blur_to_pil_image(original_image, strength, BLUR_ANGLE)
                    image_tensor, _, _ = preprocess_image_for_blur_analysis(test_image, model_type, processor, clip_preprocess)
                    attention_map = compute_multi_scale_attention(model, image_tensor, model_type)
                
                # 计算与原始注意力图的相似度 (皮尔逊相关系数)
                if attention_map.max() == 0: # 如果模糊后的注意力图也全为0
                    similarity = 0.0 # 视为不相似
                    print(f"    ⚠ Warning: Blurred attention map for {image_name} at strength {strength} is all zeros, similarity set to 0.")
                else:
                    correlation = np.corrcoef(original_attention_map.flatten(), attention_map.flatten())[0, 1]
                    similarity = correlation if not np.isnan(correlation) else 0.0
                
                image_similarities[f'{strength:.3f}'] = float(similarity) # 保存为浮点数
                print(f"    Similarity to original: {similarity:.4f}")
            
            all_image_similarities[image_name] = image_similarities
            print(f"  ✓ Completed motion blur analysis for {image_name}")
            
        except Exception as e:
            print(f"  ✗ Error analyzing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 将结果保存到JSON文件
    output_filepath = os.path.join(output_dir, MOTION_BLUR_OUTPUT_DIR, f'{model_name}_attention_similarities.json')
    Path(os.path.dirname(output_filepath)).mkdir(parents=True, exist_ok=True)
    with open(output_filepath, 'w') as f:
        json.dump(all_image_similarities, f, indent=4)
    print(f"✓ Attention similarity data saved to: {output_filepath}")
    
    print(f"Motion blur analysis completed for model {model_name}!")
    return all_image_similarities # 返回收集到的数据


def plot_attention_similarity_curve(all_models_avg_similarities, output_base_dir, 
                                   blur_strengths, subset_name="all"):  # 添加 subset_name 参数
    """
    绘制cosine similarity相似度曲线图
    """
    print(f"  Generating curve for subset: {subset_name}")

    plt.figure(figsize=(8, 6))


    # 定义模型名称到颜色和标记的映射 (更新模型列表)
    model_plot_configs = {
        'student': {'label': 'DINOV3', 'color': COL_DINO, 'marker': 'o'},
        'custom_vit': {'label': 'Community Forensics', 'color': COL_CUSTOM_VIT, 'marker': '^'},
        'clip': {'label': 'CLIP', 'color': COL_CLIP, 'marker': 's'},
        'univfd': {'label': 'UnivFD', 'color': COL_UNIVFD, 'marker': 'D'},  
        'drct': {'label': 'DRCT', 'color': COL_DRCT_CONVB, 'marker': 'v'},  
        'aide': {'label': 'AIDE', 'color': COL_AIDE, 'marker': 'p'},  
    }

    # 确保 blur_strengths 是浮点数数组
    plot_blur_strengths = np.array(blur_strengths).astype(float)
    
    for model_name, avg_similarities in all_models_avg_similarities.items():
        if model_name in model_plot_configs:
            config = model_plot_configs[model_name]
            # 确保 avg_similarities 也是浮点数数组，并且与 blur_strengths 长度匹配
            plot_similarities = np.array([
                avg_similarities.get(f'{s:.3f}', 0.0) for s in blur_strengths
            ]).astype(float)
            
            plt.plot(plot_blur_strengths, plot_similarities, 
                     marker=config['marker'], linewidth=LINEWIDTH, markersize=MARKERSIZE,
                     label=config['label'], color=config['color'])
        else:
            print(f"    Warning: Model '{model_name}' not configured for plotting, skipping.")

    # 根据子集名称调整标题
    subset_display_names = {
        "0_real": "REAL Images (0_real)",
        "1_fake": "FAKE Images (1_fake)",
        "all": "AIGCDetectBenchmark Wukong Images"
    }
    title_suffix = subset_display_names.get(subset_name, subset_name)
    
    plt.title(f'Cosine Similarity Consistency Against Motion Blur - {title_suffix}',  # 修改
              fontsize=16, pad=14)
    plt.xlabel('Motion Blur Strength', fontsize=13, labelpad=8)
    plt.ylabel('Cosine Similarity Matrix Correlation', fontsize=13, labelpad=8)  # 修改

    plt.ylim(0.0, 1.05)
    plt.xlim(-0.005, 0.205)
    plt.xticks(plot_blur_strengths)
    plt.tick_params(axis='both', which='major', labelsize=11, length=6, width=1)

    plt.grid(True, which='major', color=GRID_COLOR, alpha=GRID_ALPHA,
             linestyle=GRID_LS, linewidth=GRID_LW)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    leg = plt.legend(loc='lower left', frameon=True, fancybox=True, framealpha=0.9,
                     fontsize=11, title='Models')
    plt.setp(leg.get_title(), fontsize=11)

    plt.tight_layout()  

    # 根据子集名称生成文件名
    output_filename = f'cosine_similarity_vs_blur_{subset_name}.png'  # 修改
    output_path = os.path.join(output_base_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Curve saved to: {output_path}")


# ============================================================================
# Main Function - 支持多个模型和多个子集分析
# ============================================================================
def main(batch_size=8, use_batch_processing=False, max_workers=4):
    """主函数 - 支持多模型、多子集批量处理"""
    print("Dense Feature Visualization - Multi-Model Multi-Subset Analysis")
    print("=" * 80)
    
    # 定义要测试的模型 (添加 Effort 和 AIDE)
    models_to_test = [
        {
            "name": "student",
            "display_name": "Dinov3",
            "type": "dinov3_checkpoint",
            "path": STUDENT_CHECKPOINT
        },
        {
            "name": "custom_vit",
            "display_name": "Community Forensics",
            "type": "custom"
        },
        {
            "name": "clip",
            "display_name": "CLIP ViT-H/14",
            "type": "clip"
        },
        {
            "name": "univfd", 
            "display_name": "UnivFD",
            "type": "univfd",
            "path": UNIVFD_CHECKPOINT
        },
        {
            "name": "drct_convnext",  # 新增DRCT ConvNeXt
            "display_name": "DRCT ConvB",
            "type": "drct_convnext",
            "path": DRCT_CONVNEXT_CHECKPOINT
        }
    ]
    
    
    # 创建主输出文件夹
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)
        print(f"Created main output directory: {OUTPUT_BASE_DIR}")
    
    # 加载DINOv3专用的processor
    print("Loading DINOv3 image processor...")
    dinov3_processor = AutoImageProcessor.from_pretrained(BASE_MODEL_PATH)
    print("✓ DINOv3 Processor loaded\n")
    
    # 收集图片文件 - 读取所有图片
    print("=" * 80)
    print("Collecting REAL images (0_real)...")
    print("=" * 80)
    real_image_files = get_image_files_from_directory(REAL_ORIGINAL_DIR, num_images=None)  # None表示读取所有
    
    print("" + "=" * 80)
    print("Collecting FAKE images (1_fake)...")
    print("=" * 80)
    fake_image_files = get_image_files_from_directory(FAKE_ORIGINAL_DIR, num_images=None)  # None表示读取所有
    
    # 合并所有图片
    all_image_files = real_image_files + fake_image_files
    
    if not all_image_files:
        print("No images found! Exiting.")
        return
    
    total_count = len(all_image_files)
    real_count = len(real_image_files)
    fake_count = len(fake_image_files)
    
    print(f"{'='*80}")
    print(f"Dataset Summary:")
    print(f"  REAL images (0_real): {real_count}")
    print(f"  FAKE images (1_fake): {fake_count}")
    print(f"  Total images: {total_count}")
    print(f"{'='*80}")
    
    # 计算blur_strengths列表
    min_strength, max_strength, step = BLUR_STRENGTH_RANGE
    blur_strengths = np.arange(min_strength, max_strength + step, step)
    blur_strengths = [round(s, 3) for s in blur_strengths]
    
    # 定义要分析的子集
    subsets_to_analyze = [
        {
            "name": "all",
            "display_name": "All Images (0_real + 1_fake)",
            "image_files": all_image_files
        }
    ]
    
    # 存储所有模型、所有子集的平均相似度数据
    # 结构: {subset_name: {model_name: {blur_strength_str: avg_similarity, ...}}}
    all_subsets_all_models_data = {}
    
    # 处理每个模型
    for model_config in models_to_test:
        model_name = model_config["name"]
        model_type = model_config["type"]
        
        print(f"{'#'*100}")
        print(f"# Processing Model: {model_config['display_name']} (Type: {model_type})")
        print(f"{'#'*100}")
        
        # 创建模型专属输出文件夹
        model_output_dir = os.path.join(OUTPUT_BASE_DIR, model_name)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
            print(f"Created model output directory: {model_output_dir}")
        
        current_model = None
        current_processor = None
        current_clip_preprocess = None
    
        try:
            if model_type == "dinov3_checkpoint":
                current_model = load_model_from_checkpoint(model_config["path"], BASE_MODEL_PATH)
                current_processor = dinov3_processor
                model_type = "dinov3"
                if current_model is None:
                    print(f"✗ Failed to load DINOv3 model '{model_config['display_name']}', skipping...")
                    continue
                    
            elif model_type == "custom":
                current_model, _ = load_custom_vit_model()
                if current_model is None:
                    print(f"✗ Failed to load Custom ViT model '{model_config['display_name']}', skipping...")
                    continue
                    
            elif model_type == "clip":
                current_model, current_clip_preprocess = load_clip_model()
                if current_model is None:
                    print(f"✗ Failed to load CLIP model '{model_config['display_name']}', skipping...")
                    continue
                    
            elif model_type == "univfd":  # 新增
                current_model, current_clip_preprocess = load_univfd_model()
                if current_model is None:
                    print(f"✗ Failed to load UnivFD model '{model_config['display_name']}', skipping...")
                    continue
                    
            elif model_type == "drct_convnext":  # 新增处理
                current_model, _ = load_drct_convnext_model()
                if current_model is None:
                    print(f"✗ Failed to load DRCT ConvNeXt model '{model_display_name}', skipping...")
                    continue
                    
                    
            elif model_type == "aide":
                current_model, _ = load_aide_model()
                model_type = "custom"
                if current_model is None:
                    print(f"✗ Failed to load AIDE model '{model_config['display_name']}', skipping...")
                    continue
                    
            
            else:
                print(f"✗ Unknown model type '{model_type}', skipping...")
                continue

            current_model.eval()
            
            # 确认模型在GPU上
            if next(current_model.parameters()).is_cuda:
                print(f"✓ Model confirmed on GPU: {DEVICE}")
            else:
                print(f"⚠ Warning: Model is on CPU, moving to {DEVICE}...")
                current_model = current_model.to(DEVICE)
            
        except Exception as e:
            print(f"✗ Error loading model '{model_config['display_name']}': {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 对每个子集进行分析
        for subset_config in subsets_to_analyze:
            subset_name = subset_config["name"]
            subset_display_name = subset_config["display_name"]
            subset_image_files = subset_config["image_files"]
            
            if len(subset_image_files) == 0:
                print(f"\n⚠ Warning: No images in subset '{subset_name}', skipping...")
                continue
            
            print(f"{'='*80}")
            print(f"Analyzing subset: {subset_display_name}")
            print(f"Model: {model_config['display_name']}")
            print('='*80)
            
            # 调用子集分析函数
            subset_avg_similarities, all_image_similarities = analyze_subset(
                current_model, current_processor, subset_image_files, 
                model_name, model_type, current_clip_preprocess, 
                blur_strengths, subset_display_name
            )
            
            # 保存该模型、该子集的详细数据到JSON
            subset_output_dir = os.path.join(model_output_dir, MOTION_BLUR_OUTPUT_DIR, subset_name)
            Path(subset_output_dir).mkdir(parents=True, exist_ok=True)
            
            # 保存每张图片的相似度数据
            detailed_filepath = os.path.join(
                subset_output_dir, 
                f'{model_name}_{subset_name}_cosine_similarities.json'  # 修改
            )
            with open(detailed_filepath, 'w') as f:
                json.dump(all_image_similarities, f, indent=4)
            print(f"  ✓ Detailed similarity data saved to: {detailed_filepath}")
            
            # 保存平均相似度数据
            avg_filepath = os.path.join(
                subset_output_dir, 
                f'{model_name}_{subset_name}_avg_cosine_similarities.json'  # 修改
            )
            with open(avg_filepath, 'w') as f:
                json.dump(subset_avg_similarities, f, indent=4)
            print(f"  ✓ Average similarity data saved to: {avg_filepath}")
            
            # 存储到总数据结构中
            if subset_name not in all_subsets_all_models_data:
                all_subsets_all_models_data[subset_name] = {}
            all_subsets_all_models_data[subset_name][model_name] = subset_avg_similarities
        
        # 清理内存
        del current_model
        if current_clip_preprocess is not None:
            del current_clip_preprocess
        torch.cuda.empty_cache()
        print(f"✓ Model '{model_config['display_name']}' cleaned from memory")
    
    # 所有模型、所有子集处理完毕后，为每个子集绘制折线图
    print(f"{'='*80}")
    print("All models and subsets processed. Generating final curves...")
    print('='*80)
    
    final_output_dir = os.path.join(OUTPUT_BASE_DIR, MOTION_BLUR_OUTPUT_DIR)
    Path(final_output_dir).mkdir(parents=True, exist_ok=True)
    
    for subset_name, models_data in all_subsets_all_models_data.items():
        if models_data:
            print(f"Generating curve for subset: {subset_name}")
            
            # 保存该子集所有模型的汇总数据
            subset_summary_filepath = os.path.join(
                final_output_dir, 
                f'{subset_name}_all_models_avg_similarities.json'
            )
            with open(subset_summary_filepath, 'w') as f:
                json.dump(models_data, f, indent=4)
            print(f"  ✓ Subset summary saved to: {subset_summary_filepath}")
            
            # 绘制该子集的折线图
            plot_attention_similarity_curve(
                models_data, 
                final_output_dir, 
                blur_strengths, 
                subset_name=subset_name  # 传入子集名称用于文件命名
            )
    
    # 最终总结
    print(f"{'#'*100}")
    print("ALL PROCESSING COMPLETED!")
    print(f"{'#'*100}")
    print(f"Total models tested: {len(models_to_test)}")
    print(f"Total subsets analyzed: {len(subsets_to_analyze)}")
    print(f"Dataset summary:")
    print(f"  REAL images (0_real): {real_count}")
    print(f"  FAKE images (1_fake): {fake_count}")
    print(f"  Total images: {total_count}")
    print(f"Main output directory: {OUTPUT_BASE_DIR}")
    print(f"Blur intensity range: {BLUR_STRENGTH_RANGE}")
    print(f"Final curves and data saved in: {final_output_dir}")
    print(f"{'#'*100}")

if __name__ == "__main__":
    # 配置参数
    BATCH_SIZE = 8 # 不再使用，但可以保留
    USE_BATCH_PROCESSING = False  # 关闭常规批量可视化
    MAX_WORKERS = 4 # 不再使用，但可以保留
    
    main(batch_size=BATCH_SIZE, use_batch_processing=USE_BATCH_PROCESSING, max_workers=MAX_WORKERS)
