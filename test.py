import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
import time
from datetime import datetime
from transformers import AutoModel
warnings.filterwarnings('ignore')

# ===============================
# 配置参数
# ===============================
# 模糊测试配置
BLUR_MODE = "both"  # "no_blur", "global", "both" - both表示同时测试两种情况
BLUR_PROB = 1.0
BLUR_STRENGTH_RANGE = (0.1, 0.3)
MODEL_TEST_MODE = "both"# "teacher", "student", "both" 

"""
"""

# 测试集配置
TEST_DATASETS = {
        'cyclegan': {
        'type': 'multi_class',
        'base_path': '/home/work/xueyunqi/11ar_datasets/test/cyclegan',
        'classes': None  # None 表示自动发现所有类别
    },
    'progan': {
        'type': 'multi_class', 
        'base_path': '/home/work/xueyunqi/11ar_datasets/test/progan',
        'classes': None
    },
    'stylegan2': {
        'type': 'multi_class',
        'base_path': '/home/work/xueyunqi/11ar_datasets/test/stylegan2', 
        'classes': None
    },
    'stylegan': {
        'type': 'multi_class',
        'base_path': '/home/work/xueyunqi/11ar_datasets/test/stylegan',
        'classes': None
    },
    'adm': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/ADM/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/ADM/1_fake"
    },
    'vqdm': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/VQDM/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/VQDM/1_fake"
    },
        'sdv14': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/stable_diffusion_v_1_4/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/stable_diffusion_v_1_4/1_fake"
    },

        'sdv15': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/stable_diffusion_v_1_5/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/stable_diffusion_v_1_5/1_fake"
    },
    'stargan': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/stargan/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/stargan/1_fake"
    },
        'wukong': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/wukong/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/wukong/1_fake"
    },
            'dalle2': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/datasets/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test/DALLE2/0_real",
        'fake_folder': "/home/work/xueyunqi/datasets/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test/DALLE2/1_fake"
    },
                'midjourney': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/Midjourney/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/Midjourney/1_fake"
    },
                'biggan': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/biggan/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/biggan/1_fake"
    },
                    'sd-xl': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/sd_xl/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/sd_xl/1_fake"
    },
                    'gaugan': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/gaugan/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/gaugan/1_fake"
    },
                    'whichfaceisreal': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/whichfaceisreal/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/whichfaceisreal/1_fake"
    },
    'glide': {
        'type': 'simple',
        'real_folder': "/home/work/xueyunqi/11ar_datasets/test/Glide/0_real",
        'fake_folder': "/home/work/xueyunqi/11ar_datasets/test/Glide/1_fake"
    }
}



# 测试模式配置
TEST_ALL_DATASETS = True
SELECTED_TEST_DATASET = 'adm'

# 模型路径配置 - 支持Teacher和Student模型
TEACHER_MODEL_PATH = "/home/work/xueyunqi/11ar/checkpoints/teacher/0917_dinov3_teacher_student/nofocal_best_teacher_model.pth"
STUDENT_MODEL_PATH = "/home/work/xueyunqi/11ar/checkpoints/student/0917_dinov3_teacher_student/nofocal_best_student_model.pth"


#TEACHER_MODEL_PATH = "/home/work/xueyunqi/11ar/final_dinov3/checkpoints/teacher/0917_dinov3_teacher_student/best_teacher_model.pth"
#STUDENT_MODEL_PATH = "/home/work/xueyunqi/11ar/final_dinov3/checkpoints/student/0917_dinov3_teacher_student/best_student_model.pth"
# STUDENT_MODEL_PATH = "/home/work/xueyunqi/11ar/final_dinov3/checkpoints/student/0918_teacher7b_student_vitl/best_student_vitl_model.pth"
DINOV3_MODEL_ID = "/home/work/xueyunqi/dinov3-vit7b16-pretrain-lvd1689m"

# 测试配置
DEVICE = torch.device('cuda:0') 
BATCH_SIZE = 256
NUM_WORKERS = 1
PROJECTION_DIM = 512

# 输出目录
OUTPUT_DIR = "1026_simclr_student_test_results"

# ===============================
# 从train代码导入的网络架构
# ===============================
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class ImprovedDinoV3Adapter(nn.Module):
    """改进的DinoV3模型适配器 - 从train代码复制"""
    
    def __init__(self, model_path, num_classes=2, projection_dim=512, 
                 adapter_layers=2, dropout_rate=0.1, device="cuda"):
        super().__init__()
        
        self.device = device
        self.projection_dim = projection_dim
        
        print(f"Loading DinoV3 model: {model_path}")
        
        # 加载DinoV3 backbone
        self.backbone = self._load_dinov3_backbone(model_path, device)
        self.hidden_size = getattr(self.backbone.config, "hidden_size", 4096)
        
        print(f"DinoV3 feature dimension: {self.hidden_size}")
        
        # 特征投影器 - 多层设计
        projection_layers = []
        current_dim = self.hidden_size
        
        for i in range(adapter_layers):
            if i == adapter_layers - 1:  # 最后一层
                projection_layers.extend([
                    nn.Linear(current_dim, projection_dim),
                    nn.LayerNorm(projection_dim)
                ])
            else:
                next_dim = max(projection_dim, current_dim // 2)
                projection_layers.extend([
                    nn.Linear(current_dim, next_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.LayerNorm(next_dim)
                ])
                current_dim = next_dim
        
        self.projection = nn.Sequential(*projection_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim // 2, num_classes)
        )
        
        # 移动到指定设备
        self.projection = self.projection.to(device)
        self.classifier = self.classifier.to(device)
        
        print(f"✓ DinoV3 Adapter initialized:")
        print(f"  - Backbone: {self.hidden_size}D")
        print(f"  - Projection: {projection_dim}D ({adapter_layers} layers)")
        print(f"  - Classes: {num_classes}")
        
        # 初始化权重
        self._initialize_weights()
    
    def _load_dinov3_backbone(self, model_path, device):
        """加载DinoV3 backbone"""
        try:
            # 加载到CPU后再转移到目标设备
            backbone = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            
            # 移动到目标设备并设置为eval模式
            backbone = backbone.to(device).eval()
            
            # 冻结backbone参数
            for param in backbone.parameters():
                param.requires_grad = False
            
            return backbone
            
        except Exception as e:
            print(f"Error loading DinoV3 backbone: {e}")
            raise e
    
    def _initialize_weights(self):
        """初始化投影和分类层权重"""
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.projection.apply(init_layer)
        self.classifier.apply(init_layer)
    
    def extract_features(self, pixel_values):
        """提取DinoV3特征"""
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
            
            # 提取CLS token特征
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0]  # CLS token
            
            return features.float()  # 确保输出为float32
    
    def forward(self, pixel_values, return_features=False):
        """前向传播"""
        # 特征提取（frozen backbone）
        raw_features = self.extract_features(pixel_values)
        
        # 投影和分类（trainable layers）
        projected_features = self.projection(raw_features)
        logits = self.classifier(projected_features)
        
        if return_features:
            return {
                'logits': logits,
                'projected_features': projected_features,
                'raw_features': raw_features
            }
        else:
            return logits
    
    def get_projected_features(self, pixel_values):
        """获取投影特征"""
        raw_features = self.extract_features(pixel_values)
        projected_features = self.projection(raw_features)
        return projected_features

class TeacherStudentNetwork(nn.Module):
    """Teacher-Student网络 - 从train代码复制"""
    
    def __init__(self, dinov3_model_path, num_classes=2, projection_dim=512, device="cuda"):
        super().__init__()
        
        self.device = device
        
        print(f"{'='*60}")
        print("INITIALIZING TEACHER-STUDENT NETWORK")
        print(f"{'='*60}")
        
        # Teacher模型 - DinoV3 Adapter
        print("Creating Teacher model (DinoV3 + Adapter)...")
        self.teacher = ImprovedDinoV3Adapter(
            model_path=dinov3_model_path,
            num_classes=num_classes,
            projection_dim=projection_dim,
            adapter_layers=3,
            dropout_rate=0.1,
            device=device
        )
        
        # Student模型 - 共享backbone但独立的投影和分类层
        print("Creating Student model (shared backbone + independent head)...")
        self.student_projection = nn.Sequential(
            nn.Linear(self.teacher.hidden_size, projection_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim)
        ).to(device)
        
        self.student_classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim // 2, num_classes)
        ).to(device)
        
        # 初始化Student权重
        self._initialize_student_weights()
        
        print(f"✓ Teacher-Student network initialized")
        print(f"  - Shared DinoV3 backbone: {self.teacher.hidden_size}D")
        print(f"  - Projection dimension: {projection_dim}D")
        print(f"  - Number of classes: {num_classes}")
    
    def _initialize_student_weights(self):
        """初始化Student层权重"""
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.student_projection.apply(init_layer)
        self.student_classifier.apply(init_layer)
    
    def forward_teacher(self, pixel_values):
        """Teacher前向传播"""
        outputs = self.teacher(pixel_values, return_features=True)
        return outputs['projected_features'], outputs['logits']
    
    def forward_student(self, pixel_values):
        """Student前向传播 - 使用Teacher的backbone"""
        # 使用Teacher的backbone提取特征
        raw_features = self.teacher.extract_features(pixel_values)
        
        # Student独有的投影和分类
        projected_features = self.student_projection(raw_features)
        logits = self.student_classifier(projected_features)
        
        return projected_features, logits

# ===============================
# 运动模糊函数
# ===============================
def apply_motion_blur_batch_efficient(images, strength):
    """高效的批量运动模糊处理"""
    if not isinstance(images, torch.Tensor):
        return images
    
    device = images.device
    batch_size, channels, height, width = images.shape
    
    # 计算kernel_size
    kernel_size = int(5 + (strength - 0.05) * 44.44)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_size = max(3, min(kernel_size, 31))
    
    # 创建运动模糊核 (水平方向)
    kernel = torch.zeros(kernel_size, kernel_size, device=device)
    kernel[kernel_size//2, :] = 1.0
    kernel = kernel / kernel.sum()
    
    # 扩展kernel为4D: (out_channels, in_channels, h, w)
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)
    
    # 使用depthwise convolution应用模糊
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel, padding=padding, groups=channels)
    
    return blurred

# ===============================
# 数据变换
# ===============================
def get_test_transforms():
    """测试时的数据变换"""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# ===============================
# 通用测试数据集类
# ===============================
class MultiTestDataset(Dataset):
    """支持多种测试数据集结构的通用数据集类"""

    def __init__(self, dataset_config, transform, 
                 blur_mode="no_blur", blur_prob=0.0, blur_strength_range=(0.1, 0.5)):
        self.transform = transform
        self.blur_mode = blur_mode
        self.blur_prob = blur_prob
        self.blur_strength_range = blur_strength_range
        self.dataset_config = dataset_config
        
        # 根据数据集类型加载图像路径
        self.data = []
        
        if dataset_config['type'] == 'simple':
            # 简单结构：直接的real和fake文件夹
            self._load_simple_dataset(dataset_config)
        elif dataset_config['type'] == 'multi_class':
            # 多类别结构：每个类别下有real和fake文件夹
            self._load_multi_class_dataset(dataset_config)
        
        print(f"Total test samples loaded: {len(self.data)}")
    
    def _load_simple_dataset(self, config):
        """加载简单结构的数据集"""
        real_images = self._get_image_files(config['real_folder'])
        fake_images = self._get_image_files(config['fake_folder'])
        
        # 添加真实图像
        for img_path in real_images:
            self.data.append((img_path, 0, 'real'))  # (path, label, category)
        
        # 添加假图像
        for img_path in fake_images:
            self.data.append((img_path, 1, 'fake'))  # (path, label, category)
        
        print(f"Simple dataset loaded: {len(real_images)} real, {len(fake_images)} fake images")
    
    def _load_multi_class_dataset(self, config):
        """加载多类别结构的数据集（如StyleGAN）- 支持自动发现类别"""
        base_path = Path(config['base_path'])
        
        # 如果没有指定 classes，自动发现所有类别文件夹
        if config.get('classes') is None:
            print(f"Auto-discovering classes in {base_path}")
            discovered_classes = []
            
            for item in base_path.iterdir():
                if item.is_dir():
                    # 检查是否有 0_real 或 1_fake 子文件夹
                    real_subfolder = item / "0_real"
                    fake_subfolder = item / "1_fake"
                    if real_subfolder.exists() or fake_subfolder.exists():
                        discovered_classes.append(item.name)
            
            discovered_classes = sorted(discovered_classes)  # 排序保证一致性
            print(f"Discovered classes: {discovered_classes}")
            classes_to_process = discovered_classes
        else:
            classes_to_process = config['classes']
        
        total_real = 0
        total_fake = 0
        
        for class_name in classes_to_process:
            class_path = base_path / class_name
            
            # 检查类别文件夹是否存在
            if not class_path.exists():
                print(f"Warning: Class folder not found: {class_path}")
                continue
            
            real_folder = class_path / "0_real"
            fake_folder = class_path / "1_fake"
            
            # 加载该类别的真实图像
            if real_folder.exists():
                real_images = self._get_image_files(str(real_folder))
                for img_path in real_images:
                    self.data.append((img_path, 0, f'{class_name}_real'))
                total_real += len(real_images)
                print(f"  {class_name} real: {len(real_images)} images")
            else:
                print(f"Warning: Real folder not found: {real_folder}")
            
            # 加载该类别的假图像
            if fake_folder.exists():
                fake_images = self._get_image_files(str(fake_folder))
                for img_path in fake_images:
                    self.data.append((img_path, 1, f'{class_name}_fake'))
                total_fake += len(fake_images)
                print(f"  {class_name} fake: {len(fake_images)} images")
            else:
                print(f"Warning: Fake folder not found: {fake_folder}")
        
        print(f"Multi-class dataset loaded: {total_real} real, {total_fake} fake images across {len(classes_to_process)} classes")
        
    def _get_image_files(self, folder_path):
        """获取图像文件列表"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Warning: Folder not found at {folder_path}")
            return []
        
        image_files = []
        for img_file in folder.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in extensions:
                image_files.append(img_file)
        
        return sorted(image_files)

    def apply_blur_augmentation(self, tensor_image):
        """应用模糊增强到tensor图像"""
        if self.blur_mode == "no_blur" or random.random() > self.blur_prob:
            return tensor_image, {'mode': 'no_blur', 'blur_applied': False}
        
        blur_start_time = time.time()
        blur_strength = random.uniform(*self.blur_strength_range)
        blurred_tensor = apply_motion_blur_batch_efficient(tensor_image.unsqueeze(0), blur_strength).squeeze(0)
        
        blur_info = {
            'mode': 'global',
            'blur_strength': blur_strength,
            'blur_applied': True,
            'blur_time': time.time() - blur_start_time
        }
        
        return blurred_tensor, blur_info
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label, category = self.data[idx]
        
        # 加载原始图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用基础变换
        tensor_image = self.transform(image)
        
        # 返回图像、标签、文件名（包含类别信息）
        filename_with_category = f"{category}_{img_path.stem}"
        return tensor_image, label, filename_with_category

# ===============================
# 模型加载函数
# ===============================
def remove_module_prefix(state_dict):
    """移除state_dict中的'module.'前缀（DDP产生的）"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 移除'module.'前缀（7个字符）
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def load_teacher_student_model(teacher_model_path, student_model_path, device):
    """加载Teacher-Student模型"""
    print("Loading Teacher-Student model from checkpoint...")
    
    # 检查模型文件是否存在
    if not os.path.exists(teacher_model_path):
        print(f"Error: Teacher model not found at {teacher_model_path}")
        return None, None
    
    if not os.path.exists(student_model_path):
        print(f"Error: Student model not found at {student_model_path}")
        return None, None
    
    print(f"Loading Teacher checkpoint from: {teacher_model_path}")
    print(f"Loading Student checkpoint from: {student_model_path}")
    
    # 创建完整的Teacher-Student网络
    print("Creating Teacher-Student network...")
    model = TeacherStudentNetwork(
        dinov3_model_path=DINOV3_MODEL_ID,
        num_classes=2,
        projection_dim=PROJECTION_DIM,
        device=device
    )
    
    # 加载Teacher权重
    teacher_checkpoint = torch.load(teacher_model_path, map_location='cpu')
    if 'model_state_dict' in teacher_checkpoint:
        teacher_state_dict = teacher_checkpoint['model_state_dict']
    else:
        print("Error: No model_state_dict found in teacher checkpoint")
        return None, None
    
    teacher_state_dict = remove_module_prefix(teacher_state_dict)
    
    # 加载Student权重
    student_checkpoint = torch.load(student_model_path, map_location='cpu')
    if 'model_state_dict' in student_checkpoint:
        student_state_dict = student_checkpoint['model_state_dict']
    else:
        print("Error: No model_state_dict found in student checkpoint")
        return None, None
    
    student_state_dict = remove_module_prefix(student_state_dict)
    
    # 分别加载Teacher和Student的权重
    # Teacher权重
    teacher_weights = {}
    for key, value in teacher_state_dict.items():
        if key.startswith('teacher.'):
            teacher_weights[key] = value
    
    # Student权重
    student_projection_weights = {}
    student_classifier_weights = {}
    for key, value in student_state_dict.items():
        if key.startswith('student_projection.'):
            student_projection_weights[key[19:]] = value
        elif key.startswith('student_classifier.'):
            student_classifier_weights[key[19:]] = value
    
    # 加载权重到模型
    try:
        # 加载Teacher权重
        missing_keys, unexpected_keys = model.load_state_dict(teacher_weights, strict=False)
        print(f"Teacher weights loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        # 加载Student投影层权重
        if student_projection_weights:
            model.student_projection.load_state_dict(student_projection_weights, strict=False)
            print("✓ Student projection weights loaded")
        
        # 加载Student分类器权重
        if student_classifier_weights:
            model.student_classifier.load_state_dict(student_classifier_weights, strict=False)
            print("✓ Student classifier weights loaded")
        
        model = model.to(device)
        model.eval()
        
        # 创建独立的Teacher和Student模型用于推理
        teacher_model = TeacherWrapper(model)
        student_model = StudentWrapper(model)
        
        # 设置为多GPU推理
        if torch.cuda.device_count() > 1:
            print(f"Found {torch.cuda.device_count()} GPUs, using DataParallel for inference")
            teacher_model = nn.DataParallel(teacher_model)
            student_model = nn.DataParallel(student_model)
        
        print("✓ Teacher-Student models loaded successfully!")
        
        if 'best_acc' in teacher_checkpoint:
            print(f"Teacher training accuracy: {teacher_checkpoint['best_acc']:.2f}%")
        if 'best_acc' in student_checkpoint:
            print(f"Student training accuracy: {student_checkpoint['best_acc']:.2f}%")
        
        return teacher_model, student_model
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None, None

class TeacherWrapper(nn.Module):
    """Teacher模型包装器"""
    def __init__(self, teacher_student_model):
        super().__init__()
        self.model = teacher_student_model
    
    def forward(self, x):
        return self.model.forward_teacher(x)

class StudentWrapper(nn.Module):
    """Student模型包装器"""
    def __init__(self, teacher_student_model):
        super().__init__()
        self.model = teacher_student_model
    
    def forward(self, x):
        return self.model.forward_student(x)

# ===============================
# 评估函数 - 支持Teacher和Student
# ===============================
def evaluate_models(teacher_model, student_model, test_loader, dataset_name, blur_mode="no_blur"):
    """评估Teacher和Student模型"""
    print(f"Evaluating models on {dataset_name} with blur_mode: {blur_mode}, test_mode: {MODEL_TEST_MODE}...")
    
    teacher_model.eval()
    student_model.eval()
    
    # 根据测试模式决定要收集的结果
    results = {}
    
    # Teacher结果
    if MODEL_TEST_MODE in ["teacher", "both"]:
        teacher_predictions = []
        teacher_labels = []
        teacher_correct = 0
    
    # Student结果
    if MODEL_TEST_MODE in ["student", "both"]:
        student_predictions = []
        student_labels = []
        student_correct = 0
    
    all_filenames = []
    total_samples = 0
    
    # blur统计信息
    blur_stats = {
        'total_blur_time': 0,
        'blur_count': 0,
        'no_blur_count': 0
    }
    
    with torch.no_grad():
        for batch_idx, (tensor_image, labels, filenames) in enumerate(test_loader):
            batch_start_time = time.time()
            
            # 根据blur_mode处理图像（保持原有逻辑）
            if blur_mode == "no_blur":
                processed_imgs = tensor_image.to(DEVICE, non_blocking=True)
                batch_blur_info = [{'mode': 'no_blur', 'blur_applied': False}] * len(tensor_image)
            else:
                # 应用blur处理
                processed_imgs_list = []
                batch_blur_info = []
                
                for img_tensor in tensor_image:
                    if blur_mode == "global":
                        test_loader.dataset.blur_mode = "global"
                        test_loader.dataset.blur_prob = 1.0
                        blurred_tensor, blur_info = test_loader.dataset.apply_blur_augmentation(img_tensor)
                        processed_imgs_list.append(blurred_tensor)
                        batch_blur_info.append(blur_info)
                        
                        if blur_info['blur_applied']:
                            blur_stats['blur_count'] += 1
                            blur_stats['total_blur_time'] += blur_info.get('blur_time', 0)
                        else:
                            blur_stats['no_blur_count'] += 1
                
                processed_imgs = torch.stack(processed_imgs_list).to(DEVICE, non_blocking=True)
            
            # 根据测试模式进行推理
            if MODEL_TEST_MODE in ["teacher", "both"]:
                # Teacher推理
                teacher_features, teacher_logits = teacher_model(processed_imgs)
                _, teacher_pred = torch.max(teacher_logits.data, 1)
                
                # 统计准确率
                teacher_correct += (teacher_pred.cpu() == labels).sum().item()
                teacher_predictions.extend(teacher_pred.cpu().numpy())
                teacher_labels.extend(labels.numpy())
            
            if MODEL_TEST_MODE in ["student", "both"]:
                # Student推理
                student_features, student_logits = student_model(processed_imgs)
                _, student_pred = torch.max(student_logits.data, 1)
                
                # 统计准确率
                student_correct += (student_pred.cpu() == labels).sum().item()
                student_predictions.extend(student_pred.cpu().numpy())
                student_labels.extend(labels.numpy())
            
            total_samples += labels.size(0)
            all_filenames.extend(filenames)
            
            # 每10个batch显示一次进度
            if (batch_idx + 1) % 10 == 0:
                progress_msg = f"  [{dataset_name}-{blur_mode}] Batch {batch_idx+1}/{len(test_loader)} - "
                
                if MODEL_TEST_MODE in ["teacher", "both"]:
                    teacher_acc = 100 * teacher_correct / total_samples
                    progress_msg += f"Teacher Acc: {teacher_acc:.2f}%"
                
                if MODEL_TEST_MODE == "both":
                    progress_msg += " | "
                
                if MODEL_TEST_MODE in ["student", "both"]:
                    student_acc = 100 * student_correct / total_samples
                    progress_msg += f"Student Acc: {student_acc:.2f}%"
                
                print(progress_msg)
                
            # 清理GPU内存
            del processed_imgs
            if MODEL_TEST_MODE in ["teacher", "both"]:
                del teacher_logits, teacher_pred
            if MODEL_TEST_MODE in ["student", "both"]:
                del student_logits, student_pred
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 计算指标并构建结果
    result = {
        'dataset_name': dataset_name,
        'blur_mode': blur_mode,
        'model_test_mode': MODEL_TEST_MODE,
        'blur_stats': blur_stats,
        'total_samples': total_samples
    }
    
    # Teacher指标
    if MODEL_TEST_MODE in ["teacher", "both"]:
        teacher_accuracy = accuracy_score(teacher_labels, teacher_predictions)
        teacher_precision, teacher_recall, teacher_f1, _ = precision_recall_fscore_support(
            teacher_labels, teacher_predictions, average='binary')
        teacher_cm = confusion_matrix(teacher_labels, teacher_predictions)
        
        result['teacher'] = {
            'accuracy': float(teacher_accuracy),
            'precision': float(teacher_precision),
            'recall': float(teacher_recall),
            'f1_score': float(teacher_f1),
            'confusion_matrix': teacher_cm.tolist(),
            'predictions': teacher_predictions,
            'true_labels': teacher_labels
        }
        
        print(f"  Results for {dataset_name} ({blur_mode}):")
        print(f"    TEACHER:")
        print(f"      Accuracy: {teacher_accuracy:.4f} ({teacher_accuracy*100:.2f}%)")
        print(f"      Precision: {teacher_precision:.4f}")
        print(f"      Recall: {teacher_recall:.4f}")
        print(f"      F1-Score: {teacher_f1:.4f}")
        print(f"      Confusion Matrix: {teacher_cm.tolist()}")
    
    # Student指标
    if MODEL_TEST_MODE in ["student", "both"]:
        student_accuracy = accuracy_score(student_labels, student_predictions)
        student_precision, student_recall, student_f1, _ = precision_recall_fscore_support(
            student_labels, student_predictions, average='binary')
        student_cm = confusion_matrix(student_labels, student_predictions)
        
        result['student'] = {
            'accuracy': float(student_accuracy),
            'precision': float(student_precision),
            'recall': float(student_recall),
            'f1_score': float(student_f1),
            'confusion_matrix': student_cm.tolist(),
            'predictions': student_predictions,
            'true_labels': student_labels
        }
        
        print(f"  Results for {dataset_name} ({blur_mode}):")
        print(f"    STUDENT:")
        print(f"      Accuracy: {student_accuracy:.4f} ({student_accuracy*100:.2f}%)")
        print(f"      Precision: {student_precision:.4f}")
        print(f"      Recall: {student_recall:.4f}")
        print(f"      F1-Score: {student_f1:.4f}")
        print(f"      Confusion Matrix: {student_cm.tolist()}")
    
    # 比较性能（仅在both模式下）
    if MODEL_TEST_MODE == "both":
        acc_diff = student_accuracy - teacher_accuracy
        result['comparison'] = {
            'accuracy_difference': float(acc_diff),
            'student_better': bool(acc_diff > 0)
        }
        
        print(f"    COMPARISON:")
        print(f"      Student vs Teacher Accuracy: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
    
    # 显示blur统计
    if blur_mode != "no_blur" and blur_stats['blur_count'] > 0:
        avg_blur_time = blur_stats['total_blur_time'] / blur_stats['blur_count'] * 1000
        print(f"    Blur ops: {blur_stats['blur_count']}, Avg time: {avg_blur_time:.2f}ms")
    
    return result


def validate_dataset(dataset_name, dataset_config):
    """验证数据集配置是否有效"""
    print(f"Validating dataset: {dataset_name}")
    
    if dataset_config['type'] == 'simple':
        real_folder = dataset_config['real_folder']
        fake_folder = dataset_config['fake_folder']
        
        if not os.path.exists(real_folder):
            print(f"   Real folder not found: {real_folder}")
            return False
        if not os.path.exists(fake_folder):
            print(f"   Fake folder not found: {fake_folder}")
            return False
        
        print(f"  ✓ Folders exist")
        return True
        
    elif dataset_config['type'] == 'multi_class':
        base_path = dataset_config['base_path']
        if not os.path.exists(base_path):
            print(f"   Base path not found: {base_path}")
            return False
        
        # 如果 classes 为 None，验证是否至少有一个有效的类别文件夹
        if dataset_config.get('classes') is None:
            base_path_obj = Path(base_path)
            valid_classes = []
            
            for item in base_path_obj.iterdir():
                if item.is_dir():
                    real_subfolder = item / "0_real"
                    fake_subfolder = item / "1_fake"
                    if real_subfolder.exists() or fake_subfolder.exists():
                        valid_classes.append(item.name)
            
            if not valid_classes:
                print(f"   No valid class folders found (no 0_real or 1_fake subdirectories)")
                return False
            
            print(f"  ✓ Found {len(valid_classes)} valid class folders: {valid_classes[:5]}{'...' if len(valid_classes) > 5 else ''}")
            return True
        
        else:
            # 原有的指定 classes 验证逻辑
            missing_classes = []
            base_path_obj = Path(base_path)
            for class_name in dataset_config['classes']:
                class_path = base_path_obj / class_name
                if not class_path.exists():
                    missing_classes.append(class_name)
            
            if missing_classes:
                print(f"   Missing class folders: {missing_classes}")
                return False
            
            print(f"  ✓ All specified class folders exist")
            return True
    
    return False


def test_single_dataset(dataset_name, dataset_config, teacher_model, student_model):
    """测试单个数据集"""
    print(f"{'='*60}")
    print(f"TESTING DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # 验证数据集
    if not validate_dataset(dataset_name, dataset_config):
        print(f" Dataset {dataset_name} validation failed, skipping...")
        return None
    
    # 创建数据集和数据加载器
    test_transform = get_test_transforms()
    test_dataset = MultiTestDataset(dataset_config, test_transform)
    
    if len(test_dataset) == 0:
        print(f" No test samples found in {dataset_name}, skipping...")
        return None
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # 根据BLUR_MODE决定测试模式
    if BLUR_MODE == "both":
        test_modes = ["no_blur", "global"]
    elif BLUR_MODE in ["no_blur", "global"]:
        test_modes = [BLUR_MODE]
    else:
        print(f"Unknown BLUR_MODE: {BLUR_MODE}")
        return None
    
    dataset_results = {}
    
    # 测试每种模式
    for mode in test_modes:
        print(f"  Testing mode: {mode}")
        start_time = time.time()
        
        results = evaluate_models(teacher_model, student_model, test_loader, dataset_name, blur_mode=mode)
        
        end_time = time.time()
        eval_time = end_time - start_time
        results['evaluation_time_seconds'] = eval_time
        
        dataset_results[mode] = results
        print(f"  Mode {mode} completed in {eval_time:.2f}s")
        
        # 清理GPU内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return dataset_results

def test_all_datasets():
    """测试所有数据集"""
    print("="*70)
    print("TEACHER-STUDENT MODEL EVALUATION - ALL DATASETS")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Teacher model: {TEACHER_MODEL_PATH}")
    print(f"Student model: {STUDENT_MODEL_PATH}")
    print(f"DinoV3 backbone: {DINOV3_MODEL_ID}")
    print(f"Blur mode: {BLUR_MODE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Datasets to test: {list(TEST_DATASETS.keys())}")
    print("="*70)
    
    # 检查模型文件是否存在
    if not os.path.exists(TEACHER_MODEL_PATH):
        print(f" Teacher model not found at {TEACHER_MODEL_PATH}")
        return None
    
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f" Student model not found at {STUDENT_MODEL_PATH}")
        return None
    
    if not os.path.exists(DINOV3_MODEL_ID):
        print(f" DinoV3 backbone not found at {DINOV3_MODEL_ID}")
        return None
    
    # 加载模型（只加载一次）
    print("Loading Teacher-Student models...")
    teacher_model, student_model = load_teacher_student_model(TEACHER_MODEL_PATH, STUDENT_MODEL_PATH, DEVICE)
    
    if teacher_model is None or student_model is None:
        print(" Failed to load Teacher-Student models. Exiting...")
        return None
    
    print(" Teacher-Student models loaded successfully!")
    
    # 测试所有数据集
    all_results = {}
    successful_tests = 0
    failed_tests = 0
    
    for dataset_name, dataset_config in TEST_DATASETS.items():
        try:
            dataset_results = test_single_dataset(dataset_name, dataset_config, teacher_model, student_model)
            
            if dataset_results is not None:
                all_results[dataset_name] = dataset_results
                successful_tests += 1
                print(f" {dataset_name} completed successfully")
            else:
                failed_tests += 1
                print(f" {dataset_name} failed")
                
        except Exception as e:
            print(f" Error testing {dataset_name}: {str(e)}")
            failed_tests += 1
            continue
    
    # 保存综合结果
    if all_results:
        combined_results = {
            'test_config': {
                'blur_mode': BLUR_MODE,
                'batch_size': BATCH_SIZE,
                'blur_strength_range': BLUR_STRENGTH_RANGE,
                'teacher_model_path': TEACHER_MODEL_PATH,
                'student_model_path': STUDENT_MODEL_PATH,
                'dinov3_model_path': DINOV3_MODEL_ID,
                'projection_dim': PROJECTION_DIM,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'model_source': 'TeacherStudent_Checkpoint_Direct'
            },
            'results_by_dataset': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_file = os.path.join(OUTPUT_DIR, f'teacher_student_all_datasets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        with open(results_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # 显示总结
        print_summary(all_results, results_file)
        
        return combined_results
    else:
        print(" No datasets were successfully tested!")
        return None

def print_summary(all_results, results_file):
    """打印测试总结"""
    print(f"{'='*70}")
    print("COMPREHENSIVE TEACHER-STUDENT TEST SUMMARY")
    print(f"{'='*70}")
    
    # 根据BLUR_MODE决定要显示的模式
    if BLUR_MODE == "both":
        test_modes = ["no_blur", "global"]
    else:
        test_modes = [BLUR_MODE]
    
    # 按数据集显示结果
    for dataset_name, dataset_results in all_results.items():
        print(f"Dataset: {dataset_name.upper()}")
        print("-" * 50)
        
        for mode in test_modes:
            if mode in dataset_results:
                result = dataset_results[mode]
                teacher_result = result['teacher']
                student_result = result['student']
                comparison = result['comparison']
                
                print(f"  {mode.upper()} Mode:")
                print(f"    TEACHER - Acc: {teacher_result['accuracy']:.4f} ({teacher_result['accuracy']*100:.2f}%) | F1: {teacher_result['f1_score']:.4f}")
                print(f"    STUDENT - Acc: {student_result['accuracy']:.4f} ({student_result['accuracy']*100:.2f}%) | F1: {student_result['f1_score']:.4f}")
                print(f"    DIFF    - Acc: {comparison['accuracy_difference']:+.4f} ({comparison['accuracy_difference']*100:+.2f}%) | Student {'Better' if comparison['student_better'] else 'Worse'}")
                print(f"    Samples: {result['total_samples']} | Time: {result['evaluation_time_seconds']:.2f}s")
                
                # 显示Confusion Matrix
                teacher_cm = teacher_result['confusion_matrix']
                student_cm = student_result['confusion_matrix']
                print(f"    Teacher CM: {teacher_cm}")
                print(f"    Student CM: {student_cm}")
        
        print()
    
    # 跨数据集比较
    print(f"{'='*50}")
    print("CROSS-DATASET COMPARISON")
    print(f"{'='*50}")
    
    for mode in test_modes:
        print(f"{mode.upper()} Mode Results:")
        print("-" * 30)
        
        teacher_results = []
        student_results = []
        comparison_results = []
        
        for dataset_name, dataset_results in all_results.items():
            if mode in dataset_results:
                result = dataset_results[mode]
                teacher_acc = result['teacher']['accuracy']
                student_acc = result['student']['accuracy']
                acc_diff = result['comparison']['accuracy_difference']
                
                teacher_results.append((dataset_name, teacher_acc, result['teacher']['f1_score']))
                student_results.append((dataset_name, student_acc, result['student']['f1_score']))
                comparison_results.append((dataset_name, acc_diff))
        
        # 按Teacher准确率排序
        teacher_results.sort(key=lambda x: x[1], reverse=True)
        student_results.sort(key=lambda x: x[1], reverse=True)
        comparison_results.sort(key=lambda x: x[1], reverse=True)
        
        print("  TEACHER Rankings:")
        for i, (dataset_name, accuracy, f1_score) in enumerate(teacher_results, 1):
            print(f"    {i}. {dataset_name:15} - Acc: {accuracy:.4f} | F1: {f1_score:.4f}")
        
        print("  STUDENT Rankings:")
        for i, (dataset_name, accuracy, f1_score) in enumerate(student_results, 1):
            print(f"    {i}. {dataset_name:15} - Acc: {accuracy:.4f} | F1: {f1_score:.4f}")
        
        print("  STUDENT vs TEACHER Performance Difference:")
        for i, (dataset_name, acc_diff) in enumerate(comparison_results, 1):
            status = "Better" if acc_diff > 0 else "Worse"
            print(f"    {i}. {dataset_name:15} - Diff: {acc_diff:+.4f} ({acc_diff*100:+.2f}%) [{status}]")
        
        # 计算平均值
        if teacher_results:
            avg_teacher_acc = sum(result[1] for result in teacher_results) / len(teacher_results)
            avg_teacher_f1 = sum(result[2] for result in teacher_results) / len(teacher_results)
            avg_student_acc = sum(result[1] for result in student_results) / len(student_results)
            avg_student_f1 = sum(result[2] for result in student_results) / len(student_results)
            avg_diff = sum(result[1] for result in comparison_results) / len(comparison_results)
            
            print(f"  AVERAGES:")
            print(f"    Teacher - Acc: {avg_teacher_acc:.4f} | F1: {avg_teacher_f1:.4f}")
            print(f"    Student - Acc: {avg_student_acc:.4f} | F1: {avg_student_f1:.4f}")
            print(f"    Difference - Acc: {avg_diff:+.4f} ({avg_diff*100:+.2f}%)")
        
        print()
    
    # 详细的Confusion Matrix分析
    print(f"{'='*50}")
    print("DETAILED CONFUSION MATRIX ANALYSIS")
    print(f"{'='*50}")
    
    for mode in test_modes:
        print(f"{mode.upper()} Mode - Confusion Matrices:")
        print("-" * 40)
        
        for dataset_name, dataset_results in all_results.items():
            if mode in dataset_results:
                result = dataset_results[mode]
                teacher_cm = result['teacher']['confusion_matrix']
                student_cm = result['student']['confusion_matrix']
                
                # Teacher CM分析
                teacher_tn, teacher_fp, teacher_fn, teacher_tp = teacher_cm[0][0], teacher_cm[0][1], teacher_cm[1][0], teacher_cm[1][1]
                teacher_total = teacher_tn + teacher_fp + teacher_fn + teacher_tp
                teacher_tpr = teacher_tp / (teacher_tp + teacher_fn) if (teacher_tp + teacher_fn) > 0 else 0
                teacher_tnr = teacher_tn / (teacher_tn + teacher_fp) if (teacher_tn + teacher_fp) > 0 else 0
                
                # Student CM分析
                student_tn, student_fp, student_fn, student_tp = student_cm[0][0], student_cm[0][1], student_cm[1][0], student_cm[1][1]
                student_total = student_tn + student_fp + student_fn + student_tp
                student_tpr = student_tp / (student_tp + student_fn) if (student_tp + student_fn) > 0 else 0
                student_tnr = student_tn / (student_tn + student_fp) if (student_tn + student_fp) > 0 else 0
                
                print(f"  {dataset_name.upper()}:")
                print(f"    TEACHER Matrix: [[{teacher_tn:4d}, {teacher_fp:4d}], [{teacher_fn:4d}, {teacher_tp:4d}]]")
                print(f"            TPR: {teacher_tpr:.4f} | TNR: {teacher_tnr:.4f}")
                print(f"    STUDENT Matrix: [[{student_tn:4d}, {student_fp:4d}], [{student_fn:4d}, {student_tp:4d}]]")
                print(f"            TPR: {student_tpr:.4f} | TNR: {student_tnr:.4f}")
                print(f"    COMPARISON:")
                print(f"            TPR Diff: {student_tpr - teacher_tpr:+.4f}")
                print(f"            TNR Diff: {student_tnr - teacher_tnr:+.4f}")
                print()
    
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"Total datasets tested: {len(all_results)}")
    print("="*70)

def main():
    """主函数 - 根据配置决定测试单个还是所有数据集"""
    if TEST_ALL_DATASETS:
        # 测试所有数据集
        return test_all_datasets()
    else:
        # 测试单个数据集
        print("="*70)
        print("TEACHER-STUDENT MODEL EVALUATION - SINGLE DATASET")
        print("="*70)
        print(f"Device: {DEVICE}")
        print(f"Selected test dataset: {SELECTED_TEST_DATASET}")
        print(f"Teacher model: {TEACHER_MODEL_PATH}")
        print(f"Student model: {STUDENT_MODEL_PATH}")
        print(f"Model test mode: {MODEL_TEST_MODE}")
        print(f"DinoV3 backbone: {DINOV3_MODEL_ID}")
        print(f"Blur mode: {BLUR_MODE}")
        print(f"Batch size: {BATCH_SIZE}")
        print("="*70)
        
        # 检查选择的测试集是否存在
        if SELECTED_TEST_DATASET not in TEST_DATASETS:
            print(f" Unknown test dataset '{SELECTED_TEST_DATASET}'")
            print(f"Available datasets: {list(TEST_DATASETS.keys())}")
            return None
        
        # 检查模型文件是否存在
        if not os.path.exists(TEACHER_MODEL_PATH):
            print(f" Teacher model not found at {TEACHER_MODEL_PATH}")
            return None
        
        if not os.path.exists(STUDENT_MODEL_PATH):
            print(f" Student model not found at {STUDENT_MODEL_PATH}")
            return None
        
        if not os.path.exists(DINOV3_MODEL_ID):
            print(f" DinoV3 backbone not found at {DINOV3_MODEL_ID}")
            return None
        
        # 加载模型
        teacher_model, student_model = load_teacher_student_model(TEACHER_MODEL_PATH, STUDENT_MODEL_PATH, DEVICE)
        if teacher_model is None or student_model is None:
            print(" Failed to load Teacher-Student models. Exiting...")
            return None
        
        # 测试单个数据集
        dataset_config = TEST_DATASETS[SELECTED_TEST_DATASET]
        results = test_single_dataset(SELECTED_TEST_DATASET, dataset_config, teacher_model, student_model)
        
        if results is not None:
            # 保存单个数据集结果
            combined_results = {
                'test_config': {
                    'dataset_name': SELECTED_TEST_DATASET,
                    'dataset_config': dataset_config,
                    'batch_size': BATCH_SIZE,
                    'blur_strength_range': BLUR_STRENGTH_RANGE,
                    'blur_mode': BLUR_MODE,
                    'teacher_model_path': TEACHER_MODEL_PATH,
                    'student_model_path': STUDENT_MODEL_PATH,
                    'dinov3_model_path': DINOV3_MODEL_ID,
                    'projection_dim': PROJECTION_DIM,
                    'model_source': 'TeacherStudent_Checkpoint_Direct'
                },
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            results_file = os.path.join(OUTPUT_DIR, f'teacher_student_single_{SELECTED_TEST_DATASET}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            
            with open(results_file, 'w') as f:
                json.dump(combined_results, f, indent=2)
            
            # 显示单个数据集的总结
            print_summary({SELECTED_TEST_DATASET: results}, results_file)
            
            return combined_results
        else:
            print(f" Failed to test dataset {SELECTED_TEST_DATASET}")
            return None

if __name__ == "__main__":
    # 显示启动信息
    print(" Starting Teacher-Student Model Testing - Optimized Version")
    print(f" Device: {DEVICE}")
    print(f" Teacher Model: {TEACHER_MODEL_PATH}")
    print(f" Student Model: {STUDENT_MODEL_PATH}")
    print(f" DinoV3 Backbone: {DINOV3_MODEL_ID}")
    print(f" Test mode: {'All datasets' if TEST_ALL_DATASETS else f'Single dataset ({SELECTED_TEST_DATASET})'}")
    print(f" Blur mode: {BLUR_MODE}")
    print(f" Batch size: {BATCH_SIZE}")
    print("="*70)
    
    # 内存优化设置
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name()}")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # 设置内存分配策略
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(" GPU memory cleared and optimized")
        
        # 显示多GPU信息
        if torch.cuda.device_count() > 1:
            print(f" Multiple GPUs detected: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 运行测试
    start_time = time.time()
    
    try:
        results = main()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("="*70)
            print("GPU OUT OF MEMORY ERROR!")
            print("="*70)
            print("Suggestions:")
            print("1. Reduce BATCH_SIZE (current: {})".format(BATCH_SIZE))
            print("2. Reduce NUM_WORKERS (current: {})".format(NUM_WORKERS))
            print("3. Free up GPU memory by closing other processes")
            print("4. Use a machine with more GPU memory")
            print("5. Test datasets one by one instead of all at once")
            print("="*70)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            results = None
        else:
            print(f"Runtime error: {e}")
            results = None
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        results = None
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if results is not None:
        print("="*70)
        print("✓ TEACHER-STUDENT TESTING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"  Total time: {total_time:.2f} seconds")
        
        # 显示简要统计
        if TEST_ALL_DATASETS and 'results_by_dataset' in results:
            successful_count = len(results['results_by_dataset'])
            total_count = len(TEST_DATASETS)
            print(f" Datasets tested: {successful_count}/{total_count}")
            
            # 计算平均准确率
            if BLUR_MODE == "both":
                modes = ["no_blur", "global"]
            else:
                modes = [BLUR_MODE]
            
            for mode in modes:
                teacher_accuracies = []
                student_accuracies = []
                differences = []
                
                for dataset_results in results['results_by_dataset'].values():
                    if mode in dataset_results:
                        result = dataset_results[mode]
                        teacher_accuracies.append(result['teacher']['accuracy'])
                        student_accuracies.append(result['student']['accuracy'])
                        differences.append(result['comparison']['accuracy_difference'])
                
                if teacher_accuracies:
                    avg_teacher_acc = sum(teacher_accuracies) / len(teacher_accuracies)
                    avg_student_acc = sum(student_accuracies) / len(student_accuracies)
                    avg_diff = sum(differences) / len(differences)
                    
                    print(f" Average accuracy ({mode}):")
                    print(f"   Teacher: {avg_teacher_acc:.4f} ({avg_teacher_acc*100:.2f}%)")
                    print(f"   Student: {avg_student_acc:.4f} ({avg_student_acc*100:.2f}%)")
                    print(f"   Difference: {avg_diff:+.4f} ({avg_diff*100:+.2f}%)")
                    
                    # 统计Student表现更好的数据集数量
                    better_count = sum(1 for diff in differences if diff > 0)
                    print(f"   Student better on: {better_count}/{len(differences)} datasets")
        
    else:
        print("="*70)
        print("✗ TESTING FAILED!")
        print("="*70)
        print(f"  Time elapsed: {total_time:.2f} seconds")
        
    print("="*70)
    print(" Teacher-Student testing session completed.")
    print("="*70)

