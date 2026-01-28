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
import time
import argparse
from datetime import datetime
from transformers import AutoModel
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 配置参数
# ===============================

# 1. 模型路径配置 (请修改这里指向您 gaussian_direct_train.py或者motion_direct_train.py 训练出的模型)
MODEL_PATH = "teacher_blur_training/dinov3-vit7b16-pretrain-lvd1689m_blur02/best_teacher_blur_model.pth"

# DinoV3 预训练模型路径(这个也要修改成为您自己的路径！)
DINOV3_MODEL_ID = "/nas_train/app.e0016372/models/dinov3-vit7b16-pretrain-lvd1689m"

# 2. 模糊测试配置
BLUR_MODE = "both"  # "no_blur", "global", "both"
BLUR_PROB = 1.0
BLUR_STRENGTH_RANGE = (0.1, 0.3)

# 3. 测试集配置 （我看了一下，wildrf里面三个子集全部都是simple的，您看着修改文件夹路径即可）
# TEST_DATASETS = {
#     'reddit': {
#         'type': 'simple',
#         'real_folder': "/data/app.e0016372/WildRF/test/reddit/0_real",
#         'fake_folder': "/data/app.e0016372/WildRF/test/reddit/1_fake"
#     },
#     'facebook': {
#         'type': 'simple',
#         'real_folder': "/data/app.e0016372/WildRF/test/facebook/0_real",
#         'fake_folder': "/data/app.e0016372/WildRF/test/facebook/1_fake"
#     },
#     'twitter': {
#         'type': 'simple',
#         'real_folder': "/data/app.e0016372/WildRF/test/twitter/0_real",
#         'fake_folder': "/data/app.e0016372/WildRF/test/twitter/1_fake"
#     }
# }

TEST_DATASETS = {
    'cyclegan': {
        'type': 'multi_class',
        'base_path': '/data/app.e0016372/11ar_datasets/test/cyclegan',
        'classes': None  # None 表示自动发现所有类别
    },
    'progan': {
        'type': 'multi_class',
        'base_path': '/data/app.e0016372/11ar_datasets/test/progan',
        'classes': None
    },
    'stylegan2': {
        'type': 'multi_class',
        'base_path': '/data/app.e0016372/11ar_datasets/test/stylegan2',
        'classes': None
    },
    'stylegan': {
        'type': 'multi_class',
        'base_path': '/data/app.e0016372/11ar_datasets/test/stylegan',
        'classes': None
    },
    'adm': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/ADM/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/ADM/1_fake'
    },
    'vqdm': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/VQDM/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/VQDM/1_fake'
    },
    'sdv14': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/stable_diffusion_v_1_4/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/stable_diffusion_v_1_4/1_fake'
    },
    'sdv15': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/stable_diffusion_v_1_5/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/stable_diffusion_v_1_5/1_fake'
    },
    'stargan': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/stargan/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/stargan/1_fake'
    },
    'wukong': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/wukong/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/wukong/1_fake'
    },
    'dalle2': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/datasets/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test/DALLE2/0_real',
        'fake_folder': '/data/app.e0016372/datasets/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test/DALLE2/1_fake'
    },
    'midjourney': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/Midjourney/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/Midjourney/1_fake'
    },
    'biggan': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/biggan/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/biggan/1_fake'
    },
    'sd-xl': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/sd_xl/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/sd_xl/1_fake'
    },
    'gaugan': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/gaugan/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/gaugan/1_fake'
    },
    'whichfaceisreal': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/whichfaceisreal/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/whichfaceisreal/1_fake'
    },
    'glide': {
        'type': 'simple',
        'real_folder': '/data/app.e0016372/11ar_datasets/test/Glide/0_real',
        'fake_folder': '/data/app.e0016372/11ar_datasets/test/Glide/1_fake'
    }
}

# 4. 运行配置
TEST_ALL_DATASETS = True
SELECTED_TEST_DATASET = 'reddit'  # 如果 TEST_ALL_DATASETS = False，则只测这个

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
NUM_WORKERS = 4
PROJECTION_DIM = 512
OUTPUT_DIR = "gaussian_test_results"

# ===============================
# 模型架构 (从 gaussian_direct_train.py 复制)
# ===============================
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class ImprovedDinoV3Adapter(nn.Module):
    """改进的DinoV3模型适配器"""
    
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
        
        # 初始化权重
        self._initialize_weights()
    
    def _load_dinov3_backbone(self, model_path, device):
        try:
            backbone = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            backbone = backbone.to(device).eval()
            for param in backbone.parameters():
                param.requires_grad = False
            return backbone
        except Exception as e:
            print(f"Error loading DinoV3 backbone: {e}")
            raise e
    
    def _initialize_weights(self):
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
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0]
            return features.float()
    
    def forward(self, pixel_values, return_features=False):
        raw_features = self.extract_features(pixel_values)
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

# ===============================
# 模糊处理工具
# ===============================
def apply_motion_blur_batch_efficient(images, strength):
    """
    高效的批量运动模糊处理 (保持原有test.py逻辑用于测试)
    注意: 原来test.py中使用的是Motion Blur进行测试，这里保持一致以便对比。
    如果需要改成高斯模糊测试，请替换此函数。
    """
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
    
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel, padding=padding, groups=channels)
    
    return blurred

# ===============================
# 数据变换与数据集
# ===============================
def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

class MultiTestDataset(Dataset):
    """支持多种测试数据集结构的通用数据集类"""

    def __init__(self, dataset_config, transform, 
                 blur_mode="no_blur", blur_prob=0.0, blur_strength_range=(0.1, 0.5)):
        self.transform = transform
        self.blur_mode = blur_mode
        self.blur_prob = blur_prob
        self.blur_strength_range = blur_strength_range
        self.dataset_config = dataset_config
        self.data = []
        
        if dataset_config['type'] == 'simple':
            self._load_simple_dataset(dataset_config)
        elif dataset_config['type'] == 'multi_class':
            self._load_multi_class_dataset(dataset_config)
        
        print(f"Total test samples loaded: {len(self.data)}")
    
    def _load_simple_dataset(self, config):
        real_images = self._get_image_files(config['real_folder'])
        fake_images = self._get_image_files(config['fake_folder'])
        for img_path in real_images:
            self.data.append((img_path, 0, 'real'))
        for img_path in fake_images:
            self.data.append((img_path, 1, 'fake'))
        print(f"Simple dataset loaded: {len(real_images)} real, {len(fake_images)} fake images")
    
    def _load_multi_class_dataset(self, config):
        base_path = Path(config['base_path'])
        if config.get('classes') is None:
            discovered_classes = []
            for item in base_path.iterdir():
                if item.is_dir():
                    if (item / "0_real").exists() or (item / "1_fake").exists():
                        discovered_classes.append(item.name)
            classes_to_process = sorted(discovered_classes)
        else:
            classes_to_process = config['classes']
        
        total_real, total_fake = 0, 0
        for class_name in classes_to_process:
            class_path = base_path / class_name
            real_folder, fake_folder = class_path / "0_real", class_path / "1_fake"
            
            if real_folder.exists():
                imgs = self._get_image_files(str(real_folder))
                for img_path in imgs: self.data.append((img_path, 0, f'{class_name}_real'))
                total_real += len(imgs)
            if fake_folder.exists():
                imgs = self._get_image_files(str(fake_folder))
                for img_path in imgs: self.data.append((img_path, 1, f'{class_name}_fake'))
                total_fake += len(imgs)
        print(f"Multi-class dataset loaded: {total_real} real, {total_fake} fake images")
        
    def _get_image_files(self, folder_path):
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        folder = Path(folder_path)
        if not folder.exists(): return []
        return sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in extensions])

    def apply_blur_augmentation(self, tensor_image):
        if self.blur_mode == "no_blur" or random.random() > self.blur_prob:
            return tensor_image, {'mode': 'no_blur', 'blur_applied': False}
        
        blur_start_time = time.time()
        blur_strength = random.uniform(*self.blur_strength_range)
        # 注意：这里使用原test.py中的apply_motion_blur_batch_efficient
        # 如果训练使用的是gaussian，但测试想测motion robustness，保持不变
        # 如果想测高斯模糊，需要修改apply_motion_blur_batch_efficient内部实现
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
        try:
            image = Image.open(img_path).convert('RGB')
            tensor_image = self.transform(image)
            return tensor_image, label, f"{category}_{img_path.stem}"
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回一个占位符或处理异常
            return torch.zeros((3, 224, 224)), label, "error"

# ===============================
# 模型加载与辅助函数
# ===============================
def remove_module_prefix(state_dict):
    """移除DDP产生的 module. 前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def load_single_model(model_path, device):
    """加载单个Gaussian训练出来的模型"""
    print(f"Loading Single model from checkpoint: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    print("Creating ImprovedDinoV3Adapter...")
    # 注意: gaussian_direct_train.py 中使用了 adapter_layers=3
    model = ImprovedDinoV3Adapter(
        model_path=DINOV3_MODEL_ID,
        num_classes=2,
        projection_dim=PROJECTION_DIM,
        adapter_layers=3,  # 关键：必须与训练时一致
        dropout_rate=0.1,
        device=device
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # 有些保存方式直接保存state_dict
        state_dict = checkpoint
    
    # 移除 DDP 前缀
    state_dict = remove_module_prefix(state_dict)
    
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print(f"Missing keys example: {missing[:5]}")
        
        model = model.to(device)
        model.eval()
        
        # 多卡推理支持
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for inference")
            model = nn.DataParallel(model)
            
        print("✓ Model loaded successfully!")
        if 'best_acc' in checkpoint:
            print(f"Best Training Accuracy: {checkpoint['best_acc']:.2f}%")
            
        return model
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_model(model, test_loader, dataset_name, blur_mode="no_blur"):
    """评估单个模型"""
    print(f"Evaluating model on {dataset_name} with blur_mode: {blur_mode}...")
    
    model.eval()
    
    predictions = []
    true_labels = []
    correct = 0
    total_samples = 0
    
    blur_stats = {'blur_count': 0, 'total_blur_time': 0}
    
    with torch.no_grad():
        for batch_idx, (tensor_image, labels, filenames) in enumerate(test_loader):
            # 处理 Blur
            if blur_mode == "no_blur":
                processed_imgs = tensor_image.to(DEVICE, non_blocking=True)
            else:
                processed_imgs_list = []
                for img_tensor in tensor_image:
                    if blur_mode == "global":
                        test_loader.dataset.blur_mode = "global"
                        test_loader.dataset.blur_prob = 1.0
                        blurred_tensor, blur_info = test_loader.dataset.apply_blur_augmentation(img_tensor)
                        processed_imgs_list.append(blurred_tensor)
                        if blur_info['blur_applied']:
                            blur_stats['blur_count'] += 1
                            blur_stats['total_blur_time'] += blur_info.get('blur_time', 0)
                processed_imgs = torch.stack(processed_imgs_list).to(DEVICE, non_blocking=True)
            
            # 推理
            logits = model(processed_imgs)
            _, pred = torch.max(logits.data, 1)
            
            correct += (pred.cpu() == labels).sum().item()
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(labels.numpy())
            total_samples += labels.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                acc = 100 * correct / total_samples
                print(f"  [{dataset_name}-{blur_mode}] Batch {batch_idx+1}/{len(test_loader)} - Acc: {acc:.2f}%")
                
            del processed_imgs, logits, pred
    
    # 指标计算
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    cm = confusion_matrix(true_labels, predictions)
    
    result = {
        'dataset_name': dataset_name,
        'blur_mode': blur_mode,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': total_samples
    }
    
    print(f"  Result ({blur_mode}): Acc: {accuracy:.4f}, F1: {f1:.4f}")
    return result

def test_single_dataset(dataset_name, dataset_config, model):
    print(f"{'='*60}\nTESTING DATASET: {dataset_name.upper()}\n{'='*60}")
    
    # 验证
    if not validate_dataset(dataset_name, dataset_config): return None
    
    test_transform = get_test_transforms()
    test_dataset = MultiTestDataset(dataset_config, test_transform)
    
    if len(test_dataset) == 0: return None
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    test_modes = ["no_blur", "global"] if BLUR_MODE == "both" else [BLUR_MODE]
    
    dataset_results = {}
    for mode in test_modes:
        start_time = time.time()
        res = evaluate_model(model, test_loader, dataset_name, blur_mode=mode)
        res['time_seconds'] = time.time() - start_time
        dataset_results[mode] = res
        
    return dataset_results

def validate_dataset(dataset_name, dataset_config):
    # (简化版验证逻辑，保持原功能)
    if dataset_config['type'] == 'simple':
        return os.path.exists(dataset_config['real_folder']) and os.path.exists(dataset_config['fake_folder'])
    elif dataset_config['type'] == 'multi_class':
        return os.path.exists(dataset_config['base_path'])
    return False

def print_summary(all_results, results_file):
    print(f"{'='*70}\nTEST SUMMARY\n{'='*70}")
    test_modes = ["no_blur", "global"] if BLUR_MODE == "both" else [BLUR_MODE]
    
    for mode in test_modes:
        print(f"\nMode: {mode.upper()}")
        results_list = []
        for name, res in all_results.items():
            if mode in res:
                r = res[mode]
                results_list.append((name, r['accuracy'], r['f1_score']))
        
        # 排序
        results_list.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, acc, f1) in enumerate(results_list, 1):
            print(f"  {i}. {name:15} - Acc: {acc:.4f} | F1: {f1:.4f}")
            
        if results_list:
            avg_acc = sum(x[1] for x in results_list) / len(results_list)
            print(f"  AVERAGE ACC: {avg_acc:.4f}")
            
    print(f"\nResults saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dinov3_model_id', type=str, required=True)
    args = parser.parse_args()
    
    # 动态设置路径
    global MODEL_PATH, DINOV3_MODEL_ID, OUTPUT_DIR
    MODEL_PATH = args.model_path
    DINOV3_MODEL_ID = args.dinov3_model_id
    OUTPUT_DIR = str(Path(MODEL_PATH).parent)

    print("="*70)
    print("GAUSSIAN MODEL EVALUATION")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print("="*70)
    
    # 1. 加载模型
    model = load_single_model(MODEL_PATH, DEVICE)
    if model is None: return
    
    # 2. 确定测试集
    if TEST_ALL_DATASETS:
        datasets_to_test = TEST_DATASETS
    else:
        if SELECTED_TEST_DATASET not in TEST_DATASETS:
            print("Invalid selected dataset")
            return
        datasets_to_test = {SELECTED_TEST_DATASET: TEST_DATASETS[SELECTED_TEST_DATASET]}
    
    # 3. 执行测试
    all_results = {}
    for name, config in datasets_to_test.items():
        try:
            res = test_single_dataset(name, config, model)
            if res: all_results[name] = res
        except Exception as e:
            print(f"Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
            
    # 4. 保存结果
    if all_results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f'gaussian_eval_{timestamp}.json')
        
        final_data = {
            'config': {
                'model_path': MODEL_PATH,
                'blur_mode': BLUR_MODE,
                'batch_size': BATCH_SIZE
            },
            'results': all_results
        }
        
        with open(save_path, 'w') as f:
            json.dump(final_data, f, indent=2)
            
        print_summary(all_results, save_path)
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
