# DINO-Detect 公开版代码整理说明

这个文件夹是我根据论文 **DINO-Detect: Towards Blur-Robust AI-Generated Image Detection** 对原始实验代码做的公开发布版重构。

目标不是简单“拆文件”，而是把原本偏实验脚本风格的实现整理成更适合放在 GitHub 上展示和维护的形式：

- 保留论文主线：`DINOv3 teacher -> blur student -> sharp/blur alignment`
- 去掉原始脚本中的硬编码路径
- 将训练、测试、模型、数据处理、模糊增强、checkpoint 等模块拆开
- 给出可直接修改的配置文件和更像论文源码的 README

## 你现在应该看哪里

- 训练入口: [train.py](train.py)
- 测试入口: [test.py](test.py)
- 模型定义: [network.py](src/dino_detect/models/network.py)
- 训练逻辑: [trainer.py](src/dino_detect/training/trainer.py)
- 评测逻辑: [evaluator.py](src/dino_detect/evaluation/evaluator.py)
- 数据说明: [DATA_LAYOUT.md](docs/DATA_LAYOUT.md)

## 目前这版的特点

- 训练配置和测试配置都改成 JSON 文件管理
- 支持 `torchrun` 多卡启动
- 训练输出会自动保存配置、checkpoint 和历史指标
- 评测输出会自动写成结构化 JSON，方便后续画表格和作图

## 和原始脚本的关系

仓库根目录下的原始 `train.py`、`test.py` 没有被覆盖。  
`github_release/` 是面向公开发布的新版本，便于你后续继续美化、补 figure、补 benchmark 表格或直接整理成真正的 GitHub 仓库。
