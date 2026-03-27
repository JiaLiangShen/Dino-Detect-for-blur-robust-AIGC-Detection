# 0327 Blur Generalization Expansion

这一个文件夹专门放 2026-03-27 这轮补实验脚本，保持“训练一个文件、测试一个文件、分析一个文件”的单文件风格。

包含的脚本如下：

1. `train_lora_backbones.py`
   - 固定 backbone，只训练 LoRA + projection/classifier。
   - 支持 `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` 和 `timm/eva_giant_patch14_336.m30m_ft_in22k_in1k`。
   - 默认在 GenImage SDV1.4 上训练，CCMBA 路径默认保持 `/home/work/xueyunqi/11ar_datasets/progan_ccmba_train`。

2. `eval_lora_cross_dataset.py`
   - 加载 LoRA checkpoint。
   - 在 clean / blur 两种设置下做 cross-dataset 测试。
   - 额外导出 `table4_clean_accuracy.csv`、`table5_blur_accuracy.csv`、`table6_clean_f1.csv`、`table7_blur_f1.csv`。

3. `train_teacher_student_backbones.py`
   - 保持原 `train_motion.py` 的 teacher-student 训练思路。
   - 只替换 DINOv3 backbone 规模，默认补 `ViT-Large-300M` 和 `ViT-Huge-840M`。
   - 默认在 SDV1.4 上训练。

4. `eval_teacher_student_aigc.py`
   - 加载 teacher/student checkpoint。
   - 默认在 AIGCBenchmark 上评测，可选 `teacher` 或 `student` 分支。

5. `analyze_dinov3_blur_consistency.py`
   - 分析不同 blur strength 下 DINOv3 特征稳定性。
   - 输出逐图 JSON、平均结果 JSON，以及一张折线图。

说明：

- 本地 Hugging Face 模型目录我先约定成 `/nas_train/app.e0016372/models/hf_backbones/...`，后续你们只需要改每个脚本顶部的默认路径即可。
- 脚本没有额外做层级化封装，尽量保留你现有工程里“直接在一个文件里完成主要流程”的风格。
- 这轮我没有替你实际跑测试或训练。
