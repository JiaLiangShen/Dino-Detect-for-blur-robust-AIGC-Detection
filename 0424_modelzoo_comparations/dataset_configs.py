from copy import deepcopy
from typing import Dict


CROSS_DATASET_TEST_DATASETS: Dict[str, Dict[str, object]] = {
    "cyclegan": {
        "type": "multi_class",
        "base_path": "/data/app.e0016372/11ar_datasets/test/cyclegan",
        "classes": None,
    },
    "progan": {
        "type": "multi_class",
        "base_path": "/data/app.e0016372/11ar_datasets/test/progan",
        "classes": None,
    },
    "stylegan2": {
        "type": "multi_class",
        "base_path": "/data/app.e0016372/11ar_datasets/test/stylegan2",
        "classes": None,
    },
    "stylegan": {
        "type": "multi_class",
        "base_path": "/data/app.e0016372/11ar_datasets/test/stylegan",
        "classes": None,
    },
    "adm": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/ADM/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/ADM/1_fake",
    },
    "vqdm": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/VQDM/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/VQDM/1_fake",
    },
    "sdv14": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/stable_diffusion_v_1_4/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/stable_diffusion_v_1_4/1_fake",
    },
    "sdv15": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/stable_diffusion_v_1_5/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/stable_diffusion_v_1_5/1_fake",
    },
    "stargan": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/stargan/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/stargan/1_fake",
    },
    "wukong": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/wukong/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/wukong/1_fake",
    },
    "dalle2": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/datasets/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test/DALLE2/0_real",
        "fake_folder": "/data/app.e0016372/datasets/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test/DALLE2/1_fake",
    },
    "midjourney": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/Midjourney/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/Midjourney/1_fake",
    },
    "biggan": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/biggan/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/biggan/1_fake",
    },
    "sd-xl": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/sd_xl/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/sd_xl/1_fake",
    },
    "gaugan": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/gaugan/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/gaugan/1_fake",
    },
    "whichfaceisreal": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/whichfaceisreal/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/whichfaceisreal/1_fake",
    },
    "glide": {
        "type": "simple",
        "real_folder": "/data/app.e0016372/11ar_datasets/test/Glide/0_real",
        "fake_folder": "/data/app.e0016372/11ar_datasets/test/Glide/1_fake",
    },
    "own_benchmark": {
        "type": "simple",
        "real_folder": "own_benchmark/0_real",
        "fake_folder": "own_benchmark/1_fake",
    },
}


AIGC_BENCHMARK_DATASET_NAMES = [
    "dalle2",
    "midjourney",
    "sd-xl",
    "wukong",
    "gaugan",
    "whichfaceisreal",
    "glide",
]


TABLE_EXPORT_SPECS = [
    ("table4_clean_accuracy", "no_blur", "accuracy"),
    ("table5_blur_accuracy", "global", "accuracy"),
    ("table6_clean_f1", "no_blur", "f1_score"),
    ("table7_blur_f1", "global", "f1_score"),
]


def select_datasets(group: str = "all") -> Dict[str, Dict[str, object]]:
    if group == "all":
        return deepcopy(CROSS_DATASET_TEST_DATASETS)
    if group == "aigc_benchmark":
        return {name: deepcopy(CROSS_DATASET_TEST_DATASETS[name]) for name in AIGC_BENCHMARK_DATASET_NAMES}
    if group in CROSS_DATASET_TEST_DATASETS:
        return {group: deepcopy(CROSS_DATASET_TEST_DATASETS[group])}
    raise ValueError(f"Unknown dataset group: {group}")
