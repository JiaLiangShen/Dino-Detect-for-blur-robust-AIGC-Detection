# Data Layout

## Training Layout

The public training pipeline expects one category directory per generator family or domain.

```text
train_root/
├── ProGAN/
│   ├── 0_real/
│   └── 1_fake/
├── StyleGAN2/
│   ├── 0_real/
│   └── 1_fake/
├── ADM/
│   ├── 0_real/
│   └── 1_fake/
└── ...
```

## Evaluation Layout

Two layouts are supported.

### Simple Binary Layout

```text
dataset_name/
├── 0_real/
└── 1_fake/
```

### Multi-Category Layout

```text
dataset_name/
├── category_a/
│   ├── 0_real/
│   └── 1_fake/
├── category_b/
│   ├── 0_real/
│   └── 1_fake/
└── ...
```

## CCMBA Layout

If you use `ccmba` or `mixed` blur mode, the code expects:

```text
ccmba_root/
├── category_name/
│   ├── nature/
│   │   ├── blurred_images/
│   │   ├── blur_masks/
│   │   └── metadata/
│   └── ai/
│       ├── blurred_images/
│       ├── blur_masks/
│       └── metadata/
└── ...
```

Each blurred image should share the same stem as its blur mask and metadata file.
