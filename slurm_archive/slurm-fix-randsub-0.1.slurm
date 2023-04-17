The following modules were not unloaded:
  (Use "module --force purge" to unload all):

  1) slurm/latest
/data/gpfs/projects/punim1896/coresets/repositories/HRNet-Semantic-Segmentation-Coreset/venv/lib/python3.8/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torch.distributed.run.
Note that --use_env is set by default in torch.distributed.run.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
WARNING:torch.distributed.run:*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
Seeding with 304
=> creating output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484
=> creating log/subset/0.1/cityscapes/seg_hrnet_ocr/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484_2023-04-12-12-43
Seeding with 304
Namespace(cfg='experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml', local_rank=0, opts=[], seed=304)
=> creating output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484
AUTO_RESUME: False
CUDNN:
  BENCHMARK: True
  DETERMINISTIC: False
  ENABLED: True
DATASET:
  DATASET: cityscapes
  EXTRA_TRAIN_SET: 
  NUM_CLASSES: 19
  ROOT: data/
  TEST_SET: list/cityscapes/val.lst
  TRAIN_SET: list/cityscapes/train.lst
DEBUG:
  DEBUG: False
  SAVE_BATCH_IMAGES_GT: False
  SAVE_BATCH_IMAGES_PRED: False
  SAVE_HEATMAPS_GT: False
  SAVE_HEATMAPS_PRED: False
GPUS: (0, 1, 2, 3)
LOG_DIR: log/subset/0.1/
LOSS:
  BALANCE_WEIGHTS: [0.4, 1]
  CLASS_BALANCE: False
  OHEMKEEP: 131072
  OHEMTHRES: 0.9
  USE_OHEM: False
MODEL:
  ALIGN_CORNERS: True
  EXTRA:
    FINAL_CONV_KERNEL: 1
    STAGE1:
      BLOCK: BOTTLENECK
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4]
      NUM_CHANNELS: [64]
      NUM_MODULES: 1
      NUM_RANCHES: 1
    STAGE2:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4]
      NUM_BRANCHES: 2
      NUM_CHANNELS: [48, 96]
      NUM_MODULES: 1
    STAGE3:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4, 4]
      NUM_BRANCHES: 3
      NUM_CHANNELS: [48, 96, 192]
      NUM_MODULES: 4
    STAGE4:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4, 4, 4]
      NUM_BRANCHES: 4
      NUM_CHANNELS: [48, 96, 192, 384]
      NUM_MODULES: 3
  NAME: seg_hrnet_ocr
  NUM_OUTPUTS: 2
  OCR:
    DROPOUT: 0.05
    KEY_CHANNELS: 256
    MID_CHANNELS: 512
    SCALE: 1
  PRETRAINED: pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
OUTPUT_DIR: output/subset/0.1/
PIN_MEMORY: True
PRINT_FREQ: 10
RANK: 0
TEST:
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 4
  FLIP_TEST: False
  IMAGE_SIZE: [2048, 1024]
  MODEL_FILE: 
  MULTI_SCALE: False
  NUM_SAMPLES: 0
  OUTPUT_INDEX: -1
  SCALE_LIST: [1]
TRAIN:
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 3
  BEGIN_EPOCH: 0
  DOWNSAMPLERATE: 1
  END_EPOCH: 484
  EXTRA_EPOCH: 0
  EXTRA_LR: 0.001
  FLIP: True
  FREEZE_EPOCHS: -1
  FREEZE_LAYERS: 
  IGNORE_LABEL: 255
  IMAGE_SIZE: [1024, 512]
  LR: 0.01
  LR_FACTOR: 0.1
  LR_STEP: [90, 110]
  MOMENTUM: 0.9
  MULTI_SCALE: True
  NESTEROV: False
  NONBACKBONE_KEYWORDS: []
  NONBACKBONE_MULT: 10
  NUM_SAMPLES: 0
  OPTIMIZER: sgd
  RANDOM_BRIGHTNESS: False
  RANDOM_BRIGHTNESS_SHIFT_VALUE: 10
  RANDOM_SUBSET: 0.1
  RESUME: True
  SCALE_FACTOR: 16
  SHUFFLE: True
  WD: 0.0005
WORKERS: 4
=> creating log/subset/0.1/cityscapes/seg_hrnet_ocr/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484_2023-04-12-12-43
Namespace(cfg='experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml', local_rank=1, opts=[], seed=304)
args.local_rank:  0
AUTO_RESUME: False
CUDNN:
  BENCHMARK: True
  DETERMINISTIC: False
  ENABLED: True
DATASET:
  DATASET: cityscapes
  EXTRA_TRAIN_SET: 
  NUM_CLASSES: 19
  ROOT: data/
  TEST_SET: list/cityscapes/val.lst
  TRAIN_SET: list/cityscapes/train.lst
DEBUG:
  DEBUG: False
  SAVE_BATCH_IMAGES_GT: False
  SAVE_BATCH_IMAGES_PRED: False
  SAVE_HEATMAPS_GT: False
  SAVE_HEATMAPS_PRED: False
GPUS: (0, 1, 2, 3)
LOG_DIR: log/subset/0.1/
LOSS:
  BALANCE_WEIGHTS: [0.4, 1]
  CLASS_BALANCE: False
  OHEMKEEP: 131072
  OHEMTHRES: 0.9
  USE_OHEM: False
MODEL:
  ALIGN_CORNERS: True
  EXTRA:
    FINAL_CONV_KERNEL: 1
    STAGE1:
      BLOCK: BOTTLENECK
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4]
      NUM_CHANNELS: [64]
      NUM_MODULES: 1
      NUM_RANCHES: 1
    STAGE2:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4]
      NUM_BRANCHES: 2
      NUM_CHANNELS: [48, 96]
      NUM_MODULES: 1
    STAGE3:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4, 4]
      NUM_BRANCHES: 3
      NUM_CHANNELS: [48, 96, 192]
      NUM_MODULES: 4
    STAGE4:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4, 4, 4]
      NUM_BRANCHES: 4
      NUM_CHANNELS: [48, 96, 192, 384]
      NUM_MODULES: 3
  NAME: seg_hrnet_ocr
  NUM_OUTPUTS: 2
  OCR:
    DROPOUT: 0.05
    KEY_CHANNELS: 256
    MID_CHANNELS: 512
    SCALE: 1
  PRETRAINED: pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
OUTPUT_DIR: output/subset/0.1/
PIN_MEMORY: True
PRINT_FREQ: 10
RANK: 0
TEST:
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 4
  FLIP_TEST: False
  IMAGE_SIZE: [2048, 1024]
  MODEL_FILE: 
  MULTI_SCALE: False
  NUM_SAMPLES: 0
  OUTPUT_INDEX: -1
  SCALE_LIST: [1]
TRAIN:
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 3
  BEGIN_EPOCH: 0
  DOWNSAMPLERATE: 1
  END_EPOCH: 484
  EXTRA_EPOCH: 0
  EXTRA_LR: 0.001
  FLIP: True
  FREEZE_EPOCHS: -1
  FREEZE_LAYERS: 
  IGNORE_LABEL: 255
  IMAGE_SIZE: [1024, 512]
  LR: 0.01
  LR_FACTOR: 0.1
  LR_STEP: [90, 110]
  MOMENTUM: 0.9
  MULTI_SCALE: True
  NESTEROV: False
  NONBACKBONE_KEYWORDS: []
  NONBACKBONE_MULT: 10
  NUM_SAMPLES: 0
  OPTIMIZER: sgd
  RANDOM_BRIGHTNESS: False
  RANDOM_BRIGHTNESS_SHIFT_VALUE: 10
  RANDOM_SUBSET: 0.1
  RESUME: True
  SCALE_FACTOR: 16
  SHUFFLE: True
  WD: 0.0005
WORKERS: 4
args.local_rank:  1
Seeding with 304
=> creating output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484
=> creating log/subset/0.1/cityscapes/seg_hrnet_ocr/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484_2023-04-12-12-43
Namespace(cfg='experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml', local_rank=2, opts=[], seed=304)
AUTO_RESUME: False
CUDNN:
  BENCHMARK: True
  DETERMINISTIC: False
  ENABLED: True
DATASET:
  DATASET: cityscapes
  EXTRA_TRAIN_SET: 
  NUM_CLASSES: 19
  ROOT: data/
  TEST_SET: list/cityscapes/val.lst
  TRAIN_SET: list/cityscapes/train.lst
DEBUG:
  DEBUG: False
  SAVE_BATCH_IMAGES_GT: False
  SAVE_BATCH_IMAGES_PRED: False
  SAVE_HEATMAPS_GT: False
  SAVE_HEATMAPS_PRED: False
GPUS: (0, 1, 2, 3)
LOG_DIR: log/subset/0.1/
LOSS:
  BALANCE_WEIGHTS: [0.4, 1]
  CLASS_BALANCE: False
  OHEMKEEP: 131072
  OHEMTHRES: 0.9
  USE_OHEM: False
MODEL:
  ALIGN_CORNERS: True
  EXTRA:
    FINAL_CONV_KERNEL: 1
    STAGE1:
      BLOCK: BOTTLENECK
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4]
      NUM_CHANNELS: [64]
      NUM_MODULES: 1
      NUM_RANCHES: 1
    STAGE2:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4]
      NUM_BRANCHES: 2
      NUM_CHANNELS: [48, 96]
      NUM_MODULES: 1
    STAGE3:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4, 4]
      NUM_BRANCHES: 3
      NUM_CHANNELS: [48, 96, 192]
      NUM_MODULES: 4
    STAGE4:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4, 4, 4]
      NUM_BRANCHES: 4
      NUM_CHANNELS: [48, 96, 192, 384]
      NUM_MODULES: 3
  NAME: seg_hrnet_ocr
  NUM_OUTPUTS: 2
  OCR:
    DROPOUT: 0.05
    KEY_CHANNELS: 256
    MID_CHANNELS: 512
    SCALE: 1
  PRETRAINED: pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
OUTPUT_DIR: output/subset/0.1/
PIN_MEMORY: True
PRINT_FREQ: 10
RANK: 0
TEST:
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 4
  FLIP_TEST: False
  IMAGE_SIZE: [2048, 1024]
  MODEL_FILE: 
  MULTI_SCALE: False
  NUM_SAMPLES: 0
  OUTPUT_INDEX: -1
  SCALE_LIST: [1]
TRAIN:
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 3
  BEGIN_EPOCH: 0
  DOWNSAMPLERATE: 1
  END_EPOCH: 484
  EXTRA_EPOCH: 0
  EXTRA_LR: 0.001
  FLIP: True
  FREEZE_EPOCHS: -1
  FREEZE_LAYERS: 
  IGNORE_LABEL: 255
  IMAGE_SIZE: [1024, 512]
  LR: 0.01
  LR_FACTOR: 0.1
  LR_STEP: [90, 110]
  MOMENTUM: 0.9
  MULTI_SCALE: True
  NESTEROV: False
  NONBACKBONE_KEYWORDS: []
  NONBACKBONE_MULT: 10
  NUM_SAMPLES: 0
  OPTIMIZER: sgd
  RANDOM_BRIGHTNESS: False
  RANDOM_BRIGHTNESS_SHIFT_VALUE: 10
  RANDOM_SUBSET: 0.1
  RESUME: True
  SCALE_FACTOR: 16
  SHUFFLE: True
  WD: 0.0005
WORKERS: 4
args.local_rank:  2
Seeding with 304
=> creating output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484
=> creating log/subset/0.1/cityscapes/seg_hrnet_ocr/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484_2023-04-12-12-43
Namespace(cfg='experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml', local_rank=3, opts=[], seed=304)
AUTO_RESUME: False
CUDNN:
  BENCHMARK: True
  DETERMINISTIC: False
  ENABLED: True
DATASET:
  DATASET: cityscapes
  EXTRA_TRAIN_SET: 
  NUM_CLASSES: 19
  ROOT: data/
  TEST_SET: list/cityscapes/val.lst
  TRAIN_SET: list/cityscapes/train.lst
DEBUG:
  DEBUG: False
  SAVE_BATCH_IMAGES_GT: False
  SAVE_BATCH_IMAGES_PRED: False
  SAVE_HEATMAPS_GT: False
  SAVE_HEATMAPS_PRED: False
GPUS: (0, 1, 2, 3)
LOG_DIR: log/subset/0.1/
LOSS:
  BALANCE_WEIGHTS: [0.4, 1]
  CLASS_BALANCE: False
  OHEMKEEP: 131072
  OHEMTHRES: 0.9
  USE_OHEM: False
MODEL:
  ALIGN_CORNERS: True
  EXTRA:
    FINAL_CONV_KERNEL: 1
    STAGE1:
      BLOCK: BOTTLENECK
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4]
      NUM_CHANNELS: [64]
      NUM_MODULES: 1
      NUM_RANCHES: 1
    STAGE2:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4]
      NUM_BRANCHES: 2
      NUM_CHANNELS: [48, 96]
      NUM_MODULES: 1
    STAGE3:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4, 4]
      NUM_BRANCHES: 3
      NUM_CHANNELS: [48, 96, 192]
      NUM_MODULES: 4
    STAGE4:
      BLOCK: BASIC
      FUSE_METHOD: SUM
      NUM_BLOCKS: [4, 4, 4, 4]
      NUM_BRANCHES: 4
      NUM_CHANNELS: [48, 96, 192, 384]
      NUM_MODULES: 3
  NAME: seg_hrnet_ocr
  NUM_OUTPUTS: 2
  OCR:
    DROPOUT: 0.05
    KEY_CHANNELS: 256
    MID_CHANNELS: 512
    SCALE: 1
  PRETRAINED: pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
OUTPUT_DIR: output/subset/0.1/
PIN_MEMORY: True
PRINT_FREQ: 10
RANK: 0
TEST:
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 4
  FLIP_TEST: False
  IMAGE_SIZE: [2048, 1024]
  MODEL_FILE: 
  MULTI_SCALE: False
  NUM_SAMPLES: 0
  OUTPUT_INDEX: -1
  SCALE_LIST: [1]
TRAIN:
  BASE_SIZE: 2048
  BATCH_SIZE_PER_GPU: 3
  BEGIN_EPOCH: 0
  DOWNSAMPLERATE: 1
  END_EPOCH: 484
  EXTRA_EPOCH: 0
  EXTRA_LR: 0.001
  FLIP: True
  FREEZE_EPOCHS: -1
  FREEZE_LAYERS: 
  IGNORE_LABEL: 255
  IMAGE_SIZE: [1024, 512]
  LR: 0.01
  LR_FACTOR: 0.1
  LR_STEP: [90, 110]
  MOMENTUM: 0.9
  MULTI_SCALE: True
  NESTEROV: False
  NONBACKBONE_KEYWORDS: []
  NONBACKBONE_MULT: 10
  NUM_SAMPLES: 0
  OPTIMIZER: sgd
  RANDOM_BRIGHTNESS: False
  RANDOM_BRIGHTNESS_SHIFT_VALUE: 10
  RANDOM_SUBSET: 0.1
  RESUME: True
  SCALE_FACTOR: 16
  SHUFFLE: True
  WD: 0.0005
WORKERS: 4
args.local_rank:  3
Added key: store_based_barrier_key:1 to store for rank: 0
Added key: store_based_barrier_key:1 to store for rank: 1
Added key: store_based_barrier_key:1 to store for rank: 2
Added key: store_based_barrier_key:1 to store for rank: 3
Rank 3: Completed store-based barrier for 4 nodes.
Rank 1: Completed store-based barrier for 4 nodes.
Rank 2: Completed store-based barrier for 4 nodes.
Rank 0: Completed store-based barrier for 4 nodes.
=> init weights from normal distribution
=> init weights from normal distribution
=> init weights from normal distribution
=> init weights from normal distribution
=> loading pretrained model pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
=> loading pretrained model pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
=> loading pretrained model pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
=> loading pretrained model pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
{'ocr_distri_head.object_context_block.f_up.1.0.running_var', 'cls_head.bias', 'ocr_distri_head.object_context_block.f_pixel.3.0.bias', 'aux_head.1.running_var', 'conv3x3_ocr.0.bias', 'conv3x3_ocr.1.weight', 'ocr_distri_head.object_context_block.f_down.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_up.1.0.bias', 'ocr_distri_head.object_context_block.f_pixel.3.0.running_mean', 'ocr_distri_head.object_context_block.f_object.3.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_up.0.weight', 'ocr_distri_head.conv_bn_dropout.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_down.1.0.running_mean', 'conv3x3_ocr.1.bias', 'aux_head.0.weight', 'ocr_distri_head.object_context_block.f_up.1.0.num_batches_tracked', 'aux_head.3.bias', 'ocr_distri_head.object_context_block.f_pixel.0.weight', 'ocr_distri_head.object_context_block.f_pixel.3.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_down.1.0.bias', 'ocr_distri_head.object_context_block.f_object.2.weight', 'ocr_distri_head.object_context_block.f_up.1.0.running_mean', 'ocr_distri_head.object_context_block.f_pixel.2.weight', 'aux_head.1.weight', 'ocr_distri_head.conv_bn_dropout.1.0.bias', 'ocr_distri_head.object_context_block.f_pixel.1.0.weight', 'ocr_distri_head.object_context_block.f_object.1.0.running_var', 'ocr_distri_head.conv_bn_dropout.1.0.weight', 'aux_head.1.running_mean', 'ocr_distri_head.object_context_block.f_down.0.weight', 'ocr_distri_head.object_context_block.f_object.3.0.bias', 'ocr_distri_head.object_context_block.f_object.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_pixel.3.0.weight', 'ocr_distri_head.object_context_block.f_object.1.0.running_mean', 'aux_head.3.weight', 'ocr_distri_head.object_context_block.f_pixel.1.0.bias', 'ocr_distri_head.conv_bn_dropout.1.0.running_var', 'ocr_distri_head.object_context_block.f_object.3.0.running_mean', 'ocr_distri_head.conv_bn_dropout.1.0.running_mean', 'ocr_distri_head.object_context_block.f_object.1.0.weight', 'aux_head.1.num_batches_tracked', 'ocr_distri_head.object_context_block.f_pixel.1.0.num_batches_tracked', 'cls_head.weight', 'ocr_distri_head.object_context_block.f_object.0.weight', 'conv3x3_ocr.1.running_mean', 'aux_head.0.bias', 'conv3x3_ocr.1.running_var', 'ocr_distri_head.object_context_block.f_object.3.0.running_var', 'ocr_distri_head.conv_bn_dropout.0.weight', 'ocr_distri_head.object_context_block.f_down.1.0.running_var', 'ocr_distri_head.object_context_block.f_pixel.1.0.running_var', 'aux_head.1.bias', 'conv3x3_ocr.1.num_batches_tracked', 'ocr_distri_head.object_context_block.f_down.1.0.weight', 'ocr_distri_head.object_context_block.f_pixel.1.0.running_mean', 'ocr_distri_head.object_context_block.f_up.1.0.weight', 'conv3x3_ocr.0.weight', 'ocr_distri_head.object_context_block.f_object.3.0.weight', 'ocr_distri_head.object_context_block.f_object.1.0.bias', 'ocr_distri_head.object_context_block.f_pixel.3.0.running_var'}
{'ocr_distri_head.object_context_block.f_pixel.3.0.running_var', 'ocr_distri_head.object_context_block.f_pixel.1.0.running_mean', 'conv3x3_ocr.0.bias', 'ocr_distri_head.object_context_block.f_up.1.0.running_mean', 'ocr_distri_head.object_context_block.f_object.1.0.weight', 'ocr_distri_head.conv_bn_dropout.1.0.running_mean', 'ocr_distri_head.object_context_block.f_pixel.1.0.weight', 'ocr_distri_head.object_context_block.f_pixel.1.0.num_batches_tracked', 'aux_head.3.bias', 'ocr_distri_head.object_context_block.f_pixel.3.0.weight', 'ocr_distri_head.object_context_block.f_up.1.0.num_batches_tracked', 'conv3x3_ocr.1.weight', 'ocr_distri_head.object_context_block.f_object.3.0.weight', 'ocr_distri_head.object_context_block.f_object.1.0.num_batches_tracked', 'conv3x3_ocr.1.num_batches_tracked', 'aux_head.1.running_var', 'ocr_distri_head.object_context_block.f_pixel.2.weight', 'ocr_distri_head.object_context_block.f_up.1.0.weight', 'aux_head.1.num_batches_tracked', 'ocr_distri_head.object_context_block.f_pixel.3.0.running_mean', 'cls_head.weight', 'ocr_distri_head.object_context_block.f_object.3.0.running_mean', 'conv3x3_ocr.1.running_var', 'ocr_distri_head.conv_bn_dropout.1.0.weight', 'ocr_distri_head.object_context_block.f_down.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_up.1.0.bias', 'ocr_distri_head.object_context_block.f_object.1.0.running_mean', 'ocr_distri_head.object_context_block.f_pixel.3.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_up.0.weight', 'ocr_distri_head.object_context_block.f_pixel.1.0.running_var', 'aux_head.0.weight', 'aux_head.1.running_mean', 'ocr_distri_head.object_context_block.f_object.1.0.running_var', 'ocr_distri_head.object_context_block.f_object.0.weight', 'aux_head.0.bias', 'ocr_distri_head.object_context_block.f_down.0.weight', 'ocr_distri_head.conv_bn_dropout.1.0.running_var', 'aux_head.1.bias', 'ocr_distri_head.conv_bn_dropout.1.0.bias', 'conv3x3_ocr.0.weight', 'cls_head.bias', 'ocr_distri_head.object_context_block.f_object.2.weight', 'ocr_distri_head.object_context_block.f_pixel.0.weight', 'ocr_distri_head.object_context_block.f_object.3.0.running_var', 'aux_head.1.weight', 'conv3x3_ocr.1.running_mean', 'ocr_distri_head.object_context_block.f_down.1.0.weight', 'ocr_distri_head.object_context_block.f_object.3.0.bias', 'ocr_distri_head.object_context_block.f_object.1.0.bias', 'ocr_distri_head.object_context_block.f_down.1.0.running_var', 'ocr_distri_head.conv_bn_dropout.1.0.num_batches_tracked', 'conv3x3_ocr.1.bias', 'ocr_distri_head.object_context_block.f_pixel.3.0.bias', 'ocr_distri_head.conv_bn_dropout.0.weight', 'ocr_distri_head.object_context_block.f_object.3.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_down.1.0.bias', 'ocr_distri_head.object_context_block.f_pixel.1.0.bias', 'aux_head.3.weight', 'ocr_distri_head.object_context_block.f_up.1.0.running_var', 'ocr_distri_head.object_context_block.f_down.1.0.running_mean'}
{'incre_modules.0.0.conv3.weight', 'incre_modules.3.0.bn2.running_mean', 'incre_modules.2.0.bn1.running_mean', 'downsamp_modules.0.1.num_batches_tracked', 'incre_modules.1.0.bn2.running_mean', 'incre_modules.3.0.bn1.num_batches_tracked', 'incre_modules.2.0.conv3.weight', 'incre_modules.3.0.downsample.0.weight', 'incre_modules.1.0.bn2.weight', 'downsamp_modules.1.1.num_batches_tracked', 'incre_modules.0.0.downsample.1.running_var', 'incre_modules.0.0.bn3.running_var', 'incre_modules.1.0.bn2.num_batches_tracked', 'incre_modules.0.0.bn1.weight', 'downsamp_modules.1.0.bias', 'incre_modules.0.0.bn2.num_batches_tracked', 'downsamp_modules.0.1.weight', 'incre_modules.0.0.bn2.running_var', 'incre_modules.3.0.bn2.bias', 'incre_modules.0.0.bn1.bias', 'incre_modules.2.0.bn3.running_var', 'downsamp_modules.2.1.num_batches_tracked', 'incre_modules.0.0.bn3.running_mean', 'downsamp_modules.0.0.weight', 'incre_modules.1.0.downsample.1.bias', 'downsamp_modules.1.1.bias', 'incre_modules.1.0.bn3.running_var', 'final_layer.1.bias', 'incre_modules.0.0.bn2.weight', 'incre_modules.1.0.downsample.1.num_batches_tracked', 'incre_modules.0.0.downsample.0.weight', 'incre_modules.0.0.bn1.running_var', 'incre_modules.3.0.bn2.weight', 'downsamp_modules.2.1.running_var', 'incre_modules.3.0.bn2.num_batches_tracked', 'downsamp_modules.2.0.bias', 'downsamp_modules.2.1.weight', 'downsamp_modules.0.1.running_var', 'incre_modules.0.0.bn3.num_batches_tracked', 'incre_modules.3.0.bn3.running_mean', 'incre_modules.2.0.bn3.running_mean', 'incre_modules.1.0.bn3.weight', 'downsamp_modules.1.1.weight', 'classifier.weight', 'incre_modules.0.0.bn3.weight', 'incre_modules.3.0.conv1.weight', 'incre_modules.1.0.bn1.running_mean', 'incre_modules.3.0.downsample.1.bias', 'incre_modules.2.0.bn1.num_batches_tracked', 'incre_modules.1.0.conv1.weight', 'incre_modules.2.0.conv2.weight', 'incre_modules.1.0.downsample.1.running_var', 'final_layer.1.running_mean', 'incre_modules.1.0.conv3.weight', 'incre_modules.0.0.downsample.1.num_batches_tracked', 'incre_modules.3.0.bn1.running_mean', 'final_layer.1.weight', 'final_layer.1.running_var', 'incre_modules.2.0.downsample.1.num_batches_tracked', 'incre_modules.3.0.bn1.running_var', 'incre_modules.2.0.bn1.bias', 'downsamp_modules.2.1.bias', 'incre_modules.3.0.bn2.running_var', 'incre_modules.2.0.bn3.weight', 'incre_modules.2.0.bn2.bias', 'incre_modules.2.0.downsample.1.running_var', 'incre_modules.2.0.downsample.1.weight', 'incre_modules.3.0.bn3.bias', 'incre_modules.2.0.conv1.weight', 'incre_modules.1.0.bn3.bias', 'incre_modules.2.0.bn1.weight', 'incre_modules.0.0.bn2.running_mean', 'incre_modules.0.0.bn2.bias', 'incre_modules.3.0.conv2.weight', 'incre_modules.1.0.bn3.running_mean', 'incre_modules.2.0.downsample.1.bias', 'incre_modules.3.0.downsample.1.num_batches_tracked', 'downsamp_modules.0.1.bias', 'incre_modules.0.0.bn1.running_mean', 'downsamp_modules.0.0.bias', 'incre_modules.1.0.bn1.num_batches_tracked', 'incre_modules.2.0.bn2.running_mean', 'incre_modules.2.0.downsample.1.running_mean', 'incre_modules.0.0.downsample.1.bias', 'incre_modules.0.0.bn3.bias', 'incre_modules.3.0.conv3.weight', 'incre_modules.1.0.bn1.weight', 'final_layer.1.num_batches_tracked', 'incre_modules.1.0.conv2.weight', 'incre_modules.2.0.bn1.running_var', 'incre_modules.3.0.bn1.bias', 'incre_modules.0.0.conv2.weight', 'downsamp_modules.1.1.running_mean', 'incre_modules.1.0.bn3.num_batches_tracked', 'incre_modules.1.0.downsample.0.weight', 'incre_modules.0.0.downsample.1.running_mean', 'incre_modules.1.0.downsample.1.running_mean', 'incre_modules.3.0.bn1.weight', 'incre_modules.3.0.bn3.num_batches_tracked', 'incre_modules.0.0.bn1.num_batches_tracked', 'incre_modules.3.0.downsample.1.running_var', 'incre_modules.1.0.downsample.1.weight', 'downsamp_modules.2.0.weight', 'incre_modules.3.0.bn3.weight', 'incre_modules.0.0.downsample.1.weight', 'incre_modules.2.0.downsample.0.weight', 'incre_modules.1.0.bn1.running_var', 'incre_modules.2.0.bn2.running_var', 'incre_modules.0.0.conv1.weight', 'classifier.bias', 'incre_modules.3.0.downsample.1.running_mean', 'final_layer.0.weight', 'downsamp_modules.1.1.running_var', 'incre_modules.3.0.bn3.running_var', 'downsamp_modules.2.1.running_mean', 'incre_modules.3.0.downsample.1.weight', 'incre_modules.2.0.bn3.num_batches_tracked', 'incre_modules.2.0.bn3.bias', 'downsamp_modules.1.0.weight', 'incre_modules.2.0.bn2.weight', 'final_layer.0.bias', 'incre_modules.2.0.bn2.num_batches_tracked', 'incre_modules.1.0.bn2.bias', 'downsamp_modules.0.1.running_mean', 'incre_modules.1.0.bn2.running_var', 'incre_modules.1.0.bn1.bias'}
{'incre_modules.0.0.downsample.1.running_var', 'incre_modules.0.0.downsample.1.num_batches_tracked', 'classifier.bias', 'incre_modules.3.0.downsample.1.bias', 'final_layer.1.num_batches_tracked', 'incre_modules.3.0.bn2.running_var', 'incre_modules.2.0.bn3.running_var', 'downsamp_modules.2.1.num_batches_tracked', 'incre_modules.3.0.bn2.bias', 'incre_modules.2.0.bn2.weight', 'incre_modules.2.0.bn3.bias', 'incre_modules.2.0.downsample.1.running_mean', 'incre_modules.3.0.bn1.weight', 'incre_modules.3.0.conv3.weight', 'final_layer.1.running_mean', 'incre_modules.1.0.downsample.0.weight', 'incre_modules.2.0.bn1.running_var', 'downsamp_modules.0.1.bias', 'incre_modules.1.0.bn3.num_batches_tracked', 'downsamp_modules.0.0.bias', 'incre_modules.1.0.bn1.weight', 'incre_modules.3.0.bn2.weight', 'incre_modules.0.0.downsample.1.weight', 'incre_modules.2.0.bn3.num_batches_tracked', 'downsamp_modules.0.1.num_batches_tracked', 'incre_modules.3.0.bn3.running_mean', 'incre_modules.0.0.bn1.bias', 'incre_modules.2.0.downsample.1.bias', 'incre_modules.3.0.bn3.weight', 'downsamp_modules.0.1.running_mean', 'incre_modules.2.0.bn3.weight', 'incre_modules.0.0.bn2.bias', 'incre_modules.3.0.bn3.bias', 'incre_modules.1.0.bn3.weight', 'incre_modules.0.0.bn2.running_mean', 'incre_modules.3.0.bn1.running_var', 'incre_modules.3.0.bn2.running_mean', 'downsamp_modules.0.0.weight', 'incre_modules.1.0.downsample.1.bias', 'incre_modules.0.0.downsample.1.running_mean', 'final_layer.0.bias', 'incre_modules.0.0.conv1.weight', 'incre_modules.0.0.downsample.1.bias', 'incre_modules.3.0.downsample.1.running_mean', 'incre_modules.0.0.downsample.0.weight', 'downsamp_modules.2.1.bias', 'incre_modules.2.0.bn2.bias', 'incre_modules.2.0.conv1.weight', 'downsamp_modules.1.1.running_var', 'incre_modules.0.0.bn3.running_mean', 'incre_modules.2.0.bn1.bias', 'incre_modules.1.0.bn2.bias', 'incre_modules.0.0.bn3.bias', 'downsamp_modules.1.0.bias', 'incre_modules.0.0.bn1.running_mean', 'incre_modules.1.0.bn3.running_var', 'incre_modules.1.0.bn3.bias', 'classifier.weight', 'incre_modules.1.0.conv2.weight', 'incre_modules.3.0.bn1.num_batches_tracked', 'downsamp_modules.2.1.running_mean', 'incre_modules.2.0.downsample.0.weight', 'downsamp_modules.2.0.weight', 'downsamp_modules.2.1.weight', 'incre_modules.0.0.bn2.num_batches_tracked', 'incre_modules.0.0.bn1.running_var', 'final_layer.1.weight', 'incre_modules.0.0.bn1.num_batches_tracked', 'incre_modules.1.0.downsample.1.running_mean', 'incre_modules.0.0.bn2.weight', 'incre_modules.3.0.conv1.weight', 'incre_modules.1.0.downsample.1.num_batches_tracked', 'downsamp_modules.1.0.weight', 'incre_modules.3.0.downsample.1.weight', 'incre_modules.3.0.bn3.num_batches_tracked', 'incre_modules.3.0.bn3.running_var', 'downsamp_modules.1.1.running_mean', 'incre_modules.1.0.bn2.running_mean', 'incre_modules.2.0.bn2.running_mean', 'final_layer.1.running_var', 'incre_modules.0.0.conv2.weight', 'incre_modules.1.0.bn1.num_batches_tracked', 'incre_modules.1.0.bn2.weight', 'incre_modules.3.0.bn2.num_batches_tracked', 'incre_modules.3.0.bn1.bias', 'incre_modules.1.0.downsample.1.running_var', 'downsamp_modules.1.1.num_batches_tracked', 'incre_modules.1.0.bn1.running_var', 'incre_modules.0.0.bn3.running_var', 'incre_modules.2.0.downsample.1.running_var', 'final_layer.0.weight', 'downsamp_modules.0.1.running_var', 'incre_modules.1.0.bn1.bias', 'incre_modules.1.0.bn3.running_mean', 'incre_modules.1.0.downsample.1.weight', 'incre_modules.2.0.conv3.weight', 'incre_modules.0.0.bn1.weight', 'incre_modules.1.0.conv3.weight', 'downsamp_modules.1.1.weight', 'incre_modules.2.0.bn1.weight', 'incre_modules.2.0.conv2.weight', 'incre_modules.2.0.bn2.running_var', 'incre_modules.3.0.bn1.running_mean', 'incre_modules.0.0.conv3.weight', 'downsamp_modules.1.1.bias', 'incre_modules.1.0.bn2.num_batches_tracked', 'incre_modules.2.0.bn1.num_batches_tracked', 'final_layer.1.bias', 'incre_modules.3.0.downsample.1.num_batches_tracked', 'incre_modules.2.0.bn3.running_mean', 'downsamp_modules.0.1.weight', 'incre_modules.2.0.bn1.running_mean', 'incre_modules.2.0.downsample.1.weight', 'incre_modules.0.0.bn2.running_var', 'downsamp_modules.2.1.running_var', 'incre_modules.0.0.bn3.weight', 'incre_modules.1.0.bn2.running_var', 'incre_modules.3.0.downsample.1.running_var', 'incre_modules.0.0.bn3.num_batches_tracked', 'incre_modules.3.0.downsample.0.weight', 'incre_modules.3.0.conv2.weight', 'downsamp_modules.2.0.bias', 'incre_modules.1.0.bn1.running_mean', 'incre_modules.2.0.bn2.num_batches_tracked', 'incre_modules.1.0.conv1.weight', 'incre_modules.2.0.downsample.1.num_batches_tracked'}
{'ocr_distri_head.conv_bn_dropout.1.0.weight', 'conv3x3_ocr.1.running_var', 'ocr_distri_head.object_context_block.f_object.2.weight', 'ocr_distri_head.object_context_block.f_pixel.1.0.running_mean', 'conv3x3_ocr.0.weight', 'ocr_distri_head.object_context_block.f_pixel.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_up.1.0.bias', 'ocr_distri_head.object_context_block.f_pixel.3.0.running_var', 'ocr_distri_head.object_context_block.f_pixel.1.0.weight', 'ocr_distri_head.object_context_block.f_up.1.0.running_var', 'ocr_distri_head.object_context_block.f_object.3.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_object.3.0.bias', 'ocr_distri_head.object_context_block.f_object.3.0.running_var', 'ocr_distri_head.conv_bn_dropout.1.0.bias', 'ocr_distri_head.object_context_block.f_object.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_up.1.0.running_mean', 'aux_head.1.weight', 'ocr_distri_head.conv_bn_dropout.1.0.running_mean', 'ocr_distri_head.object_context_block.f_pixel.2.weight', 'conv3x3_ocr.1.weight', 'ocr_distri_head.object_context_block.f_object.1.0.running_mean', 'aux_head.1.running_var', 'ocr_distri_head.object_context_block.f_object.0.weight', 'ocr_distri_head.conv_bn_dropout.1.0.num_batches_tracked', 'aux_head.1.num_batches_tracked', 'ocr_distri_head.object_context_block.f_pixel.1.0.bias', 'aux_head.1.bias', 'ocr_distri_head.conv_bn_dropout.0.weight', 'ocr_distri_head.object_context_block.f_pixel.3.0.num_batches_tracked', 'aux_head.3.weight', 'ocr_distri_head.object_context_block.f_down.1.0.running_mean', 'ocr_distri_head.object_context_block.f_down.1.0.bias', 'aux_head.0.bias', 'ocr_distri_head.object_context_block.f_object.3.0.weight', 'conv3x3_ocr.1.running_mean', 'ocr_distri_head.conv_bn_dropout.1.0.running_var', 'ocr_distri_head.object_context_block.f_object.1.0.bias', 'ocr_distri_head.object_context_block.f_pixel.1.0.running_var', 'ocr_distri_head.object_context_block.f_up.1.0.weight', 'cls_head.bias', 'aux_head.0.weight', 'ocr_distri_head.object_context_block.f_down.0.weight', 'aux_head.1.running_mean', 'ocr_distri_head.object_context_block.f_object.1.0.weight', 'ocr_distri_head.object_context_block.f_pixel.0.weight', 'ocr_distri_head.object_context_block.f_down.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_pixel.3.0.running_mean', 'ocr_distri_head.object_context_block.f_down.1.0.running_var', 'aux_head.3.bias', 'ocr_distri_head.object_context_block.f_pixel.3.0.weight', 'ocr_distri_head.object_context_block.f_up.0.weight', 'conv3x3_ocr.0.bias', 'ocr_distri_head.object_context_block.f_up.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_pixel.3.0.bias', 'cls_head.weight', 'conv3x3_ocr.1.num_batches_tracked', 'ocr_distri_head.object_context_block.f_object.1.0.running_var', 'ocr_distri_head.object_context_block.f_down.1.0.weight', 'conv3x3_ocr.1.bias', 'ocr_distri_head.object_context_block.f_object.3.0.running_mean'}{'ocr_distri_head.object_context_block.f_down.1.0.weight', 'ocr_distri_head.conv_bn_dropout.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_object.2.weight', 'ocr_distri_head.object_context_block.f_object.3.0.running_mean', 'ocr_distri_head.object_context_block.f_object.1.0.num_batches_tracked', 'ocr_distri_head.conv_bn_dropout.1.0.running_mean', 'ocr_distri_head.object_context_block.f_up.1.0.num_batches_tracked', 'conv3x3_ocr.1.num_batches_tracked', 'conv3x3_ocr.1.weight', 'ocr_distri_head.conv_bn_dropout.1.0.bias', 'ocr_distri_head.object_context_block.f_up.1.0.weight', 'ocr_distri_head.object_context_block.f_pixel.1.0.running_var', 'aux_head.1.running_var', 'ocr_distri_head.object_context_block.f_object.0.weight', 'ocr_distri_head.object_context_block.f_pixel.3.0.bias', 'ocr_distri_head.object_context_block.f_object.3.0.num_batches_tracked', 'aux_head.1.bias', 'ocr_distri_head.conv_bn_dropout.1.0.weight', 'aux_head.3.weight', 'aux_head.3.bias', 'ocr_distri_head.object_context_block.f_up.1.0.running_mean', 'conv3x3_ocr.0.weight', 'conv3x3_ocr.1.bias', 'ocr_distri_head.object_context_block.f_pixel.3.0.weight', 'aux_head.0.weight', 'conv3x3_ocr.1.running_var', 'ocr_distri_head.object_context_block.f_object.3.0.weight', 'ocr_distri_head.object_context_block.f_up.1.0.running_var', 'conv3x3_ocr.0.bias', 'ocr_distri_head.object_context_block.f_down.1.0.bias', 'aux_head.1.num_batches_tracked', 'ocr_distri_head.object_context_block.f_down.1.0.running_mean', 'ocr_distri_head.object_context_block.f_pixel.3.0.running_var', 'ocr_distri_head.object_context_block.f_pixel.3.0.running_mean', 'conv3x3_ocr.1.running_mean', 'ocr_distri_head.object_context_block.f_object.3.0.running_var', 'ocr_distri_head.object_context_block.f_pixel.2.weight', 'ocr_distri_head.object_context_block.f_down.1.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_down.0.weight', 'ocr_distri_head.object_context_block.f_up.1.0.bias', 'ocr_distri_head.object_context_block.f_pixel.1.0.running_mean', 'cls_head.bias', 'aux_head.1.running_mean', 'ocr_distri_head.object_context_block.f_object.1.0.weight', 'ocr_distri_head.conv_bn_dropout.0.weight', 'ocr_distri_head.object_context_block.f_up.0.weight', 'ocr_distri_head.conv_bn_dropout.1.0.running_var', 'ocr_distri_head.object_context_block.f_pixel.1.0.num_batches_tracked', 'aux_head.1.weight', 'ocr_distri_head.object_context_block.f_down.1.0.running_var', 'ocr_distri_head.object_context_block.f_object.1.0.running_var', 'ocr_distri_head.object_context_block.f_pixel.1.0.weight', 'ocr_distri_head.object_context_block.f_pixel.3.0.num_batches_tracked', 'ocr_distri_head.object_context_block.f_pixel.0.weight', 'ocr_distri_head.object_context_block.f_object.1.0.running_mean', 'cls_head.weight', 'ocr_distri_head.object_context_block.f_object.3.0.bias', 'aux_head.0.bias', 'ocr_distri_head.object_context_block.f_pixel.1.0.bias', 'ocr_distri_head.object_context_block.f_object.1.0.bias'}

{'downsamp_modules.0.1.num_batches_tracked', 'incre_modules.0.0.bn2.weight', 'incre_modules.3.0.bn3.bias', 'incre_modules.0.0.conv3.weight', 'incre_modules.1.0.bn3.num_batches_tracked', 'downsamp_modules.0.1.running_var', 'incre_modules.2.0.bn2.bias', 'downsamp_modules.1.1.num_batches_tracked', 'incre_modules.2.0.conv1.weight', 'final_layer.0.weight', 'incre_modules.1.0.bn1.weight', 'incre_modules.2.0.bn3.running_mean', 'downsamp_modules.1.1.running_var', 'incre_modules.2.0.downsample.1.running_mean', 'incre_modules.3.0.bn1.bias', 'incre_modules.0.0.bn1.running_mean', 'downsamp_modules.1.0.weight', 'classifier.bias', 'incre_modules.1.0.downsample.1.running_var', 'incre_modules.0.0.bn3.running_var', 'incre_modules.1.0.bn1.num_batches_tracked', 'downsamp_modules.0.1.weight', 'incre_modules.2.0.downsample.0.weight', 'final_layer.1.weight', 'incre_modules.0.0.bn2.bias', 'incre_modules.0.0.bn2.running_var', 'downsamp_modules.1.1.running_mean', 'incre_modules.1.0.bn1.bias', 'incre_modules.3.0.bn2.running_mean', 'downsamp_modules.2.1.num_batches_tracked', 'downsamp_modules.2.1.weight', 'final_layer.0.bias', 'downsamp_modules.0.0.weight', 'incre_modules.1.0.conv2.weight', 'incre_modules.2.0.bn2.running_mean', 'incre_modules.3.0.bn2.bias', 'incre_modules.2.0.bn2.running_var', 'classifier.weight', 'downsamp_modules.2.0.bias', 'incre_modules.0.0.bn1.running_var', 'incre_modules.0.0.downsample.1.num_batches_tracked', 'incre_modules.2.0.bn1.running_mean', 'final_layer.1.running_var', 'incre_modules.2.0.bn1.weight', 'incre_modules.0.0.bn2.running_mean', 'incre_modules.2.0.bn1.bias', 'final_layer.1.bias', 'incre_modules.3.0.bn3.weight', 'incre_modules.3.0.conv3.weight', 'downsamp_modules.2.0.weight', 'incre_modules.3.0.bn2.weight', 'incre_modules.1.0.bn3.weight', 'final_layer.1.num_batches_tracked', 'downsamp_modules.0.0.bias', 'incre_modules.0.0.bn1.bias', 'incre_modules.1.0.bn2.running_mean', 'downsamp_modules.2.1.running_var', 'incre_modules.1.0.conv3.weight', 'downsamp_modules.1.1.bias', 'incre_modules.1.0.bn2.num_batches_tracked', 'incre_modules.2.0.downsample.1.running_var', 'incre_modules.1.0.bn2.running_var', 'incre_modules.0.0.downsample.1.running_mean', 'incre_modules.3.0.downsample.1.weight', 'incre_modules.3.0.downsample.1.num_batches_tracked', 'incre_modules.2.0.conv3.weight', 'downsamp_modules.1.1.weight', 'final_layer.1.running_mean', 'incre_modules.0.0.downsample.0.weight', 'incre_modules.2.0.downsample.1.bias', 'incre_modules.3.0.downsample.1.bias', 'incre_modules.0.0.bn2.num_batches_tracked', 'incre_modules.0.0.conv2.weight', 'incre_modules.1.0.bn2.weight', 'incre_modules.3.0.downsample.1.running_var', 'incre_modules.3.0.bn3.running_var', 'incre_modules.1.0.conv1.weight', 'incre_modules.2.0.conv2.weight', 'incre_modules.1.0.downsample.0.weight', 'incre_modules.1.0.bn3.running_var', 'incre_modules.1.0.downsample.1.bias', 'incre_modules.1.0.bn3.running_mean', 'downsamp_modules.0.1.running_mean', 'incre_modules.0.0.bn1.weight', 'incre_modules.2.0.bn2.num_batches_tracked', 'incre_modules.3.0.downsample.0.weight', 'incre_modules.2.0.bn3.bias', 'incre_modules.3.0.bn2.num_batches_tracked', 'incre_modules.0.0.bn3.weight', 'incre_modules.2.0.bn1.num_batches_tracked', 'incre_modules.3.0.bn1.num_batches_tracked', 'incre_modules.3.0.conv2.weight', 'incre_modules.0.0.bn3.num_batches_tracked', 'incre_modules.2.0.bn1.running_var', 'incre_modules.2.0.downsample.1.num_batches_tracked', 'incre_modules.1.0.bn2.bias', 'incre_modules.3.0.conv1.weight', 'downsamp_modules.1.0.bias', 'incre_modules.2.0.bn3.running_var', 'incre_modules.3.0.bn3.running_mean', 'incre_modules.1.0.downsample.1.running_mean', 'incre_modules.3.0.bn1.running_mean', 'incre_modules.1.0.bn3.bias', 'incre_modules.1.0.downsample.1.num_batches_tracked', 'incre_modules.0.0.bn3.bias', 'incre_modules.0.0.downsample.1.bias', 'incre_modules.2.0.bn3.num_batches_tracked', 'incre_modules.3.0.downsample.1.running_mean', 'incre_modules.1.0.bn1.running_var', 'incre_modules.0.0.bn3.running_mean', 'incre_modules.2.0.bn3.weight', 'incre_modules.2.0.bn2.weight', 'downsamp_modules.0.1.bias', 'incre_modules.0.0.bn1.num_batches_tracked', 'incre_modules.0.0.downsample.1.running_var', 'incre_modules.3.0.bn2.running_var', 'incre_modules.1.0.downsample.1.weight', 'incre_modules.0.0.downsample.1.weight', 'incre_modules.3.0.bn1.weight', 'incre_modules.3.0.bn1.running_var', 'incre_modules.1.0.bn1.running_mean', 'downsamp_modules.2.1.running_mean', 'incre_modules.0.0.conv1.weight', 'downsamp_modules.2.1.bias', 'incre_modules.2.0.downsample.1.weight', 'incre_modules.3.0.bn3.num_batches_tracked'}{'downsamp_modules.1.1.weight', 'downsamp_modules.0.1.running_var', 'incre_modules.2.0.bn3.weight', 'incre_modules.0.0.downsample.1.running_mean', 'downsamp_modules.2.1.num_batches_tracked', 'incre_modules.1.0.downsample.1.bias', 'incre_modules.2.0.downsample.1.running_var', 'incre_modules.0.0.downsample.1.running_var', 'incre_modules.1.0.conv1.weight', 'incre_modules.3.0.downsample.1.num_batches_tracked', 'final_layer.0.bias', 'incre_modules.3.0.bn3.running_mean', 'incre_modules.2.0.bn3.bias', 'incre_modules.2.0.bn3.num_batches_tracked', 'incre_modules.1.0.bn2.bias', 'incre_modules.1.0.downsample.1.running_var', 'incre_modules.1.0.downsample.1.weight', 'downsamp_modules.2.0.weight', 'incre_modules.3.0.downsample.0.weight', 'downsamp_modules.2.1.bias', 'incre_modules.3.0.downsample.1.running_mean', 'downsamp_modules.1.1.running_var', 'incre_modules.3.0.bn1.running_mean', 'incre_modules.0.0.bn1.weight', 'incre_modules.1.0.bn2.running_var', 'incre_modules.3.0.downsample.1.running_var', 'incre_modules.2.0.downsample.1.bias', 'incre_modules.2.0.bn1.running_var', 'incre_modules.0.0.downsample.1.bias', 'downsamp_modules.1.1.bias', 'incre_modules.0.0.bn1.running_mean', 'incre_modules.0.0.bn2.running_mean', 'incre_modules.3.0.bn1.bias', 'incre_modules.3.0.conv2.weight', 'incre_modules.2.0.downsample.0.weight', 'final_layer.1.weight', 'incre_modules.3.0.bn1.num_batches_tracked', 'incre_modules.2.0.downsample.1.weight', 'incre_modules.0.0.conv2.weight', 'incre_modules.1.0.bn3.running_var', 'incre_modules.2.0.bn2.running_var', 'incre_modules.0.0.downsample.0.weight', 'incre_modules.0.0.bn2.running_var', 'incre_modules.2.0.bn2.weight', 'incre_modules.2.0.bn2.bias', 'downsamp_modules.1.1.running_mean', 'incre_modules.3.0.bn2.bias', 'incre_modules.2.0.conv3.weight', 'incre_modules.0.0.downsample.1.num_batches_tracked', 'final_layer.0.weight', 'incre_modules.3.0.bn3.running_var', 'incre_modules.0.0.conv1.weight', 'incre_modules.2.0.downsample.1.num_batches_tracked', 'incre_modules.2.0.bn1.num_batches_tracked', 'final_layer.1.num_batches_tracked', 'incre_modules.2.0.bn1.running_mean', 'incre_modules.3.0.bn3.bias', 'incre_modules.2.0.bn1.bias', 'incre_modules.1.0.conv3.weight', 'incre_modules.0.0.bn2.weight', 'downsamp_modules.0.0.bias', 'final_layer.1.bias', 'incre_modules.2.0.bn3.running_var', 'incre_modules.1.0.bn3.num_batches_tracked', 'incre_modules.0.0.bn3.bias', 'incre_modules.3.0.bn3.weight', 'incre_modules.0.0.downsample.1.weight', 'incre_modules.0.0.bn1.num_batches_tracked', 'incre_modules.3.0.downsample.1.bias', 'downsamp_modules.0.1.num_batches_tracked', 'incre_modules.2.0.bn2.running_mean', 'incre_modules.2.0.conv2.weight', 'incre_modules.0.0.conv3.weight', 'incre_modules.0.0.bn3.weight', 'downsamp_modules.0.1.weight', 'incre_modules.2.0.bn3.running_mean', 'incre_modules.3.0.bn1.running_var', 'classifier.weight', 'incre_modules.0.0.bn3.running_mean', 'incre_modules.0.0.bn3.num_batches_tracked', 'incre_modules.3.0.downsample.1.weight', 'incre_modules.1.0.bn1.num_batches_tracked', 'incre_modules.3.0.conv1.weight', 'incre_modules.1.0.downsample.1.num_batches_tracked', 'incre_modules.0.0.bn3.running_var', 'downsamp_modules.0.1.bias', 'downsamp_modules.0.1.running_mean', 'incre_modules.3.0.bn2.running_mean', 'final_layer.1.running_var', 'downsamp_modules.1.0.weight', 'downsamp_modules.2.1.weight', 'incre_modules.1.0.bn3.running_mean', 'incre_modules.2.0.downsample.1.running_mean', 'incre_modules.3.0.bn2.num_batches_tracked', 'incre_modules.1.0.bn2.num_batches_tracked', 'downsamp_modules.1.1.num_batches_tracked', 'incre_modules.1.0.bn1.running_mean', 'incre_modules.3.0.bn3.num_batches_tracked', 'incre_modules.2.0.bn1.weight', 'incre_modules.1.0.bn2.running_mean', 'incre_modules.3.0.bn2.weight', 'incre_modules.3.0.bn1.weight', 'downsamp_modules.0.0.weight', 'incre_modules.1.0.downsample.0.weight', 'downsamp_modules.2.0.bias', 'incre_modules.0.0.bn1.bias', 'incre_modules.0.0.bn2.num_batches_tracked', 'incre_modules.3.0.bn2.running_var', 'incre_modules.0.0.bn2.bias', 'incre_modules.1.0.bn1.bias', 'incre_modules.1.0.bn1.running_var', 'classifier.bias', 'incre_modules.1.0.bn3.bias', 'incre_modules.3.0.conv3.weight', 'downsamp_modules.2.1.running_mean', 'incre_modules.1.0.downsample.1.running_mean', 'incre_modules.1.0.conv2.weight', 'incre_modules.2.0.bn2.num_batches_tracked', 'incre_modules.0.0.bn1.running_var', 'downsamp_modules.1.0.bias', 'incre_modules.1.0.bn1.weight', 'incre_modules.2.0.conv1.weight', 'downsamp_modules.2.1.running_var', 'final_layer.1.running_mean', 'incre_modules.1.0.bn2.weight', 'incre_modules.1.0.bn3.weight'}

tools/train.py:250: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  epoch_iters = np.int(train_dataset.__len__() /
tools/train.py:250: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  epoch_iters = np.int(train_dataset.__len__() /
tools/train.py:250: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  epoch_iters = np.int(train_dataset.__len__() /
tools/train.py:250: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  epoch_iters = np.int(train_dataset.__len__() /
[W reducer.cpp:1158] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[W reducer.cpp:1158] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[W reducer.cpp:1158] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[W reducer.cpp:1158] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
Epoch: [0/484] Iter:[0/24], Time: 8.00, lr: [0.01], Loss: 4.256449
Epoch: [0/484] Iter:[10/24], Time: 1.09, lr: [0.009992251732507718], Loss: 2.818506
Epoch: [0/484] Iter:[20/24], Time: 0.76, lr: [0.00998450279737781], Loss: 2.374220
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.84596833 0.093472   0.68175349 0.         0.         0.
 0.         0.         0.64337736 0.         0.40830983 0.00232658
 0.         0.57009024 0.         0.         0.         0.
 0.        ] 0.17080514807643637
1 [0.85483282 0.29132609 0.67715891 0.         0.         0.00824997
 0.         0.         0.75889812 0.         0.52975936 0.20979496
 0.         0.63484559 0.         0.         0.         0.
 0.        ] 0.20867714798192172
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 1.294, MeanIU:  0.2087, Best_mIoU:  0.2087
[0.85483282 0.29132609 0.67715891 0.         0.         0.00824997
 0.         0.         0.75889812 0.         0.52975936 0.20979496
 0.         0.63484559 0.         0.         0.         0.
 0.        ]
Epoch: [1/484] Iter:[0/24], Time: 0.94, lr: [0.009981403036245477], Loss: 1.679380
Epoch: [1/484] Iter:[10/24], Time: 0.44, lr: [0.009973653165358603], Loss: 1.204424
Epoch: [1/484] Iter:[20/24], Time: 0.41, lr: [0.00996590262531213], Loss: 1.192457
0
0
0
0
10
10
10
10
20
20
20
20
3030

30
30
40
40
40
40
0 [8.72523145e-01 4.08801886e-01 7.52546710e-01 0.00000000e+00
 0.00000000e+00 2.49354281e-01 0.00000000e+00 7.81496985e-04
 8.46432299e-01 1.01796123e-01 7.30314620e-01 4.34647561e-01
 0.00000000e+00 7.52736686e-01 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 2.90518392e-03] 0.27120210483884805
1 [8.43901536e-01 4.20018334e-01 7.31975936e-01 0.00000000e+00
 2.79130230e-06 2.73185116e-01 0.00000000e+00 5.09965647e-05
 8.47977305e-01 2.14376255e-01 7.35453808e-01 4.54037344e-01
 0.00000000e+00 7.55512707e-01 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 1.77606612e-01] 0.2870578285014378
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.852, MeanIU:  0.2871, Best_mIoU:  0.2871
[8.43901536e-01 4.20018334e-01 7.31975936e-01 0.00000000e+00
 2.79130230e-06 2.73185116e-01 0.00000000e+00 5.09965647e-05
 8.47977305e-01 2.14376255e-01 7.35453808e-01 4.54037344e-01
 0.00000000e+00 7.55512707e-01 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 1.77606612e-01]
Epoch: [2/484] Iter:[0/24], Time: 0.95, lr: [0.009962802221786402], Loss: 1.755512
Epoch: [2/484] Iter:[10/24], Time: 0.44, lr: [0.009955050743847565], Loss: 1.105612
Epoch: [2/484] Iter:[20/24], Time: 0.41, lr: [0.009947298595220522], Loss: 1.000685
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.85592019 0.43911367 0.73328598 0.         0.04297326 0.25135991
 0.         0.12379662 0.72249241 0.11695867 0.71132035 0.46437326
 0.         0.7323909  0.         0.         0.         0.
 0.10006234] 0.27863408180757804
1 [0.84759847 0.44002093 0.74396603 0.         0.0584867  0.25207475
 0.         0.22338724 0.71481767 0.16125691 0.745583   0.47090993
 0.         0.73876698 0.         0.         0.         0.
 0.29514455] 0.2995796395883066
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.859, MeanIU:  0.2996, Best_mIoU:  0.2996
[0.84759847 0.44002093 0.74396603 0.         0.0584867  0.25207475
 0.         0.22338724 0.71481767 0.16125691 0.745583   0.47090993
 0.         0.73876698 0.         0.         0.         0.
 0.29514455]
Epoch: [3/484] Iter:[0/24], Time: 0.93, lr: [0.009944197547833935], Loss: 1.016206
Epoch: [3/484] Iter:[10/24], Time: 0.44, lr: [0.009936444459169807], Loss: 0.983757
Epoch: [3/484] Iter:[20/24], Time: 0.41, lr: [0.009928690698282186], Loss: 0.921776
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.89410271 0.51508513 0.78486135 0.         0.16548685 0.29201249
 0.         0.33507527 0.85265392 0.36819679 0.77004535 0.53180665
 0.         0.8159679  0.         0.         0.         0.
 0.46386935] 0.35732441030186957
1 [0.87576918 0.52726064 0.75850666 0.         0.0761035  0.29353482
 0.         0.31383761 0.8510309  0.36307359 0.71375662 0.54689485
 0.         0.8142605  0.         0.         0.         0.
 0.52046379] 0.35023645654528773
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.686, MeanIU:  0.3502, Best_mIoU:  0.3502
[0.87576918 0.52726064 0.75850666 0.         0.0761035  0.29353482
 0.         0.31383761 0.8510309  0.36307359 0.71375662 0.54689485
 0.         0.8142605  0.         0.         0.         0.
 0.52046379]
Epoch: [4/484] Iter:[0/24], Time: 0.94, lr: [0.00992558900556086], Loss: 0.636691
Epoch: [4/484] Iter:[10/24], Time: 0.44, lr: [0.009917834302482052], Loss: 0.691609
Epoch: [4/484] Iter:[20/24], Time: 0.41, lr: [0.009910078925637736], Loss: 0.636185
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [9.16294066e-01 5.74313058e-01 7.89274649e-01 7.01003641e-05
 2.73945188e-01 3.25802845e-01 0.00000000e+00 4.03339609e-01
 8.55521658e-01 2.88176970e-01 5.41728107e-01 5.53854454e-01
 0.00000000e+00 8.30123770e-01 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 5.51463913e-01] 0.3633635992704519
1 [9.19343363e-01 5.97422545e-01 7.76150731e-01 7.48668992e-05
 2.84613033e-01 3.29797022e-01 0.00000000e+00 3.90454710e-01
 8.51171419e-01 2.64203143e-01 5.24267492e-01 5.40231306e-01
 0.00000000e+00 8.19109673e-01 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 5.45173183e-01] 0.3601059203776967
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.619, MeanIU:  0.3601, Best_mIoU:  0.3601
[9.19343363e-01 5.97422545e-01 7.76150731e-01 7.48668992e-05
 2.84613033e-01 3.29797022e-01 0.00000000e+00 3.90454710e-01
 8.51171419e-01 2.64203143e-01 5.24267492e-01 5.40231306e-01
 0.00000000e+00 8.19109673e-01 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 5.45173183e-01]
Epoch: [5/484] Iter:[0/24], Time: 0.93, lr: [0.009906976586101339], Loss: 0.797838
Epoch: [5/484] Iter:[10/24], Time: 0.44, lr: [0.009899220264902293], Loss: 0.648935
Epoch: [5/484] Iter:[20/24], Time: 0.41, lr: [0.00989146326838896], Loss: 0.688055
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [9.16860048e-01 6.11106032e-01 7.78183318e-01 3.27622951e-03
 2.05048740e-01 3.53870191e-01 0.00000000e+00 3.84712170e-01
 8.61308325e-01 2.91345769e-01 7.87210660e-01 5.95406484e-01
 0.00000000e+00 8.06693267e-01 2.68085151e-05 0.00000000e+00
 0.00000000e+00 0.00000000e+00 5.65925925e-01] 0.37689336672441603
1 [9.08423726e-01 6.18112879e-01 7.56141774e-01 1.05602105e-02
 1.30824770e-01 3.49855189e-01 0.00000000e+00 4.15742776e-01
 8.56930551e-01 2.66097798e-01 8.01014622e-01 5.94398861e-01
 0.00000000e+00 8.08178834e-01 1.14093561e-04 0.00000000e+00
 0.00000000e+00 0.00000000e+00 5.83611326e-01] 0.3736846005883879
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.595, MeanIU:  0.3737, Best_mIoU:  0.3737
[9.08423726e-01 6.18112879e-01 7.56141774e-01 1.05602105e-02
 1.30824770e-01 3.49855189e-01 0.00000000e+00 4.15742776e-01
 8.56930551e-01 2.66097798e-01 8.01014622e-01 5.94398861e-01
 0.00000000e+00 8.08178834e-01 1.14093561e-04 0.00000000e+00
 0.00000000e+00 0.00000000e+00 5.83611326e-01]
Epoch: [6/484] Iter:[0/24], Time: 0.92, lr: [0.009888360280550665], Loss: 0.686962
Epoch: [6/484] Iter:[10/24], Time: 0.44, lr: [0.009880602337509549], Loss: 0.789408
Epoch: [6/484] Iter:[20/24], Time: 0.41, lr: [0.00987284371759856], Loss: 0.720874
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [8.95662674e-01 5.33333263e-01 8.07700141e-01 1.12433950e-02
 2.42121411e-01 3.20787896e-01 0.00000000e+00 4.65493873e-01
 7.84069625e-01 2.96546296e-01 8.10525135e-01 5.70950861e-01
 0.00000000e+00 7.64280610e-01 1.41288274e-05 0.00000000e+00
 0.00000000e+00 0.00000000e+00 5.49719301e-01] 0.37118150576274855
1 [8.88288416e-01 5.27636707e-01 8.02982067e-01 3.25499135e-03
 2.14646926e-01 3.36000791e-01 0.00000000e+00 4.58580534e-01
 7.58564124e-01 2.73595107e-01 8.17597611e-01 5.65866400e-01
 0.00000000e+00 7.47115263e-01 3.32804102e-04 0.00000000e+00
 0.00000000e+00 0.00000000e+00 4.86297662e-01] 0.3621452317738057
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.733, MeanIU:  0.3621, Best_mIoU:  0.3737
[8.88288416e-01 5.27636707e-01 8.02982067e-01 3.25499135e-03
 2.14646926e-01 3.36000791e-01 0.00000000e+00 4.58580534e-01
 7.58564124e-01 2.73595107e-01 8.17597611e-01 5.65866400e-01
 0.00000000e+00 7.47115263e-01 3.32804102e-04 0.00000000e+00
 0.00000000e+00 0.00000000e+00 4.86297662e-01]
Epoch: [7/484] Iter:[0/24], Time: 0.94, lr: [0.009869740079964998], Loss: 0.661313
Epoch: [7/484] Iter:[10/24], Time: 0.44, lr: [0.00986198051134361], Loss: 0.609235
Epoch: [7/484] Iter:[20/24], Time: 0.42, lr: [0.009854220264289901], Loss: 0.596165
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [9.37513527e-01 6.23374379e-01 8.38732655e-01 7.43196093e-02
 2.67783463e-01 3.75811865e-01 3.91507819e-04 4.23396211e-01
 8.61871890e-01 3.97078625e-01 8.56306014e-01 6.14235550e-01
 0.00000000e+00 8.45922294e-01 6.17290420e-04 1.38629738e-04
 0.00000000e+00 0.00000000e+00 6.06019127e-01] 0.4065006651029533
1 [9.31613659e-01 6.04767082e-01 8.24693213e-01 1.89129451e-01
 2.34685114e-01 3.57676783e-01 5.34723578e-04 4.72426110e-01
 8.49495186e-01 3.64238758e-01 8.52646134e-01 6.11511310e-01
 1.01235273e-06 8.52252205e-01 2.07828383e-02 4.14951613e-04
 0.00000000e+00 0.00000000e+00 6.04860477e-01] 0.40903836874162414
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.527, MeanIU:  0.4090, Best_mIoU:  0.4090
[9.31613659e-01 6.04767082e-01 8.24693213e-01 1.89129451e-01
 2.34685114e-01 3.57676783e-01 5.34723578e-04 4.72426110e-01
 8.49495186e-01 3.64238758e-01 8.52646134e-01 6.11511310e-01
 1.01235273e-06 8.52252205e-01 2.07828383e-02 4.14951613e-04
 0.00000000e+00 0.00000000e+00 6.04860477e-01]
Epoch: [8/484] Iter:[0/24], Time: 0.95, lr: [0.009851115975361124], Loss: 0.570440
Epoch: [8/484] Iter:[10/24], Time: 0.44, lr: [0.009843354777404771], Loss: 0.641823
Epoch: [8/484] Iter:[20/24], Time: 0.41, lr: [0.009835592899446752], Loss: 0.650936
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [9.36344540e-01 6.15969113e-01 8.41442943e-01 1.64160908e-01
 3.00627480e-01 3.65004260e-01 3.00601978e-03 3.40978517e-01
 8.64668855e-01 2.43646590e-01 8.81614494e-01 6.22205946e-01
 0.00000000e+00 8.31186319e-01 5.39881921e-04 1.58978224e-02
 0.00000000e+00 1.22015588e-04 6.09383792e-01] 0.40193681569764056
1 [9.31328079e-01 6.20004954e-01 8.40791231e-01 4.85766250e-02
 3.18913637e-01 3.63143766e-01 1.49493938e-03 4.21976836e-01
 8.62267720e-01 3.17092083e-01 8.80501833e-01 6.29053928e-01
 1.87284307e-05 8.37928702e-01 0.00000000e+00 1.49886270e-01
 0.00000000e+00 1.68657026e-02 6.01956168e-01] 0.41272637914883764
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.491, MeanIU:  0.4127, Best_mIoU:  0.4127
[9.31328079e-01 6.20004954e-01 8.40791231e-01 4.85766250e-02
 3.18913637e-01 3.63143766e-01 1.49493938e-03 4.21976836e-01
 8.62267720e-01 3.17092083e-01 8.80501833e-01 6.29053928e-01
 1.87284307e-05 8.37928702e-01 0.00000000e+00 1.49886270e-01
 0.00000000e+00 1.68657026e-02 6.01956168e-01]
Epoch: [9/484] Iter:[0/24], Time: 0.94, lr: [0.00983248795771619], Loss: 0.595016
Epoch: [9/484] Iter:[10/24], Time: 0.44, lr: [0.009824725126653592], Loss: 0.550816
Epoch: [9/484] Iter:[20/24], Time: 0.42, lr: [0.009816961614013039], Loss: 0.535436
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [9.21151878e-01 5.09644951e-01 8.35797041e-01 9.62060771e-02
 2.55171088e-01 3.39858861e-01 1.10585278e-03 4.53784742e-01
 8.70984061e-01 3.91765232e-01 8.41638410e-01 6.23430226e-01
 2.02469418e-06 8.42408891e-01 4.99872950e-04 4.03354874e-02
 0.00000000e+00 1.24099729e-03 5.99685597e-01] 0.4013005941930784
1 [9.17498180e-01 4.78426066e-01 8.27439876e-01 1.23030929e-01
 2.74978626e-01 3.33569223e-01 6.01974117e-03 4.59051946e-01
 8.67307549e-01 4.13956797e-01 7.97393525e-01 6.35834680e-01
 0.00000000e+00 8.54250900e-01 6.62743563e-05 1.50434204e-01
 0.00000000e+00 4.98360081e-02 6.20076958e-01] 0.4110090254205191
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.522, MeanIU:  0.4110, Best_mIoU:  0.4127
[9.17498180e-01 4.78426066e-01 8.27439876e-01 1.23030929e-01
 2.74978626e-01 3.33569223e-01 6.01974117e-03 4.59051946e-01
 8.67307549e-01 4.13956797e-01 7.97393525e-01 6.35834680e-01
 0.00000000e+00 8.54250900e-01 6.62743563e-05 1.50434204e-01
 0.00000000e+00 4.98360081e-02 6.20076958e-01]
Epoch: [10/484] Iter:[0/24], Time: 0.95, lr: [0.00981385601796745], Loss: 0.284833
Epoch: [10/484] Iter:[10/24], Time: 0.44, lr: [0.009806091550010627], Loss: 0.520401
Epoch: [10/484] Iter:[20/24], Time: 0.42, lr: [0.009798326398892568], Loss: 0.575429
0
0
00

10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [8.99058722e-01 4.84679128e-01 7.30654979e-01 5.25075774e-02
 1.58533643e-01 2.70491191e-01 2.38529016e-03 4.14890168e-01
 7.76950691e-01 2.68986571e-01 8.53934391e-01 5.64736805e-01
 9.17470170e-04 7.37223947e-01 5.81401224e-04 5.02857130e-03
 0.00000000e+00 2.22489397e-02 5.43303352e-01] 0.35721646509514565
1 [0.90461424 0.45163401 0.69412478 0.05235125 0.1114083  0.29727422
 0.01144577 0.40403152 0.58921291 0.227006   0.83290696 0.57875609
 0.00932411 0.59235622 0.00445634 0.01133414 0.         0.04310164
 0.55128593] 0.3350854956476796
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.830, MeanIU:  0.3351, Best_mIoU:  0.4127
[0.90461424 0.45163401 0.69412478 0.05235125 0.1114083  0.29727422
 0.01144577 0.40403152 0.58921291 0.227006   0.83290696 0.57875609
 0.00932411 0.59235622 0.00445634 0.01133414 0.         0.04310164
 0.55128593]
Epoch: [11/484] Iter:[0/24], Time: 0.95, lr: [0.009795220147012004], Loss: 0.346029
Epoch: [11/484] Iter:[10/24], Time: 0.44, lr: [0.009787454038356163], Loss: 0.671616
Epoch: [11/484] Iter:[20/24], Time: 0.42, lr: [0.009779687244948772], Loss: 0.582280
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.8829891  0.5008063  0.79486542 0.18351692 0.27163178 0.37742073
 0.03488404 0.49718747 0.85268268 0.40663868 0.82331492 0.62599648
 0.0111707  0.70269281 0.01116357 0.00256375 0.         0.02901312
 0.6137597 ] 0.4011735879655896
1 [8.79136749e-01 4.75285068e-01 7.69651212e-01 6.39626063e-02
 2.60806492e-01 3.69862405e-01 2.11519596e-01 5.05991089e-01
 8.54534467e-01 3.95670021e-01 7.83059589e-01 6.41556562e-01
 3.47573797e-02 7.68539168e-01 9.22396463e-02 7.66182447e-05
 0.00000000e+00 4.62494993e-02 6.17556498e-01] 0.4089712981808166
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.665, MeanIU:  0.4090, Best_mIoU:  0.4127
[8.79136749e-01 4.75285068e-01 7.69651212e-01 6.39626063e-02
 2.60806492e-01 3.69862405e-01 2.11519596e-01 5.05991089e-01
 8.54534467e-01 3.95670021e-01 7.83059589e-01 6.41556562e-01
 3.47573797e-02 7.68539168e-01 9.22396463e-02 7.66182447e-05
 0.00000000e+00 4.62494993e-02 6.17556498e-01]
Epoch: [12/484] Iter:[0/24], Time: 0.96, lr: [0.009776580335706525], Loss: 0.457465
Epoch: [12/484] Iter:[10/24], Time: 0.44, lr: [0.00976881258252996], Loss: 0.503422
Epoch: [12/484] Iter:[20/24], Time: 0.42, lr: [0.009761044143004445], Loss: 0.509414
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94263908 0.64453357 0.83823437 0.17947885 0.30965281 0.39906644
 0.10104835 0.51431464 0.86717331 0.39430297 0.84705465 0.64703226
 0.03801886 0.83488177 0.10249238 0.14707598 0.         0.06881634
 0.61726911] 0.4470045122742538
1 [0.94201685 0.64486958 0.83332119 0.13810342 0.24456431 0.37687029
 0.17337695 0.51646214 0.87108067 0.4274422  0.86392605 0.65965618
 0.02881012 0.84279278 0.12609574 0.20036834 0.         0.11074263
 0.62003139] 0.4537121492568614
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.465, MeanIU:  0.4537, Best_mIoU:  0.4537
[0.94201685 0.64486958 0.83332119 0.13810342 0.24456431 0.37687029
 0.17337695 0.51646214 0.87108067 0.4274422  0.86392605 0.65965618
 0.02881012 0.84279278 0.12609574 0.20036834 0.         0.11074263
 0.62003139]
Epoch: [13/484] Iter:[0/24], Time: 0.93, lr: [0.009757936574867013], Loss: 0.524465
Epoch: [13/484] Iter:[10/24], Time: 0.44, lr: [0.009750167173330982], Loss: 0.588376
Epoch: [13/484] Iter:[20/24], Time: 0.42, lr: [0.009742397083841472], Loss: 0.564991
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94571851 0.66288597 0.84794467 0.19579087 0.3620961  0.38962174
 0.08782552 0.51766775 0.87704359 0.39315775 0.87822146 0.64088828
 0.00657665 0.86173289 0.1343128  0.21420985 0.         0.08660343
 0.62549192] 0.4593573564652696
1 [0.93040269 0.67010537 0.83955834 0.16982125 0.35681758 0.38732296
 0.09695818 0.51381906 0.85011999 0.44381826 0.87460145 0.64218885
 0.00189436 0.86519099 0.16536069 0.18635546 0.         0.16148364
 0.63253254] 0.4625448230676442
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.453, MeanIU:  0.4625, Best_mIoU:  0.4625
[0.93040269 0.67010537 0.83955834 0.16982125 0.35681758 0.38732296
 0.09695818 0.51381906 0.85011999 0.44381826 0.87460145 0.64218885
 0.00189436 0.86519099 0.16536069 0.18635546 0.         0.16148364
 0.63253254]
Epoch: [14/484] Iter:[0/24], Time: 0.95, lr: [0.009739288855268508], Loss: 0.386024
Epoch: [14/484] Iter:[10/24], Time: 0.44, lr: [0.009731517801517127], Loss: 0.622145
Epoch: [14/484] Iter:[20/24], Time: 0.42, lr: [0.009723746058200566], Loss: 0.576221
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9336984  0.64896568 0.79767697 0.19691312 0.1135083  0.38450774
 0.17310541 0.39681811 0.87598349 0.41568607 0.85447439 0.62371446
 0.03438364 0.85465216 0.03184311 0.34972966 0.         0.02743288
 0.60626   ] 0.43786071434830937
1 [0.93915977 0.64133202 0.78057705 0.2517933  0.02369244 0.34190612
 0.23157815 0.30935875 0.87354073 0.27297735 0.83599875 0.58593086
 0.1433566  0.85786497 0.0853554  0.26009311 0.         0.02298187
 0.60050555] 0.42410540920294454
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.519, MeanIU:  0.4241, Best_mIoU:  0.4625
[0.93915977 0.64133202 0.78057705 0.2517933  0.02369244 0.34190612
 0.23157815 0.30935875 0.87354073 0.27297735 0.83599875 0.58593086
 0.1433566  0.85786497 0.0853554  0.26009311 0.         0.02298187
 0.60050555]
Epoch: [15/484] Iter:[0/24], Time: 0.95, lr: [0.00972063716764483], Loss: 0.317913
Epoch: [15/484] Iter:[10/24], Time: 0.44, lr: [0.00971286445780496], Loss: 0.590358
Epoch: [15/484] Iter:[20/24], Time: 0.42, lr: [0.009705091056780982], Loss: 0.517143
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [9.45294676e-01 6.75018743e-01 8.46244343e-01 1.27999143e-01
 3.38894041e-01 3.60781675e-01 2.68334512e-01 5.23582717e-01
 8.79340464e-01 4.76545495e-01 8.89531663e-01 6.36550065e-01
 3.15176846e-02 8.60589878e-01 7.39035689e-02 1.34861083e-01
 1.67671565e-04 3.27319640e-02 6.35619089e-01] 0.45986886715099634
1 [0.94585642 0.68058963 0.84278707 0.09521608 0.36837603 0.33172033
 0.32509371 0.53664801 0.87444276 0.43451338 0.89235638 0.63620438
 0.03479124 0.85904337 0.06902729 0.09251243 0.00276516 0.0425919
 0.64145565] 0.4582100637845767
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.423, MeanIU:  0.4582, Best_mIoU:  0.4625
[0.94585642 0.68058963 0.84278707 0.09521608 0.36837603 0.33172033
 0.32509371 0.53664801 0.87444276 0.43451338 0.89235638 0.63620438
 0.03479124 0.85904337 0.06902729 0.09251243 0.00276516 0.0425919
 0.64145565]
Epoch: [16/484] Iter:[0/24], Time: 0.96, lr: [0.0097019815026883], Loss: 0.299477
Epoch: [16/484] Iter:[10/24], Time: 0.44, lr: [0.009694207132869434], Loss: 0.481875
Epoch: [16/484] Iter:[20/24], Time: 0.42, lr: [0.009686432070240256], Loss: 0.454545
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95512454 0.70651044 0.84815771 0.25949706 0.38175544 0.42864593
 0.38307102 0.55685545 0.88108126 0.44129116 0.78009337 0.65584417
 0.11409197 0.87262148 0.19002264 0.30653358 0.         0.06077642
 0.63967252] 0.49798137595544406
1 [0.95606858 0.7108267  0.84685144 0.26495225 0.41267608 0.42984647
 0.43501036 0.56499035 0.87897995 0.44569021 0.76115013 0.66997174
 0.18807524 0.87909641 0.1786996  0.14528062 0.         0.12601748
 0.64155064] 0.5018807492240738
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.399, MeanIU:  0.5019, Best_mIoU:  0.5019
[0.95606858 0.7108267  0.84685144 0.26495225 0.41267608 0.42984647
 0.43501036 0.56499035 0.87897995 0.44569021 0.76115013 0.66997174
 0.18807524 0.87909641 0.1786996  0.14528062 0.         0.12601748
 0.64155064]
Epoch: [17/484] Iter:[0/24], Time: 0.94, lr: [0.009683321851049479], Loss: 0.319322
Epoch: [17/484] Iter:[10/24], Time: 0.44, lr: [0.009675545817343619], Loss: 0.397855
Epoch: [17/484] Iter:[20/24], Time: 0.42, lr: [0.009667769089193928], Loss: 0.426283
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94359659 0.67295927 0.84576989 0.22112523 0.38445924 0.3872759
 0.3344103  0.56879332 0.8767469  0.44887934 0.84608777 0.62738118
 0.07904624 0.82480127 0.0931954  0.09683411 0.00294436 0.10925519
 0.62495303] 0.4730797114646327
1 [9.46103742e-01 6.82011153e-01 8.45448722e-01 1.97251796e-01
 4.12215202e-01 4.07458593e-01 4.04371663e-01 5.60927467e-01
 8.76714082e-01 4.65629985e-01 8.48580570e-01 6.27545454e-01
 1.07699444e-01 8.16209409e-01 9.96994563e-02 9.86506267e-02
 5.04046915e-05 1.89739025e-01 6.22968462e-01] 0.4846986977191026
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.440, MeanIU:  0.4847, Best_mIoU:  0.5019
[9.46103742e-01 6.82011153e-01 8.45448722e-01 1.97251796e-01
 4.12215202e-01 4.07458593e-01 4.04371663e-01 5.60927467e-01
 8.76714082e-01 4.65629985e-01 8.48580570e-01 6.27545454e-01
 1.07699444e-01 8.16209409e-01 9.96994563e-02 9.86506267e-02
 5.04046915e-05 1.89739025e-01 6.22968462e-01]
Epoch: [18/484] Iter:[0/24], Time: 0.97, lr: [0.009664658203336873], Loss: 0.376027
Epoch: [18/484] Iter:[10/24], Time: 0.44, lr: [0.00965688050181842], Loss: 0.455750
Epoch: [18/484] Iter:[20/24], Time: 0.42, lr: [0.009649102104215247], Loss: 0.457947
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94014573 0.67896536 0.82687472 0.18117931 0.23891099 0.39254314
 0.35963609 0.5493336  0.86540678 0.43905658 0.87806369 0.64864452
 0.18939089 0.85030929 0.12472762 0.20997438 0.00186497 0.09303065
 0.63376206] 0.479043178457792
1 [0.94232143 0.67917074 0.82568758 0.2001502  0.21846159 0.38711354
 0.39786566 0.53900802 0.86569786 0.44596045 0.87908565 0.64755094
 0.16876962 0.86707369 0.11928509 0.29907909 0.06220074 0.13053709
 0.63226635] 0.4898571207799696
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.470, MeanIU:  0.4899, Best_mIoU:  0.5019
[0.94232143 0.67917074 0.82568758 0.2001502  0.21846159 0.38711354
 0.39786566 0.53900802 0.86569786 0.44596045 0.87908565 0.64755094
 0.16876962 0.86707369 0.11928509 0.29907909 0.06220074 0.13053709
 0.63226635]
Epoch: [19/484] Iter:[0/24], Time: 0.96, lr: [0.00964599055011666], Loss: 0.356974
Epoch: [19/484] Iter:[10/24], Time: 0.44, lr: [0.009638211176842298], Loss: 0.413134
Epoch: [19/484] Iter:[20/24], Time: 0.42, lr: [0.009630431105834912], Loss: 0.406253
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95180118 0.69902605 0.86049716 0.25263124 0.38115223 0.40530304
 0.34453413 0.522365   0.88340178 0.4381491  0.87593492 0.67967239
 0.20348881 0.8645838  0.1848221  0.23356486 0.18265726 0.12332886
 0.65149712] 0.512547948092278
1 [0.95530021 0.7014471  0.86138667 0.3019609  0.40003234 0.40953918
 0.40458488 0.5273243  0.88641521 0.4591085  0.86904867 0.68522907
 0.25908871 0.87131398 0.18608855 0.15749982 0.24002087 0.19637348
 0.6578629 ] 0.5278750183586342
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.390, MeanIU:  0.5279, Best_mIoU:  0.5279
[0.95530021 0.7014471  0.86138667 0.3019609  0.40003234 0.40953918
 0.40458488 0.5273243  0.88641521 0.4591085  0.86904867 0.68522907
 0.25908871 0.87131398 0.18608855 0.15749982 0.24002087 0.19637348
 0.6578629 ]
Epoch: [20/484] Iter:[0/24], Time: 0.96, lr: [0.009627318881912417], Loss: 0.548987
Epoch: [20/484] Iter:[10/24], Time: 0.45, lr: [0.009619537832920992], Loss: 0.472142
Epoch: [20/484] Iter:[20/24], Time: 0.42, lr: [0.009611756084540764], Loss: 0.485471
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95110183 0.70382697 0.86170915 0.22489797 0.29840955 0.4126371
 0.4145993  0.55638227 0.87505825 0.42451199 0.88642241 0.66762838
 0.22034443 0.86772445 0.20886834 0.34442897 0.13854518 0.23031972
 0.63519386] 0.5222426387816285
1 [0.95021678 0.70448776 0.86237457 0.2086161  0.31371877 0.40102289
 0.44404588 0.56685029 0.87578778 0.38138895 0.88372198 0.67388681
 0.22426961 0.86782028 0.22856817 0.31076981 0.19549626 0.27595678
 0.63635615] 0.5265976639140078
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.417, MeanIU:  0.5266, Best_mIoU:  0.5279
[0.95021678 0.70448776 0.86237457 0.2086161  0.31371877 0.40102289
 0.44404588 0.56685029 0.87578778 0.38138895 0.88372198 0.67388681
 0.22426961 0.86782028 0.22856817 0.31076981 0.19549626 0.27595678
 0.63635615]
Epoch: [21/484] Iter:[0/24], Time: 0.99, lr: [0.00960864318920482], Loss: 0.405151
Epoch: [21/484] Iter:[10/24], Time: 0.44, lr: [0.009600860460517217], Loss: 0.426191
Epoch: [21/484] Iter:[20/24], Time: 0.42, lr: [0.009593077030777523], Loss: 0.433954
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94835713 0.67454228 0.85352075 0.2246596  0.34308147 0.42900183
 0.39418507 0.56689303 0.87970757 0.46875828 0.8871087  0.66599876
 0.16249697 0.8568496  0.14287714 0.22119594 0.02633379 0.16602207
 0.64597841] 0.503029914983156
1 [0.94447883 0.66060718 0.85586146 0.28159712 0.3734887  0.44131771
 0.41608332 0.56672929 0.88180377 0.47696924 0.89531361 0.67351975
 0.14520237 0.85507606 0.17885886 0.17211467 0.01125972 0.17132183
 0.64450659] 0.5076900050522998
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.411, MeanIU:  0.5077, Best_mIoU:  0.5279
[0.94447883 0.66060718 0.85586146 0.28159712 0.3734887  0.44131771
 0.41608332 0.56672929 0.88180377 0.47696924 0.89531361 0.67351975
 0.14520237 0.85507606 0.17885886 0.17211467 0.01125972 0.17132183
 0.64450659]
Epoch: [22/484] Iter:[0/24], Time: 0.94, lr: [0.009589963462431366], Loss: 0.418947
Epoch: [22/484] Iter:[10/24], Time: 0.44, lr: [0.009582179050050398], Loss: 0.405885
Epoch: [22/484] Iter:[20/24], Time: 0.42, lr: [0.00957439393494648], Loss: 0.515169
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.72930737 0.25496655 0.59747114 0.11698623 0.14749446 0.19308822
 0.03689976 0.23007993 0.75150572 0.05359578 0.27724547 0.32725634
 0.06076226 0.49226326 0.02767084 0.1664667  0.00237257 0.04413248
 0.46378966] 0.26175551241387485
1 [7.87499859e-01 3.66228315e-01 1.58705828e-01 6.97548864e-02
 1.01177671e-01 1.40709367e-01 2.92471345e-02 7.53626727e-02
 1.21833859e-01 3.76620372e-04 1.99357583e-01 5.61648427e-02
 1.84285847e-02 1.41532340e-01 2.47575621e-02 6.84569110e-04
 0.00000000e+00 1.74546180e-02 1.65604795e-01] 0.1302569004393099
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 3.657, MeanIU:  0.1303, Best_mIoU:  0.5279
[7.87499859e-01 3.66228315e-01 1.58705828e-01 6.97548864e-02
 1.01177671e-01 1.40709367e-01 2.92471345e-02 7.53626727e-02
 1.21833859e-01 3.76620372e-04 1.99357583e-01 5.61648427e-02
 1.84285847e-02 1.41532340e-01 2.47575621e-02 6.84569110e-04
 0.00000000e+00 1.74546180e-02 1.65604795e-01]
Epoch: [23/484] Iter:[0/24], Time: 0.96, lr: [0.009571279691986082], Loss: 0.694431
Epoch: [23/484] Iter:[10/24], Time: 0.44, lr: [0.009563493591896361], Loss: 0.669040
Epoch: [23/484] Iter:[20/24], Time: 0.42, lr: [0.009555706787405211], Loss: 0.652440
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9292083  0.6316614  0.84196772 0.17107356 0.34825242 0.39546484
 0.27926845 0.55996016 0.8666277  0.37135352 0.87315853 0.65354115
 0.28262352 0.81548306 0.05686346 0.22628826 0.17287792 0.10954316
 0.59708638] 0.48327913139207473
1 [0.92414874 0.64223981 0.83874775 0.19131514 0.31155948 0.38987386
 0.32050449 0.56469511 0.86020448 0.39884556 0.8752291  0.62669986
 0.31181204 0.82311024 0.01500218 0.34606692 0.19925507 0.08831909
 0.58585463] 0.4901833451263046
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.495, MeanIU:  0.4902, Best_mIoU:  0.5279
[0.92414874 0.64223981 0.83874775 0.19131514 0.31155948 0.38987386
 0.32050449 0.56469511 0.86020448 0.39884556 0.8752291  0.62669986
 0.31181204 0.82311024 0.01500218 0.34606692 0.19925507 0.08831909
 0.58585463]
Epoch: [24/484] Iter:[0/24], Time: 0.95, lr: [0.00955259186821923], Loss: 0.470831
Epoch: [24/484] Iter:[10/24], Time: 0.44, lr: [0.009544804076387048], Loss: 0.494284
Epoch: [24/484] Iter:[20/24], Time: 0.41, lr: [0.009537015578467286], Loss: 0.485266
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.90856495 0.57518991 0.84257138 0.17825491 0.29846332 0.40718061
 0.25246238 0.5361961  0.87646974 0.38197402 0.88945524 0.65771824
 0.22370415 0.85892269 0.04535561 0.31544049 0.07546182 0.13966005
 0.59355894] 0.4766633972692371
1 [0.89965682 0.55728651 0.831663   0.12045464 0.29245515 0.41671319
 0.32223304 0.53589963 0.8772622  0.36539048 0.88911444 0.66147366
 0.22080799 0.86507824 0.03656403 0.35880062 0.05835871 0.20921821
 0.59939512] 0.47988556139106725
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.532, MeanIU:  0.4799, Best_mIoU:  0.5279
[0.89965682 0.55728651 0.831663   0.12045464 0.29245515 0.41671319
 0.32223304 0.53589963 0.8772622  0.36539048 0.88911444 0.66147366
 0.22080799 0.86507824 0.03656403 0.35880062 0.05835871 0.20921821
 0.59939512]
Epoch: [25/484] Iter:[0/24], Time: 0.97, lr: [0.00953389998143702], Loss: 0.490537
Epoch: [25/484] Iter:[10/24], Time: 0.44, lr: [0.009526110493810221], Loss: 0.510493
Epoch: [25/484] Iter:[20/24], Time: 0.42, lr: [0.009518320298401968], Loss: 0.449294
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95336447 0.69605773 0.86681052 0.2424837  0.35485331 0.42375891
 0.38788781 0.54154511 0.8886603  0.46401239 0.90686795 0.67151454
 0.20035227 0.88799368 0.15290767 0.2827376  0.18942722 0.21547336
 0.64559671] 0.5248581701021405
1 [0.95230088 0.68097581 0.86499259 0.27592366 0.36185709 0.41752664
 0.43264092 0.53799322 0.88868156 0.46893293 0.90809194 0.67522973
 0.17313523 0.88744502 0.16931088 0.06969398 0.151804   0.27551146
 0.64253459] 0.5176095847895658
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.387, MeanIU:  0.5176, Best_mIoU:  0.5279
[0.95230088 0.68097581 0.86499259 0.27592366 0.36185709 0.41752664
 0.43264092 0.53799322 0.88868156 0.46893293 0.90809194 0.67522973
 0.17313523 0.88744502 0.16931088 0.06969398 0.151804   0.27551146
 0.64253459]
Epoch: [26/484] Iter:[0/24], Time: 0.96, lr: [0.0095152040219013], Loss: 0.288299
Epoch: [26/484] Iter:[10/24], Time: 0.44, lr: [0.009507412834409156], Loss: 0.455387
Epoch: [26/484] Iter:[20/24], Time: 0.42, lr: [0.009499620937433917], Loss: 0.464165
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94283258 0.67398539 0.80958371 0.17904767 0.35693137 0.43565899
 0.40352659 0.55452418 0.86787818 0.46707156 0.90775814 0.68206012
 0.34315401 0.87504415 0.15989309 0.20107936 0.15181697 0.26285826
 0.65471263] 0.5226008927278085
1 [0.94133832 0.68374063 0.78961076 0.07584527 0.36872548 0.43206245
 0.42465604 0.56131782 0.86532417 0.47823022 0.90467166 0.68143313
 0.38948745 0.85228342 0.16101821 0.22072794 0.10424111 0.28627462
 0.66129045] 0.5201199554340838
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.447, MeanIU:  0.5201, Best_mIoU:  0.5279
[0.94133832 0.68374063 0.78961076 0.07584527 0.36872548 0.43206245
 0.42465604 0.56131782 0.86532417 0.47823022 0.90467166 0.68143313
 0.38948745 0.85228342 0.16101821 0.22072794 0.10424111 0.28627462
 0.66129045]
Epoch: [27/484] Iter:[0/24], Time: 0.94, lr: [0.009496503979829263], Loss: 0.322142
Epoch: [27/484] Iter:[10/24], Time: 0.44, lr: [0.009488711088382356], Loss: 0.363559
Epoch: [27/484] Iter:[20/24], Time: 0.42, lr: [0.00948091748574288], Loss: 0.374813
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94722239 0.67636264 0.8696567  0.32727897 0.38027423 0.43479944
 0.42674047 0.58992878 0.88845622 0.4830889  0.89727746 0.68413039
 0.33258209 0.85467572 0.16742657 0.30898615 0.13234477 0.28885536
 0.65070011] 0.5442519653422472
1 [0.94726147 0.68308799 0.87161632 0.35970591 0.40403746 0.4544092
 0.46292846 0.5979683  0.88707279 0.49860553 0.90432616 0.68964366
 0.33968479 0.86045572 0.1451899  0.32609418 0.06379485 0.31542266
 0.65747959] 0.5509886819725593
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.378, MeanIU:  0.5510, Best_mIoU:  0.5510
[0.94726147 0.68308799 0.87161632 0.35970591 0.40403746 0.4544092
 0.46292846 0.5979683  0.88707279 0.49860553 0.90432616 0.68964366
 0.33968479 0.86045572 0.1451899  0.32609418 0.06379485 0.31542266
 0.65747959]
Epoch: [28/484] Iter:[0/24], Time: 0.94, lr: [0.009477799845393143], Loss: 0.305673
Epoch: [28/484] Iter:[10/24], Time: 0.44, lr: [0.00947000524588323], Loss: 0.374455
Epoch: [28/484] Iter:[20/24], Time: 0.42, lr: [0.009462209933463393], Loss: 0.390427
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94150527 0.67003771 0.85841164 0.22785587 0.36247997 0.44453043
 0.43673221 0.58660321 0.87960535 0.44750788 0.87094377 0.66779544
 0.2281354  0.87492221 0.14800531 0.4260803  0.12955851 0.18486499
 0.66114075] 0.528774536853685
1 [0.9374713  0.64976025 0.8553303  0.13993425 0.37374646 0.45554118
 0.48248384 0.59996816 0.88479465 0.45884943 0.87023584 0.67289063
 0.20577358 0.88926876 0.14853529 0.46036616 0.14368584 0.16315433
 0.64882405] 0.5284533843544345
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.408, MeanIU:  0.5285, Best_mIoU:  0.5510
[0.9374713  0.64976025 0.8553303  0.13993425 0.37374646 0.45554118
 0.48248384 0.59996816 0.88479465 0.45884943 0.87023584 0.67289063
 0.20577358 0.88926876 0.14853529 0.46036616 0.14368584 0.16315433
 0.64882405]
Epoch: [29/484] Iter:[0/24], Time: 0.97, lr: [0.009459091608719916], Loss: 0.447576
Epoch: [29/484] Iter:[10/24], Time: 0.44, lr: [0.0094512952970198], Loss: 0.371709
Epoch: [29/484] Iter:[20/24], Time: 0.42, lr: [0.009443498270684471], Loss: 0.364797
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95730491 0.73012618 0.8613653  0.27253109 0.38767868 0.46010759
 0.44145437 0.58971333 0.87083629 0.38821937 0.88711608 0.70024163
 0.30534846 0.88813418 0.19923155 0.46323146 0.12628396 0.33568125
 0.67028614] 0.5544679903247114
1 [0.9558928  0.73589    0.85818595 0.30970622 0.42401865 0.46665672
 0.48587738 0.60238625 0.86378419 0.35631357 0.88193995 0.71029295
 0.32624793 0.88776519 0.21280741 0.48565774 0.20311256 0.37032531
 0.68126933] 0.5693752684423675
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.369, MeanIU:  0.5694, Best_mIoU:  0.5694
[0.9558928  0.73589    0.85818595 0.30970622 0.42401865 0.46665672
 0.48587738 0.60238625 0.86378419 0.35631357 0.88193995 0.71029295
 0.32624793 0.88776519 0.21280741 0.48565774 0.20311256 0.37032531
 0.68126933]
Epoch: [30/484] Iter:[0/24], Time: 0.93, lr: [0.009440379259890974], Loss: 0.290462
Epoch: [30/484] Iter:[10/24], Time: 0.44, lr: [0.009432581231854382], Loss: 0.306337
Epoch: [30/484] Iter:[20/24], Time: 0.42, lr: [0.009424782487449301], Loss: 0.331320
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94670822 0.70017892 0.8499498  0.22530453 0.3809406  0.41589583
 0.40122822 0.58902787 0.88231723 0.25206947 0.89634968 0.70385014
 0.38075821 0.87894355 0.27089824 0.42596767 0.25358121 0.30263803
 0.66193892] 0.5483445436930272
1 [0.94527659 0.70436227 0.84650408 0.21236753 0.38780918 0.44251491
 0.47078876 0.59216342 0.87850464 0.20760241 0.89871086 0.71025407
 0.41378128 0.87907998 0.27940564 0.47218312 0.29483038 0.3346687
 0.66466657] 0.5597618104452075
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.382, MeanIU:  0.5598, Best_mIoU:  0.5694
[0.94527659 0.70436227 0.84650408 0.21236753 0.38780918 0.44251491
 0.47078876 0.59216342 0.87850464 0.20760241 0.89871086 0.71025407
 0.41378128 0.87907998 0.27940564 0.47218312 0.29483038 0.3346687
 0.66466657]
Epoch: [31/484] Iter:[0/24], Time: 0.95, lr: [0.009421662788941835], Loss: 0.264866
Epoch: [31/484] Iter:[10/24], Time: 0.44, lr: [0.00941386304040328], Loss: 0.497042
Epoch: [31/484] Iter:[20/24], Time: 0.42, lr: [0.009406062573754915], Loss: 0.479774
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94096746 0.66076451 0.83938514 0.15164965 0.35142443 0.45263558
 0.40129971 0.5667256  0.88578524 0.4539492  0.8671372  0.69533154
 0.38282624 0.8656104  0.18570357 0.28850924 0.2719335  0.18273919
 0.65970145] 0.5317936231504228
1 [0.941116   0.65317157 0.82607216 0.08634326 0.33484372 0.4550689
 0.45288032 0.56613054 0.88627967 0.45484738 0.8668572  0.68887836
 0.40638172 0.87864215 0.24039019 0.21656246 0.26594865 0.23082297
 0.66581731] 0.5324765538891427
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.423, MeanIU:  0.5325, Best_mIoU:  0.5694
[0.941116   0.65317157 0.82607216 0.08634326 0.33484372 0.4550689
 0.45288032 0.56613054 0.88627967 0.45484738 0.8668572  0.68887836
 0.40638172 0.87864215 0.24039019 0.21656246 0.26594865 0.23082297
 0.66581731]
Epoch: [32/484] Iter:[0/24], Time: 0.95, lr: [0.009402942185861811], Loss: 0.318678
Epoch: [32/484] Iter:[10/24], Time: 0.44, lr: [0.009395140712636463], Loss: 0.398172
Epoch: [32/484] Iter:[20/24], Time: 0.42, lr: [0.00938733851955189], Loss: 0.405313
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.90967318 0.55394307 0.86907839 0.1861907  0.36712095 0.42248906
 0.43293809 0.60367009 0.88196651 0.40669175 0.91077836 0.68867348
 0.37535991 0.8855354  0.30960445 0.45268596 0.307591   0.40146122
 0.66714122] 0.5596101460075835
1 [0.89532341 0.52325088 0.86860665 0.20021092 0.355556   0.43997113
 0.48127272 0.60795045 0.88586614 0.38728433 0.91212168 0.68809713
 0.37605271 0.890848   0.31984052 0.4929334  0.30420297 0.38123095
 0.66147592] 0.5616892587617839
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.439, MeanIU:  0.5617, Best_mIoU:  0.5694
[0.89532341 0.52325088 0.86860665 0.20021092 0.355556   0.43997113
 0.48127272 0.60795045 0.88586614 0.38728433 0.91212168 0.68809713
 0.37605271 0.890848   0.31984052 0.4929334  0.30420297 0.38123095
 0.66147592]
Epoch: [33/484] Iter:[0/24], Time: 0.96, lr: [0.009384217440593706], Loss: 0.343870
Epoch: [33/484] Iter:[10/24], Time: 0.44, lr: [0.009376414238477262], Loss: 0.447262
Epoch: [33/484] Iter:[20/24], Time: 0.42, lr: [0.009368610314744017], Loss: 0.400071
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94797844 0.67028195 0.86871358 0.2015198  0.31878133 0.40389923
 0.46430759 0.58621059 0.8873688  0.43751712 0.89836747 0.70043843
 0.33474498 0.86490735 0.19403563 0.29460895 0.30290687 0.27889079
 0.6640053 ] 0.5431307468866615
1 [0.94589524 0.67214203 0.87065088 0.19184153 0.33000854 0.39266266
 0.4963932  0.58988634 0.89046694 0.44833551 0.90144391 0.7008962
 0.38031625 0.87051719 0.257465   0.32599707 0.37305473 0.31499874
 0.66725217] 0.5589591649367089
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.393, MeanIU:  0.5590, Best_mIoU:  0.5694
[0.94589524 0.67214203 0.87065088 0.19184153 0.33000854 0.39266266
 0.4963932  0.58988634 0.89046694 0.44833551 0.90144391 0.7008962
 0.38031625 0.87051719 0.257465   0.32599707 0.37305473 0.31499874
 0.66725217]
Epoch: [34/484] Iter:[0/24], Time: 0.96, lr: [0.009365488543033481], Loss: 0.254558
Epoch: [34/484] Iter:[10/24], Time: 0.44, lr: [0.00935768360780203], Loss: 0.313455
Epoch: [34/484] Iter:[20/24], Time: 0.41, lr: [0.00934987794918799], Loss: 0.352787
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95689124 0.73393735 0.85846396 0.26295497 0.37194999 0.47111411
 0.46021358 0.61502631 0.89406545 0.47058118 0.77795681 0.68215125
 0.33608597 0.86490353 0.24652787 0.36851398 0.3398773  0.30337808
 0.67661049] 0.5626949170776562
1 [0.95678577 0.74376336 0.85685681 0.27547233 0.37194974 0.48290461
 0.48731224 0.62452954 0.89274111 0.48725854 0.77200448 0.67211796
 0.34073911 0.86323725 0.28160629 0.37015147 0.39798992 0.32993009
 0.68435225] 0.5732475195800951
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.370, MeanIU:  0.5732, Best_mIoU:  0.5732
[0.95678577 0.74376336 0.85685681 0.27547233 0.37194974 0.48290461
 0.48731224 0.62452954 0.89274111 0.48725854 0.77200448 0.67211796
 0.34073911 0.86323725 0.28160629 0.37015147 0.39798992 0.32993009
 0.68435225]
Epoch: [35/484] Iter:[0/24], Time: 0.92, lr: [0.009346755483029948], Loss: 0.314047
Epoch: [35/484] Iter:[10/24], Time: 0.44, lr: [0.009338948810439828], Loss: 0.309434
Epoch: [35/484] Iter:[20/24], Time: 0.41, lr: [0.00933114141269307], Loss: 0.338178
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95543686 0.72140128 0.8567655  0.27413941 0.40286735 0.4627567
 0.43560235 0.60791672 0.87620809 0.46315595 0.88270374 0.69964806
 0.35288462 0.88881083 0.28112861 0.37745045 0.23629622 0.20436669
 0.64376232] 0.5591211447877985
1 [0.95434867 0.72280217 0.85855388 0.29707433 0.42507639 0.4810159
 0.47287734 0.61763521 0.8807756  0.46472064 0.87706872 0.68272591
 0.30496272 0.8915116  0.31977177 0.38138474 0.32269262 0.21885287
 0.63132149] 0.5686932924394864
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.364, MeanIU:  0.5687, Best_mIoU:  0.5732
[0.95434867 0.72280217 0.85855388 0.29707433 0.42507639 0.4810159
 0.47287734 0.61763521 0.8807756  0.46472064 0.87706872 0.68272591
 0.30496272 0.8915116  0.31977177 0.38138474 0.32269262 0.21885287
 0.63132149]
Epoch: [36/484] Iter:[0/24], Time: 0.94, lr: [0.00932801825038443], Loss: 0.375825
Epoch: [36/484] Iter:[10/24], Time: 0.44, lr: [0.009320209836172101], Loss: 0.430239
Epoch: [36/484] Iter:[20/24], Time: 0.41, lr: [0.00931240069502076], Loss: 0.415502
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95652463 0.69186389 0.87193107 0.27608611 0.40653322 0.46968428
 0.44464549 0.61458873 0.88378895 0.42242069 0.90942907 0.70797956
 0.34606151 0.89439728 0.29933559 0.49943458 0.24269365 0.35654353
 0.67447568] 0.5772851334230373
1 [0.95468785 0.68114916 0.8745563  0.32381838 0.41984395 0.48706283
 0.49229227 0.600191   0.88847152 0.43975165 0.89871075 0.71384067
 0.41212486 0.89679216 0.30429202 0.50700952 0.15749071 0.38283503
 0.67561155] 0.5847648506949671
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.353, MeanIU:  0.5848, Best_mIoU:  0.5848
[0.95468785 0.68114916 0.8745563  0.32381838 0.41984395 0.48706283
 0.49229227 0.600191   0.88847152 0.43975165 0.89871075 0.71384067
 0.41212486 0.89679216 0.30429202 0.50700952 0.15749071 0.38283503
 0.67561155]
Epoch: [37/484] Iter:[0/24], Time: 0.94, lr: [0.009309276834850442], Loss: 0.218964
Epoch: [37/484] Iter:[10/24], Time: 0.44, lr: [0.009301466674732346], Loss: 0.312552
Epoch: [37/484] Iter:[20/24], Time: 0.42, lr: [0.009293655785884479], Loss: 0.345310
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95770579 0.72310512 0.86747897 0.274527   0.41742681 0.46978206
 0.48692079 0.62467605 0.88746858 0.46613994 0.90898402 0.71639772
 0.3999522  0.89059366 0.27562113 0.28235432 0.21632437 0.34353709
 0.68015879] 0.5731133897858408
1 [0.95822309 0.71800143 0.86258371 0.21491772 0.40807839 0.4617879
 0.50649006 0.62851513 0.88831213 0.45772617 0.91091004 0.72357132
 0.43458818 0.89454265 0.31555023 0.18619262 0.27127836 0.36358533
 0.68276981] 0.5730328573947351
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.351, MeanIU:  0.5730, Best_mIoU:  0.5848
[0.95822309 0.71800143 0.86258371 0.21491772 0.40807839 0.4617879
 0.50649006 0.62851513 0.88831213 0.45772617 0.91091004 0.72357132
 0.43458818 0.89454265 0.31555023 0.18619262 0.27127836 0.36358533
 0.68276981]
Epoch: [38/484] Iter:[0/24], Time: 0.94, lr: [0.009290531226133351], Loss: 0.575906
Epoch: [38/484] Iter:[10/24], Time: 0.44, lr: [0.00928271931580577], Loss: 0.339311
Epoch: [38/484] Iter:[20/24], Time: 0.42, lr: [0.00927490667494922], Loss: 0.326192
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96213049 0.73321166 0.87543869 0.32571698 0.42451265 0.47026523
 0.4520637  0.62084408 0.89278226 0.46956218 0.911558   0.71593104
 0.37176739 0.89899673 0.30460996 0.39456593 0.27933998 0.31360933
 0.67637024] 0.5838566603633817
1 [0.96478951 0.73810087 0.87943472 0.34599043 0.4524887  0.49239295
 0.49430655 0.63258931 0.89452067 0.46689709 0.9129344  0.72397583
 0.39494754 0.90351941 0.26581993 0.32137999 0.25088749 0.3516494
 0.68057887] 0.5877475605154996
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.319, MeanIU:  0.5877, Best_mIoU:  0.5877
[0.96478951 0.73810087 0.87943472 0.34599043 0.4524887  0.49239295
 0.49430655 0.63258931 0.89452067 0.46689709 0.9129344  0.72397583
 0.39494754 0.90351941 0.26581993 0.32137999 0.25088749 0.3516494
 0.68057887]
Epoch: [39/484] Iter:[0/24], Time: 0.94, lr: [0.009271781413890052], Loss: 0.172283
Epoch: [39/484] Iter:[10/24], Time: 0.44, lr: [0.009263967749028969], Loss: 0.290152
Epoch: [39/484] Iter:[20/24], Time: 0.42, lr: [0.009256153351831218], Loss: 0.288890
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95014313 0.67177122 0.88026203 0.31537036 0.4571241  0.48768973
 0.49094899 0.62254088 0.89468049 0.48448497 0.91545793 0.72709688
 0.43109286 0.89333608 0.304219   0.4143418  0.37562432 0.40862746
 0.69017164] 0.6007886236938119
1 [0.95174254 0.68107278 0.88351287 0.3513441  0.47852106 0.49350638
 0.52748489 0.6337591  0.89897419 0.50444901 0.91651626 0.73329281
 0.45644336 0.90236756 0.34021641 0.37142236 0.40020682 0.44029688
 0.6964991 ] 0.6137699204247384
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.327, MeanIU:  0.6138, Best_mIoU:  0.6138
[0.95174254 0.68107278 0.88351287 0.3513441  0.47852106 0.49350638
 0.52748489 0.6337591  0.89897419 0.50444901 0.91651626 0.73329281
 0.45644336 0.90236756 0.34021641 0.37142236 0.40020682 0.44029688
 0.6964991 ]
Epoch: [40/484] Iter:[0/24], Time: 0.94, lr: [0.009253027387728619], Loss: 0.267756
Epoch: [40/484] Iter:[10/24], Time: 0.44, lr: [0.009245211963989575], Loss: 0.320522
Epoch: [40/484] Iter:[20/24], Time: 0.42, lr: [0.009237395806097611], Loss: 0.319277
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96120066 0.74082028 0.84977987 0.33175466 0.32730891 0.45849893
 0.38675852 0.60846572 0.88410158 0.47190074 0.8075964  0.71896228
 0.39342415 0.89576108 0.20425524 0.4785579  0.28983149 0.25752839
 0.68632664] 0.5659386015504168
1 [0.95914521 0.73744434 0.83816138 0.3425372  0.34160919 0.45494139
 0.41440179 0.61443808 0.8794869  0.49856191 0.79378805 0.72415359
 0.4124591  0.90226106 0.21915517 0.49337614 0.41837105 0.27873388
 0.69400952] 0.5798439439845143
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.365, MeanIU:  0.5798, Best_mIoU:  0.6138
[0.95914521 0.73744434 0.83816138 0.3425372  0.34160919 0.45494139
 0.41440179 0.61443808 0.8794869  0.49856191 0.79378805 0.72415359
 0.4124591  0.90226106 0.21915517 0.49337614 0.41837105 0.27873388
 0.69400952]
Epoch: [41/484] Iter:[0/24], Time: 0.96, lr: [0.00923426913720797], Loss: 0.290366
Epoch: [41/484] Iter:[10/24], Time: 0.44, lr: [0.009226451950225931], Loss: 0.321390
Epoch: [41/484] Iter:[20/24], Time: 0.42, lr: [0.00921863402726609], Loss: 0.328925
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94512373 0.67402296 0.87114722 0.30306176 0.41543516 0.46373479
 0.47017291 0.62805004 0.89307074 0.46077564 0.90255647 0.720665
 0.41715949 0.89808118 0.2503928  0.16697892 0.1981259  0.44280415
 0.68545879] 0.5687798769604632
1 [0.94234002 0.66495894 0.87337735 0.30308301 0.45698119 0.49175992
 0.51837339 0.64011956 0.89631629 0.44634422 0.905196   0.72931475
 0.43709274 0.91089931 0.25754234 0.10986007 0.22961999 0.44885651
 0.69101583] 0.5764763913748374
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.357, MeanIU:  0.5765, Best_mIoU:  0.6138
[0.94234002 0.66495894 0.87337735 0.30308301 0.45698119 0.49175992
 0.51837339 0.64011956 0.89631629 0.44634422 0.905196   0.72931475
 0.43709274 0.91089931 0.25754234 0.10986007 0.22961999 0.44885651
 0.69101583]
Epoch: [42/484] Iter:[0/24], Time: 0.96, lr: [0.009215506651837527], Loss: 0.299472
Epoch: [42/484] Iter:[10/24], Time: 0.44, lr: [0.00920768769722672], Loss: 0.328027
Epoch: [42/484] Iter:[20/24], Time: 0.41, lr: [0.00919986800480456], Loss: 0.366956
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95795379 0.7416614  0.85022848 0.25233019 0.42339184 0.47638801
 0.48726216 0.6281794  0.89323485 0.51947567 0.91308259 0.72328681
 0.3895143  0.88581562 0.27469339 0.27741811 0.09254018 0.39217416
 0.68237984] 0.5716321471406927
1 [0.95865364 0.7448649  0.818524   0.22863629 0.44310323 0.48502222
 0.50454141 0.63195914 0.89320095 0.5341488  0.91735507 0.72934698
 0.39845505 0.8932265  0.28142771 0.19878327 0.05358643 0.38840456
 0.6886764 ] 0.5679956074995659
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.386, MeanIU:  0.5680, Best_mIoU:  0.6138
[0.95865364 0.7448649  0.818524   0.22863629 0.44310323 0.48502222
 0.50454141 0.63195914 0.89320095 0.5341488  0.91735507 0.72934698
 0.39845505 0.8932265  0.28142771 0.19878327 0.05358643 0.38840456
 0.6886764 ]
Epoch: [43/484] Iter:[0/24], Time: 0.97, lr: [0.009196739921076858], Loss: 0.238075
Epoch: [43/484] Iter:[10/24], Time: 0.44, lr: [0.009188919194430645], Loss: 0.321509
Epoch: [43/484] Iter:[20/24], Time: 0.41, lr: [0.00918109772813078], Loss: 0.310349
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96462871 0.75611195 0.87957237 0.34060469 0.42391686 0.47928926
 0.48154599 0.6289086  0.89129002 0.50466885 0.91872622 0.72374303
 0.41882927 0.88856149 0.28896095 0.39243542 0.14792964 0.40480019
 0.67945795] 0.5902095500163967
1 [0.96487652 0.7590629  0.87836558 0.34897957 0.42121174 0.49446885
 0.50929053 0.63876714 0.88833507 0.5089991  0.91939437 0.72902525
 0.44785332 0.89862017 0.32947906 0.41133282 0.01503564 0.41821741
 0.67718324] 0.5925525407844258
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.314, MeanIU:  0.5926, Best_mIoU:  0.6138
[0.96487652 0.7590629  0.87836558 0.34897957 0.42121174 0.49446885
 0.50929053 0.63876714 0.88833507 0.5089991  0.91939437 0.72902525
 0.44785332 0.89862017 0.32947906 0.41133282 0.01503564 0.41821741
 0.67718324]
Epoch: [44/484] Iter:[0/24], Time: 0.97, lr: [0.009177968934335336], Loss: 0.229046
Epoch: [44/484] Iter:[10/24], Time: 0.44, lr: [0.009170146431226055], Loss: 0.353779
Epoch: [44/484] Iter:[20/24], Time: 0.42, lr: [0.009162323186612028], Loss: 0.323830
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95977655 0.74119089 0.873175   0.3054117  0.39473767 0.48086048
 0.49172909 0.62984027 0.89232121 0.48968515 0.91649075 0.70881676
 0.39936074 0.88388464 0.27622785 0.3737753  0.20124345 0.24718239
 0.68198983] 0.5761947225502367
1 [0.95745198 0.75211886 0.87003137 0.32217943 0.40013513 0.4958131
 0.52086447 0.63697507 0.89388163 0.50050222 0.91715179 0.70977841
 0.42274107 0.88191702 0.33379228 0.38905196 0.2896479  0.24196791
 0.69036363] 0.5908613270043891
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.335, MeanIU:  0.5909, Best_mIoU:  0.6138
[0.95745198 0.75211886 0.87003137 0.32217943 0.40013513 0.4958131
 0.52086447 0.63697507 0.89388163 0.50050222 0.91715179 0.70977841
 0.42274107 0.88191702 0.33379228 0.38905196 0.2896479  0.24196791
 0.69036363]
Epoch: [45/484] Iter:[0/24], Time: 0.95, lr: [0.009159193680971784], Loss: 0.348294
Epoch: [45/484] Iter:[10/24], Time: 0.44, lr: [0.009151369396950606], Loss: 0.315204
Epoch: [45/484] Iter:[20/24], Time: 0.42, lr: [0.009143544369564717], Loss: 0.313817
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9610929  0.73790308 0.86619063 0.26425213 0.28549765 0.48882354
 0.47501314 0.63003905 0.88810195 0.43447192 0.87168956 0.7089048
 0.39399685 0.8911355  0.1799044  0.43271324 0.10125397 0.39148195
 0.68240674] 0.5623617370669082
1 [0.95910397 0.72664415 0.85972858 0.27012077 0.2325267  0.50498776
 0.50806516 0.64315596 0.88594375 0.4051529  0.86212939 0.71777572
 0.42503743 0.88661156 0.1794709  0.40358241 0.06984355 0.41681271
 0.68603553] 0.5601436271858985
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.362, MeanIU:  0.5601, Best_mIoU:  0.6138
[0.95910397 0.72664415 0.85972858 0.27012077 0.2325267  0.50498776
 0.50806516 0.64315596 0.88594375 0.4051529  0.86212939 0.71777572
 0.42503743 0.88661156 0.1794709  0.40358241 0.06984355 0.41681271
 0.68603553]
Epoch: [46/484] Iter:[0/24], Time: 0.97, lr: [0.00914041415029411], Loss: 0.291669
Epoch: [46/484] Iter:[10/24], Time: 0.44, lr: [0.009132588080890887], Loss: 0.313328
Epoch: [46/484] Iter:[20/24], Time: 0.42, lr: [0.009124761266254064], Loss: 0.378063
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.86712345 0.59280473 0.73414803 0.17696931 0.33253211 0.41094444
 0.23339053 0.5682651  0.81851062 0.26645151 0.87417616 0.64887278
 0.20916529 0.82345951 0.1836663  0.38960784 0.16090858 0.10281951
 0.53842857] 0.47011812470345116
1 [0.77862061 0.58421303 0.65270345 0.19280002 0.31371399 0.41005574
 0.26343208 0.57651925 0.8159896  0.28114278 0.85569326 0.64424128
 0.2344919  0.75120443 0.26042524 0.37254711 0.20946666 0.10549123
 0.52660964] 0.4647032260314676
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.736, MeanIU:  0.4647, Best_mIoU:  0.6138
[0.77862061 0.58421303 0.65270345 0.19280002 0.31371399 0.41005574
 0.26343208 0.57651925 0.8159896  0.28114278 0.85569326 0.64424128
 0.2344919  0.75120443 0.26042524 0.37254711 0.20946666 0.10549123
 0.52660964]
Epoch: [47/484] Iter:[0/24], Time: 0.99, lr: [0.009121630331558956], Loss: 0.660730
Epoch: [47/484] Iter:[10/24], Time: 0.44, lr: [0.009113802472282072], Loss: 0.396290
Epoch: [47/484] Iter:[20/24], Time: 0.42, lr: [0.009105973865893706], Loss: 0.403559
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95741251 0.72604059 0.85151414 0.21203396 0.32141544 0.43097341
 0.42992958 0.58363559 0.8815981  0.39422372 0.88921757 0.70403532
 0.41556687 0.85606972 0.24431126 0.40989707 0.20254373 0.37324278
 0.6565996 ] 0.5547505772618381
1 [0.95963976 0.72847021 0.84950247 0.20507618 0.3139581  0.44726093
 0.45130212 0.56172507 0.8823321  0.37097459 0.88986398 0.71222188
 0.43887891 0.87494216 0.29591236 0.42603477 0.21424007 0.3649173
 0.6453478 ] 0.5596105658484073
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.388, MeanIU:  0.5596, Best_mIoU:  0.6138
[0.95963976 0.72847021 0.84950247 0.20507618 0.3139581  0.44726093
 0.45130212 0.56172507 0.8823321  0.37097459 0.88986398 0.71222188
 0.43887891 0.87494216 0.29591236 0.42603477 0.21424007 0.3649173
 0.6453478 ]
Epoch: [48/484] Iter:[0/24], Time: 0.95, lr: [0.009102842213971334], Loss: 0.222960
Epoch: [48/484] Iter:[10/24], Time: 0.44, lr: [0.009095012560307548], Loss: 0.357757
Epoch: [48/484] Iter:[20/24], Time: 0.42, lr: [0.009087182157645346], Loss: 0.357000
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.93751339 0.67298424 0.84051897 0.19671202 0.36252809 0.45778434
 0.34344266 0.56146752 0.81116505 0.27279926 0.88660601 0.71548287
 0.40556272 0.63802754 0.19258428 0.25818663 0.26806264 0.20247753
 0.6542908 ] 0.5093787668243914
1 [0.93946327 0.68606706 0.828481   0.16618237 0.40087565 0.46485274
 0.33827598 0.53324805 0.80778268 0.277437   0.88123655 0.72539562
 0.43850123 0.59946582 0.14266879 0.24020855 0.2863303  0.19912138
 0.65895967] 0.5060291415564425
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.536, MeanIU:  0.5060, Best_mIoU:  0.6138
[0.93946327 0.68606706 0.828481   0.16618237 0.40087565 0.46485274
 0.33827598 0.53324805 0.80778268 0.277437   0.88123655 0.72539562
 0.43850123 0.59946582 0.14266879 0.24020855 0.2863303  0.19912138
 0.65895967]
Epoch: [49/484] Iter:[0/24], Time: 0.96, lr: [0.00908404978668425], Loss: 0.367119
Epoch: [49/484] Iter:[10/24], Time: 0.44, lr: [0.009076218334098544], Loss: 0.345381
Epoch: [49/484] Iter:[20/24], Time: 0.42, lr: [0.009068386130618375], Loss: 0.332329
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9608459  0.70811607 0.8783852  0.24757695 0.42508876 0.48674805
 0.47579607 0.62225884 0.88872249 0.46843708 0.90630757 0.72486471
 0.45233009 0.89230378 0.3024015  0.46355889 0.27752929 0.33939411
 0.68411002] 0.5897250187908447
1 [0.95856297 0.70255758 0.87523722 0.24762939 0.45428034 0.49604373
 0.51423504 0.61736285 0.88956391 0.49935801 0.90330874 0.72486069
 0.47178343 0.88317181 0.37051003 0.51801132 0.33266913 0.38835242
 0.67851501] 0.6066322957935161
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.334, MeanIU:  0.6066, Best_mIoU:  0.6138
[0.95856297 0.70255758 0.87523722 0.24762939 0.45428034 0.49604373
 0.51423504 0.61736285 0.88956391 0.49935801 0.90330874 0.72486069
 0.47178343 0.88317181 0.37051003 0.51801132 0.33266913 0.38835242
 0.67851501]
Epoch: [50/484] Iter:[0/24], Time: 0.95, lr: [0.00906525303879834], Loss: 0.331641
Epoch: [50/484] Iter:[10/24], Time: 0.44, lr: [0.009057419782733774], Loss: 0.384582
Epoch: [50/484] Iter:[20/24], Time: 0.42, lr: [0.009049585773869502], Loss: 0.355986
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95327663 0.69139492 0.86391827 0.22420183 0.36502359 0.47190815
 0.42779595 0.61159879 0.88295946 0.42875445 0.88714806 0.70326334
 0.40717756 0.84307702 0.20969772 0.15580055 0.23780531 0.26233994
 0.66702012] 0.5417979827044005
1 [0.95085863 0.66601311 0.86621747 0.20180954 0.35318076 0.48241279
 0.47792176 0.63406799 0.88461242 0.40092611 0.8898718  0.71800103
 0.44185801 0.84385799 0.21582105 0.18852663 0.33824846 0.2819753
 0.66745146] 0.5528227525943358
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.396, MeanIU:  0.5528, Best_mIoU:  0.6138
[0.95085863 0.66601311 0.86621747 0.20180954 0.35318076 0.48241279
 0.47792176 0.63406799 0.88461242 0.40092611 0.8898718  0.71800103
 0.44185801 0.84385799 0.21582105 0.18852663 0.33824846 0.2819753
 0.66745146]
Epoch: [51/484] Iter:[0/24], Time: 0.97, lr: [0.009046451959361499], Loss: 0.313368
Epoch: [51/484] Iter:[10/24], Time: 0.44, lr: [0.009038616895239035], Loss: 0.365183
Epoch: [51/484] Iter:[20/24], Time: 0.42, lr: [0.009030781076402377], Loss: 0.336781
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95971018 0.7367653  0.87916597 0.19725729 0.42309939 0.49530493
 0.50609716 0.63635947 0.8943446  0.50280305 0.91160653 0.72623778
 0.45458374 0.89579124 0.31543408 0.54861937 0.25772967 0.37979206
 0.69279966] 0.6007106036023153
1 [0.96039763 0.74087996 0.88352079 0.1662499  0.44660211 0.50877831
 0.53566666 0.65133957 0.89585408 0.51895663 0.91367667 0.7326793
 0.47347057 0.89974394 0.40593232 0.58332049 0.34111284 0.39377419
 0.69501387] 0.6182615692624838
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.318, MeanIU:  0.6183, Best_mIoU:  0.6183
[0.96039763 0.74087996 0.88352079 0.1662499  0.44660211 0.50877831
 0.53566666 0.65133957 0.89585408 0.51895663 0.91367667 0.7326793
 0.47347057 0.89974394 0.40593232 0.58332049 0.34111284 0.39377419
 0.69501387]
Epoch: [52/484] Iter:[0/24], Time: 0.93, lr: [0.009027646537368495], Loss: 0.194014
Epoch: [52/484] Iter:[10/24], Time: 0.44, lr: [0.009019809660586853], Loss: 0.290901
Epoch: [52/484] Iter:[20/24], Time: 0.42, lr: [0.009011972027167211], Loss: 0.279915
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95772372 0.73841558 0.87307767 0.32363706 0.40083785 0.49681409
 0.51913807 0.64409598 0.89501403 0.51652446 0.91277537 0.73182987
 0.43983393 0.89733327 0.3089762  0.48749521 0.28626279 0.36724783
 0.6907236 ] 0.6046187663742462
1 [0.95751806 0.74290181 0.8704325  0.34934357 0.40166724 0.50603638
 0.54767099 0.65560066 0.89801761 0.5351909  0.91242525 0.7415998
 0.46951935 0.90124683 0.32494879 0.50852735 0.41210241 0.42235539
 0.68777389] 0.6234146720782908
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.323, MeanIU:  0.6234, Best_mIoU:  0.6234
[0.95751806 0.74290181 0.8704325  0.34934357 0.40166724 0.50603638
 0.54767099 0.65560066 0.89801761 0.5351909  0.91242525 0.7415998
 0.46951935 0.90124683 0.32494879 0.50852735 0.41210241 0.42235539
 0.68777389]
Epoch: [53/484] Iter:[0/24], Time: 0.94, lr: [0.009008836761760597], Loss: 0.193056
Epoch: [53/484] Iter:[10/24], Time: 0.44, lr: [0.009000998067696084], Loss: 0.277256
Epoch: [53/484] Iter:[20/24], Time: 0.41, lr: [0.008993158615060387], Loss: 0.287860
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95741489 0.73401828 0.87393189 0.21018731 0.33746677 0.50146488
 0.51496768 0.64539568 0.89827881 0.51757029 0.91615491 0.73538371
 0.44922523 0.90748017 0.31035542 0.58406729 0.20588073 0.41074083
 0.70493474] 0.6007852370512498
1 [0.95811407 0.74024546 0.87572064 0.20721252 0.36165788 0.51851083
 0.55106814 0.66115533 0.89967893 0.54062635 0.91842356 0.7458804
 0.48478466 0.90794884 0.31470755 0.59546782 0.34852662 0.47333772
 0.71401044] 0.6219514614768739
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.317, MeanIU:  0.6220, Best_mIoU:  0.6234
[0.95811407 0.74024546 0.87572064 0.20721252 0.36165788 0.51851083
 0.55106814 0.66115533 0.89967893 0.54062635 0.91842356 0.7458804
 0.48478466 0.90794884 0.31470755 0.59546782 0.34852662 0.47333772
 0.71401044]
Epoch: [54/484] Iter:[0/24], Time: 0.97, lr: [0.008990022621425177], Loss: 0.160257
Epoch: [54/484] Iter:[10/24], Time: 0.44, lr: [0.00898218210543154], Loss: 0.291720
Epoch: [54/484] Iter:[20/24], Time: 0.42, lr: [0.00897434082892408], Loss: 0.274434
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96452169 0.75684195 0.88958036 0.35880449 0.45014524 0.51007816
 0.51545182 0.66297158 0.89797331 0.49411576 0.90628485 0.73721999
 0.40903109 0.90015482 0.33187154 0.46751026 0.40309383 0.38857745
 0.69892293] 0.6180605859964358
1 [0.96568674 0.76468445 0.89091756 0.38461306 0.47561829 0.52545567
 0.55408324 0.66636683 0.89911471 0.516618   0.90430386 0.74710926
 0.45252939 0.90148337 0.39414755 0.49039082 0.44723242 0.45272233
 0.69865533] 0.6385122570333056
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.295, MeanIU:  0.6385, Best_mIoU:  0.6385
[0.96568674 0.76468445 0.89091756 0.38461306 0.47561829 0.52545567
 0.55408324 0.66636683 0.89911471 0.516618   0.90430386 0.74710926
 0.45252939 0.90148337 0.39414755 0.49039082 0.44723242 0.45272233
 0.69865533]
Epoch: [55/484] Iter:[0/24], Time: 0.96, lr: [0.008971204105195337], Loss: 0.422115
Epoch: [55/484] Iter:[10/24], Time: 0.44, lr: [0.00896336176260359], Loss: 0.283038
Epoch: [55/484] Iter:[20/24], Time: 0.42, lr: [0.008955518657545855], Loss: 0.288032
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96358222 0.74134465 0.88696776 0.37964816 0.45396596 0.49488216
 0.51079772 0.65837983 0.8977822  0.46625061 0.91854526 0.7412826
 0.45181761 0.89279092 0.32842602 0.32646275 0.36596906 0.24941482
 0.68952648] 0.6009387787579692
1 [0.96335239 0.74206884 0.89030705 0.38484024 0.46016182 0.52055624
 0.54528776 0.66245191 0.89933961 0.45978399 0.91853255 0.7496362
 0.48270168 0.89720315 0.39130341 0.39325237 0.49092359 0.27777664
 0.69790159] 0.6224937383461718
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.309, MeanIU:  0.6225, Best_mIoU:  0.6385
[0.96335239 0.74206884 0.89030705 0.38484024 0.46016182 0.52055624
 0.54528776 0.66245191 0.89933961 0.45978399 0.91853255 0.7496362
 0.48270168 0.89720315 0.39130341 0.39325237 0.49092359 0.27777664
 0.69790159]
Epoch: [56/484] Iter:[0/24], Time: 0.96, lr: [0.008952381201849508], Loss: 0.248704
Epoch: [56/484] Iter:[10/24], Time: 0.44, lr: [0.008944537027967762], Loss: 0.286008
Epoch: [56/484] Iter:[20/24], Time: 0.42, lr: [0.008936692089658286], Loss: 0.290249
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96319241 0.7448121  0.88536731 0.29497285 0.41657665 0.50315185
 0.53597257 0.63729482 0.89761382 0.45942414 0.91740583 0.71408179
 0.37902487 0.89902761 0.31152219 0.53501652 0.3731771  0.40540482
 0.69972949] 0.6090930917097539
1 [0.96437935 0.75075022 0.88878179 0.25575393 0.43180514 0.51539488
 0.55555541 0.64559044 0.89633553 0.40394966 0.91874713 0.72806058
 0.42273061 0.90402927 0.37286921 0.60858285 0.46191371 0.44105863
 0.70337481] 0.6247191124132635
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.306, MeanIU:  0.6247, Best_mIoU:  0.6385
[0.96437935 0.75075022 0.88878179 0.25575393 0.43180514 0.51539488
 0.55555541 0.64559044 0.89633553 0.40394966 0.91874713 0.72806058
 0.42273061 0.90402927 0.37286921 0.60858285 0.46191371 0.44105863
 0.70337481]
Epoch: [57/484] Iter:[0/24], Time: 0.97, lr: [0.008933553900111052], Loss: 0.241965
Epoch: [57/484] Iter:[10/24], Time: 0.44, lr: [0.008925707890224366], Loss: 0.289600
Epoch: [57/484] Iter:[20/24], Time: 0.42, lr: [0.008917861113938542], Loss: 0.292583
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95649226 0.7241851  0.86813632 0.27069293 0.28437009 0.48679251
 0.48559984 0.64860267 0.88740893 0.49301103 0.86751036 0.71970258
 0.44052802 0.88606424 0.29736903 0.43293821 0.25022618 0.36965608
 0.68630444] 0.5818732016830742
1 [0.95826014 0.73263929 0.86398675 0.26702655 0.24764498 0.5017715
 0.52021256 0.66205728 0.88730384 0.49954758 0.85707451 0.72495709
 0.46247174 0.89535173 0.33306082 0.41388309 0.26830442 0.40192799
 0.68487434] 0.5885450620410106
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.357, MeanIU:  0.5885, Best_mIoU:  0.6385
[0.95826014 0.73263929 0.86398675 0.26702655 0.24764498 0.5017715
 0.52021256 0.66205728 0.88730384 0.49954758 0.85707451 0.72495709
 0.46247174 0.89535173 0.33306082 0.41388309 0.26830442 0.40192799
 0.68487434]
Epoch: [58/484] Iter:[0/24], Time: 0.95, lr: [0.008914722188647868], Loss: 0.256000
Epoch: [58/484] Iter:[10/24], Time: 0.44, lr: [0.008906874338018065], Loss: 0.382035
Epoch: [58/484] Iter:[20/24], Time: 0.42, lr: [0.00889902571900799], Loss: 0.345947
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95817791 0.71219757 0.87577817 0.26963722 0.32800802 0.46008596
 0.50622047 0.64722158 0.89348904 0.37736976 0.89067053 0.72239784
 0.3869877  0.87637428 0.26814189 0.39690214 0.29995674 0.32183876
 0.69160236] 0.5727925234994737
1 [0.95766086 0.70885906 0.88025708 0.27030856 0.33186423 0.47881258
 0.53160701 0.65745249 0.8934968  0.33542931 0.89549619 0.73232666
 0.40966511 0.87917801 0.32069108 0.47171243 0.27344948 0.30857168
 0.70203425] 0.5809933089648072
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.352, MeanIU:  0.5810, Best_mIoU:  0.6385
[0.95766086 0.70885906 0.88025708 0.27030856 0.33186423 0.47881258
 0.53160701 0.65745249 0.8934968  0.33542931 0.89549619 0.73232666
 0.40966511 0.87917801 0.32069108 0.47171243 0.27344948 0.30857168
 0.70203425]
Epoch: [59/484] Iter:[0/24], Time: 0.98, lr: [0.008895886056071986], Loss: 0.144682
Epoch: [59/484] Iter:[10/24], Time: 0.45, lr: [0.008888036359937497], Loss: 0.286974
Epoch: [59/484] Iter:[20/24], Time: 0.42, lr: [0.008880185893431793], Loss: 0.264519
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96574676 0.74233082 0.87734004 0.29628871 0.38253271 0.4943416
 0.52247271 0.65277499 0.8930483  0.48257672 0.91981116 0.73822619
 0.46454203 0.8946319  0.30955202 0.5117677  0.17783167 0.37631772
 0.7040138 ] 0.6003235551198864
1 [0.96616052 0.74907021 0.87928254 0.28093034 0.37493787 0.51046651
 0.54693219 0.66802691 0.89305045 0.50943552 0.92373973 0.74460235
 0.48781426 0.90554567 0.34656164 0.52954792 0.07642996 0.39772244
 0.70866437] 0.6052063883252591
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.311, MeanIU:  0.6052, Best_mIoU:  0.6385
[0.96616052 0.74907021 0.87928254 0.28093034 0.37493787 0.51046651
 0.54693219 0.66802691 0.89305045 0.50943552 0.92373973 0.74460235
 0.48781426 0.90554567 0.34656164 0.52954792 0.07642996 0.39772244
 0.70866437]
Epoch: [60/484] Iter:[0/24], Time: 0.98, lr: [0.00887704549093916], Loss: 0.216700
Epoch: [60/484] Iter:[10/24], Time: 0.44, lr: [0.008869193944514834], Loss: 0.315310
Epoch: [60/484] Iter:[20/24], Time: 0.42, lr: [0.008861341625718493], Loss: 0.310256
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.92147218 0.596887   0.87790306 0.24342042 0.41648859 0.48980086
 0.52151169 0.65069255 0.89435187 0.43469008 0.92321081 0.71807472
 0.45505892 0.85923478 0.31105214 0.42361439 0.26142326 0.41642659
 0.69769469] 0.5848951885614438
1 [0.92744253 0.61343541 0.87963344 0.23860581 0.44014904 0.5014134
 0.54615575 0.67037397 0.89687465 0.45059254 0.92228284 0.70840596
 0.45742919 0.8497624  0.27309321 0.39560774 0.23817416 0.45478249
 0.69110473] 0.5871220652986083
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.392, MeanIU:  0.5871, Best_mIoU:  0.6385
[0.92744253 0.61343541 0.87963344 0.23860581 0.44014904 0.5014134
 0.54615575 0.67037397 0.89687465 0.45059254 0.92228284 0.70840596
 0.45742919 0.8497624  0.27309321 0.39560774 0.23817416 0.45478249
 0.69110473]
Epoch: [61/484] Iter:[0/24], Time: 0.95, lr: [0.00885820048174845], Loss: 0.568346
Epoch: [61/484] Iter:[10/24], Time: 0.44, lr: [0.008850347080225406], Loss: 0.311155
Epoch: [61/484] Iter:[20/24], Time: 0.42, lr: [0.008842492904319594], Loss: 0.285236
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96409813 0.75359085 0.88513844 0.25392943 0.41664601 0.49750185
 0.52944888 0.66415882 0.89769844 0.46631045 0.91662261 0.72564978
 0.42010378 0.90276087 0.3440417  0.500543   0.35911277 0.45428507
 0.70198116] 0.6133485288348338
1 [0.96411179 0.7512285  0.88614225 0.22373884 0.4246205  0.50326943
 0.54699489 0.67531968 0.89534561 0.43111964 0.92016609 0.73986139
 0.46168895 0.90868936 0.43655941 0.52110552 0.35131001 0.48294488
 0.70971738] 0.6228386375735431
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.309, MeanIU:  0.6228, Best_mIoU:  0.6385
[0.96411179 0.7512285  0.88614225 0.22373884 0.4246205  0.50326943
 0.54699489 0.67531968 0.89534561 0.43111964 0.92016609 0.73986139
 0.46168895 0.90868936 0.43655941 0.52110552 0.35131001 0.48294488
 0.70971738]
Epoch: [62/484] Iter:[0/24], Time: 0.96, lr: [0.00883935101694182], Loss: 0.150716
Epoch: [62/484] Iter:[10/24], Time: 0.44, lr: [0.008831495755487254], Loss: 0.236984
Epoch: [62/484] Iter:[20/24], Time: 0.42, lr: [0.008823639717629152], Loss: 0.256538
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96635156 0.75819419 0.8867187  0.28934703 0.4542693  0.50128761
 0.54521192 0.66448048 0.89713771 0.48838937 0.91928527 0.73700094
 0.47801288 0.90040291 0.32561202 0.39162052 0.3576906  0.47520329
 0.70564513] 0.6179927085702029
1 [0.96684643 0.76372783 0.88815214 0.29173481 0.47702306 0.52200304
 0.56423077 0.68105583 0.89839749 0.49817357 0.922456   0.73217178
 0.47293838 0.90610544 0.39435837 0.39234164 0.36707321 0.48740041
 0.70917987] 0.6281773721051581
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.297, MeanIU:  0.6282, Best_mIoU:  0.6385
[0.96684643 0.76372783 0.88815214 0.29173481 0.47702306 0.52200304
 0.56423077 0.68105583 0.89839749 0.49817357 0.922456   0.73217178
 0.47293838 0.90610544 0.39435837 0.39234164 0.36707321 0.48740041
 0.70917987]
Epoch: [63/484] Iter:[0/24], Time: 0.95, lr: [0.008820497084903707], Loss: 0.191080
Epoch: [63/484] Iter:[10/24], Time: 0.44, lr: [0.008812639958660723], Loss: 0.270034
Epoch: [63/484] Iter:[20/24], Time: 0.42, lr: [0.008804782053983343], Loss: 0.280469
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95774564 0.73632843 0.8823111  0.34648019 0.42131529 0.50302402
 0.48049229 0.6594067  0.89421226 0.47320938 0.92143853 0.71895464
 0.32470356 0.895401   0.3768301  0.54184321 0.38917879 0.37027731
 0.69441814] 0.6098721361179186
1 [0.95700433 0.74124463 0.88658939 0.40114658 0.44747482 0.51778218
 0.5347054  0.67443461 0.89643098 0.48316669 0.92485529 0.73154953
 0.37447315 0.89395449 0.4843525  0.57517336 0.43741787 0.38740839
 0.70101478] 0.6342199463358064
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.312, MeanIU:  0.6342, Best_mIoU:  0.6385
[0.95700433 0.74124463 0.88658939 0.40114658 0.44747482 0.51778218
 0.5347054  0.67443461 0.89643098 0.48316669 0.92485529 0.73154953
 0.37447315 0.89395449 0.4843525  0.57517336 0.43741787 0.38740839
 0.70101478]
Epoch: [64/484] Iter:[0/24], Time: 0.96, lr: [0.008801638673960601], Loss: 0.261071
Epoch: [64/484] Iter:[10/24], Time: 0.44, lr: [0.008793779678048034], Loss: 0.292911
Epoch: [64/484] Iter:[20/24], Time: 0.42, lr: [0.008785919901660045], Loss: 0.280220
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95902913 0.72718156 0.88120611 0.31890331 0.4419508  0.51089046
 0.52840815 0.65147936 0.89692036 0.417244   0.92115793 0.73175787
 0.41377081 0.88713665 0.24873197 0.2591541  0.3061939  0.30503716
 0.68663146] 0.5838307941142373
1 [0.95804246 0.71890432 0.88503147 0.35887319 0.46667444 0.51741236
 0.55941818 0.66805088 0.89823649 0.3584168  0.92387898 0.73917978
 0.47245899 0.88845928 0.25044658 0.15070668 0.25912581 0.30002808
 0.6864001 ] 0.5820918334400301
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.337, MeanIU:  0.5821, Best_mIoU:  0.6385
[0.95804246 0.71890432 0.88503147 0.35887319 0.46667444 0.51741236
 0.55941818 0.66805088 0.89823649 0.3584168  0.92387898 0.73917978
 0.47245899 0.88845928 0.25044658 0.15070668 0.25912581 0.30002808
 0.6864001 ]
Epoch: [65/484] Iter:[0/24], Time: 0.96, lr: [0.00878277577238062], Loss: 0.313151
Epoch: [65/484] Iter:[10/24], Time: 0.44, lr: [0.00877491490189285], Loss: 0.248016
Epoch: [65/484] Iter:[20/24], Time: 0.42, lr: [0.008767053248878396], Loss: 0.264205
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96495437 0.76397755 0.88156044 0.29158128 0.39863272 0.47792387
 0.5470716  0.67165575 0.89942121 0.5184527  0.91976161 0.73337567
 0.46723046 0.90208999 0.34198168 0.32335302 0.21761702 0.339953
 0.70602323] 0.5982430087227663
1 [0.96440649 0.7647559  0.88520818 0.27385669 0.4267722  0.50834027
 0.5673581  0.68242945 0.90020161 0.51719251 0.92131555 0.73239038
 0.49672747 0.90715127 0.38906926 0.27295314 0.19199724 0.35442891
 0.71077284] 0.603543550766361
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.307, MeanIU:  0.6035, Best_mIoU:  0.6385
[0.96440649 0.7647559  0.88520818 0.27385669 0.4267722  0.50834027
 0.5673581  0.68242945 0.90020161 0.51719251 0.92131555 0.73239038
 0.49672747 0.90715127 0.38906926 0.27295314 0.19199724 0.35442891
 0.71077284]
Epoch: [66/484] Iter:[0/24], Time: 0.98, lr: [0.008763908368373067], Loss: 0.197424
Epoch: [66/484] Iter:[10/24], Time: 0.44, lr: [0.008756045618379853], Loss: 0.236157
Epoch: [66/484] Iter:[20/24], Time: 0.42, lr: [0.008748182083798366], Loss: 0.230964
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9629945  0.76014624 0.87753456 0.22196442 0.40632876 0.50326061
 0.51615481 0.67253402 0.89856264 0.52344544 0.9235889  0.74741667
 0.4346723  0.90654004 0.32986214 0.5318985  0.20140169 0.28485008
 0.69014091] 0.5996472213809175
1 [0.96473914 0.7680279  0.87933716 0.20248656 0.44324878 0.5163712
 0.5408471  0.68209765 0.89963718 0.52071183 0.9258942  0.75617338
 0.46755782 0.91232615 0.44465792 0.56868977 0.12577708 0.29924901
 0.6899019 ] 0.6109332486883374
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.304, MeanIU:  0.6109, Best_mIoU:  0.6385
[0.96473914 0.7680279  0.87933716 0.20248656 0.44324878 0.5163712
 0.5408471  0.68209765 0.89963718 0.52071183 0.9258942  0.75617338
 0.46755782 0.91232615 0.44465792 0.56868977 0.12577708 0.29924901
 0.6899019 ]
Epoch: [67/484] Iter:[0/24], Time: 0.94, lr: [0.008745036450088015], Loss: 0.237076
Epoch: [67/484] Iter:[10/24], Time: 0.44, lr: [0.008737171815634291], Loss: 0.254014
Epoch: [67/484] Iter:[20/24], Time: 0.42, lr: [0.008729306394520318], Loss: 0.257892
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96533374 0.76342075 0.87300057 0.26577294 0.36784544 0.50773746
 0.5369267  0.65596924 0.89293957 0.48905261 0.92399372 0.73552483
 0.42520127 0.89461759 0.27599118 0.42561596 0.20927302 0.37698215
 0.70285186] 0.5941079262542801
1 [0.96535116 0.76154903 0.8645647  0.24311463 0.33269895 0.50981449
 0.56188826 0.66304512 0.89367624 0.48838802 0.92593547 0.74249472
 0.46286898 0.90008221 0.35205593 0.49184023 0.25545391 0.40009287
 0.70432886] 0.6062759882666491
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.321, MeanIU:  0.6063, Best_mIoU:  0.6385
[0.96535116 0.76154903 0.8645647  0.24311463 0.33269895 0.50981449
 0.56188826 0.66304512 0.89367624 0.48838802 0.92593547 0.74249472
 0.46286898 0.90008221 0.35205593 0.49184023 0.25545391 0.40009287
 0.70432886]
Epoch: [68/484] Iter:[0/24], Time: 0.98, lr: [0.008726160005615846], Loss: 0.187241
Epoch: [68/484] Iter:[10/24], Time: 0.44, lr: [0.00871829348172155], Loss: 0.229123
Epoch: [68/484] Iter:[20/24], Time: 0.41, lr: [0.00871042616908456], Loss: 0.255034
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96991034 0.77755866 0.89218502 0.36482321 0.44071342 0.52373662
 0.54223369 0.67015697 0.90137646 0.50343842 0.92705607 0.74779982
 0.47569692 0.90219821 0.32726697 0.48718816 0.30311074 0.37077901
 0.70656476] 0.6228312356266268
1 [0.97066614 0.78026379 0.89469248 0.36398996 0.46841428 0.5392723
 0.56513687 0.68510086 0.90130395 0.49030445 0.92820023 0.75469165
 0.50759928 0.90999028 0.45209346 0.58263122 0.38736314 0.40042158
 0.71438151] 0.647185128326605
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.277, MeanIU:  0.6472, Best_mIoU:  0.6472
[0.97066614 0.78026379 0.89469248 0.36398996 0.46841428 0.5392723
 0.56513687 0.68510086 0.90130395 0.49030445 0.92820023 0.75469165
 0.50759928 0.90999028 0.45209346 0.58263122 0.38736314 0.40042158
 0.71438151]
Epoch: [69/484] Iter:[0/24], Time: 0.94, lr: [0.008707279022986812], Loss: 0.277977
Epoch: [69/484] Iter:[10/24], Time: 0.44, lr: [0.0086994106046467], Loss: 0.273412
Epoch: [69/484] Iter:[20/24], Time: 0.42, lr: [0.008691541395470896], Loss: 0.267378
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96300054 0.7651811  0.88657046 0.32880609 0.44703071 0.51474095
 0.53890229 0.66440767 0.89910927 0.50985948 0.92675806 0.74170865
 0.46291839 0.86106895 0.23191289 0.15741451 0.29796518 0.36022433
 0.70267728] 0.5926450950472407
1 [0.96144456 0.76746771 0.88886255 0.35685888 0.47740935 0.52327117
 0.56392    0.67661734 0.90202396 0.51199801 0.928054   0.75006979
 0.49720701 0.87256219 0.28318254 0.18107504 0.34216251 0.39216112
 0.70610383] 0.6096027137414343
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.326, MeanIU:  0.6096, Best_mIoU:  0.6472
[0.96144456 0.76746771 0.88886255 0.35685888 0.47740935 0.52327117
 0.56392    0.67661734 0.90202396 0.51199801 0.928054   0.75006979
 0.49720701 0.87256219 0.28318254 0.18107504 0.34216251 0.39216112
 0.70610383]
Epoch: [70/484] Iter:[0/24], Time: 0.98, lr: [0.008688393490170594], Loss: 0.286038
Epoch: [70/484] Iter:[10/24], Time: 0.44, lr: [0.008680523172354045], Loss: 0.213799
Epoch: [70/484] Iter:[20/24], Time: 0.42, lr: [0.008672652061598176], Loss: 0.217683
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96779322 0.76777057 0.89217734 0.31255365 0.45375656 0.52565552
 0.55074892 0.67071263 0.90080887 0.50955984 0.92672821 0.75043339
 0.47755342 0.89521516 0.32857655 0.37812977 0.36301289 0.41567702
 0.70352028] 0.6205465169100071
1 [0.96801768 0.76961966 0.89051968 0.32546698 0.45864363 0.53543415
 0.5579001  0.6791067  0.90087088 0.50911827 0.92825397 0.75595344
 0.50087108 0.90523283 0.42806659 0.46811511 0.42449548 0.43181279
 0.71038714] 0.6393624292948354
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.294, MeanIU:  0.6394, Best_mIoU:  0.6472
[0.96801768 0.76961966 0.89051968 0.32546698 0.45864363 0.53543415
 0.5579001  0.6791067  0.90087088 0.50911827 0.92825397 0.75595344
 0.50087108 0.90523283 0.42806659 0.46811511 0.42449548 0.43181279
 0.71038714]
Epoch: [71/484] Iter:[0/24], Time: 0.96, lr: [0.008669503395075837], Loss: 0.302951
Epoch: [71/484] Iter:[10/24], Time: 0.44, lr: [0.008661631172726666], Loss: 0.226437
Epoch: [71/484] Iter:[20/24], Time: 0.42, lr: [0.00865375815532384], Loss: 0.239278
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.8850164  0.65666918 0.77011802 0.31756849 0.3954935  0.48228935
 0.45430413 0.65090213 0.88401368 0.44850109 0.87208045 0.72295121
 0.4459205  0.87990146 0.29633664 0.45102522 0.21475058 0.3283424
 0.68613878] 0.5706485895483834
1 [0.87453655 0.66104913 0.75703453 0.31342264 0.41555184 0.50529809
 0.49721251 0.66439645 0.88712391 0.44002215 0.8633878  0.7285856
 0.46346754 0.89443371 0.39446953 0.5409009  0.172846   0.35054042
 0.68735406] 0.5848228076571589
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.472, MeanIU:  0.5848, Best_mIoU:  0.6472
[0.87453655 0.66104913 0.75703453 0.31342264 0.41555184 0.50529809
 0.49721251 0.66439645 0.88712391 0.44002215 0.8633878  0.7285856
 0.46346754 0.89443371 0.39446953 0.5409009  0.172846   0.35054042
 0.68735406]
Epoch: [72/484] Iter:[0/24], Time: 0.97, lr: [0.0086506087255497], Loss: 0.273488
Epoch: [72/484] Iter:[10/24], Time: 0.44, lr: [0.008642734593585963], Loss: 0.339979
Epoch: [72/484] Iter:[20/24], Time: 0.42, lr: [0.00863485966444345], Loss: 0.315257
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95839628 0.72376228 0.85056609 0.23907725 0.3873872  0.4844424
 0.4553068  0.62660795 0.88543413 0.5111129  0.80460196 0.72875808
 0.4351627  0.8819489  0.29108235 0.42949006 0.26689386 0.37834946
 0.68182982] 0.5800110771151585
1 [0.95801171 0.71857887 0.84931515 0.17305249 0.41260727 0.48407503
 0.47647234 0.64023492 0.88613658 0.51166015 0.79462866 0.72993311
 0.44950492 0.89179344 0.37386201 0.50396    0.29309091 0.39338982
 0.67829716] 0.5904528699273122
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.371, MeanIU:  0.5905, Best_mIoU:  0.6472
[0.95801171 0.71857887 0.84931515 0.17305249 0.41260727 0.48407503
 0.47647234 0.64023492 0.88613658 0.51166015 0.79462866 0.72993311
 0.44950492 0.89179344 0.37386201 0.50396    0.29309091 0.39338982
 0.67829716]
Epoch: [73/484] Iter:[0/24], Time: 0.97, lr: [0.008631709469377387], Loss: 0.242062
Epoch: [73/484] Iter:[10/24], Time: 0.44, lr: [0.008623833422691194], Loss: 0.481676
Epoch: [73/484] Iter:[20/24], Time: 0.42, lr: [0.008615956576690228], Loss: 0.412248
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94631954 0.64848733 0.87192223 0.21466152 0.33233217 0.48799563
 0.41988766 0.62882904 0.8765311  0.43239101 0.90848974 0.71716723
 0.44811461 0.88151981 0.27639523 0.19601027 0.26317996 0.2896165
 0.64475882] 0.5518215470132999
1 [0.9271315  0.55729854 0.87341875 0.19068871 0.32587125 0.47350208
 0.45321725 0.62609826 0.87291048 0.40710926 0.9046501  0.69922835
 0.44518635 0.87986455 0.32308024 0.24890652 0.2971731  0.27507679
 0.63735431] 0.5483034948789479
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.410, MeanIU:  0.5483, Best_mIoU:  0.6472
[0.9271315  0.55729854 0.87341875 0.19068871 0.32587125 0.47350208
 0.45321725 0.62609826 0.87291048 0.40710926 0.9046501  0.69922835
 0.44518635 0.87986455 0.32308024 0.24890652 0.2971731  0.27507679
 0.63735431]
Epoch: [74/484] Iter:[0/24], Time: 0.98, lr: [0.008612805614281687], Loss: 0.198996
Epoch: [74/484] Iter:[10/24], Time: 0.45, lr: [0.008604927647738998], Loss: 0.329185
Epoch: [74/484] Iter:[20/24], Time: 0.42, lr: [0.008597048879734588], Loss: 0.287612
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95354401 0.71719469 0.8734351  0.26054969 0.42796395 0.50685432
 0.52557182 0.64813635 0.89371117 0.51682697 0.91824601 0.70031903
 0.36737461 0.8955957  0.31233835 0.47917837 0.31214462 0.39608778
 0.68779813] 0.5996247728068149
1 [0.95277136 0.70994529 0.87294709 0.26411967 0.46410336 0.52117418
 0.55506656 0.66361911 0.88980514 0.51859625 0.91679368 0.69206597
 0.39881916 0.90562982 0.39480938 0.51607534 0.35116833 0.40937935
 0.6909503 ] 0.6151494391077093
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.331, MeanIU:  0.6151, Best_mIoU:  0.6472
[0.95277136 0.70994529 0.87294709 0.26411967 0.46410336 0.52117418
 0.55506656 0.66361911 0.88980514 0.51859625 0.91679368 0.69206597
 0.39881916 0.90562982 0.39480938 0.51607534 0.35116833 0.40937935
 0.6909503 ]
Epoch: [75/484] Iter:[0/24], Time: 0.97, lr: [0.008593897147922499], Loss: 0.270649
Epoch: [75/484] Iter:[10/24], Time: 0.44, lr: [0.008586017256362925], Loss: 0.255133
Epoch: [75/484] Iter:[20/24], Time: 0.42, lr: [0.00857813656118365], Loss: 0.272716
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95890024 0.74663362 0.87974916 0.36177835 0.43208689 0.51872329
 0.5186262  0.65323279 0.89969339 0.53485314 0.92148619 0.7289365
 0.42039106 0.89764801 0.33508066 0.528549   0.32357715 0.40170783
 0.69309294] 0.6186708633153295
1 [0.95941869 0.7494755  0.88058856 0.39134659 0.44525461 0.53669505
 0.55481095 0.67077863 0.90178738 0.55432248 0.92152211 0.74171137
 0.46090821 0.90324074 0.38573123 0.54750499 0.28318189 0.44213017
 0.70497699] 0.6334413760326812
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.299, MeanIU:  0.6334, Best_mIoU:  0.6472
[0.95941869 0.7494755  0.88058856 0.39134659 0.44525461 0.53669505
 0.55481095 0.67077863 0.90178738 0.55432248 0.92152211 0.74171137
 0.46090821 0.90324074 0.38573123 0.54750499 0.28318189 0.44213017
 0.70497699]
Epoch: [76/484] Iter:[0/24], Time: 0.96, lr: [0.008574984057896343], Loss: 0.390637
Epoch: [76/484] Iter:[10/24], Time: 0.44, lr: [0.008567102236132957], Loss: 0.293232
Epoch: [76/484] Iter:[20/24], Time: 0.42, lr: [0.008559219608580769], Loss: 0.281804
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95300365 0.71331845 0.87692674 0.19309807 0.4398233  0.49180602
 0.50840912 0.64673415 0.89076776 0.4493669  0.9189298  0.73105891
 0.40465061 0.72130731 0.24749599 0.19972821 0.04024262 0.31922235
 0.67959592] 0.5487097821812899
1 [0.95444434 0.70914591 0.8735148  0.15343449 0.46090443 0.50238704
 0.53847407 0.6570949  0.88923057 0.4042468  0.91604095 0.74190408
 0.45663951 0.69603738 0.22659827 0.20327328 0.04185677 0.34470796
 0.68052907] 0.5500244526777768
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.445, MeanIU:  0.5500, Best_mIoU:  0.6472
[0.95444434 0.70914591 0.8735148  0.15343449 0.46090443 0.50238704
 0.53847407 0.6570949  0.88923057 0.4042468  0.91604095 0.74190408
 0.45663951 0.69603738 0.22659827 0.20327328 0.04185677 0.34470796
 0.68052907]
Epoch: [77/484] Iter:[0/24], Time: 0.95, lr: [0.008556066331735904], Loss: 0.363463
Epoch: [77/484] Iter:[10/24], Time: 0.44, lr: [0.008548182574555028], Loss: 0.266698
Epoch: [77/484] Iter:[20/24], Time: 0.42, lr: [0.008540298009405041], Loss: 0.310256
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96118013 0.75111601 0.87679823 0.34666345 0.40850974 0.50884704
 0.5370106  0.64690329 0.8953616  0.4962862  0.9144952  0.71212268
 0.44664452 0.86778474 0.27875546 0.23003375 0.19044519 0.37906069
 0.68486606] 0.585941292945908
1 [0.96054288 0.74938799 0.87043648 0.31046551 0.43006533 0.52471565
 0.55953438 0.66397554 0.89464167 0.51282175 0.91376058 0.71253439
 0.46213876 0.87004962 0.30237604 0.20237636 0.0088933  0.38798513
 0.68661596] 0.5801745949989595
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.336, MeanIU:  0.5802, Best_mIoU:  0.6472
[0.96054288 0.74938799 0.87043648 0.31046551 0.43006533 0.52471565
 0.55953438 0.66397554 0.89464167 0.51282175 0.91376058 0.71253439
 0.46213876 0.87004962 0.30237604 0.20237636 0.0088933  0.38798513
 0.68661596]
Epoch: [78/484] Iter:[0/24], Time: 0.99, lr: [0.00853714395690952], Loss: 0.350090
Epoch: [78/484] Iter:[10/24], Time: 0.45, lr: [0.008529258259070529], Loss: 0.276908
Epoch: [78/484] Iter:[20/24], Time: 0.42, lr: [0.008521371751070825], Loss: 0.264454
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95577923 0.73676571 0.8782325  0.19887894 0.41540978 0.50215567
 0.53913261 0.65191975 0.89607001 0.48900795 0.91821549 0.74386711
 0.48242049 0.89207119 0.2671364  0.34763358 0.2101213  0.35760811
 0.68602317] 0.5878131047221139
1 [0.95630312 0.7357946  0.88110309 0.1825144  0.42891547 0.51445951
 0.54792238 0.65707982 0.89616866 0.46943854 0.91559711 0.74837571
 0.50774551 0.89780591 0.33456289 0.42253413 0.25320021 0.40898582
 0.68993387] 0.6025495124223598
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.328, MeanIU:  0.6025, Best_mIoU:  0.6472
[0.95630312 0.7357946  0.88110309 0.1825144  0.42891547 0.51445951
 0.54792238 0.65707982 0.89616866 0.46943854 0.91559711 0.74837571
 0.50774551 0.89780591 0.33456289 0.42253413 0.25320021 0.40898582
 0.68993387]
Epoch: [79/484] Iter:[0/24], Time: 0.97, lr: [0.00851821692082071], Loss: 0.286205
Epoch: [79/484] Iter:[10/24], Time: 0.44, lr: [0.008510329277055815], Loss: 0.260754
Epoch: [79/484] Iter:[20/24], Time: 0.42, lr: [0.008502440820927232], Loss: 0.249794
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95826488 0.71802318 0.89115846 0.30166097 0.44305238 0.52977012
 0.51353907 0.65786062 0.89784912 0.47747451 0.92253722 0.74639632
 0.44402083 0.90230618 0.28317384 0.50383228 0.30269258 0.36961401
 0.6941726 ] 0.6082841669192874
1 [0.95898405 0.72265747 0.89566131 0.29595215 0.46107741 0.54576251
 0.54476688 0.67763691 0.90108244 0.47373003 0.9231259  0.75195994
 0.49595749 0.90670508 0.32599229 0.5577621  0.30471351 0.40068435
 0.70135098] 0.6234506736846515
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.304, MeanIU:  0.6235, Best_mIoU:  0.6472
[0.95898405 0.72265747 0.89566131 0.29595215 0.46107741 0.54576251
 0.54476688 0.67763691 0.90108244 0.47373003 0.9231259  0.75195994
 0.49595749 0.90670508 0.32599229 0.5577621  0.30471351 0.40068435
 0.70135098]
Epoch: [80/484] Iter:[0/24], Time: 0.99, lr: [0.008499285210807665], Loss: 0.190266
Epoch: [80/484] Iter:[10/24], Time: 0.44, lr: [0.008491395615821715], Loss: 0.264738
Epoch: [80/484] Iter:[20/24], Time: 0.42, lr: [0.008483505206257637], Loss: 0.279605
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94524247 0.70015335 0.80005203 0.19496235 0.38184991 0.46594433
 0.34426996 0.63241359 0.80931817 0.41871708 0.78247285 0.7125063
 0.39775252 0.86447047 0.2214177  0.36752523 0.12850813 0.14405593
 0.6707162 ] 0.5253867669443995
1 [0.93489312 0.7078417  0.7891723  0.17922803 0.40951015 0.474422
 0.40676476 0.64380085 0.81672491 0.44783969 0.73697492 0.71700458
 0.44046143 0.86898595 0.25905196 0.43891058 0.08644387 0.14376171
 0.68168439] 0.5359724692766957
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.455, MeanIU:  0.5360, Best_mIoU:  0.6472
[0.93489312 0.7078417  0.7891723  0.17922803 0.40951015 0.474422
 0.40676476 0.64380085 0.81672491 0.44783969 0.73697492 0.71700458
 0.44046143 0.86898595 0.25905196 0.43891058 0.08644387 0.14376171
 0.68168439]
Epoch: [81/484] Iter:[0/24], Time: 1.00, lr: [0.008480348814142754], Loss: 0.283286
Epoch: [81/484] Iter:[10/24], Time: 0.44, lr: [0.00847245726261302], Loss: 0.276773
Epoch: [81/484] Iter:[20/24], Time: 0.42, lr: [0.008464564894279167], Loss: 0.272839
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96370037 0.74131532 0.88307157 0.23238161 0.41274011 0.51707551
 0.55168496 0.66598402 0.89231319 0.3802847  0.91052062 0.73189577
 0.46930382 0.88510501 0.32384998 0.40057342 0.27982158 0.41482693
 0.69160318] 0.597265876928997
1 [0.96264553 0.74326601 0.88077355 0.23746791 0.45483075 0.52706781
 0.56145654 0.68142504 0.89012282 0.33371429 0.91336618 0.72153943
 0.45009602 0.89403002 0.43032711 0.46100561 0.37457732 0.42159056
 0.693316  ] 0.6122430793306626
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.324, MeanIU:  0.6122, Best_mIoU:  0.6472
[0.96264553 0.74326601 0.88077355 0.23746791 0.45483075 0.52706781
 0.56145654 0.68142504 0.89012282 0.33371429 0.91336618 0.72153943
 0.45009602 0.89403002 0.43032711 0.46100561 0.37457732 0.42159056
 0.693316  ]
Epoch: [82/484] Iter:[0/24], Time: 0.96, lr: [0.008461407718032013], Loss: 0.156547
Epoch: [82/484] Iter:[10/24], Time: 0.44, lr: [0.008453514204607979], Loss: 0.250414
Epoch: [82/484] Iter:[20/24], Time: 0.42, lr: [0.008445619872142191], Loss: 0.248490
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96090535 0.74504637 0.88367312 0.23027937 0.4190985  0.52586084
 0.53160077 0.66471469 0.89774644 0.50219632 0.91243828 0.71787839
 0.34853434 0.90463134 0.3148694  0.55346522 0.31704834 0.2931617
 0.70797898] 0.6016383035070081
1 [0.96157044 0.73641184 0.8798483  0.21757943 0.45294113 0.53676935
 0.56334389 0.68056556 0.89361612 0.50966039 0.91338766 0.721437
 0.40315893 0.90954958 0.37524652 0.57891171 0.32200085 0.33688323
 0.71653235] 0.6162849605547447
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.316, MeanIU:  0.6163, Best_mIoU:  0.6472
[0.96157044 0.73641184 0.8798483  0.21757943 0.45294113 0.53676935
 0.56334389 0.68056556 0.89361612 0.50966039 0.91338766 0.721437
 0.40315893 0.90954958 0.37524652 0.57891171 0.32200085 0.33688323
 0.71653235]
Epoch: [83/484] Iter:[0/24], Time: 0.97, lr: [0.008442461909614632], Loss: 0.193739
Epoch: [83/484] Iter:[10/24], Time: 0.44, lr: [0.00843456642891778], Loss: 0.240148
Epoch: [83/484] Iter:[20/24], Time: 0.42, lr: [0.008426670126929808], Loss: 0.250095
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9656473  0.76066334 0.88507996 0.32943685 0.41661736 0.52247859
 0.53736554 0.65430178 0.89699075 0.49869812 0.91743214 0.7408775
 0.48625322 0.90401728 0.33663266 0.42089012 0.32714257 0.37163338
 0.68565413] 0.613569084185177
1 [0.96477685 0.76269834 0.88587679 0.32867157 0.42390794 0.53169975
 0.55168817 0.66050203 0.89825048 0.48163267 0.92011207 0.74438307
 0.50770616 0.90900084 0.42969774 0.51830231 0.42346673 0.38012534
 0.67405066] 0.6313973424560767
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.301, MeanIU:  0.6314, Best_mIoU:  0.6472
[0.96477685 0.76269834 0.88587679 0.32867157 0.42390794 0.53169975
 0.55168817 0.66050203 0.89825048 0.48163267 0.92011207 0.74438307
 0.50770616 0.90900084 0.42969774 0.51830231 0.42346673 0.38012534
 0.67405066]
Epoch: [84/484] Iter:[0/24], Time: 0.97, lr: [0.008423511375962452], Loss: 0.228048
Epoch: [84/484] Iter:[10/24], Time: 0.44, lr: [0.00841561392258604], Loss: 0.252534
Epoch: [84/484] Iter:[20/24], Time: 0.42, lr: [0.008407715645657317], Loss: 0.247108
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96324305 0.7489488  0.89045157 0.26031914 0.43441033 0.52642792
 0.55508024 0.67292902 0.90055813 0.53048932 0.91896096 0.72454524
 0.32936089 0.90077921 0.34584704 0.55219322 0.39171997 0.39960441
 0.70593201] 0.6185158147104657
1 [0.96311366 0.7516871  0.88735462 0.15562134 0.46908087 0.53770285
 0.5776629  0.68666958 0.90257579 0.55271904 0.92013389 0.72917567
 0.37031948 0.90933358 0.40435531 0.61775493 0.45531149 0.41776485
 0.71211329] 0.6326552761923633
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.298, MeanIU:  0.6327, Best_mIoU:  0.6472
[0.96311366 0.7516871  0.88735462 0.15562134 0.46908087 0.53770285
 0.5776629  0.68666958 0.90257579 0.55271904 0.92013389 0.72917567
 0.37031948 0.90933358 0.40435531 0.61775493 0.45531149 0.41776485
 0.71211329]
Epoch: [85/484] Iter:[0/24], Time: 0.97, lr: [0.008404556104079421], Loss: 0.262225
Epoch: [85/484] Iter:[10/24], Time: 0.44, lr: [0.008396656672588269], Loss: 0.345486
Epoch: [85/484] Iter:[20/24], Time: 0.42, lr: [0.008388756415271705], Loss: 0.291663
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95776107 0.72216007 0.88532117 0.30038646 0.43270802 0.51029483
 0.55255324 0.66059717 0.90161338 0.46411268 0.91689158 0.74407684
 0.47859745 0.89815518 0.33860714 0.47796571 0.4451701  0.42544459
 0.70366935] 0.62189926356552
1 [0.9562492  0.71667193 0.8863484  0.24208665 0.47605851 0.50829851
 0.58239422 0.6811241  0.89771127 0.40011185 0.91892082 0.74724733
 0.48988042 0.90279217 0.41250938 0.52093289 0.52889882 0.44387573
 0.70095534] 0.6322667112077123
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.314, MeanIU:  0.6323, Best_mIoU:  0.6472
[0.9562492  0.71667193 0.8863484  0.24208665 0.47605851 0.50829851
 0.58239422 0.6811241  0.89771127 0.40011185 0.91892082 0.74724733
 0.48988042 0.90279217 0.41250938 0.52093289 0.52889882 0.44387573
 0.70095534]
Epoch: [86/484] Iter:[0/24], Time: 0.97, lr: [0.008385596080901083], Loss: 0.356335
Epoch: [86/484] Iter:[10/24], Time: 0.44, lr: [0.008377694665831352], Loss: 0.258211
Epoch: [86/484] Iter:[20/24], Time: 0.42, lr: [0.008369792422651097], Loss: 0.258333
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96067387 0.74763932 0.89087626 0.27741726 0.44584296 0.53073396
 0.53556291 0.66583944 0.89913992 0.48234688 0.88833688 0.74762156
 0.47399974 0.90659273 0.39767994 0.46607688 0.39498462 0.34066525
 0.69266644] 0.6181419367536455
1 [0.96266678 0.74235997 0.89234168 0.29129609 0.45942926 0.54335903
 0.57097785 0.67861842 0.89986907 0.48662397 0.91083751 0.7534502
 0.49172394 0.91488874 0.46762843 0.51765129 0.4719337  0.36731973
 0.69503834] 0.6377902111404853
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.298, MeanIU:  0.6378, Best_mIoU:  0.6472
[0.96266678 0.74235997 0.89234168 0.29129609 0.45942926 0.54335903
 0.57097785 0.67861842 0.89986907 0.48662397 0.91083751 0.7534502
 0.49172394 0.91488874 0.46762843 0.51765129 0.4719337  0.36731973
 0.69503834]
Epoch: [87/484] Iter:[0/24], Time: 0.97, lr: [0.008366631293294041], Loss: 0.183841
Epoch: [87/484] Iter:[10/24], Time: 0.44, lr: [0.008358727889153008], Loss: 0.251778
Epoch: [87/484] Iter:[20/24], Time: 0.42, lr: [0.008350823654604236], Loss: 0.246072
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95989897 0.7532281  0.87190235 0.28382026 0.41733223 0.51330772
 0.51370202 0.67134386 0.89425915 0.51264393 0.85609642 0.73632203
 0.45151922 0.88841887 0.29618919 0.37257012 0.45287315 0.2772668
 0.68982893] 0.600659121456524
1 [0.95454423 0.7489296  0.86172782 0.2722034  0.44659235 0.5277524
 0.54093454 0.68376902 0.89384959 0.52887384 0.85394837 0.74359124
 0.48446758 0.8994436  0.35284596 0.42861219 0.51291664 0.30565112
 0.6947751 ] 0.6176541368817955
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.331, MeanIU:  0.6177, Best_mIoU:  0.6472
[0.95454423 0.7489296  0.86172782 0.2722034  0.44659235 0.5277524
 0.54093454 0.68376902 0.89384959 0.52887384 0.85394837 0.74359124
 0.48446758 0.8994436  0.35284596 0.42861219 0.51291664 0.30565112
 0.6947751 ]
Epoch: [88/484] Iter:[0/24], Time: 1.00, lr: [0.008347661728055417], Loss: 0.177485
Epoch: [88/484] Iter:[10/24], Time: 0.44, lr: [0.008339756329321246], Loss: 0.230699
Epoch: [88/484] Iter:[20/24], Time: 0.42, lr: [0.008331850097869924], Loss: 0.271783
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95634234 0.7313176  0.87521444 0.31313746 0.40497847 0.48986874
 0.50106882 0.67104836 0.89625581 0.47380141 0.88844011 0.71865042
 0.38684324 0.89358069 0.34707144 0.47724393 0.45978527 0.41987584
 0.69898083] 0.6107108020287071
1 [0.95114937 0.72862812 0.86223303 0.29609776 0.40796544 0.46575142
 0.5003105  0.680043   0.89715545 0.51652    0.86722037 0.72232686
 0.40748634 0.9033617  0.39551902 0.53718932 0.47541752 0.45021813
 0.69820944] 0.6190948849907925
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.337, MeanIU:  0.6191, Best_mIoU:  0.6472
[0.95114937 0.72862812 0.86223303 0.29609776 0.40796544 0.46575142
 0.5003105  0.680043   0.89715545 0.51652    0.86722037 0.72232686
 0.40748634 0.9033617  0.39551902 0.53718932 0.47541752 0.45021813
 0.69820944]
Epoch: [89/484] Iter:[0/24], Time: 0.96, lr: [0.00832868737191231], Loss: 0.161100
Epoch: [89/484] Iter:[10/24], Time: 0.44, lr: [0.008320779973033826], Loss: 0.271014
Epoch: [89/484] Iter:[20/24], Time: 0.42, lr: [0.008312871739116488], Loss: 0.263290
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95971203 0.73803247 0.89268335 0.37490312 0.43904132 0.52924208
 0.56271179 0.65776435 0.90173567 0.47364513 0.92160538 0.74202764
 0.42954598 0.90045078 0.34188861 0.51763007 0.41311392 0.44482354
 0.68758776] 0.6277971055677047
1 [0.95777135 0.73374833 0.89356134 0.40122327 0.45891911 0.53995036
 0.57596422 0.66606253 0.90420322 0.48737219 0.92695141 0.75155534
 0.47242611 0.91168719 0.37733715 0.58956717 0.45500833 0.46338456
 0.68733949] 0.6449490882238964
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.304, MeanIU:  0.6449, Best_mIoU:  0.6472
[0.95777135 0.73374833 0.89356134 0.40122327 0.45891911 0.53995036
 0.57596422 0.66606253 0.90420322 0.48737219 0.92695141 0.75155534
 0.47242611 0.91168719 0.37733715 0.58956717 0.45500833 0.46338456
 0.68733949]
Epoch: [90/484] Iter:[0/24], Time: 0.99, lr: [0.008309708211521239], Loss: 0.243718
Epoch: [90/484] Iter:[10/24], Time: 0.44, lr: [0.008301798806917698], Loss: 0.281481
Epoch: [90/484] Iter:[20/24], Time: 0.42, lr: [0.008293888564941215], Loss: 0.254333
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96788451 0.7626241  0.89338571 0.28187777 0.4620642  0.54400763
 0.55505873 0.67082493 0.90293741 0.50635802 0.91944742 0.74952397
 0.46661346 0.90801571 0.35919863 0.53293885 0.44467627 0.38815693
 0.71117789] 0.6329880065778419
1 [0.96713011 0.75885882 0.89393265 0.27633294 0.49618478 0.55229304
 0.58522479 0.67834193 0.90532532 0.50481736 0.91899775 0.75586329
 0.49556148 0.91597947 0.4280312  0.62769164 0.55123659 0.41897017
 0.71451911] 0.6550153923850486
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.284, MeanIU:  0.6550, Best_mIoU:  0.6550
[0.96713011 0.75885882 0.89393265 0.27633294 0.49618478 0.55229304
 0.58522479 0.67834193 0.90532532 0.50481736 0.91899775 0.75586329
 0.49556148 0.91597947 0.4280312  0.62769164 0.55123659 0.41897017
 0.71451911]
Epoch: [91/484] Iter:[0/24], Time: 0.95, lr: [0.008290724233467599], Loss: 0.225655
Epoch: [91/484] Iter:[10/24], Time: 0.44, lr: [0.00828281281752846], Loss: 0.209352
Epoch: [91/484] Iter:[20/24], Time: 0.42, lr: [0.008274900561869797], Loss: 0.214922
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96535434 0.76113589 0.89563369 0.36247792 0.47216526 0.54106971
 0.57027984 0.67659037 0.90201196 0.51807444 0.92486832 0.7530976
 0.49647738 0.90496843 0.3659678  0.48089499 0.48124881 0.45787518
 0.71123688] 0.6442857270480896
1 [0.96515001 0.75952364 0.89864068 0.36882437 0.50196782 0.5526454
 0.59193405 0.68846865 0.90388474 0.54079186 0.92990555 0.75504525
 0.50117277 0.91302798 0.42710061 0.56155085 0.57637994 0.49099161
 0.71281637] 0.6652537968504494
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.281, MeanIU:  0.6653, Best_mIoU:  0.6653
[0.96515001 0.75952364 0.89864068 0.36882437 0.50196782 0.5526454
 0.59193405 0.68846865 0.90388474 0.54079186 0.92990555 0.75504525
 0.50117277 0.91302798 0.42710061 0.56155085 0.57637994 0.49099161
 0.71281637]
Epoch: [92/484] Iter:[0/24], Time: 0.93, lr: [0.008271735424265094], Loss: 0.203647
Epoch: [92/484] Iter:[10/24], Time: 0.44, lr: [0.00826382199134977], Loss: 0.217744
Epoch: [92/484] Iter:[20/24], Time: 0.41, lr: [0.008255907716355761], Loss: 0.232652
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96860989 0.76720166 0.8992866  0.34473853 0.48737316 0.54149632
 0.57311468 0.68542328 0.90287023 0.48258757 0.92237359 0.75960573
 0.49825448 0.91466069 0.40605015 0.60662509 0.43923606 0.40682865
 0.71763245] 0.6486299362781217
1 [0.96830773 0.76402955 0.90027027 0.33867588 0.52034153 0.55309458
 0.59424983 0.69471947 0.90259628 0.42471926 0.92414423 0.76691217
 0.5168517  0.9212824  0.47776517 0.66347542 0.42698214 0.43146318
 0.72206456] 0.6585234389850371
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.274, MeanIU:  0.6585, Best_mIoU:  0.6653
[0.96830773 0.76402955 0.90027027 0.33867588 0.52034153 0.55309458
 0.59424983 0.69471947 0.90259628 0.42471926 0.92414423 0.76691217
 0.5168517  0.9212824  0.47776517 0.66347542 0.42698214 0.43146318
 0.72206456]
Epoch: [93/484] Iter:[0/24], Time: 0.97, lr: [0.008252741770355171], Loss: 0.224234
Epoch: [93/484] Iter:[10/24], Time: 0.44, lr: [0.008244826314792805], Loss: 0.239426
Epoch: [93/484] Iter:[20/24], Time: 0.42, lr: [0.0082369100147799], Loss: 0.229405
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96941291 0.773997   0.88877384 0.2842574  0.39949903 0.52649865
 0.55431655 0.67401177 0.90213761 0.48457311 0.92394824 0.75429271
 0.4710494  0.8991057  0.37168743 0.34314222 0.40338135 0.45549834
 0.71112389] 0.6205635345888904
1 [0.97058405 0.77683427 0.88888388 0.27991922 0.423437   0.53826426
 0.58375875 0.6901965  0.90368927 0.49463652 0.92519727 0.76381717
 0.50732039 0.90348681 0.41441805 0.41670107 0.52434438 0.48184901
 0.71360322] 0.6421547938667217
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.293, MeanIU:  0.6422, Best_mIoU:  0.6653
[0.97058405 0.77683427 0.88888388 0.27991922 0.423437   0.53826426
 0.58375875 0.6901965  0.90368927 0.49463652 0.92519727 0.76381717
 0.50732039 0.90348681 0.41441805 0.41670107 0.52434438 0.48184901
 0.71360322]
Epoch: [94/484] Iter:[0/24], Time: 0.96, lr: [0.008233743258106446], Loss: 0.159446
Epoch: [94/484] Iter:[10/24], Time: 0.44, lr: [0.00822582577419566], Loss: 0.235274
Epoch: [94/484] Iter:[20/24], Time: 0.42, lr: [0.008217907443449699], Loss: 0.222570
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96133382 0.73135058 0.88487037 0.22245319 0.3829753  0.53159284
 0.55932407 0.68116175 0.89956316 0.497949   0.89639948 0.75669182
 0.49815161 0.90392833 0.34443218 0.35815432 0.27962997 0.42633906
 0.7139735 ] 0.6068565454201371
1 [0.96078132 0.7300271  0.88066123 0.19154704 0.37482821 0.53675816
 0.56866158 0.68569582 0.89861109 0.4951793  0.87781714 0.76226405
 0.50657271 0.91185288 0.37823714 0.45609028 0.29231252 0.45892466
 0.71837921] 0.6150106023021086
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.324, MeanIU:  0.6150, Best_mIoU:  0.6653
[0.96078132 0.7300271  0.88066123 0.19154704 0.37482821 0.53675816
 0.56866158 0.68569582 0.89861109 0.4951793  0.87781714 0.76226405
 0.50657271 0.91185288 0.37823714 0.45609028 0.29231252 0.45892466
 0.71837921]
Epoch: [95/484] Iter:[0/24], Time: 1.00, lr: [0.00821473987381412], Loss: 0.153562
Epoch: [95/484] Iter:[10/24], Time: 0.45, lr: [0.008206820355822783], Loss: 0.370772
Epoch: [95/484] Iter:[20/24], Time: 0.42, lr: [0.008198899988598738], Loss: 0.364961
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95528717 0.70387323 0.8807244  0.35366253 0.42011021 0.52303944
 0.52140349 0.66536565 0.89057028 0.43383715 0.91436311 0.74291519
 0.48536408 0.90571865 0.33415905 0.37467383 0.27634956 0.35718473
 0.71203196] 0.602664932220644
1 [0.95447132 0.69873527 0.87914783 0.3692308  0.44032774 0.53195339
 0.54457627 0.67945248 0.88683045 0.40766789 0.90776944 0.7474108
 0.50787154 0.91091338 0.37412632 0.46587631 0.26527228 0.39764904
 0.71619953] 0.6150253729584156
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.334, MeanIU:  0.6150, Best_mIoU:  0.6653
[0.95447132 0.69873527 0.87914783 0.3692308  0.44032774 0.53195339
 0.54457627 0.67945248 0.88683045 0.40766789 0.90776944 0.7474108
 0.50787154 0.91091338 0.37412632 0.46587631 0.26527228 0.39764904
 0.71619953]
Epoch: [96/484] Iter:[0/24], Time: 0.95, lr: [0.008195731603699413], Loss: 0.225554
Epoch: [96/484] Iter:[10/24], Time: 0.44, lr: [0.008187810045864378], Loss: 0.273120
Epoch: [96/484] Iter:[20/24], Time: 0.42, lr: [0.008179887636386121], Loss: 0.279284
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96010154 0.73697706 0.87476218 0.2136487  0.40764674 0.51188615
 0.55746478 0.66755327 0.89675887 0.45557898 0.92214795 0.75023887
 0.48041254 0.91095668 0.41648837 0.54868214 0.32773348 0.42898691
 0.70823297] 0.619803061338487
1 [0.9621065  0.73885559 0.87610813 0.18433578 0.43549379 0.5079559
 0.5643104  0.68111608 0.89871825 0.44858046 0.9194407  0.75827541
 0.50964969 0.91868841 0.44197658 0.65107291 0.39876817 0.48938888
 0.71507978] 0.6368379684144366
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.313, MeanIU:  0.6368, Best_mIoU:  0.6653
[0.9621065  0.73885559 0.87610813 0.18433578 0.43549379 0.5079559
 0.5643104  0.68111608 0.89871825 0.44858046 0.9194407  0.75827541
 0.50964969 0.91868841 0.44197658 0.65107291 0.39876817 0.48938888
 0.71507978]
Epoch: [97/484] Iter:[0/24], Time: 0.99, lr: [0.008176718433908946], Loss: 0.171121
Epoch: [97/484] Iter:[10/24], Time: 0.45, lr: [0.00816879483043582], Loss: 0.248059
Epoch: [97/484] Iter:[20/24], Time: 0.42, lr: [0.008160870372895859], Loss: 0.243522
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96660959 0.76331021 0.87848603 0.38718762 0.37170195 0.51542343
 0.51066268 0.67936917 0.89919324 0.45898755 0.9232119  0.74786072
 0.47131982 0.90371355 0.39728284 0.51834829 0.37463116 0.3732406
 0.71373058] 0.6239089965799741
1 [0.96705624 0.7667575  0.87801871 0.44472856 0.38345427 0.5270259
 0.51327592 0.69439278 0.89898232 0.40908279 0.92483177 0.75583939
 0.49780275 0.91285509 0.44865693 0.62584144 0.48698033 0.42173782
 0.71981898] 0.6461652344782829
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.302, MeanIU:  0.6462, Best_mIoU:  0.6653
[0.96705624 0.7667575  0.87801871 0.44472856 0.38345427 0.5270259
 0.51327592 0.69439278 0.89898232 0.40908279 0.92483177 0.75583939
 0.49780275 0.91285509 0.44865693 0.62584144 0.48698033 0.42173782
 0.71981898]
Epoch: [98/484] Iter:[0/24], Time: 0.96, lr: [0.008157700350514161], Loss: 0.267928
Epoch: [98/484] Iter:[10/24], Time: 0.44, lr: [0.008149774695577048], Loss: 0.190405
Epoch: [98/484] Iter:[20/24], Time: 0.42, lr: [0.008141848184136281], Loss: 0.230224
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94056541 0.65884717 0.88414516 0.20487572 0.43617345 0.53484025
 0.56420322 0.68470553 0.8981594  0.46855334 0.92821359 0.75844977
 0.49570497 0.91438466 0.38877306 0.57433293 0.3650174  0.43248031
 0.69940749] 0.6227280443328668
1 [0.93831122 0.6581335  0.88317979 0.13640098 0.46456088 0.55172603
 0.57762067 0.69418783 0.89937819 0.47900535 0.92921852 0.76439734
 0.52382458 0.92090909 0.49627628 0.67258248 0.4838199  0.45212859
 0.70567042] 0.6437542980513901
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.334, MeanIU:  0.6438, Best_mIoU:  0.6653
[0.93831122 0.6581335  0.88317979 0.13640098 0.46456088 0.55172603
 0.57762067 0.69418783 0.89937819 0.47900535 0.92921852 0.76439734
 0.52382458 0.92090909 0.49627628 0.67258248 0.4838199  0.45212859
 0.70567042]
Epoch: [99/484] Iter:[0/24], Time: 0.96, lr: [0.008138677339510715], Loss: 0.199051
Epoch: [99/484] Iter:[10/24], Time: 0.44, lr: [0.008130749627251958], Loss: 0.259793
Epoch: [99/484] Iter:[20/24], Time: 0.42, lr: [0.008122821056039424], Loss: 0.258941
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96728255 0.75979825 0.89124548 0.3501237  0.46815473 0.53799447
 0.55345924 0.67575544 0.9008533  0.47012707 0.91752898 0.75980094
 0.49644275 0.90952867 0.40873674 0.55806181 0.38242223 0.45808897
 0.69766261] 0.6401614704668718
1 [0.96689961 0.76267646 0.89191506 0.36820402 0.48827201 0.54778364
 0.57169141 0.6897334  0.90079405 0.44685395 0.92030195 0.76306986
 0.51765665 0.91459011 0.51220946 0.6529097  0.41091513 0.45787492
 0.69709396] 0.6569181761758643
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.281, MeanIU:  0.6569, Best_mIoU:  0.6653
[0.96689961 0.76267646 0.89191506 0.36820402 0.48827201 0.54778364
 0.57169141 0.6897334  0.90079405 0.44685395 0.92030195 0.76306986
 0.51765665 0.91459011 0.51220946 0.6529097  0.41091513 0.45787492
 0.69709396]
Epoch: [100/484] Iter:[0/24], Time: 0.99, lr: [0.008119649386817867], Loss: 0.139181
Epoch: [100/484] Iter:[10/24], Time: 0.44, lr: [0.008111719611347793], Loss: 0.336458
Epoch: [100/484] Iter:[20/24], Time: 0.42, lr: [0.008103788974460404], Loss: 0.311158
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96112951 0.73352842 0.87086245 0.2229368  0.38114023 0.507657
 0.5419911  0.6711706  0.88581776 0.44247586 0.90593174 0.7422673
 0.46301503 0.88621935 0.37293917 0.45338365 0.26229031 0.37544737
 0.70494066] 0.5992181215259709
1 [0.96059439 0.73536315 0.87009319 0.2159137  0.38207872 0.52111055
 0.56207359 0.68217136 0.88608088 0.4620938  0.91208798 0.7429959
 0.48056251 0.89321034 0.412409   0.52816544 0.25194619 0.37847724
 0.70452328] 0.6095763803787544
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.345, MeanIU:  0.6096, Best_mIoU:  0.6653
[0.96059439 0.73536315 0.87009319 0.2159137  0.38207872 0.52111055
 0.56207359 0.68217136 0.88608088 0.4620938  0.91208798 0.7429959
 0.48056251 0.89321034 0.412409   0.52816544 0.25194619 0.37847724
 0.70452328]
Epoch: [101/484] Iter:[0/24], Time: 0.98, lr: [0.008100616478277856], Loss: 0.172062
Epoch: [101/484] Iter:[10/24], Time: 0.44, lr: [0.008092684633674527], Loss: 0.317101
Epoch: [101/484] Iter:[20/24], Time: 0.42, lr: [0.008084751925176816], Loss: 0.308651
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.94468572 0.71921946 0.83578184 0.20735422 0.39823536 0.52610507
 0.54699084 0.66552115 0.89678094 0.46369615 0.92249888 0.73650435
 0.4853642  0.89634042 0.3641516  0.53758696 0.30536291 0.39745722
 0.67218052] 0.6064114630771451
1 [0.94185349 0.69924213 0.8191751  0.19654238 0.41518596 0.52252655
 0.53664316 0.64986152 0.8969847  0.45965211 0.92241642 0.7412945
 0.47839559 0.89468833 0.38085649 0.59088131 0.33362169 0.36759015
 0.67472554] 0.6064282692381592
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.370, MeanIU:  0.6064, Best_mIoU:  0.6653
[0.94185349 0.69924213 0.8191751  0.19654238 0.41518596 0.52252655
 0.53664316 0.64986152 0.8969847  0.45965211 0.92241642 0.7412945
 0.47839559 0.89468833 0.38085649 0.59088131 0.33362169 0.36759015
 0.67472554]
Epoch: [102/484] Iter:[0/24], Time: 0.99, lr: [0.008081578599655293], Loss: 0.155017
Epoch: [102/484] Iter:[10/24], Time: 0.45, lr: [0.008073644679964227], Loss: 0.253975
Epoch: [102/484] Iter:[20/24], Time: 0.42, lr: [0.008065709893888084], Loss: 0.254399
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95736918 0.71924937 0.88994825 0.2814485  0.45144265 0.53397402
 0.53540571 0.67886016 0.90113691 0.52023942 0.92608193 0.74522162
 0.40534261 0.90991332 0.42531766 0.54754958 0.32027429 0.33216127
 0.70498723] 0.6203117726035439
1 [0.95468126 0.72198057 0.88964028 0.27298602 0.46108885 0.54698743
 0.56409873 0.69573394 0.90309479 0.53197285 0.92808427 0.75470139
 0.44573745 0.91494986 0.51340047 0.64813307 0.35601726 0.38341196
 0.71073944] 0.6419705218336534
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.306, MeanIU:  0.6420, Best_mIoU:  0.6653
[0.95468126 0.72198057 0.88964028 0.27298602 0.46108885 0.54698743
 0.56409873 0.69573394 0.90309479 0.53197285 0.92808427 0.75470139
 0.44573745 0.91494986 0.51340047 0.64813307 0.35601726 0.38341196
 0.71073944]
Epoch: [103/484] Iter:[0/24], Time: 0.98, lr: [0.008062535736636513], Loss: 0.190044
Epoch: [103/484] Iter:[10/24], Time: 0.44, lr: [0.008054599735870435], Loss: 0.246611
Epoch: [103/484] Iter:[20/24], Time: 0.42, lr: [0.008046662866214837], Loss: 0.256730
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96092588 0.74528976 0.88555436 0.278951   0.42395567 0.53721311
 0.52834973 0.67834765 0.89867992 0.47673301 0.91753993 0.74187596
 0.40310857 0.86914329 0.30081356 0.36038927 0.24594026 0.4444447
 0.69397875] 0.5995386517565006
1 [0.96319332 0.7498517  0.8860227  0.30224851 0.44209716 0.54415377
 0.55172712 0.68967574 0.90006827 0.44294071 0.91998918 0.74808886
 0.42657848 0.87540183 0.28274987 0.42498012 0.22315728 0.4959861
 0.69570115] 0.6086637835632089
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.328, MeanIU:  0.6087, Best_mIoU:  0.6653
[0.96319332 0.7498517  0.8860227  0.30224851 0.44209716 0.54415377
 0.55172712 0.68967574 0.90006827 0.44294071 0.91998918 0.74808886
 0.42657848 0.87540183 0.28274987 0.42498012 0.22315728 0.4959861
 0.69570115]
Epoch: [104/484] Iter:[0/24], Time: 0.98, lr: [0.00804348787482895], Loss: 0.254971
Epoch: [104/484] Iter:[10/24], Time: 0.44, lr: [0.008035549786967515], Loss: 0.265950
Epoch: [104/484] Iter:[20/24], Time: 0.42, lr: [0.008027610827698262], Loss: 0.247899
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96417623 0.75895589 0.89398243 0.28120308 0.43025058 0.53447064
 0.56006105 0.68162073 0.89959597 0.49856647 0.91109872 0.75151427
 0.47647679 0.88387159 0.35507039 0.4550328  0.32428781 0.46421473
 0.70849348] 0.6227865075886239
1 [0.96157206 0.74442724 0.89596245 0.26067113 0.43030669 0.54105244
 0.56952909 0.69036841 0.89854845 0.50455756 0.90869113 0.74583034
 0.45705402 0.88523957 0.39163783 0.52167547 0.47118041 0.46946138
 0.70318411] 0.6342605137201871
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.302, MeanIU:  0.6343, Best_mIoU:  0.6653
[0.96157206 0.74442724 0.89596245 0.26067113 0.43030669 0.54105244
 0.56952909 0.69036841 0.89854845 0.50455756 0.90869113 0.74583034
 0.45705402 0.88523957 0.39163783 0.52167547 0.47118041 0.46946138
 0.70318411]
Epoch: [105/484] Iter:[0/24], Time: 0.98, lr: [0.008024434999760488], Loss: 0.288477
Epoch: [105/484] Iter:[10/24], Time: 0.44, lr: [0.008016494818750018], Loss: 0.232476
Epoch: [105/484] Iter:[20/24], Time: 0.42, lr: [0.008008553763799454], Loss: 0.222501
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96733689 0.77735511 0.89347835 0.31821081 0.46131948 0.53669018
 0.55682185 0.6838937  0.90113312 0.48164438 0.9217866  0.75357684
 0.4173732  0.89591727 0.36389048 0.51526056 0.20359126 0.40477355
 0.7145105 ] 0.6193981119477013
1 [0.96837433 0.78297401 0.892701   0.32506769 0.48820388 0.54307455
 0.56527264 0.69088697 0.90136549 0.47335034 0.92434544 0.76247192
 0.46158876 0.9100033  0.39341989 0.57478323 0.20105676 0.45291735
 0.72405076] 0.6334688585415567
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.285, MeanIU:  0.6335, Best_mIoU:  0.6653
[0.96837433 0.78297401 0.892701   0.32506769 0.48820388 0.54307455
 0.56527264 0.69088697 0.90136549 0.47335034 0.92434544 0.76247192
 0.46158876 0.9100033  0.39341989 0.57478323 0.20105676 0.45291735
 0.72405076]
Epoch: [106/484] Iter:[0/24], Time: 0.95, lr: [0.008005377096878813], Loss: 0.233186
Epoch: [106/484] Iter:[10/24], Time: 0.44, lr: [0.007997434816632013], Loss: 0.205997
Epoch: [106/484] Iter:[20/24], Time: 0.42, lr: [0.007989491659898769], Loss: 0.212203
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96789094 0.77968807 0.89207846 0.26576184 0.39397052 0.53300219
 0.52213742 0.68576457 0.90111678 0.51211932 0.91901208 0.75593644
 0.50044447 0.90483506 0.41037052 0.52340979 0.27433854 0.42117625
 0.70916514] 0.6248535994949571
1 [0.96969008 0.78534384 0.89467932 0.29863469 0.41130953 0.5396839
 0.54721784 0.70063848 0.9028698  0.531838   0.92723083 0.76347392
 0.51962466 0.91618164 0.46944156 0.54468914 0.34372512 0.46205865
 0.71043186] 0.6441454139261577
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.284, MeanIU:  0.6441, Best_mIoU:  0.6653
[0.96969008 0.78534384 0.89467932 0.29863469 0.41130953 0.5396839
 0.54721784 0.70063848 0.9028698  0.531838   0.92723083 0.76347392
 0.51962466 0.91618164 0.46944156 0.54468914 0.34372512 0.46205865
 0.71043186]
Epoch: [107/484] Iter:[0/24], Time: 0.95, lr: [0.007986314151550752], Loss: 0.262324
Epoch: [107/484] Iter:[10/24], Time: 0.44, lr: [0.007978369765946448], Loss: 0.189434
Epoch: [107/484] Iter:[20/24], Time: 0.42, lr: [0.007970424501295141], Loss: 0.189169
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97171142 0.78752463 0.89703671 0.31469332 0.43265764 0.54537614
 0.56520713 0.68304133 0.90380734 0.51881584 0.92140577 0.76565787
 0.48136833 0.91194644 0.38890111 0.52567465 0.27881612 0.47271696
 0.71619371] 0.6359238120949119
1 [0.97224934 0.79273638 0.89852528 0.35195856 0.45362373 0.55786909
 0.59300264 0.69311697 0.906473   0.54173775 0.92033508 0.77271897
 0.50257714 0.92174298 0.47769741 0.57151185 0.33545089 0.47732018
 0.71895514] 0.6557685470964775
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.268, MeanIU:  0.6558, Best_mIoU:  0.6653
[0.97224934 0.79273638 0.89852528 0.35195856 0.45362373 0.55786909
 0.59300264 0.69311697 0.906473   0.54173775 0.92033508 0.77271897
 0.50257714 0.92174298 0.47769741 0.57151185 0.33545089 0.47732018
 0.71895514]
Epoch: [108/484] Iter:[0/24], Time: 0.97, lr: [0.00796724614906161], Loss: 0.176217
Epoch: [108/484] Iter:[10/24], Time: 0.44, lr: [0.00795929965194446], Loss: 0.187942
Epoch: [108/484] Iter:[20/24], Time: 0.42, lr: [0.00795135227320543], Loss: 0.195094
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96871684 0.77510287 0.89146568 0.28986338 0.42541362 0.54410615
 0.56273928 0.69127127 0.90190123 0.5136809  0.91871983 0.75998231
 0.49171899 0.91188355 0.37777354 0.50355349 0.35802419 0.44189462
 0.70625829] 0.6333721074510109
1 [0.96827948 0.77616132 0.88926223 0.26711919 0.42088017 0.54957743
 0.5888849  0.6972641  0.90053437 0.50653247 0.91913714 0.7644213
 0.52251846 0.92038794 0.48155856 0.57859598 0.49987011 0.46436736
 0.71245767] 0.6540952729509689
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.285, MeanIU:  0.6541, Best_mIoU:  0.6653
[0.96827948 0.77616132 0.88926223 0.26711919 0.42088017 0.54957743
 0.5888849  0.6972641  0.90053437 0.50653247 0.91913714 0.7644213
 0.52251846 0.92038794 0.48155856 0.57859598 0.49987011 0.46436736
 0.71245767]
Epoch: [109/484] Iter:[0/24], Time: 0.98, lr: [0.0079481730746145], Loss: 0.148761
Epoch: [109/484] Iter:[10/24], Time: 0.45, lr: [0.007940224459794714], Loss: 0.198677
Epoch: [109/484] Iter:[20/24], Time: 0.42, lr: [0.007932274960763732], Loss: 0.214750
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96696658 0.77254378 0.89645675 0.35670721 0.45322747 0.50923282
 0.55576517 0.69376671 0.9035886  0.51637307 0.91458065 0.75499653
 0.50497572 0.88829577 0.39134629 0.46175107 0.371985   0.4742156
 0.69933171] 0.6361108685982811
1 [0.96838011 0.7766614  0.89870974 0.31696995 0.46616572 0.51191261
 0.58610068 0.70568885 0.90498903 0.51859148 0.92206894 0.75908953
 0.51972619 0.90427148 0.49231059 0.5588758  0.53991864 0.48841144
 0.69951041] 0.6599132944344617
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.279, MeanIU:  0.6599, Best_mIoU:  0.6653
[0.96838011 0.7766614  0.89870974 0.31696995 0.46616572 0.51191261
 0.58610068 0.70568885 0.90498903 0.51859148 0.92206894 0.75908953
 0.51972619 0.90427148 0.49231059 0.5588758  0.53991864 0.48841144
 0.69951041]
Epoch: [110/484] Iter:[0/24], Time: 0.98, lr: [0.007929094913329658], Loss: 0.191384
Epoch: [110/484] Iter:[10/24], Time: 0.45, lr: [0.007921144174582711], Loss: 0.188162
Epoch: [110/484] Iter:[20/24], Time: 0.42, lr: [0.007913192549020698], Loss: 0.193453
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96894387 0.78125231 0.89522811 0.31204116 0.44574858 0.54627845
 0.55838392 0.69387957 0.90500452 0.54081056 0.92356166 0.7532898
 0.48457471 0.90806602 0.39796934 0.45267799 0.39508928 0.44986875
 0.71575327] 0.6383379922708706
1 [0.9692002  0.78445204 0.89765941 0.38416396 0.46842248 0.55690424
 0.58999056 0.70626599 0.90700222 0.56470974 0.92697031 0.76309315
 0.51494976 0.92135529 0.55821461 0.56128408 0.56352744 0.48092418
 0.72142684] 0.6758166583228831
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.270, MeanIU:  0.6758, Best_mIoU:  0.6758
[0.9692002  0.78445204 0.89765941 0.38416396 0.46842248 0.55690424
 0.58999056 0.70626599 0.90700222 0.56470974 0.92697031 0.76309315
 0.51494976 0.92135529 0.55821461 0.56128408 0.56352744 0.48092418
 0.72142684]
Epoch: [111/484] Iter:[0/24], Time: 0.95, lr: [0.007910011650243763], Loss: 0.278036
Epoch: [111/484] Iter:[10/24], Time: 0.44, lr: [0.00790205878131011], Loss: 0.216010
Epoch: [111/484] Iter:[20/24], Time: 0.42, lr: [0.00789410502294284], Loss: 0.198832
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97111679 0.77947465 0.89972644 0.40582901 0.45667303 0.53097364
 0.56203685 0.69416442 0.90327513 0.42357029 0.92657058 0.7570063
 0.49643205 0.90712284 0.39769077 0.45041031 0.31041615 0.5169075
 0.70721929] 0.6366640016996844
1 [0.9722307  0.77912102 0.89957239 0.45647539 0.46853141 0.55021025
 0.59115565 0.70473301 0.90374478 0.41228508 0.926895   0.76352435
 0.51878933 0.92085538 0.43212754 0.5243624  0.33881708 0.51445055
 0.7041241 ] 0.6516844942831442
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.274, MeanIU:  0.6517, Best_mIoU:  0.6758
[0.9722307  0.77912102 0.89957239 0.45647539 0.46853141 0.55021025
 0.59115565 0.70473301 0.90374478 0.41228508 0.926895   0.76352435
 0.51878933 0.92085538 0.43212754 0.5243624  0.33881708 0.51445055
 0.7041241 ]
Epoch: [112/484] Iter:[0/24], Time: 0.97, lr: [0.007890923270309234], Loss: 0.177626
Epoch: [112/484] Iter:[10/24], Time: 0.44, lr: [0.007882968264894017], Loss: 0.212577
Epoch: [112/484] Iter:[20/24], Time: 0.42, lr: [0.007875012367411834], Loss: 0.225980
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97042241 0.77968326 0.88963913 0.24309444 0.39280496 0.54642578
 0.5685878  0.67902519 0.90293827 0.46095104 0.92473444 0.76412882
 0.48621558 0.89835659 0.37205804 0.44459754 0.43713363 0.50636593
 0.71573499] 0.6306788335009882
1 [0.97133875 0.78287716 0.88699534 0.18167747 0.39035977 0.55403691
 0.58526914 0.68829558 0.90342788 0.4585422  0.92742804 0.77387628
 0.52017162 0.90919202 0.44253217 0.54962865 0.56235254 0.51935734
 0.71779863] 0.6486924989776149
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.292, MeanIU:  0.6487, Best_mIoU:  0.6758
[0.97133875 0.78287716 0.88699534 0.18167747 0.39035977 0.55403691
 0.58526914 0.68829558 0.90342788 0.4585422  0.92742804 0.77387628
 0.52017162 0.90919202 0.44253217 0.54962865 0.56235254 0.51935734
 0.71779863]
Epoch: [113/484] Iter:[0/24], Time: 0.96, lr: [0.007871829758393536], Loss: 0.209808
Epoch: [113/484] Iter:[10/24], Time: 0.44, lr: [0.007863872610166286], Loss: 0.203287
Epoch: [113/484] Iter:[20/24], Time: 0.42, lr: [0.007855914567223801], Loss: 0.209459
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9708827  0.77368672 0.89396275 0.34709501 0.44904952 0.55079711
 0.57445627 0.69375478 0.90013591 0.47929081 0.93114874 0.76518646
 0.5065826  0.91124742 0.40438046 0.5493228  0.4457068  0.51531995
 0.70665344] 0.6509821194645615
1 [0.97144088 0.77801529 0.89432275 0.32784728 0.47201059 0.56228662
 0.59814964 0.70870763 0.90331263 0.50351325 0.93363647 0.76739419
 0.51781971 0.91970631 0.51478547 0.63969411 0.53445596 0.52685974
 0.70793953] 0.6727314763074269
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.272, MeanIU:  0.6727, Best_mIoU:  0.6758
[0.97144088 0.77801529 0.89432275 0.32784728 0.47201059 0.56228662
 0.59814964 0.70870763 0.90331263 0.50351325 0.93363647 0.76739419
 0.51781971 0.91970631 0.51478547 0.63969411 0.53445596 0.52685974
 0.70793953]
Epoch: [114/484] Iter:[0/24], Time: 0.97, lr: [0.007852731099278462], Loss: 0.367012
Epoch: [114/484] Iter:[10/24], Time: 0.44, lr: [0.007844771801872812], Loss: 0.224669
Epoch: [114/484] Iter:[20/24], Time: 0.42, lr: [0.007836811607088597], Loss: 0.222002
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97038317 0.76263766 0.89637916 0.22818943 0.45039517 0.55346657
 0.57861374 0.70178436 0.90166812 0.4596844  0.92187171 0.76424074
 0.51985517 0.91591097 0.39664463 0.59913297 0.41585036 0.35754387
 0.69320174] 0.6361817854562496
1 [0.97134606 0.76503022 0.89972271 0.243073   0.47526021 0.56232455
 0.60280158 0.70660723 0.90306171 0.47024361 0.92288582 0.76778538
 0.53014141 0.92165239 0.4428332  0.61427842 0.52900332 0.36960155
 0.69930775] 0.6524715856037075
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.277, MeanIU:  0.6525, Best_mIoU:  0.6758
[0.97134606 0.76503022 0.89972271 0.243073   0.47526021 0.56232455
 0.60280158 0.70660723 0.90306171 0.47024361 0.92288582 0.76778538
 0.53014141 0.92165239 0.4428332  0.61427842 0.52900332 0.36960155
 0.69930775]
Epoch: [115/484] Iter:[0/24], Time: 0.97, lr: [0.007833627277659422], Loss: 0.361800
Epoch: [115/484] Iter:[10/24], Time: 0.44, lr: [0.00782566582467279], Loss: 0.224986
Epoch: [115/484] Iter:[20/24], Time: 0.42, lr: [0.007817703471629092], Loss: 0.232889
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95986794 0.71809245 0.86691095 0.25255242 0.37193129 0.50827332
 0.53860626 0.68572283 0.89643719 0.43005922 0.91998447 0.75096758
 0.50277206 0.90145514 0.32989884 0.28082248 0.22242817 0.49970119
 0.70511632] 0.5969263212246302
1 [0.95956796 0.7189592  0.86807355 0.31264289 0.38781209 0.51610583
 0.57007171 0.69257885 0.89989917 0.45819647 0.92707992 0.75616983
 0.51111488 0.9097339  0.40743684 0.33436168 0.30384523 0.5203455
 0.71232452] 0.6192800009159718
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.328, MeanIU:  0.6193, Best_mIoU:  0.6758
[0.95956796 0.7189592  0.86807355 0.31264289 0.38781209 0.51610583
 0.57007171 0.69257885 0.89989917 0.45819647 0.92707992 0.75616983
 0.51111488 0.9097339  0.40743684 0.33436168 0.30384523 0.5203455
 0.71232452]
Epoch: [116/484] Iter:[0/24], Time: 0.96, lr: [0.00781451827814472], Loss: 0.214058
Epoch: [116/484] Iter:[10/24], Time: 0.44, lr: [0.00780655466313801], Loss: 0.253886
Epoch: [116/484] Iter:[20/24], Time: 0.42, lr: [0.00779859014538043], Loss: 0.238463
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97029171 0.77083049 0.89417359 0.26940155 0.42137615 0.545365
 0.56011107 0.68511787 0.90456986 0.50283844 0.93054935 0.76737871
 0.50612154 0.90829794 0.33652802 0.32588326 0.2984325  0.46080028
 0.71862035] 0.6198256657862401
1 [0.97134822 0.77768794 0.89624317 0.29431715 0.43760066 0.53769096
 0.57675752 0.69086159 0.90685206 0.51581813 0.93096944 0.77386046
 0.53103064 0.91624634 0.42944816 0.36703528 0.32935233 0.50009952
 0.72588983] 0.6373215468087757
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.280, MeanIU:  0.6373, Best_mIoU:  0.6758
[0.97134822 0.77768794 0.89624317 0.29431715 0.43760066 0.53769096
 0.57675752 0.69086159 0.90685206 0.51581813 0.93096944 0.77386046
 0.53103064 0.91624634 0.42944816 0.36703528 0.32935233 0.50009952
 0.72588983]
Epoch: [117/484] Iter:[0/24], Time: 0.95, lr: [0.007795404085254805], Loss: 0.188723
Epoch: [117/484] Iter:[10/24], Time: 0.44, lr: [0.007787438301752103], Loss: 0.209633
Epoch: [117/484] Iter:[20/24], Time: 0.42, lr: [0.007779471612789294], Loss: 0.205932
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97018228 0.77170203 0.89897624 0.32572362 0.4640314  0.53944611
 0.57052381 0.70015685 0.90437456 0.50355108 0.92922338 0.76892609
 0.52184133 0.91047416 0.38764782 0.53141679 0.45160304 0.50748254
 0.71236417] 0.6510340695439658
1 [0.97091577 0.7774353  0.90011699 0.35519404 0.49828719 0.54164234
 0.59355429 0.71015282 0.90806819 0.52483194 0.93089896 0.77283542
 0.53511562 0.92029543 0.49621037 0.61583293 0.49867373 0.52653565
 0.71926951] 0.6734666583709146
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.266, MeanIU:  0.6735, Best_mIoU:  0.6758
[0.97091577 0.7774353  0.90011699 0.35519404 0.49828719 0.54164234
 0.59355429 0.71015282 0.90806819 0.52483194 0.93089896 0.77283542
 0.53511562 0.92029543 0.49621037 0.61583293 0.49867373 0.52653565
 0.71926951]
Epoch: [118/484] Iter:[0/24], Time: 0.96, lr: [0.007776284683421543], Loss: 0.094581
Epoch: [118/484] Iter:[10/24], Time: 0.44, lr: [0.007768316724909803], Loss: 0.191809
Epoch: [118/484] Iter:[20/24], Time: 0.42, lr: [0.007760347858213155], Loss: 0.195806
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96897561 0.76441691 0.89610833 0.35627191 0.44929061 0.54751754
 0.56416533 0.69006794 0.90434478 0.49209585 0.92395881 0.76125692
 0.48408887 0.90898121 0.40731989 0.5414844  0.47849688 0.46387826
 0.71710034] 0.6484115990907359
1 [0.96851132 0.76494739 0.89897281 0.41222159 0.4896915  0.55879571
 0.58702909 0.69861098 0.90605037 0.49591691 0.92582247 0.770837
 0.51524666 0.91831956 0.50209092 0.62091179 0.54284425 0.50489048
 0.7249948 ] 0.674037136820196
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.270, MeanIU:  0.6740, Best_mIoU:  0.6758
[0.96851132 0.76494739 0.89897281 0.41222159 0.4896915  0.55879571
 0.58702909 0.69861098 0.90605037 0.49591691 0.92582247 0.770837
 0.51524666 0.91831956 0.50209092 0.62091179 0.54284425 0.50489048
 0.7249948 ]
Epoch: [119/484] Iter:[0/24], Time: 0.96, lr: [0.0077571600569874615], Loss: 0.297114
Epoch: [119/484] Iter:[10/24], Time: 0.44, lr: [0.007749189916916193], Loss: 0.203925
Epoch: [119/484] Iter:[20/24], Time: 0.42, lr: [0.0077412188659195125], Loss: 0.215684
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97119005 0.76600433 0.8990616  0.32285993 0.43203836 0.555475
 0.57781071 0.69296146 0.90315333 0.47530121 0.92465738 0.7609264
 0.50430929 0.90958243 0.38433925 0.51379874 0.48322776 0.48081859
 0.72001233] 0.6461856913061061
1 [0.97096925 0.7675408  0.89967504 0.30071801 0.45039898 0.56902679
 0.59961971 0.70211252 0.90253371 0.44504404 0.92651654 0.76793967
 0.51706624 0.91667099 0.47528156 0.58034371 0.59165098 0.50651091
 0.72226169] 0.6637832167543598
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.274, MeanIU:  0.6638, Best_mIoU:  0.6758
[0.97096925 0.7675408  0.89967504 0.30071801 0.45039898 0.56902679
 0.59961971 0.70211252 0.90253371 0.44504404 0.92651654 0.76793967
 0.51706624 0.91667099 0.47528156 0.58034371 0.59165098 0.50651091
 0.72226169]
Epoch: [120/484] Iter:[0/24], Time: 0.98, lr: [0.007738030190204992], Loss: 0.159904
Epoch: [120/484] Iter:[10/24], Time: 0.44, lr: [0.007730057861985933], Loss: 0.186916
Epoch: [120/484] Iter:[20/24], Time: 0.42, lr: [0.007722084620085131], Loss: 0.209657
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95732219 0.72206018 0.89148005 0.35600225 0.4370603  0.53118653
 0.57109903 0.69144442 0.89784471 0.52241851 0.92309633 0.76095639
 0.49096393 0.90093688 0.36703016 0.48833438 0.46308793 0.4360705
 0.70809461] 0.6377099609802821
1 [0.95555167 0.72295078 0.89326773 0.42134459 0.47403222 0.53955666
 0.58977147 0.70169178 0.89910095 0.54750255 0.92285932 0.76627126
 0.5103672  0.91109881 0.43457823 0.59425504 0.61339372 0.46471724
 0.71389996] 0.6671690097911782
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.299, MeanIU:  0.6672, Best_mIoU:  0.6758
[0.95555167 0.72295078 0.89326773 0.42134459 0.47403222 0.53955666
 0.58977147 0.70169178 0.89910095 0.54750255 0.92285932 0.76627126
 0.5103672  0.91109881 0.43457823 0.59425504 0.61339372 0.46471724
 0.71389996]
Epoch: [121/484] Iter:[0/24], Time: 0.98, lr: [0.007718895067235705], Loss: 0.179205
Epoch: [121/484] Iter:[10/24], Time: 0.44, lr: [0.007710920544242506], Loss: 0.196581
Epoch: [121/484] Iter:[20/24], Time: 0.42, lr: [0.007702945104795266], Loss: 0.216482
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96436366 0.75342966 0.89074111 0.24656839 0.42184977 0.53155568
 0.53413039 0.6984041  0.90368297 0.49766744 0.9152597  0.74790388
 0.45057102 0.91458119 0.30661635 0.57997398 0.40585627 0.37518595
 0.71273469] 0.6237408520845552
1 [0.9653947  0.75734678 0.88703969 0.19560574 0.42321276 0.53058978
 0.56470423 0.7123603  0.90346775 0.50743413 0.91657773 0.75671793
 0.49073835 0.92221675 0.30256596 0.63791255 0.50128327 0.40390616
 0.71846596] 0.6367126586169567
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.299, MeanIU:  0.6367, Best_mIoU:  0.6758
[0.9653947  0.75734678 0.88703969 0.19560574 0.42321276 0.53058978
 0.56470423 0.7123603  0.90346775 0.50743413 0.91657773 0.75671793
 0.49073835 0.92221675 0.30256596 0.63791255 0.50128327 0.40390616
 0.71846596]
Epoch: [122/484] Iter:[0/24], Time: 0.99, lr: [0.007699754672149522], Loss: 0.249665
Epoch: [122/484] Iter:[10/24], Time: 0.45, lr: [0.00769177794771742], Loss: 0.219381
Epoch: [122/484] Iter:[20/24], Time: 0.42, lr: [0.007683800304042877], Loss: 0.236078
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96699844 0.75812864 0.88993861 0.21832426 0.43615098 0.52796179
 0.53860742 0.68079437 0.89636203 0.51427846 0.88233311 0.74744611
 0.48123791 0.89081414 0.36047505 0.45938191 0.40387178 0.42369279
 0.70835483] 0.6202711923424354
1 [0.96645043 0.7596018  0.88832016 0.2102026  0.45954162 0.53406791
 0.54572078 0.69388678 0.89556659 0.53718375 0.86693205 0.75343757
 0.5000724  0.87477414 0.37654276 0.45699326 0.48783304 0.42152175
 0.71148222] 0.6284279787523614
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.312, MeanIU:  0.6284, Best_mIoU:  0.6758
[0.96645043 0.7596018  0.88832016 0.2102026  0.45954162 0.53406791
 0.54572078 0.69388678 0.89556659 0.53718375 0.86693205 0.75343757
 0.5000724  0.87477414 0.37654276 0.45699326 0.48783304 0.42152175
 0.71148222]
Epoch: [123/484] Iter:[0/24], Time: 0.97, lr: [0.007680608988923949], Loss: 0.390157
Epoch: [123/484] Iter:[10/24], Time: 0.45, lr: [0.007672630056349435], Loss: 0.213649
Epoch: [123/484] Iter:[20/24], Time: 0.42, lr: [0.00766465020172784], Loss: 0.219991
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97019331 0.76944363 0.89311835 0.29298913 0.42408308 0.54962903
 0.56842092 0.69308931 0.90369444 0.48388123 0.92553    0.74639494
 0.451738   0.91226217 0.38320626 0.55801292 0.41657105 0.41541935
 0.69834517] 0.6345274896560037
1 [0.97138634 0.77702381 0.89083322 0.26752057 0.4407901  0.5524835
 0.58958787 0.70485143 0.90119604 0.47324664 0.92621396 0.75530151
 0.49284115 0.91785011 0.49378443 0.63093937 0.50411491 0.41454689
 0.70183671] 0.6529657133622743
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.283, MeanIU:  0.6530, Best_mIoU:  0.6758
[0.97138634 0.77702381 0.89083322 0.26752057 0.4407901  0.5524835
 0.58958787 0.70485143 0.90119604 0.47324664 0.92621396 0.75530151
 0.49284115 0.91785011 0.49378443 0.63093937 0.50411491 0.41454689
 0.70183671]
Epoch: [124/484] Iter:[0/24], Time: 0.99, lr: [0.007661458001443266], Loss: 0.219511
Epoch: [124/484] Iter:[10/24], Time: 0.45, lr: [0.007653476853983756], Loss: 0.215372
Epoch: [124/484] Iter:[20/24], Time: 0.42, lr: [0.00764549478165614], Loss: 0.214061
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97013385 0.76793037 0.89755398 0.34059613 0.4486811  0.55748398
 0.56521168 0.69874941 0.90349956 0.51439114 0.92813224 0.75379352
 0.44113695 0.91277434 0.3969354  0.53447805 0.32792443 0.41505317
 0.71012916] 0.6360309717095399
1 [0.9694582  0.7691391  0.89870575 0.38046655 0.47720869 0.56793289
 0.59020233 0.71484967 0.9072256  0.52260004 0.92765366 0.76509338
 0.47489135 0.92394959 0.52095738 0.57252529 0.30677606 0.41968972
 0.71019875] 0.6536591583954126
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.270, MeanIU:  0.6537, Best_mIoU:  0.6758
[0.9694582  0.7691391  0.89870575 0.38046655 0.47720869 0.56793289
 0.59020233 0.71484967 0.9072256  0.52260004 0.92765366 0.76509338
 0.47489135 0.92394959 0.52095738 0.57252529 0.30677606 0.41968972
 0.71019875]
Epoch: [125/484] Iter:[0/24], Time: 0.95, lr: [0.007642301693497736], Loss: 0.142804
Epoch: [125/484] Iter:[10/24], Time: 0.44, lr: [0.007634318324371229], Loss: 0.177855
Epoch: [125/484] Iter:[20/24], Time: 0.41, lr: [0.00762633402753907], Loss: 0.205823
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95168026 0.65246624 0.89595524 0.28612126 0.462011   0.55152168
 0.57893442 0.70664964 0.90427942 0.46774183 0.92826614 0.75947753
 0.48896788 0.90623853 0.39830567 0.4354069  0.44256835 0.48714918
 0.70438923] 0.632006862103842
1 [0.95230507 0.65942557 0.89515969 0.26727125 0.48938103 0.56317629
 0.59798786 0.72009689 0.90443678 0.46924634 0.92678913 0.76435105
 0.51853221 0.91770495 0.51862384 0.51805056 0.52251601 0.50319576
 0.7036788 ] 0.6532594260853012
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.319, MeanIU:  0.6533, Best_mIoU:  0.6758
[0.95230507 0.65942557 0.89515969 0.26727125 0.48938103 0.56317629
 0.59798786 0.72009689 0.90443678 0.46924634 0.92678913 0.76435105
 0.51853221 0.91770495 0.51862384 0.51805056 0.52251601 0.50319576
 0.7036788 ]
Epoch: [126/484] Iter:[0/24], Time: 0.97, lr: [0.007623140048782783], Loss: 0.112684
Epoch: [126/484] Iter:[10/24], Time: 0.44, lr: [0.007615154451167525], Loss: 0.224031
Epoch: [126/484] Iter:[20/24], Time: 0.42, lr: [0.007607167922992398], Loss: 0.221832
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96371258 0.72985442 0.88874587 0.25806546 0.43644425 0.53049199
 0.55957832 0.67954867 0.9034739  0.48446622 0.92657118 0.76085022
 0.50509616 0.89525719 0.40142725 0.5181967  0.34352913 0.49450444
 0.71413096] 0.6312602585925923
1 [0.96359146 0.73173536 0.8876195  0.25067354 0.46446133 0.52879881
 0.57248258 0.69326932 0.90562584 0.49978441 0.92778716 0.7675423
 0.51307461 0.91358741 0.45416237 0.62559524 0.38543397 0.51107799
 0.7195205 ] 0.6482012471652782
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.300, MeanIU:  0.6482, Best_mIoU:  0.6758
[0.96359146 0.73173536 0.8876195  0.25067354 0.46446133 0.52879881
 0.57248258 0.69326932 0.90562584 0.49978441 0.92778716 0.7675423
 0.51307461 0.91358741 0.45416237 0.62559524 0.38543397 0.51107799
 0.7195205 ]
Epoch: [127/484] Iter:[0/24], Time: 0.93, lr: [0.007603973050898182], Loss: 0.203282
Epoch: [127/484] Iter:[10/24], Time: 0.44, lr: [0.007595985217932315], Loss: 0.192759
Epoch: [127/484] Iter:[20/24], Time: 0.41, lr: [0.007587996451535555], Loss: 0.190730
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96935926 0.76778924 0.8942772  0.31294596 0.45225091 0.55594704
 0.57115395 0.6808451  0.90533474 0.51488962 0.91567056 0.76549254
 0.49780923 0.91502904 0.38225035 0.59082315 0.38620103 0.47685323
 0.71925745] 0.6460094525518768
1 [0.96931704 0.76953816 0.8938393  0.34623034 0.47352811 0.56316676
 0.58032322 0.68782993 0.90681914 0.52772629 0.91231779 0.77383768
 0.52118386 0.92078036 0.38993996 0.68848613 0.46143014 0.49600391
 0.72638284] 0.6636147882293398
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.276, MeanIU:  0.6636, Best_mIoU:  0.6758
[0.96931704 0.76953816 0.8938393  0.34623034 0.47352811 0.56316676
 0.58032322 0.68782993 0.90681914 0.52772629 0.91231779 0.77383768
 0.52118386 0.92078036 0.38993996 0.68848613 0.46143014 0.49600391
 0.72638284]
Epoch: [128/484] Iter:[0/24], Time: 0.95, lr: [0.007584800683347218], Loss: 0.131508
Epoch: [128/484] Iter:[10/24], Time: 0.44, lr: [0.007576810608128436], Loss: 0.206393
Epoch: [128/484] Iter:[20/24], Time: 0.42, lr: [0.007568819596590777], Loss: 0.196103
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96908662 0.76220762 0.90210819 0.31057184 0.47368378 0.56273419
 0.58588592 0.69600737 0.89832773 0.47743549 0.93038998 0.75312547
 0.4505886  0.90585925 0.37595855 0.50761444 0.40683057 0.47783546
 0.70425964] 0.6395005637922799
1 [0.96741752 0.76104572 0.9037206  0.33596109 0.49670291 0.57397207
 0.59708979 0.70908398 0.90307536 0.5037255  0.93096346 0.76353706
 0.48784424 0.91682873 0.44010995 0.60687332 0.51177909 0.48598485
 0.70479538] 0.6631847688631768
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.274, MeanIU:  0.6632, Best_mIoU:  0.6758
[0.96741752 0.76104572 0.9037206  0.33596109 0.49670291 0.57397207
 0.59708979 0.70908398 0.90307536 0.5037255  0.93096346 0.76353706
 0.48784424 0.91682873 0.44010995 0.60687332 0.51177909 0.48598485
 0.70479538]
Epoch: [129/484] Iter:[0/24], Time: 0.96, lr: [0.007565622929535854], Loss: 0.191453
Epoch: [129/484] Iter:[10/24], Time: 0.44, lr: [0.007557630605121043], Loss: 0.204847
Epoch: [129/484] Iter:[20/24], Time: 0.42, lr: [0.007549637341482268], Loss: 0.191021
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97102412 0.76909181 0.89994458 0.34031943 0.4521473  0.54365163
 0.58359558 0.69170999 0.90660663 0.50893445 0.92562554 0.7604381
 0.50084417 0.89661693 0.36583421 0.52324612 0.36308395 0.45269032
 0.70810415] 0.6401846842322658
1 [0.97109329 0.77064749 0.90147162 0.36708341 0.4663365  0.55449812
 0.59528847 0.70166941 0.90657977 0.50515128 0.92132146 0.76219706
 0.50706983 0.89902744 0.42697647 0.57188286 0.37557945 0.47566097
 0.71461406] 0.6523236293974339
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.278, MeanIU:  0.6523, Best_mIoU:  0.6758
[0.97109329 0.77064749 0.90147162 0.36708341 0.4663365  0.55449812
 0.59528847 0.70166941 0.90657977 0.50515128 0.92132146 0.76219706
 0.50706983 0.89902744 0.42697647 0.57188286 0.37557945 0.47566097
 0.71461406]
Epoch: [130/484] Iter:[0/24], Time: 0.96, lr: [0.007546439772771869], Loss: 0.164565
Epoch: [130/484] Iter:[10/24], Time: 0.44, lr: [0.007538445192176764], Loss: 0.162593
Epoch: [130/484] Iter:[20/24], Time: 0.42, lr: [0.007530449669435347], Loss: 0.212160
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.91289144 0.58801795 0.88024271 0.31418875 0.4352358  0.53373708
 0.5607372  0.67130422 0.89725762 0.48643075 0.92092892 0.74279168
 0.4999518  0.88812453 0.32672494 0.34144491 0.32060015 0.40493517
 0.67823519] 0.600198989924227
1 [0.91002964 0.5823039  0.87763735 0.2859763  0.45070546 0.53786001
 0.5775502  0.68441266 0.90109776 0.50062406 0.92362365 0.74115458
 0.50602549 0.89082597 0.37617429 0.40631292 0.37476072 0.39373221
 0.68217584] 0.6106833161346426
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.397, MeanIU:  0.6107, Best_mIoU:  0.6758
[0.91002964 0.5823039  0.87763735 0.2859763  0.45070546 0.53786001
 0.5775502  0.68441266 0.90109776 0.50062406 0.92362365 0.74115458
 0.50602549 0.89082597 0.37617429 0.40631292 0.37476072 0.39373221
 0.68217584]
Epoch: [131/484] Iter:[0/24], Time: 0.95, lr: [0.007527251196264013], Loss: 0.260197
Epoch: [131/484] Iter:[10/24], Time: 0.44, lr: [0.007519254352462823], Loss: 0.280173
Epoch: [131/484] Iter:[20/24], Time: 0.42, lr: [0.0075112565635755656], Loss: 0.250171
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96538464 0.75623109 0.89144309 0.29840266 0.45786838 0.54590683
 0.53842884 0.68143228 0.90118048 0.47279246 0.92757145 0.74867052
 0.45679137 0.89233668 0.38124103 0.44267024 0.36377986 0.45153604
 0.70989247] 0.6254505480316537
1 [0.96511035 0.75849694 0.88901998 0.26759504 0.47107239 0.55505736
 0.56522816 0.69983515 0.90223554 0.48333577 0.92785987 0.75501441
 0.4822302  0.9053736  0.44983166 0.60907236 0.36312111 0.45610863
 0.70378331] 0.6425990444678927
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.291, MeanIU:  0.6426, Best_mIoU:  0.6758
[0.96511035 0.75849694 0.88901998 0.26759504 0.47107239 0.55505736
 0.56522816 0.69983515 0.90223554 0.48333577 0.92785987 0.75501441
 0.4822302  0.9053736  0.44983166 0.60907236 0.36312111 0.45610863
 0.70378331]
Epoch: [132/484] Iter:[0/24], Time: 1.00, lr: [0.007508057183121132], Loss: 0.289037
Epoch: [132/484] Iter:[10/24], Time: 0.45, lr: [0.007500058069046176], Loss: 0.234007
Epoch: [132/484] Iter:[20/24], Time: 0.42, lr: [0.007492058006927841], Loss: 0.225454
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96829643 0.77412433 0.89420561 0.27777277 0.46531688 0.55605155
 0.57176124 0.6770938  0.90132732 0.45570284 0.91522443 0.75636187
 0.50972732 0.90085103 0.39282172 0.4605712  0.39058162 0.4651441
 0.71211629] 0.6339501238487857
1 [0.96955285 0.77668917 0.89169668 0.27751736 0.48482349 0.56516175
 0.57498987 0.68973591 0.90432357 0.42664852 0.91345822 0.76267492
 0.52947094 0.91219943 0.41359152 0.5861128  0.41656335 0.48286297
 0.71405627] 0.6469541895122499
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.284, MeanIU:  0.6470, Best_mIoU:  0.6758
[0.96955285 0.77668917 0.89169668 0.27751736 0.48482349 0.56516175
 0.57498987 0.68973591 0.90432357 0.42664852 0.91345822 0.76267492
 0.52947094 0.91219943 0.41359152 0.5861128  0.41656335 0.48286297
 0.71405627]
Epoch: [133/484] Iter:[0/24], Time: 0.95, lr: [0.0074888577163512784], Loss: 0.198462
Epoch: [133/484] Iter:[10/24], Time: 0.44, lr: [0.007480856324892625], Loss: 0.208604
Epoch: [133/484] Iter:[20/24], Time: 0.42, lr: [0.007472853982415559], Loss: 0.208488
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96895538 0.77759209 0.8974352  0.26832392 0.47636424 0.54833637
 0.57843644 0.69411884 0.90390683 0.51864906 0.92344921 0.76403423
 0.49367713 0.9097289  0.3935042  0.49611023 0.27161002 0.42861049
 0.70599069] 0.6325701818506386
1 [0.9695917  0.78128574 0.89763705 0.26920162 0.49600923 0.56108741
 0.59266212 0.70840251 0.90621589 0.53037325 0.92460192 0.77315227
 0.52929255 0.91732617 0.45826206 0.57283292 0.20752805 0.45254482
 0.71665555] 0.6455085692977875
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.276, MeanIU:  0.6455, Best_mIoU:  0.6758
[0.9695917  0.78128574 0.89763705 0.26920162 0.49600923 0.56108741
 0.59266212 0.70840251 0.90621589 0.53037325 0.92460192 0.77315227
 0.52929255 0.91732617 0.45826206 0.57283292 0.20752805 0.45254482
 0.71665555]
Epoch: [134/484] Iter:[0/24], Time: 0.99, lr: [0.00746965277886083], Loss: 0.107419
Epoch: [134/484] Iter:[10/24], Time: 0.45, lr: [0.007461649102865918], Loss: 0.213829
Epoch: [134/484] Iter:[20/24], Time: 0.42, lr: [0.007453644472859683], Loss: 0.206497
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96990793 0.75709825 0.89341111 0.24335856 0.44115075 0.54389008
 0.55288431 0.67513728 0.90480187 0.44047616 0.93250269 0.76580075
 0.49453879 0.91036605 0.37564029 0.48251511 0.30668914 0.45957771
 0.716245  ] 0.6245258849876496
1 [0.97014865 0.75533078 0.89271763 0.24734939 0.45419559 0.55704295
 0.58500627 0.68053135 0.90771102 0.42911911 0.93476422 0.77430484
 0.51921249 0.91539765 0.4240945  0.53401373 0.22265933 0.47061956
 0.71903559] 0.6312239290310333
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.289, MeanIU:  0.6312, Best_mIoU:  0.6758
[0.97014865 0.75533078 0.89271763 0.24734939 0.45419559 0.55704295
 0.58500627 0.68053135 0.90771102 0.42911911 0.93476422 0.77430484
 0.51921249 0.91539765 0.4240945  0.53401373 0.22265933 0.47061956
 0.71903559]
Epoch: [135/484] Iter:[0/24], Time: 1.00, lr: [0.007450442353453598], Loss: 0.186969
Epoch: [135/484] Iter:[10/24], Time: 0.45, lr: [0.007442436385726851], Loss: 0.215511
Epoch: [135/484] Iter:[20/24], Time: 0.42, lr: [0.007434429460977843], Loss: 0.210957
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96971453 0.76624428 0.8961723  0.29989689 0.46783318 0.54711562
 0.5815242  0.68313208 0.90556999 0.47908968 0.92751954 0.75512706
 0.43989818 0.90961815 0.37299849 0.4780413  0.34861582 0.40649658
 0.70339708] 0.6283160504699569
1 [0.96964278 0.7699387  0.89626274 0.29791891 0.48691931 0.55524431
 0.60220611 0.69529664 0.90659805 0.48924914 0.92839587 0.76455451
 0.46816523 0.9183268  0.43554341 0.53715025 0.2960368  0.41818585
 0.71233401] 0.6393668110591695
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.280, MeanIU:  0.6394, Best_mIoU:  0.6758
[0.96964278 0.7699387  0.89626274 0.29791891 0.48691931 0.55524431
 0.60220611 0.69529664 0.90659805 0.48924914 0.92839587 0.76455451
 0.46816523 0.9183268  0.43554341 0.53715025 0.2960368  0.41818585
 0.71233401]
Epoch: [136/484] Iter:[0/24], Time: 0.96, lr: [0.007431226422829892], Loss: 0.197790
Epoch: [136/484] Iter:[10/24], Time: 0.44, lr: [0.007423218156132346], Loss: 0.180331
Epoch: [136/484] Iter:[20/24], Time: 0.42, lr: [0.0074152089293834066], Loss: 0.186154
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96547867 0.75237731 0.89738767 0.40406873 0.43394235 0.56220187
 0.56844806 0.70438492 0.90637906 0.49094675 0.93063503 0.76498571
 0.50098503 0.91340947 0.40202744 0.43449779 0.26755334 0.44676581
 0.71541442] 0.6348362847244707
1 [0.96333952 0.74497606 0.89604519 0.44015341 0.43863594 0.57141583
 0.59444679 0.71805895 0.90630642 0.47354314 0.93033352 0.77293925
 0.52743747 0.92047271 0.44306402 0.46649235 0.1816472  0.46689953
 0.72179105] 0.6409472820665041
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.282, MeanIU:  0.6409, Best_mIoU:  0.6758
[0.96333952 0.74497606 0.89604519 0.44015341 0.43863594 0.57141583
 0.59444679 0.71805895 0.90630642 0.47354314 0.93033352 0.77293925
 0.52743747 0.92047271 0.44306402 0.46649235 0.1816472  0.46689953
 0.72179105]
Epoch: [137/484] Iter:[0/24], Time: 0.96, lr: [0.007412004969585618], Loss: 0.099099
Epoch: [137/484] Iter:[10/24], Time: 0.44, lr: [0.007403994396634532], Loss: 0.210405
Epoch: [137/484] Iter:[20/24], Time: 0.42, lr: [0.007395982860584562], Loss: 0.213194
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96987163 0.77624801 0.89062392 0.27945897 0.43982034 0.55190858
 0.57218683 0.69725879 0.90489899 0.51969904 0.92559899 0.76531094
 0.51371324 0.91698338 0.36815799 0.42170502 0.08347296 0.47045781
 0.72197062] 0.6204918973314445
1 [0.96944704 0.7797126  0.89002938 0.26612324 0.44554821 0.56451057
 0.59291772 0.71066107 0.90620814 0.52939481 0.92590423 0.77205448
 0.52282772 0.92139044 0.42439556 0.45466278 0.05557655 0.48278644
 0.72801499] 0.628535050971135
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.287, MeanIU:  0.6285, Best_mIoU:  0.6758
[0.96944704 0.7797126  0.89002938 0.26612324 0.44554821 0.56451057
 0.59291772 0.71066107 0.90620814 0.52939481 0.92590423 0.77205448
 0.52282772 0.92139044 0.42439556 0.45466278 0.05557655 0.48278644
 0.72801499]
Epoch: [138/484] Iter:[0/24], Time: 0.95, lr: [0.00739277797621134], Loss: 0.104052
Epoch: [138/484] Iter:[10/24], Time: 0.44, lr: [0.007384765089679799], Loss: 0.163841
Epoch: [138/484] Iter:[20/24], Time: 0.42, lr: [0.0073767512369833635], Loss: 0.179436
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97057657 0.78610086 0.89716136 0.31867112 0.45347588 0.54521121
 0.59337911 0.70200814 0.90925642 0.54858995 0.9328928  0.76293358
 0.50608081 0.91488986 0.40682679 0.51714317 0.12633214 0.5035154
 0.71990896] 0.6376291649181473
1 [0.97072301 0.78752741 0.89510783 0.27258531 0.47349205 0.55196118
 0.61314903 0.71375617 0.91133339 0.57305122 0.934743   0.77077827
 0.52523506 0.92257    0.48445079 0.59549932 0.10200328 0.52514557
 0.72873195] 0.6500970445488817
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.267, MeanIU:  0.6501, Best_mIoU:  0.6758
[0.97072301 0.78752741 0.89510783 0.27258531 0.47349205 0.55196118
 0.61314903 0.71375617 0.91133339 0.57305122 0.934743   0.77077827
 0.52523506 0.92257    0.48445079 0.59549932 0.10200328 0.52514557
 0.72873195]
Epoch: [139/484] Iter:[0/24], Time: 0.98, lr: [0.007373545425091332], Loss: 0.220041
Epoch: [139/484] Iter:[10/24], Time: 0.45, lr: [0.007365530217607856], Loss: 0.210589
Epoch: [139/484] Iter:[20/24], Time: 0.42, lr: [0.007357514040874784], Loss: 0.224949
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96946719 0.76671367 0.89568341 0.38032793 0.44613664 0.51746
 0.57024967 0.68243461 0.90211439 0.50387917 0.92330006 0.75169269
 0.47054897 0.90086358 0.35980331 0.50336411 0.18206257 0.35191862
 0.706969  ] 0.6202626102554637
1 [0.96881508 0.76782845 0.89845562 0.4224259  0.46197026 0.50742112
 0.5858187  0.69517238 0.90380885 0.50641268 0.91798117 0.7614241
 0.49985143 0.91118    0.43399367 0.58901848 0.20788591 0.39217454
 0.71006919] 0.6390372382823605
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.280, MeanIU:  0.6390, Best_mIoU:  0.6758
[0.96881508 0.76782845 0.89845562 0.4224259  0.46197026 0.50742112
 0.5858187  0.69517238 0.90380885 0.50641268 0.91798117 0.7614241
 0.49985143 0.91118    0.43399367 0.58901848 0.20788591 0.39217454
 0.71006919]
Epoch: [140/484] Iter:[0/24], Time: 0.95, lr: [0.007354307298502631], Loss: 0.346069
Epoch: [140/484] Iter:[10/24], Time: 0.44, lr: [0.007346289762650766], Loss: 0.208257
Epoch: [140/484] Iter:[20/24], Time: 0.42, lr: [0.007338271254445752], Loss: 0.226596
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96346559 0.75306487 0.88988143 0.30484977 0.43038361 0.54824529
 0.57366213 0.68596116 0.90510686 0.49706152 0.92827379 0.75978521
 0.46635934 0.90710442 0.39180005 0.49613171 0.17171895 0.44023188
 0.71623353] 0.6225958477012349
1 [0.96132571 0.74367951 0.88845959 0.2972837  0.45044414 0.55193502
 0.59128973 0.69772923 0.90619515 0.50087614 0.93023357 0.76865484
 0.4929682  0.91412861 0.45063412 0.5879068  0.21301019 0.47066361
 0.72322866] 0.6389813949525205
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.290, MeanIU:  0.6390, Best_mIoU:  0.6758
[0.96132571 0.74367951 0.88845959 0.2972837  0.45044414 0.55193502
 0.59128973 0.69772923 0.90619515 0.50087614 0.93023357 0.76865484
 0.4929682  0.91412861 0.45063412 0.5879068  0.21301019 0.47066361
 0.72322866]
Epoch: [141/484] Iter:[0/24], Time: 0.98, lr: [0.007335063578614055], Loss: 0.186532
Epoch: [141/484] Iter:[10/24], Time: 0.44, lr: [0.007327043706931981], Loss: 0.206509
Epoch: [141/484] Iter:[20/24], Time: 0.42, lr: [0.00731902285977417], Loss: 0.205495
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96909645 0.76602358 0.89564768 0.29924907 0.44591902 0.55525264
 0.58554144 0.70643969 0.90547189 0.50872917 0.9312258  0.68876322
 0.38327209 0.9114909  0.40584943 0.49925601 0.18634973 0.46605454
 0.70106722] 0.6216157667865165
1 [0.97009173 0.76625678 0.89506064 0.29366322 0.46663086 0.56645191
 0.60072148 0.71325688 0.90689854 0.50467162 0.93313444 0.66347987
 0.35573425 0.92021485 0.52156658 0.55977578 0.15620561 0.46808912
 0.70176012] 0.6296665409930722
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.287, MeanIU:  0.6297, Best_mIoU:  0.6758
[0.97009173 0.76625678 0.89506064 0.29366322 0.46663086 0.56645191
 0.60072148 0.71325688 0.90689854 0.50467162 0.93313444 0.66347987
 0.35573425 0.92021485 0.52156658 0.55977578 0.15620561 0.46808912
 0.70176012]
Epoch: [142/484] Iter:[0/24], Time: 0.96, lr: [0.007315814247485246], Loss: 0.168793
Epoch: [142/484] Iter:[10/24], Time: 0.44, lr: [0.0073077920324653494], Loss: 0.195908
Epoch: [142/484] Iter:[20/24], Time: 0.42, lr: [0.007299768838827935], Loss: 0.189052
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96813954 0.77024995 0.90103833 0.31009554 0.47737532 0.56435073
 0.59228646 0.70225821 0.90726763 0.5370431  0.93232256 0.76136171
 0.46558254 0.90647536 0.41136204 0.51912361 0.31960621 0.41296367
 0.71343711] 0.6406494545223476
1 [0.96837454 0.77302113 0.9045774  0.36496418 0.50368801 0.57221437
 0.61395167 0.71128039 0.91064388 0.54970511 0.93428245 0.76670368
 0.48606957 0.9157374  0.51566081 0.62563654 0.35597015 0.42049802
 0.71852899] 0.6637635948814321
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.267, MeanIU:  0.6638, Best_mIoU:  0.6758
[0.96837454 0.77302113 0.9045774  0.36496418 0.50368801 0.57221437
 0.61395167 0.71128039 0.91064388 0.54970511 0.93428245 0.76670368
 0.48606957 0.9157374  0.51566081 0.62563654 0.35597015 0.42049802
 0.71852899]
Epoch: [143/484] Iter:[0/24], Time: 1.01, lr: [0.007296559287065666], Loss: 0.133347
Epoch: [143/484] Iter:[10/24], Time: 0.45, lr: [0.007288534721154132], Loss: 0.169896
Epoch: [143/484] Iter:[20/24], Time: 0.42, lr: [0.007280509173463924], Loss: 0.178196
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96878331 0.77582999 0.88800961 0.29603728 0.46716374 0.4812684
 0.55112658 0.689672   0.9034722  0.52426096 0.93267853 0.7480529
 0.4717147  0.90288023 0.3723906  0.51756817 0.36791248 0.36516422
 0.71220214] 0.6282204220934521
1 [0.96971429 0.77896316 0.88281262 0.33882075 0.48902257 0.44441052
 0.56029133 0.6958373  0.90426295 0.51892736 0.93116735 0.75376538
 0.48599898 0.89628637 0.39303491 0.50043452 0.40670316 0.37666826
 0.71356397] 0.6337203033670453
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.294, MeanIU:  0.6337, Best_mIoU:  0.6758
[0.96971429 0.77896316 0.88281262 0.33882075 0.48902257 0.44441052
 0.56029133 0.6958373  0.90426295 0.51892736 0.93116735 0.75376538
 0.48599898 0.89628637 0.39303491 0.50043452 0.40670316 0.37666826
 0.71356397]
Epoch: [144/484] Iter:[0/24], Time: 1.00, lr: [0.007277298679193594], Loss: 0.170306
Epoch: [144/484] Iter:[10/24], Time: 0.45, lr: [0.007269271754789991], Loss: 0.182820
Epoch: [144/484] Iter:[20/24], Time: 0.42, lr: [0.007261243845427], Loss: 0.194040
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97162353 0.78868296 0.89448266 0.32088121 0.49020923 0.56404078
 0.58758747 0.71010414 0.90703304 0.5534486  0.90206245 0.759828
 0.51464507 0.90780008 0.37332011 0.47595057 0.41338051 0.44285448
 0.70902826] 0.6466822693955784
1 [0.97265156 0.79122251 0.89094554 0.30440525 0.50633851 0.56951632
 0.60712077 0.7190009  0.90772321 0.5601352  0.89943001 0.75916687
 0.51574053 0.91319343 0.42221635 0.52604531 0.48808372 0.44861196
 0.71405905] 0.6587161585199087
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.272, MeanIU:  0.6587, Best_mIoU:  0.6758
[0.97265156 0.79122251 0.89094554 0.30440525 0.50633851 0.56951632
 0.60712077 0.7190009  0.90772321 0.5601352  0.89943001 0.75916687
 0.51574053 0.91319343 0.42221635 0.52604531 0.48808372 0.44861196
 0.71405905]
Epoch: [145/484] Iter:[0/24], Time: 0.97, lr: [0.0072580324055951245], Loss: 0.169304
Epoch: [145/484] Iter:[10/24], Time: 0.44, lr: [0.007250003115051965], Loss: 0.193790
Epoch: [145/484] Iter:[20/24], Time: 0.42, lr: [0.007241972836348978], Loss: 0.219743
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9609856  0.77473218 0.87748604 0.36028445 0.43505935 0.55352
 0.56962078 0.68380702 0.90476589 0.52272845 0.91697301 0.76192434
 0.47250704 0.90054061 0.35099392 0.48835237 0.37386011 0.42404095
 0.71956265] 0.6343023550319354
1 [0.9590699  0.77534228 0.87122428 0.37068873 0.46470249 0.56092447
 0.58121879 0.69665395 0.90710131 0.50818233 0.9037544  0.76872349
 0.494178   0.91329046 0.39632142 0.57184572 0.42495322 0.45074175
 0.7268121 ] 0.6497752138181807
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.305, MeanIU:  0.6498, Best_mIoU:  0.6758
[0.9590699  0.77534228 0.87122428 0.37068873 0.46470249 0.56092447
 0.58121879 0.69665395 0.90710131 0.50818233 0.9037544  0.76872349
 0.494178   0.91329046 0.39632142 0.57184572 0.42495322 0.45074175
 0.7268121 ]
Epoch: [146/484] Iter:[0/24], Time: 0.97, lr: [0.007238760447883127], Loss: 0.174675
Epoch: [146/484] Iter:[10/24], Time: 0.44, lr: [0.007230728783505443], Loss: 0.172550
Epoch: [146/484] Iter:[20/24], Time: 0.42, lr: [0.007222696127747586], Loss: 0.184405
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97160805 0.7841809  0.90002816 0.33414768 0.47988861 0.56487489
 0.58412507 0.70258574 0.9065912  0.51249212 0.93370079 0.76566766
 0.52204166 0.90971386 0.40912069 0.50598555 0.38166549 0.46043674
 0.70906393] 0.6493641464557829
1 [0.97269142 0.79013257 0.90225151 0.37054218 0.50945257 0.57678353
 0.60339865 0.71674943 0.90833735 0.51983616 0.93477537 0.76148621
 0.51840203 0.91811274 0.52346807 0.6021675  0.47687124 0.47081269
 0.71187326] 0.6730602361940063
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.263, MeanIU:  0.6731, Best_mIoU:  0.6758
[0.97269142 0.79013257 0.90225151 0.37054218 0.50945257 0.57678353
 0.60339865 0.71674943 0.90833735 0.51983616 0.93477537 0.76148621
 0.51840203 0.91811274 0.52346807 0.6021675  0.47687124 0.47081269
 0.71187326]
Epoch: [147/484] Iter:[0/24], Time: 0.93, lr: [0.0072194827875562174], Loss: 0.219870
Epoch: [147/484] Iter:[10/24], Time: 0.44, lr: [0.007211448741601132], Loss: 0.219737
Epoch: [147/484] Iter:[20/24], Time: 0.41, lr: [0.007203413701025422], Loss: 0.216137
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97159267 0.77584751 0.86770966 0.25787652 0.39263313 0.56169776
 0.56234458 0.69900148 0.90368288 0.53138817 0.92662117 0.76265449
 0.48118633 0.91527804 0.398762   0.51459775 0.37956566 0.37579169
 0.71552931] 0.6312505689764808
1 [0.97133585 0.78091836 0.86394898 0.25964155 0.38018832 0.57373274
 0.59517732 0.70835486 0.90513878 0.55060328 0.92451393 0.76838668
 0.5008293  0.92276068 0.52477234 0.58685626 0.44460329 0.38816401
 0.71730543] 0.6509069450398463
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.297, MeanIU:  0.6509, Best_mIoU:  0.6758
[0.97133585 0.78091836 0.86394898 0.25964155 0.38018832 0.57373274
 0.59517732 0.70835486 0.90513878 0.55060328 0.92451393 0.76838668
 0.5008293  0.92276068 0.52477234 0.58685626 0.44460329 0.38816401
 0.71730543]
Epoch: [148/484] Iter:[0/24], Time: 0.98, lr: [0.007200199405997702], Loss: 0.208761
Epoch: [148/484] Iter:[10/24], Time: 0.44, lr: [0.007192162970673974], Loss: 0.189754
Epoch: [148/484] Iter:[20/24], Time: 0.42, lr: [0.007184125537468891], Loss: 0.191964
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97095616 0.78144123 0.90004099 0.32476606 0.4961014  0.56422995
 0.58509774 0.70368489 0.90429848 0.52771176 0.93042265 0.76186887
 0.49228568 0.90530818 0.36417621 0.36109612 0.28407552 0.4690077
 0.71472105] 0.6337521396602096
1 [0.97147923 0.7843749  0.9001498  0.34483597 0.51009919 0.57218471
 0.60097659 0.71749665 0.906268   0.54387909 0.9320177  0.76532822
 0.5048465  0.91441153 0.43559683 0.43521553 0.24111362 0.48149273
 0.71836339] 0.6463226406598943
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.273, MeanIU:  0.6463, Best_mIoU:  0.6758
[0.97147923 0.7843749  0.9001498  0.34483597 0.51009919 0.57218471
 0.60097659 0.71749665 0.906268   0.54387909 0.9320177  0.76532822
 0.5048465  0.91441153 0.43559683 0.43521553 0.24111362 0.48149273
 0.71836339]
Epoch: [149/484] Iter:[0/24], Time: 0.97, lr: [0.007180910284474515], Loss: 0.188620
Epoch: [149/484] Iter:[10/24], Time: 0.44, lr: [0.007172871451942098], Loss: 0.201174
Epoch: [149/484] Iter:[20/24], Time: 0.42, lr: [0.0071648316182471306], Loss: 0.194573
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97119117 0.78149046 0.89864932 0.2653406  0.45381847 0.56201371
 0.58263384 0.71134187 0.90591649 0.53159928 0.9333718  0.77013248
 0.509061   0.9051369  0.37581475 0.43202923 0.2797952  0.43847651
 0.72142566] 0.6331178289548298
1 [0.97154654 0.7845532  0.89635889 0.28407672 0.45845838 0.56966247
 0.60029042 0.7204552  0.907645   0.52654188 0.93245428 0.7744637
 0.52776332 0.91433504 0.50036078 0.52868805 0.25439799 0.4651249
 0.72868079] 0.6497819751870988
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.271, MeanIU:  0.6498, Best_mIoU:  0.6758
[0.97154654 0.7845532  0.89635889 0.28407672 0.45845838 0.56966247
 0.60029042 0.7204552  0.907645   0.52654188 0.93245428 0.7744637
 0.52776332 0.91433504 0.50036078 0.52868805 0.25439799 0.4651249
 0.72868079]
Epoch: [150/484] Iter:[0/24], Time: 0.97, lr: [0.007161615404136143], Loss: 0.155933
Epoch: [150/484] Iter:[10/24], Time: 0.44, lr: [0.007153574166505734], Loss: 0.146355
Epoch: [150/484] Iter:[20/24], Time: 0.42, lr: [0.007145531924410919], Loss: 0.164040
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97182111 0.78084824 0.89686455 0.30588922 0.48342431 0.57066499
 0.57599822 0.70265487 0.90706302 0.52512086 0.91927302 0.76310116
 0.49675155 0.90817774 0.366303   0.4670207  0.29542998 0.48744001
 0.72431091] 0.6393767073250572
1 [0.97165441 0.78294211 0.89586598 0.33854048 0.49172903 0.5826181
 0.6042055  0.709201   0.90964323 0.5257745  0.91655482 0.77188513
 0.51557589 0.91438452 0.42710328 0.5452033  0.30215955 0.50949331
 0.72971442] 0.6549604504062926
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.272, MeanIU:  0.6550, Best_mIoU:  0.6758
[0.97165441 0.78294211 0.89586598 0.33854048 0.49172903 0.5826181
 0.6042055  0.709201   0.90964323 0.5257745  0.91655482 0.77188513
 0.51557589 0.91438452 0.42710328 0.5452033  0.30215955 0.50949331
 0.72971442]
Epoch: [151/484] Iter:[0/24], Time: 0.97, lr: [0.007142314746013531], Loss: 0.160478
Epoch: [151/484] Iter:[10/24], Time: 0.44, lr: [0.0071342710953461105], Loss: 0.170939
Epoch: [151/484] Iter:[20/24], Time: 0.42, lr: [0.007126226436891574], Loss: 0.183349
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96717013 0.7610391  0.90061045 0.31806733 0.49035223 0.57616211
 0.59593903 0.70354524 0.90639084 0.5303408  0.93157657 0.76489649
 0.47761701 0.9136423  0.39692896 0.46589707 0.35019975 0.48924883
 0.72597074] 0.6455576305766461
1 [0.96536165 0.75930913 0.90105201 0.34809032 0.50874804 0.58590152
 0.60998093 0.70466112 0.90906781 0.53466445 0.9318169  0.7729473
 0.49900283 0.92077662 0.50252388 0.54692929 0.38785799 0.50781071
 0.72878454] 0.6644887922253102
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.279, MeanIU:  0.6645, Best_mIoU:  0.6758
[0.96536165 0.75930913 0.90105201 0.34809032 0.50874804 0.58590152
 0.60998093 0.70466112 0.90906781 0.53466445 0.9318169  0.7729473
 0.49900283 0.92077662 0.50252388 0.54692929 0.38785799 0.50781071
 0.72878454]
Epoch: [152/484] Iter:[0/24], Time: 0.96, lr: [0.007123008291017982], Loss: 0.182242
Epoch: [152/484] Iter:[10/24], Time: 0.44, lr: [0.007114962219324351], Loss: 0.162813
Epoch: [152/484] Iter:[20/24], Time: 0.42, lr: [0.007106915136499847], Loss: 0.178732
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.92841398 0.6900097  0.86121584 0.1191875  0.43589269 0.53827015
 0.51754777 0.67246139 0.85877973 0.48823813 0.91901095 0.74406177
 0.41790925 0.86005722 0.24490352 0.36140334 0.32752034 0.33454789
 0.69585808] 0.5797520640069866
1 [0.92371882 0.68885304 0.86210609 0.13245273 0.45466525 0.54080518
 0.55282532 0.68253768 0.87747585 0.48987827 0.91995157 0.74797648
 0.43755014 0.8641254  0.26341896 0.44899414 0.442794   0.33539898
 0.69691048] 0.5980230722611993
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.494, MeanIU:  0.5980, Best_mIoU:  0.6758
[0.92371882 0.68885304 0.86210609 0.13245273 0.45466525 0.54080518
 0.55282532 0.68253768 0.87747585 0.48987827 0.91995157 0.74797648
 0.43755014 0.8641254  0.26341896 0.44899414 0.442794   0.33539898
 0.69691048]
Epoch: [153/484] Iter:[0/24], Time: 0.98, lr: [0.007103696019940041], Loss: 0.206737
Epoch: [153/484] Iter:[10/24], Time: 0.45, lr: [0.007095647519180347], Loss: 0.315238
Epoch: [153/484] Iter:[20/24], Time: 0.42, lr: [0.007087598003924782], Loss: 0.289541
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95245062 0.72605717 0.87812302 0.20307771 0.4322435  0.51653117
 0.54271506 0.6708825  0.89308461 0.47675396 0.92000232 0.74527749
 0.48678516 0.90113251 0.36036037 0.39741422 0.20355943 0.46305231
 0.69928892] 0.6036206343000871
1 [0.9503767  0.71290745 0.87965035 0.1426052  0.45299025 0.52754214
 0.55030016 0.67778848 0.89454645 0.50114279 0.91686548 0.75179497
 0.50224126 0.90676779 0.46637434 0.51628279 0.22564926 0.47441444
 0.70236272] 0.6185580535285835
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.340, MeanIU:  0.6186, Best_mIoU:  0.6758
[0.9503767  0.71290745 0.87965035 0.1426052  0.45299025 0.52754214
 0.55030016 0.67778848 0.89454645 0.50114279 0.91686548 0.75179497
 0.50224126 0.90676779 0.46637434 0.51628279 0.22564926 0.47441444
 0.70236272]
Epoch: [154/484] Iter:[0/24], Time: 0.94, lr: [0.007084377913448356], Loss: 1.019266
Epoch: [154/484] Iter:[10/24], Time: 0.44, lr: [0.007076326975531624], Loss: 0.379792
Epoch: [154/484] Iter:[20/24], Time: 0.42, lr: [0.007068275019732578], Loss: 0.325178
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96023766 0.74963509 0.87278478 0.25396067 0.42409108 0.52337272
 0.54574213 0.68591872 0.89090537 0.46660657 0.92473771 0.73079674
 0.47943062 0.90420106 0.33129156 0.46238577 0.19027797 0.39214895
 0.66327952] 0.6027265625258941
1 [0.95564135 0.74607614 0.86811011 0.25145396 0.44141784 0.53282289
 0.56158364 0.70064947 0.89055036 0.44745843 0.92517952 0.72313559
 0.47287188 0.89782954 0.34262469 0.5115254  0.23092633 0.38005356
 0.65924556] 0.6073240135967268
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.337, MeanIU:  0.6073, Best_mIoU:  0.6758
[0.95564135 0.74607614 0.86811011 0.25145396 0.44141784 0.53282289
 0.56158364 0.70064947 0.89055036 0.44745843 0.92517952 0.72313559
 0.47287188 0.89782954 0.34262469 0.5115254  0.23092633 0.38005356
 0.65924556]
Epoch: [155/484] Iter:[0/24], Time: 0.95, lr: [0.0070650539520885425], Loss: 0.178833
Epoch: [155/484] Iter:[10/24], Time: 0.44, lr: [0.007057000568872191], Loss: 0.215854
Epoch: [155/484] Iter:[20/24], Time: 0.42, lr: [0.007048946164365434], Loss: 0.209702
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96791584 0.77078772 0.89516681 0.24122869 0.48832746 0.5511846
 0.56713506 0.69255495 0.90227901 0.49282042 0.92397617 0.75258552
 0.48048821 0.90624393 0.38207246 0.55441701 0.2835198  0.40017886
 0.70809936] 0.6295253618891329
1 [0.96592042 0.76550117 0.89788643 0.27323105 0.50581636 0.56186732
 0.59690386 0.70907278 0.90293352 0.47843196 0.9241323  0.76037488
 0.51428966 0.91389081 0.50401697 0.66470483 0.3776591  0.42161597
 0.71128132] 0.6552384581974224
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.282, MeanIU:  0.6552, Best_mIoU:  0.6758
[0.96592042 0.76550117 0.89788643 0.27323105 0.50581636 0.56186732
 0.59690386 0.70907278 0.90293352 0.47843196 0.9241323  0.76037488
 0.51428966 0.91389081 0.50401697 0.66470483 0.3776591  0.42161597
 0.71128132]
Epoch: [156/484] Iter:[0/24], Time: 0.96, lr: [0.007045724116282016], Loss: 0.174103
Epoch: [156/484] Iter:[10/24], Time: 0.44, lr: [0.007037668279571367], Loss: 0.258198
Epoch: [156/484] Iter:[20/24], Time: 0.42, lr: [0.007029611418140375], Loss: 0.224111
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96943053 0.77735368 0.89333046 0.33355147 0.47377271 0.55496755
 0.57509769 0.70028519 0.90260645 0.50421587 0.92915072 0.76511581
 0.49478241 0.91754933 0.45974295 0.56987955 0.33115118 0.48106872
 0.71929735] 0.6501236647072024
1 [0.96912187 0.77662186 0.8963636  0.36590168 0.47351283 0.56437872
 0.5968369  0.71464501 0.90548996 0.50960149 0.93063309 0.77310864
 0.51787898 0.92554754 0.56076515 0.6704025  0.41507554 0.50266489
 0.72194412] 0.6731839144882847
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.267, MeanIU:  0.6732, Best_mIoU:  0.6758
[0.96912187 0.77662186 0.8963636  0.36590168 0.47351283 0.56437872
 0.5968369  0.71464501 0.90548996 0.50960149 0.93063309 0.77310864
 0.51787898 0.92554754 0.56076515 0.6704025  0.41507554 0.50266489
 0.72194412]
Epoch: [157/484] Iter:[0/24], Time: 0.94, lr: [0.007026388386324827], Loss: 0.187924
Epoch: [157/484] Iter:[10/24], Time: 0.44, lr: [0.007018330087872615], Loss: 0.191335
Epoch: [157/484] Iter:[20/24], Time: 0.41, lr: [0.007010270761248064], Loss: 0.185122
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96960671 0.77845919 0.89957776 0.34299502 0.49960362 0.55953481
 0.58926784 0.70140423 0.90454084 0.48072748 0.92602788 0.76096359
 0.50859344 0.91845828 0.45306916 0.55909308 0.34765162 0.4264961
 0.70708935] 0.6491136843472143
1 [0.96962455 0.77758077 0.90145613 0.38842035 0.51246621 0.56433836
 0.60183152 0.71002954 0.90641133 0.47999619 0.92640281 0.76770776
 0.52493231 0.92782791 0.5662107  0.67023674 0.42266353 0.4522363
 0.70778574] 0.6725346702849753
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.263, MeanIU:  0.6725, Best_mIoU:  0.6758
[0.96962455 0.77758077 0.90145613 0.38842035 0.51246621 0.56433836
 0.60183152 0.71002954 0.90641133 0.47999619 0.92640281 0.76770776
 0.52493231 0.92782791 0.5662107  0.67023674 0.42266353 0.4522363
 0.70778574]
Epoch: [158/484] Iter:[0/24], Time: 0.96, lr: [0.007007046742386459], Loss: 0.193471
Epoch: [158/484] Iter:[10/24], Time: 0.44, lr: [0.006998985973892337], Loss: 0.192877
Epoch: [158/484] Iter:[20/24], Time: 0.42, lr: [0.0069909241737516024], Loss: 0.212358
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96398097 0.75931341 0.89005959 0.26470884 0.46004928 0.54877989
 0.59572829 0.70898693 0.90108728 0.51096738 0.91498756 0.76185297
 0.48316377 0.90412921 0.40516552 0.55432349 0.3810065  0.44383448
 0.70835205] 0.6400251250705827
1 [0.96468471 0.75950008 0.88847653 0.27175683 0.46063616 0.55223919
 0.60633633 0.72073548 0.89980545 0.52643492 0.91484674 0.77017228
 0.51748186 0.91395706 0.54085046 0.67446157 0.47602089 0.46439358
 0.71236755] 0.6650082975845603
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.290, MeanIU:  0.6650, Best_mIoU:  0.6758
[0.96468471 0.75950008 0.88847653 0.27175683 0.46063616 0.55223919
 0.60633633 0.72073548 0.89980545 0.52643492 0.91484674 0.77017228
 0.51748186 0.91395706 0.54085046 0.67446157 0.47602089 0.46439358
 0.71236755]
Epoch: [159/484] Iter:[0/24], Time: 0.96, lr: [0.006987699164508641], Loss: 0.212370
Epoch: [159/484] Iter:[10/24], Time: 0.44, lr: [0.006979635917618663], Loss: 0.197369
Epoch: [159/484] Iter:[20/24], Time: 0.42, lr: [0.006971571635585322], Loss: 0.199210
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96711481 0.76020416 0.8839059  0.23795077 0.43998613 0.54432346
 0.54058665 0.70721887 0.89637634 0.43960469 0.92374706 0.75522495
 0.47064051 0.90665966 0.36275574 0.46231923 0.27152557 0.48800925
 0.71262652] 0.6195147510149832
1 [0.96687716 0.76027825 0.8839037  0.29713394 0.45221161 0.54316624
 0.57321307 0.71652073 0.89822448 0.44230454 0.92474706 0.75801258
 0.50779562 0.91438446 0.44909505 0.58593813 0.3229301  0.51134602
 0.71152392] 0.643137192177119
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.298, MeanIU:  0.6431, Best_mIoU:  0.6758
[0.96687716 0.76027825 0.8839037  0.29713394 0.45221161 0.54316624
 0.57321307 0.71652073 0.89822448 0.44230454 0.92474706 0.75801258
 0.50779562 0.91438446 0.44909505 0.58593813 0.3229301  0.51134602
 0.71152392]
Epoch: [160/484] Iter:[0/24], Time: 0.97, lr: [0.006968345632604116], Loss: 0.293286
Epoch: [160/484] Iter:[10/24], Time: 0.44, lr: [0.006960279898910229], Loss: 0.196983
Epoch: [160/484] Iter:[20/24], Time: 0.42, lr: [0.006952213126553536], Loss: 0.195730
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96309573 0.73203654 0.8967685  0.21760287 0.46151988 0.5469373
 0.58773415 0.70801    0.90227102 0.4580535  0.92349624 0.76237571
 0.52523288 0.9066465  0.4074592  0.54872469 0.32579738 0.48008662
 0.70209283] 0.634523240130761
1 [0.96375461 0.73549467 0.89839947 0.25702246 0.4813219  0.5495874
 0.60219446 0.70916196 0.90362221 0.43694303 0.92238119 0.76299845
 0.5228668  0.91577334 0.47364413 0.66809025 0.43293691 0.47933308
 0.70436758] 0.653678626709187
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.287, MeanIU:  0.6537, Best_mIoU:  0.6758
[0.96375461 0.73549467 0.89839947 0.25702246 0.4813219  0.5495874
 0.60219446 0.70916196 0.90362221 0.43694303 0.92238119 0.76299845
 0.5228668  0.91577334 0.47364413 0.66809025 0.43293691 0.47933308
 0.70436758]
Epoch: [161/484] Iter:[0/24], Time: 0.95, lr: [0.006948986126455412], Loss: 0.262008
Epoch: [161/484] Iter:[10/24], Time: 0.44, lr: [0.006940917897494942], Loss: 0.214124
Epoch: [161/484] Iter:[20/24], Time: 0.42, lr: [0.0069328486263293045], Loss: 0.218755
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96934805 0.76930176 0.89738662 0.34320254 0.49351607 0.56014761
 0.58779514 0.70658933 0.90630196 0.54425633 0.92906799 0.76682945
 0.50878067 0.91740057 0.43573497 0.59013265 0.2893852  0.48515718
 0.71915815] 0.653657485267232
1 [0.96881117 0.76816621 0.89945017 0.40187753 0.50875143 0.56639093
 0.60682793 0.71480163 0.90949348 0.56303867 0.93265621 0.7742372
 0.53135151 0.92513496 0.55217211 0.66866383 0.31998832 0.50753101
 0.7239556 ] 0.6759631511425043
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.262, MeanIU:  0.6760, Best_mIoU:  0.6760
[0.96881117 0.76816621 0.89945017 0.40187753 0.50875143 0.56639093
 0.60682793 0.71480163 0.90949348 0.56303867 0.93265621 0.7742372
 0.53135151 0.92513496 0.55217211 0.66866383 0.31998832 0.50753101
 0.7239556 ]
Epoch: [162/484] Iter:[0/24], Time: 0.95, lr: [0.00692962062571359], Loss: 0.490992
Epoch: [162/484] Iter:[10/24], Time: 0.44, lr: [0.006921549892968713], Loss: 0.230457
Epoch: [162/484] Iter:[20/24], Time: 0.42, lr: [0.006913478114453169], Loss: 0.200946
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96570282 0.75357141 0.89266417 0.27311801 0.42168389 0.56410297
 0.58249702 0.71318434 0.90483712 0.51473318 0.93032914 0.75592266
 0.50528492 0.91168575 0.40870904 0.56663364 0.24052258 0.48740604
 0.71539865] 0.637262492909416
1 [0.9638801  0.74623097 0.89130013 0.2889904  0.43916272 0.57125462
 0.60171327 0.72342554 0.9064004  0.52536305 0.93125956 0.75388368
 0.48530616 0.92018293 0.49554053 0.63100262 0.219202   0.513398
 0.71887266] 0.6487562820480571
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.287, MeanIU:  0.6488, Best_mIoU:  0.6760
[0.9638801  0.74623097 0.89130013 0.2889904  0.43916272 0.57125462
 0.60171327 0.72342554 0.9064004  0.52536305 0.93125956 0.75388368
 0.48530616 0.92018293 0.49554053 0.63100262 0.219202   0.513398
 0.71887266]
Epoch: [163/484] Iter:[0/24], Time: 0.94, lr: [0.006910249109896981], Loss: 0.222490
Epoch: [163/484] Iter:[10/24], Time: 0.44, lr: [0.006902175864794188], Loss: 0.172578
Epoch: [163/484] Iter:[20/24], Time: 0.42, lr: [0.006894101570331872], Loss: 0.169896
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96672342 0.76797121 0.89687214 0.28294219 0.44069764 0.55694284
 0.57317698 0.68454792 0.908026   0.51430167 0.92885353 0.75659689
 0.38330983 0.90999127 0.36157544 0.52528222 0.30890225 0.44922787
 0.71727064] 0.6280637880791913
1 [0.96600262 0.76545122 0.89899474 0.31725007 0.458593   0.56647168
 0.60132388 0.69908337 0.90862122 0.50169405 0.928695   0.76534039
 0.40226488 0.91309416 0.42747092 0.56964512 0.27211654 0.48281304
 0.72265353] 0.640398918579775
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.280, MeanIU:  0.6404, Best_mIoU:  0.6760
[0.96600262 0.76545122 0.89899474 0.31725007 0.458593   0.56647168
 0.60132388 0.69908337 0.90862122 0.50169405 0.928695   0.76534039
 0.40226488 0.91309416 0.42747092 0.56964512 0.27211654 0.48281304
 0.72265353]
Epoch: [164/484] Iter:[0/24], Time: 0.97, lr: [0.006890871558389902], Loss: 0.120021
Epoch: [164/484] Iter:[10/24], Time: 0.44, lr: [0.006882795792299464], Loss: 0.203823
Epoch: [164/484] Iter:[20/24], Time: 0.42, lr: [0.006874718973237062], Loss: 0.194980
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95969966 0.71161403 0.89776352 0.2849453  0.45299632 0.56010149
 0.59621918 0.71417857 0.90555905 0.51680329 0.93101941 0.77000721
 0.52691516 0.90905828 0.41066437 0.51009685 0.45787107 0.52266532
 0.72523543] 0.6507059739283804
1 [0.9592341  0.70634029 0.89948351 0.3243757  0.47558301 0.56770975
 0.61564983 0.72510163 0.90877523 0.53496704 0.93382256 0.77754519
 0.53229607 0.91711895 0.566139   0.64129331 0.58562217 0.53327362
 0.73072404] 0.6807923681850564
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.289, MeanIU:  0.6808, Best_mIoU:  0.6808
[0.9592341  0.70634029 0.89948351 0.3243757  0.47558301 0.56770975
 0.61564983 0.72510163 0.90877523 0.53496704 0.93382256 0.77754519
 0.53229607 0.91711895 0.566139   0.64129331 0.58562217 0.53327362
 0.73072404]
Epoch: [165/484] Iter:[0/24], Time: 0.95, lr: [0.0068714879504413575], Loss: 0.187467
Epoch: [165/484] Iter:[10/24], Time: 0.44, lr: [0.006863409654676778], Loss: 0.165646
Epoch: [165/484] Iter:[20/24], Time: 0.42, lr: [0.006855330302303976], Loss: 0.176396
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96563708 0.73505713 0.88501558 0.22810979 0.39773552 0.55912074
 0.58816314 0.71942386 0.90373615 0.51783918 0.93012947 0.77428951
 0.5360761  0.91136419 0.37633419 0.49901791 0.38541929 0.49321181
 0.72521418] 0.6384681488999184
1 [0.96397758 0.7275838  0.87971161 0.24255739 0.38182501 0.55653671
 0.6033123  0.72841804 0.90292983 0.52663714 0.92935925 0.779319
 0.54712411 0.91777801 0.51608788 0.61873056 0.47886866 0.51630318
 0.73096133] 0.6604221783165642
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.300, MeanIU:  0.6604, Best_mIoU:  0.6808
[0.96397758 0.7275838  0.87971161 0.24255739 0.38182501 0.55653671
 0.6033123  0.72841804 0.90292983 0.52663714 0.92935925 0.779319
 0.54712411 0.91777801 0.51608788 0.61873056 0.47886866 0.51630318
 0.73096133]
Epoch: [166/484] Iter:[0/24], Time: 1.00, lr: [0.006852098265163722], Loss: 0.235373
Epoch: [166/484] Iter:[10/24], Time: 0.45, lr: [0.006844017430981187], Loss: 0.208817
Epoch: [166/484] Iter:[20/24], Time: 0.42, lr: [0.00683593553653012], Loss: 0.210247
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96326628 0.73599615 0.89094094 0.25245537 0.45575028 0.54870498
 0.57404306 0.70576017 0.89794011 0.47529405 0.92305786 0.74737337
 0.51275334 0.90483278 0.44752869 0.56673459 0.20468666 0.4703767
 0.71030015] 0.6309366065809044
1 [0.9632029  0.73121635 0.89223131 0.25354465 0.46838548 0.54915997
 0.58857285 0.71549532 0.90005641 0.47909848 0.92420268 0.75447886
 0.52705517 0.91762105 0.57364643 0.64287155 0.07675055 0.48000705
 0.71754617] 0.639744380693134
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.298, MeanIU:  0.6397, Best_mIoU:  0.6808
[0.9632029  0.73121635 0.89223131 0.25354465 0.46838548 0.54915997
 0.58857285 0.71549532 0.90005641 0.47909848 0.92420268 0.75447886
 0.52705517 0.91762105 0.57364643 0.64287155 0.07675055 0.48000705
 0.71754617]
Epoch: [167/484] Iter:[0/24], Time: 0.99, lr: [0.006832702481531417], Loss: 0.196488
Epoch: [167/484] Iter:[10/24], Time: 0.45, lr: [0.006824619100129227], Loss: 0.215239
Epoch: [167/484] Iter:[20/24], Time: 0.42, lr: [0.006816534654773916], Loss: 0.214453
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96317699 0.74294981 0.88650764 0.22061005 0.42170973 0.55134125
 0.56823632 0.7007713  0.89637052 0.46537424 0.91346246 0.7609659
 0.5082732  0.90372327 0.38933618 0.56045192 0.26656794 0.49915807
 0.71067155] 0.6278767552024479
1 [0.96295325 0.7443632  0.88967309 0.24591137 0.45151398 0.559826
 0.59528071 0.71986522 0.89924297 0.45657622 0.91358634 0.76371071
 0.51895225 0.91122711 0.44773984 0.62492718 0.18491336 0.50463944
 0.71812278] 0.6375276317858042
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.310, MeanIU:  0.6375, Best_mIoU:  0.6808
[0.96295325 0.7443632  0.88967309 0.24591137 0.45151398 0.559826
 0.59528071 0.71986522 0.89924297 0.45657622 0.91358634 0.76371071
 0.51895225 0.91122711 0.44773984 0.62492718 0.18491336 0.50463944
 0.71812278]
Epoch: [168/484] Iter:[0/24], Time: 0.97, lr: [0.006813300578379548], Loss: 0.166119
Epoch: [168/484] Iter:[10/24], Time: 0.45, lr: [0.006805214640897559], Loss: 0.234614
Epoch: [168/484] Iter:[20/24], Time: 0.42, lr: [0.006797127635753337], Loss: 0.227301
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96644579 0.76621804 0.88806309 0.22618876 0.45383204 0.55532389
 0.57049724 0.71693041 0.90004457 0.52966684 0.91079436 0.76276158
 0.45190486 0.90972825 0.36764926 0.5937663  0.30595867 0.40835122
 0.69090519] 0.630264756208838
1 [0.96674452 0.76349572 0.8857951  0.23661538 0.45925874 0.55464662
 0.59008908 0.72725826 0.90228641 0.54930841 0.90679303 0.76992638
 0.4890824  0.91800734 0.43409227 0.65442792 0.25260731 0.42904115
 0.69364082] 0.6412166776110955
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.293, MeanIU:  0.6412, Best_mIoU:  0.6808
[0.96674452 0.76349572 0.8857951  0.23661538 0.45925874 0.55464662
 0.59008908 0.72725826 0.90228641 0.54930841 0.90679303 0.76992638
 0.4890824  0.91800734 0.43409227 0.65442792 0.25260731 0.42904115
 0.69364082]
Epoch: [169/484] Iter:[0/24], Time: 0.99, lr: [0.006793892534402549], Loss: 0.200467
Epoch: [169/484] Iter:[10/24], Time: 0.45, lr: [0.006785804031921597], Loss: 0.209629
Epoch: [169/484] Iter:[20/24], Time: 0.42, lr: [0.006777714458044529], Loss: 0.207089
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96268073 0.73808129 0.89609411 0.28301335 0.47681146 0.56965825
 0.58978022 0.71966612 0.9034196  0.45817819 0.92166134 0.75074567
 0.52037676 0.91914126 0.41949879 0.62909806 0.26728142 0.41934281
 0.70126046] 0.6392520989854965
1 [0.96277032 0.73057134 0.89419884 0.30059596 0.4863683  0.57672845
 0.59778995 0.72915407 0.90473844 0.42547811 0.9228141  0.75045345
 0.51732596 0.92850492 0.57092372 0.67827376 0.16716616 0.38527804
 0.68996204] 0.6431103123382252
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.294, MeanIU:  0.6431, Best_mIoU:  0.6808
[0.96277032 0.73057134 0.89419884 0.30059596 0.4863683  0.57672845
 0.59778995 0.72915407 0.90473844 0.42547811 0.9228141  0.75045345
 0.51732596 0.92850492 0.57092372 0.67827376 0.16716616 0.38527804
 0.68996204]
Epoch: [170/484] Iter:[0/24], Time: 0.94, lr: [0.006774478328152794], Loss: 0.187415
Epoch: [170/484] Iter:[10/24], Time: 0.44, lr: [0.006766387251694103], Loss: 0.212248
Epoch: [170/484] Iter:[20/24], Time: 0.42, lr: [0.006758295100080414], Loss: 0.197188
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96079946 0.76418856 0.89442084 0.28269087 0.47415613 0.56492184
 0.59239243 0.71558644 0.90656523 0.53468461 0.89460578 0.74584773
 0.50225105 0.91093576 0.39423548 0.55610882 0.28502203 0.47873574
 0.71571781] 0.6407298219751357
1 [0.96026228 0.75739811 0.89739317 0.32072174 0.49245812 0.57378468
 0.61538139 0.72511965 0.90699087 0.54296132 0.8992527  0.75175529
 0.52241176 0.91823837 0.48326577 0.61908158 0.22864945 0.48848512
 0.7174324 ] 0.6537391458714028
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.289, MeanIU:  0.6537, Best_mIoU:  0.6808
[0.96026228 0.75739811 0.89739317 0.32072174 0.49245812 0.57378468
 0.61538139 0.72511965 0.90699087 0.54296132 0.8992527  0.75175529
 0.52241176 0.91823837 0.48326577 0.61908158 0.22864945 0.48848512
 0.7174324 ]
Epoch: [171/484] Iter:[0/24], Time: 0.95, lr: [0.00675505793803919], Loss: 0.192010
Epoch: [171/484] Iter:[10/24], Time: 0.44, lr: [0.0067469642785637956], Loss: 0.205202
Epoch: [171/484] Iter:[20/24], Time: 0.42, lr: [0.006738869540149259], Loss: 0.215199
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96251307 0.7541212  0.85759341 0.27082362 0.45026324 0.53631333
 0.53965297 0.69878033 0.90100923 0.53314238 0.89179397 0.76119825
 0.49826155 0.88291978 0.14059131 0.48931284 0.26724318 0.47781037
 0.71658393] 0.612101471508789
1 [0.9609103  0.75158392 0.84186497 0.30525322 0.46588995 0.54232277
 0.56319043 0.70698393 0.90246296 0.54046606 0.88285522 0.77058746
 0.51709581 0.89742298 0.1018997  0.50387827 0.17784864 0.49869112
 0.72613197] 0.6135441935565598
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.345, MeanIU:  0.6135, Best_mIoU:  0.6808
[0.9609103  0.75158392 0.84186497 0.30525322 0.46588995 0.54232277
 0.56319043 0.70698393 0.90246296 0.54046606 0.88285522 0.77058746
 0.51709581 0.89742298 0.1018997  0.50387827 0.17784864 0.49869112
 0.72613197]
Epoch: [172/484] Iter:[0/24], Time: 0.99, lr: [0.006735631342325764], Loss: 0.309670
Epoch: [172/484] Iter:[10/24], Time: 0.45, lr: [0.006727535090733907], Loss: 0.212504
Epoch: [172/484] Iter:[20/24], Time: 0.42, lr: [0.006719437756393256], Loss: 0.260960
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.69180847 0.55029259 0.77339261 0.18755509 0.28903981 0.41403488
 0.37585278 0.42101167 0.80297195 0.31848033 0.8047254  0.5754133
 0.19187352 0.36262983 0.06511204 0.04552222 0.0233914  0.15457499
 0.57558233] 0.40122448343370853
1 [9.40362218e-02 6.97132436e-02 2.29513887e-01 9.48953897e-04
 2.25535849e-03 1.36938173e-02 0.00000000e+00 8.03806993e-04
 1.10060536e-03 6.81846819e-03 1.50951383e-03 5.06669885e-03
 9.06985407e-04 2.29085151e-01 0.00000000e+00 0.00000000e+00
 0.00000000e+00 2.02826043e-05 3.77049223e-03] 0.034697025632187
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 24.117, MeanIU:  0.0347, Best_mIoU:  0.6808
[9.40362218e-02 6.97132436e-02 2.29513887e-01 9.48953897e-04
 2.25535849e-03 1.36938173e-02 0.00000000e+00 8.03806993e-04
 1.10060536e-03 6.81846819e-03 1.50951383e-03 5.06669885e-03
 9.06985407e-04 2.29085151e-01 0.00000000e+00 0.00000000e+00
 0.00000000e+00 2.02826043e-05 3.77049223e-03]
Epoch: [173/484] Iter:[0/24], Time: 0.95, lr: [0.006716198519130217], Loss: 0.261386
Epoch: [173/484] Iter:[10/24], Time: 0.44, lr: [0.006708099666260742], Loss: 0.319625
Epoch: [173/484] Iter:[20/24], Time: 0.42, lr: [0.006699999726807055], Loss: 0.296199
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.95737197 0.70892012 0.88937753 0.25132436 0.49115473 0.56107101
 0.57007979 0.71497877 0.8996432  0.48093089 0.90806163 0.76320804
 0.50560929 0.90454354 0.38542865 0.53990301 0.3009317  0.44470554
 0.70811053] 0.6308081208347369
1 [0.95426795 0.69728309 0.88416369 0.25466763 0.50788747 0.56774956
 0.58898566 0.72735323 0.90124063 0.43301566 0.91332937 0.76655331
 0.51754152 0.90821378 0.4731709  0.61149042 0.23751363 0.46508589
 0.71336298] 0.6380461247354665
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.317, MeanIU:  0.6380, Best_mIoU:  0.6808
[0.95426795 0.69728309 0.88416369 0.25466763 0.50788747 0.56774956
 0.58898566 0.72735323 0.90124063 0.43301566 0.91332937 0.76655331
 0.51754152 0.90821378 0.4731709  0.61149042 0.23751363 0.46508589
 0.71336298]
Epoch: [174/484] Iter:[0/24], Time: 0.96, lr: [0.006696759446422467], Loss: 0.230679
Epoch: [174/484] Iter:[10/24], Time: 0.44, lr: [0.006688657983052205], Loss: 0.200027
Epoch: [174/484] Iter:[20/24], Time: 0.42, lr: [0.006680555429236297], Loss: 0.212910
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96662123 0.76496036 0.89138873 0.30600723 0.39262004 0.56337845
 0.57845174 0.6989838  0.89964549 0.49751981 0.91895894 0.76001319
 0.46515316 0.91029046 0.38590275 0.52179484 0.37949607 0.36053574
 0.70871544] 0.6300230252605616
1 [0.96432097 0.75707717 0.88739364 0.34700158 0.36009906 0.5665575
 0.59760824 0.70348525 0.90236574 0.51903778 0.92087012 0.76524255
 0.48875443 0.91667538 0.47164108 0.61060821 0.46391542 0.36647739
 0.7097065 ] 0.6483598955130422
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.292, MeanIU:  0.6484, Best_mIoU:  0.6808
[0.96432097 0.75707717 0.88739364 0.34700158 0.36009906 0.5665575
 0.59760824 0.70348525 0.90236574 0.51903778 0.92087012 0.76524255
 0.48875443 0.91667538 0.47164108 0.61060821 0.46391542 0.36647739
 0.7097065 ]
Epoch: [175/484] Iter:[0/24], Time: 0.98, lr: [0.006677314102023172], Loss: 0.165253
Epoch: [175/484] Iter:[10/24], Time: 0.45, lr: [0.006669210018866322], Loss: 0.199714
Epoch: [175/484] Iter:[20/24], Time: 0.42, lr: [0.006661104841376109], Loss: 0.187739
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96964488 0.77618847 0.90101553 0.36138727 0.45916102 0.56910935
 0.58349464 0.71908166 0.90572201 0.51187745 0.92404047 0.75455562
 0.45937124 0.91310118 0.41976341 0.56918668 0.38312206 0.45024663
 0.71805882] 0.649901493712045
1 [0.96927939 0.77850552 0.90321902 0.43345222 0.48738723 0.57676103
 0.60494285 0.72934677 0.9077048  0.5171653  0.92401148 0.75790471
 0.47338503 0.92277285 0.49851622 0.67808421 0.48858095 0.46895673
 0.72229762] 0.6759091540032852
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.261, MeanIU:  0.6759, Best_mIoU:  0.6808
[0.96927939 0.77850552 0.90321902 0.43345222 0.48738723 0.57676103
 0.60494285 0.72934677 0.9077048  0.5171653  0.92401148 0.75790471
 0.47338503 0.92277285 0.49851622 0.67808421 0.48858095 0.46895673
 0.72229762]
Epoch: [176/484] Iter:[0/24], Time: 0.96, lr: [0.006657862463602231], Loss: 0.157817
Epoch: [176/484] Iter:[10/24], Time: 0.44, lr: [0.006649755751309731], Loss: 0.214081
Epoch: [176/484] Iter:[20/24], Time: 0.42, lr: [0.0066416479407696], Loss: 0.209227
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9632911  0.75597535 0.89671928 0.27519089 0.44630313 0.56921747
 0.59319133 0.72047291 0.90296107 0.42997183 0.92299624 0.75782236
 0.5041688  0.91499279 0.42369802 0.58463384 0.35107041 0.46618129
 0.71344122] 0.6416999635405233
1 [0.95977381 0.74376946 0.89442425 0.28090139 0.45913879 0.57838653
 0.60930753 0.73319038 0.90353146 0.3748044  0.9208907  0.7542375
 0.48482752 0.92379551 0.47893076 0.68508348 0.43471919 0.47813993
 0.714289  ] 0.6532706100623047
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.290, MeanIU:  0.6533, Best_mIoU:  0.6808
[0.95977381 0.74376946 0.89442425 0.28090139 0.45913879 0.57838653
 0.60930753 0.73319038 0.90353146 0.3748044  0.9208907  0.7542375
 0.48482752 0.92379551 0.47893076 0.68508348 0.43471919 0.47813993
 0.714289  ]
Epoch: [177/484] Iter:[0/24], Time: 0.97, lr: [0.006638404508677266], Loss: 0.214416
Epoch: [177/484] Iter:[10/24], Time: 0.44, lr: [0.006630295157836152], Loss: 0.184568
Epoch: [177/484] Iter:[20/24], Time: 0.42, lr: [0.006622184704806322], Loss: 0.184730
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96856391 0.77131219 0.90270943 0.30761713 0.51795707 0.57773786
 0.59558421 0.71754748 0.90625808 0.53606789 0.92561205 0.76481772
 0.42974686 0.91081763 0.41694292 0.54954502 0.43232934 0.45342336
 0.72365758] 0.6530656703540437
1 [0.96860436 0.7733654  0.90362747 0.35267654 0.52815463 0.58759825
 0.61851121 0.73150379 0.90891024 0.55881333 0.92799879 0.77313413
 0.46416859 0.92016898 0.46426807 0.66778151 0.53762671 0.46976883
 0.72923236] 0.6782059576930323
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.261, MeanIU:  0.6782, Best_mIoU:  0.6808
[0.96860436 0.7733654  0.90362747 0.35267654 0.52815463 0.58759825
 0.61851121 0.73150379 0.90891024 0.55881333 0.92799879 0.77313413
 0.46416859 0.92016898 0.46426807 0.66778151 0.53762671 0.46976883
 0.72923236]
Epoch: [178/484] Iter:[0/24], Time: 0.95, lr: [0.006618940214612088], Loss: 0.171243
Epoch: [178/484] Iter:[10/24], Time: 0.44, lr: [0.006610828215744843], Loss: 0.179601
Epoch: [178/484] Iter:[20/24], Time: 0.42, lr: [0.006602715110720718], Loss: 0.190019
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96362408 0.74871105 0.90047845 0.28590442 0.48468726 0.56841803
 0.59069836 0.71456278 0.90753679 0.54719673 0.92542731 0.7598683
 0.49331286 0.89437177 0.41773292 0.57563277 0.40206186 0.25643399
 0.7153027 ] 0.6395769704030351
1 [0.96232759 0.74358028 0.90256864 0.35260851 0.50810531 0.5785872
 0.61068652 0.72856771 0.91041472 0.5454917  0.92721785 0.76731499
 0.5099185  0.89579561 0.55495545 0.71263255 0.52677704 0.21427889
 0.72095394] 0.6669885800247503
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.281, MeanIU:  0.6670, Best_mIoU:  0.6808
[0.96232759 0.74358028 0.90256864 0.35260851 0.50810531 0.5785872
 0.61068652 0.72856771 0.91041472 0.5454917  0.92721785 0.76731499
 0.5099185  0.89579561 0.55495545 0.71263255 0.52677704 0.21427889
 0.72095394]
Epoch: [179/484] Iter:[0/24], Time: 0.94, lr: [0.006599469558615132], Loss: 0.154581
Epoch: [179/484] Iter:[10/24], Time: 0.44, lr: [0.006591354902179042], Loss: 0.222051
Epoch: [179/484] Iter:[20/24], Time: 0.42, lr: [0.006583239135590537], Loss: 0.201646
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96801123 0.76739665 0.88798246 0.24395757 0.48921038 0.55062958
 0.57303104 0.71600173 0.9032875  0.5128063  0.8817227  0.75845236
 0.50494425 0.8986568  0.413672   0.47739134 0.37605173 0.36289288
 0.71815039] 0.6318025728967379
1 [0.96707125 0.76336821 0.88709252 0.25770708 0.49770373 0.55792676
 0.59330456 0.72881229 0.90363861 0.48994136 0.88401259 0.76488217
 0.51543096 0.90428042 0.44699058 0.57306294 0.44147452 0.36861046
 0.72597258] 0.645857030790855
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.303, MeanIU:  0.6459, Best_mIoU:  0.6808
[0.96707125 0.76336821 0.88709252 0.25770708 0.49770373 0.55792676
 0.59330456 0.72881229 0.90363861 0.48994136 0.88401259 0.76488217
 0.51543096 0.90428042 0.44699058 0.57306294 0.44147452 0.36861046
 0.72597258]
Epoch: [180/484] Iter:[0/24], Time: 0.96, lr: [0.00657999251773788], Loss: 0.289734
Epoch: [180/484] Iter:[10/24], Time: 0.44, lr: [0.006571875194124363], Loss: 0.203067
Epoch: [180/484] Iter:[20/24], Time: 0.42, lr: [0.006563756756335252], Loss: 0.207888
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96760856 0.7687418  0.89741388 0.24578251 0.51465327 0.56332044
 0.59434556 0.70864062 0.90372373 0.45735486 0.92111954 0.76321064
 0.50579726 0.90162078 0.42883238 0.56262933 0.41957398 0.45673051
 0.70996935] 0.6468983686219222
1 [0.96766838 0.76798266 0.89572508 0.25452195 0.53408659 0.57458919
 0.59452771 0.71686354 0.90320102 0.45897952 0.91733091 0.76777258
 0.50984736 0.91672992 0.50956535 0.71734253 0.5815229  0.46212043
 0.70991832] 0.6715945226273311
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.284, MeanIU:  0.6716, Best_mIoU:  0.6808
[0.96766838 0.76798266 0.89572508 0.25452195 0.53408659 0.57458919
 0.59452771 0.71686354 0.90320102 0.45897952 0.91733091 0.76777258
 0.50984736 0.91672992 0.50956535 0.71734253 0.5815229  0.46212043
 0.70991832]
Epoch: [181/484] Iter:[0/24], Time: 0.93, lr: [0.006560509068873266], Loss: 0.207362
Epoch: [181/484] Iter:[10/24], Time: 0.44, lr: [0.0065523890684071944], Loss: 0.195533
Epoch: [181/484] Iter:[20/24], Time: 0.42, lr: [0.006544267949714427], Loss: 0.178342
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96980508 0.77552881 0.89281835 0.22847477 0.50974936 0.57213222
 0.58334414 0.71427907 0.8998872  0.51890323 0.93089791 0.77132857
 0.51914032 0.9145352  0.4901318  0.58283053 0.4842023  0.49097288
 0.7285786 ] 0.661975806800791
1 [0.96862969 0.77475972 0.88489467 0.19091048 0.5268816  0.57611073
 0.60322166 0.71759579 0.89828849 0.52451989 0.93361858 0.77700529
 0.53262577 0.9263352  0.59611043 0.70524283 0.5213163  0.52892777
 0.73155374] 0.6799236124261239
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.284, MeanIU:  0.6799, Best_mIoU:  0.6808
[0.96862969 0.77475972 0.88489467 0.19091048 0.5268816  0.57611073
 0.60322166 0.71759579 0.89828849 0.52451989 0.93361858 0.77700529
 0.53262577 0.9263352  0.59611043 0.70524283 0.5213163  0.52892777
 0.73155374]
Epoch: [182/484] Iter:[0/24], Time: 1.00, lr: [0.006541019188754046], Loss: 0.157042
Epoch: [182/484] Iter:[10/24], Time: 0.45, lr: [0.00653289650169307], Loss: 0.184696
Epoch: [182/484] Iter:[20/24], Time: 0.42, lr: [0.0065247726923260835], Loss: 0.184867
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97052944 0.77894367 0.89896742 0.35039248 0.4705397  0.57203387
 0.58913746 0.72424746 0.9083015  0.52578019 0.93421994 0.77647805
 0.51985748 0.91451183 0.46542554 0.54639169 0.39393134 0.47248057
 0.72576166] 0.6598911203007474
1 [0.96903493 0.77997844 0.89735864 0.40463613 0.48588472 0.5862962
 0.61177556 0.73124049 0.91092617 0.54362068 0.93489042 0.78262879
 0.5384642  0.92468862 0.60097181 0.62110854 0.30785735 0.50348947
 0.73105432] 0.6771529205501204
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.264, MeanIU:  0.6772, Best_mIoU:  0.6808
[0.96903493 0.77997844 0.89735864 0.40463613 0.48588472 0.5862962
 0.61177556 0.73124049 0.91092617 0.54362068 0.93489042 0.78262879
 0.5384642  0.92468862 0.60097181 0.62110854 0.30785735 0.50348947
 0.73105432]
Epoch: [183/484] Iter:[0/24], Time: 0.97, lr: [0.006521522853951154], Loss: 0.221046
Epoch: [183/484] Iter:[10/24], Time: 0.44, lr: [0.006513397470485011], Loss: 0.156822
Epoch: [183/484] Iter:[20/24], Time: 0.42, lr: [0.006505270960605032], Loss: 0.163110
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9721643  0.7885619  0.9039405  0.36013329 0.51078079 0.57941435
 0.60809901 0.72641141 0.90816763 0.53705473 0.93225862 0.77563473
 0.49858913 0.91770006 0.45191229 0.54931414 0.39198123 0.5315823
 0.72087337] 0.6665565144151969
1 [0.97174291 0.78860135 0.90406388 0.39049474 0.52659385 0.58480755
 0.61918341 0.73676259 0.91112772 0.56925481 0.93510633 0.78264008
 0.52448594 0.92713114 0.59286496 0.62320495 0.3183213  0.55208882
 0.72791563] 0.6834943140860752
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.6835, Best_mIoU:  0.6835
[0.97174291 0.78860135 0.90406388 0.39049474 0.52659385 0.58480755
 0.61918341 0.73676259 0.91112772 0.56925481 0.93510633 0.78264008
 0.52448594 0.92713114 0.59286496 0.62320495 0.3183213  0.55208882
 0.72791563]
Epoch: [184/484] Iter:[0/24], Time: 0.96, lr: [0.00650202004087204], Loss: 0.164269
Epoch: [184/484] Iter:[10/24], Time: 0.44, lr: [0.006493891951121848], Loss: 0.165758
Epoch: [184/484] Iter:[20/24], Time: 0.42, lr: [0.0064857627308211995], Loss: 0.159542
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96925147 0.76811236 0.9027654  0.31883461 0.50985615 0.58538694
 0.60021329 0.72129527 0.90766994 0.47615946 0.93042818 0.77298725
 0.49063283 0.91904218 0.44765907 0.60239852 0.42580795 0.48879707
 0.72348245] 0.6610937035604226
1 [0.96996158 0.76725444 0.9033913  0.3323195  0.52550492 0.59169416
 0.61588569 0.72505297 0.90965649 0.47280965 0.93248397 0.78004614
 0.51498802 0.92692546 0.58435911 0.69017293 0.41235991 0.51073628
 0.73088328] 0.6787624099781023
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.263, MeanIU:  0.6788, Best_mIoU:  0.6835
[0.96996158 0.76725444 0.9033913  0.3323195  0.52550492 0.59169416
 0.61588569 0.72505297 0.90965649 0.47280965 0.93248397 0.78004614
 0.51498802 0.92692546 0.58435911 0.69017293 0.41235991 0.51073628
 0.73088328]
Epoch: [185/484] Iter:[0/24], Time: 0.96, lr: [0.00648251072575898], Loss: 0.238658
Epoch: [185/484] Iter:[10/24], Time: 0.44, lr: [0.006474379919776527], Loss: 0.191345
Epoch: [185/484] Iter:[20/24], Time: 0.42, lr: [0.006466247979077902], Loss: 0.183058
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96793062 0.76749493 0.90718735 0.38163407 0.50680728 0.58275726
 0.6029285  0.71894405 0.90742385 0.50788841 0.93108847 0.76944525
 0.52234578 0.91208131 0.45539965 0.60279809 0.47483281 0.4856254
 0.72516827] 0.6699884918318544
1 [0.96728503 0.76589677 0.90923893 0.43339405 0.52039786 0.59229177
 0.62674207 0.73304914 0.91040631 0.52640971 0.93353126 0.7761343
 0.53925804 0.92374879 0.52298746 0.69764094 0.51712798 0.50844932
 0.73115716] 0.6913235207443984
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.257, MeanIU:  0.6913, Best_mIoU:  0.6913
[0.96728503 0.76589677 0.90923893 0.43339405 0.52039786 0.59229177
 0.62674207 0.73304914 0.91040631 0.52640971 0.93353126 0.7761343
 0.53925804 0.92374879 0.52298746 0.69764094 0.51712798 0.50844932
 0.73115716]
Epoch: [186/484] Iter:[0/24], Time: 0.95, lr: [0.006462994884687355], Loss: 0.093086
Epoch: [186/484] Iter:[10/24], Time: 0.44, lr: [0.006454861352454385], Loss: 0.165958
Epoch: [186/484] Iter:[20/24], Time: 0.42, lr: [0.006446726681310124], Loss: 0.158957
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97210541 0.78491784 0.90427056 0.31874002 0.5147468  0.58119315
 0.60600729 0.72171846 0.90815244 0.49630479 0.93045518 0.77270078
 0.50919651 0.91400143 0.45750918 0.57810448 0.45303008 0.4842812
 0.72701907] 0.6649712983536243
1 [0.97295581 0.78713246 0.90447423 0.35149609 0.53217175 0.58655746
 0.60059139 0.72133983 0.91048949 0.51376089 0.93044822 0.77849429
 0.52759805 0.921491   0.53437147 0.65612277 0.45774715 0.50404251
 0.73644265] 0.6804067120135036
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.6804, Best_mIoU:  0.6913
[0.97295581 0.78713246 0.90447423 0.35149609 0.53217175 0.58655746
 0.60059139 0.72133983 0.91048949 0.51376089 0.93044822 0.77849429
 0.52759805 0.921491   0.53437147 0.65612277 0.45774715 0.50404251
 0.73644265]
Epoch: [187/484] Iter:[0/24], Time: 0.96, lr: [0.0064434724935639245], Loss: 0.190043
Epoch: [187/484] Iter:[10/24], Time: 0.44, lr: [0.006435336224991404], Loss: 0.164775
Epoch: [187/484] Iter:[20/24], Time: 0.42, lr: [0.006427198813282757], Loss: 0.166758
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97217485 0.78304703 0.90452796 0.30998651 0.47254066 0.58106821
 0.60941313 0.72595464 0.90850178 0.50923367 0.93203328 0.77014986
 0.51836002 0.92027346 0.47554016 0.59534699 0.41735204 0.51250256
 0.72332295] 0.6653331450321183
1 [0.97223549 0.78462339 0.90658946 0.36458436 0.49888924 0.5940348
 0.6291683  0.7360565  0.9112092  0.530171   0.93129072 0.76957925
 0.5210267  0.92917089 0.5736068  0.67186518 0.36497591 0.51820541
 0.73329691] 0.6810831321651869
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.258, MeanIU:  0.6811, Best_mIoU:  0.6913
[0.97223549 0.78462339 0.90658946 0.36458436 0.49888924 0.5940348
 0.6291683  0.7360565  0.9112092  0.530171   0.93129072 0.76957925
 0.5210267  0.92917089 0.5736068  0.67186518 0.36497591 0.51820541
 0.73329691]
Epoch: [188/484] Iter:[0/24], Time: 0.94, lr: [0.006423943528125062], Loss: 0.176525
Epoch: [188/484] Iter:[10/24], Time: 0.44, lr: [0.006415804513052432], Loss: 0.172467
Epoch: [188/484] Iter:[20/24], Time: 0.42, lr: [0.0064076643505888275], Loss: 0.164692
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97259012 0.78932102 0.90106556 0.28493793 0.47014414 0.5840537
 0.61439676 0.72663738 0.90700499 0.51807831 0.92546336 0.77565993
 0.51856703 0.92033506 0.47801305 0.6044145  0.45920887 0.51366928
 0.73004138] 0.6680843349152001
1 [0.9712853  0.7864714  0.90258043 0.33285426 0.49660363 0.59402765
 0.6260872  0.739134   0.90989049 0.52849084 0.92961121 0.78340079
 0.53511523 0.92838599 0.57537702 0.69921982 0.44710972 0.53294187
 0.73740654] 0.6871575469202424
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.263, MeanIU:  0.6872, Best_mIoU:  0.6913
[0.9712853  0.7864714  0.90258043 0.33285426 0.49660363 0.59402765
 0.6260872  0.739134   0.90989049 0.52849084 0.92961121 0.78340079
 0.53511523 0.92838599 0.57537702 0.69921982 0.44710972 0.53294187
 0.73740654]
Epoch: [189/484] Iter:[0/24], Time: 0.99, lr: [0.006404407963934968], Loss: 0.102716
Epoch: [189/484] Iter:[10/24], Time: 0.44, lr: [0.0063962661921294055], Loss: 0.170235
Epoch: [189/484] Iter:[20/24], Time: 0.42, lr: [0.006388123268647679], Loss: 0.162532
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96702857 0.76158963 0.85913279 0.23319913 0.4315978  0.57104358
 0.59374246 0.72059823 0.90152448 0.46217302 0.69982763 0.76766975
 0.51915971 0.91665319 0.47277078 0.53642976 0.27350623 0.47238846
 0.72194514] 0.6253673863779337
1 [0.96517888 0.75955573 0.85338838 0.22307469 0.42836901 0.57532106
 0.61230477 0.73004515 0.90295103 0.45314374 0.70431856 0.77396186
 0.52910362 0.92475776 0.58143574 0.62217091 0.17198861 0.49253623
 0.72945229] 0.6333188436336509
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.340, MeanIU:  0.6333, Best_mIoU:  0.6913
[0.96517888 0.75955573 0.85338838 0.22307469 0.42836901 0.57532106
 0.61230477 0.73004515 0.90295103 0.45314374 0.70431856 0.77396186
 0.52910362 0.92475776 0.58143574 0.62217091 0.17198861 0.49253623
 0.72945229]
Epoch: [190/484] Iter:[0/24], Time: 1.01, lr: [0.006384865776383867], Loss: 0.116515
Epoch: [190/484] Iter:[10/24], Time: 0.45, lr: [0.0063767212375395125], Loss: 0.196576
Epoch: [190/484] Iter:[20/24], Time: 0.42, lr: [0.006368575542703148], Loss: 0.191036
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96973377 0.77583323 0.87448047 0.27349291 0.30014162 0.56520192
 0.58604975 0.71597208 0.90115008 0.50999821 0.85630215 0.77328982
 0.51676396 0.91893942 0.44832108 0.59512513 0.31574354 0.51049131
 0.7251344 ] 0.6385349924921795
1 [0.97006282 0.77921238 0.87381883 0.31523332 0.29711944 0.57497492
 0.60467918 0.72812267 0.90172589 0.52460285 0.84908702 0.77976248
 0.53479593 0.92668765 0.55467208 0.68055412 0.27143243 0.54147875
 0.73467659] 0.6548789132317336
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.305, MeanIU:  0.6549, Best_mIoU:  0.6913
[0.97006282 0.77921238 0.87381883 0.31523332 0.29711944 0.57497492
 0.60467918 0.72812267 0.90172589 0.52460285 0.84908702 0.77976248
 0.53479593 0.92668765 0.55467208 0.68055412 0.27143243 0.54147875
 0.73467659]
Epoch: [191/484] Iter:[0/24], Time: 0.97, lr: [0.006365316940686164], Loss: 0.238451
Epoch: [191/484] Iter:[10/24], Time: 0.44, lr: [0.006357169624423356], Loss: 0.227764
Epoch: [191/484] Iter:[20/24], Time: 0.42, lr: [0.0063490211478217096], Loss: 0.227590
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96508097 0.76686444 0.89169812 0.27850169 0.47546455 0.56625593
 0.60173307 0.70793548 0.90155026 0.50266158 0.9019597  0.76917194
 0.4874469  0.89426858 0.37560442 0.59238457 0.36948811 0.38824978
 0.7127838 ] 0.6394265199572439
1 [0.9650558  0.77036812 0.89424116 0.31062246 0.49503105 0.57776893
 0.62105832 0.71442113 0.90309389 0.52128214 0.91327384 0.77419863
 0.49953781 0.90520717 0.44608411 0.64639509 0.32509861 0.37640635
 0.72080305] 0.6515761937254227
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.284, MeanIU:  0.6516, Best_mIoU:  0.6913
[0.9650558  0.77036812 0.89424116 0.31062246 0.49503105 0.57776893
 0.62105832 0.71442113 0.90309389 0.52128214 0.91327384 0.77419863
 0.49953781 0.90520717 0.44608411 0.64639509 0.32509861 0.37640635
 0.72080305]
Epoch: [192/484] Iter:[0/24], Time: 0.97, lr: [0.00634576143187859], Loss: 0.287933
Epoch: [192/484] Iter:[10/24], Time: 0.44, lr: [0.00633761132774308], Loss: 0.171759
Epoch: [192/484] Iter:[20/24], Time: 0.42, lr: [0.006329460058890594], Loss: 0.185839
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96731749 0.76976366 0.88474356 0.21723552 0.30379175 0.55244382
 0.58619131 0.70702147 0.90674816 0.49263488 0.92527904 0.74562719
 0.45037034 0.91570352 0.45145386 0.5632742  0.37430274 0.41713387
 0.69679327] 0.6277805070302247
1 [0.96711239 0.76961279 0.87734008 0.19752143 0.26749809 0.56081466
 0.60559841 0.71042344 0.90804079 0.50205741 0.92794049 0.75669908
 0.48993888 0.92059273 0.43747467 0.62892244 0.2957277  0.43696828
 0.69714753] 0.6293384890508992
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.315, MeanIU:  0.6293, Best_mIoU:  0.6913
[0.96711239 0.76961279 0.87734008 0.19752143 0.26749809 0.56081466
 0.60559841 0.71042344 0.90804079 0.50205741 0.92794049 0.75669908
 0.48993888 0.92059273 0.43747467 0.62892244 0.2957277  0.43696828
 0.69714753]
Epoch: [193/484] Iter:[0/24], Time: 0.95, lr: [0.006326199224818318], Loss: 0.203486
Epoch: [193/484] Iter:[10/24], Time: 0.44, lr: [0.006318046322280474], Loss: 0.175423
Epoch: [193/484] Iter:[20/24], Time: 0.42, lr: [0.006309892250615873], Loss: 0.174350
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96996751 0.77958979 0.89851666 0.26722878 0.45403453 0.57473884
 0.59466737 0.69949845 0.89978644 0.48568506 0.92954195 0.76991999
 0.50893536 0.91522313 0.45491273 0.56892437 0.42349895 0.43718733
 0.70420846] 0.6492666158508182
1 [0.96991807 0.78034512 0.89936714 0.28841497 0.47944779 0.58208661
 0.61605921 0.70513366 0.90300471 0.50629874 0.92876096 0.77201044
 0.52895475 0.91828236 0.47463597 0.61964173 0.47921818 0.44531304
 0.69978894] 0.6629832838362973
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.276, MeanIU:  0.6630, Best_mIoU:  0.6913
[0.96991807 0.78034512 0.89936714 0.28841497 0.47944779 0.58208661
 0.61605921 0.70513366 0.90300471 0.50629874 0.92876096 0.77201044
 0.52895475 0.91828236 0.47463597 0.61964173 0.47921818 0.44531304
 0.69978894]
Epoch: [194/484] Iter:[0/24], Time: 0.97, lr: [0.006306630294181036], Loss: 0.174262
Epoch: [194/484] Iter:[10/24], Time: 0.44, lr: [0.006298474582635048], Loss: 0.149353
Epoch: [194/484] Iter:[20/24], Time: 0.42, lr: [0.00629031769752053], Loss: 0.167570
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97210546 0.79548148 0.90149939 0.30036793 0.46271873 0.58472852
 0.60989781 0.72204474 0.90715646 0.53066298 0.931615   0.7738606
 0.51014931 0.91268851 0.4590712  0.5468838  0.45641222 0.49810528
 0.71869785] 0.6628498565881844
1 [0.97257299 0.79748619 0.90197956 0.32108105 0.47213748 0.59295876
 0.62437692 0.73062326 0.90905606 0.52941808 0.93031056 0.78088014
 0.54058576 0.91464453 0.48699695 0.58181224 0.43411515 0.51832421
 0.7241568 ] 0.6717640364578158
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.263, MeanIU:  0.6718, Best_mIoU:  0.6913
[0.97257299 0.79748619 0.90197956 0.32108105 0.47213748 0.59295876
 0.62437692 0.73062326 0.90905606 0.52941808 0.93031056 0.78088014
 0.54058576 0.91464453 0.48699695 0.58181224 0.43411515 0.51832421
 0.7241568 ]
Epoch: [195/484] Iter:[0/24], Time: 1.00, lr: [0.0062870546144590264], Loss: 0.312923
Epoch: [195/484] Iter:[10/24], Time: 0.45, lr: [0.006278896083222074], Loss: 0.194912
Epoch: [195/484] Iter:[20/24], Time: 0.42, lr: [0.0062707363739424975], Loss: 0.208340
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97017967 0.76847493 0.89766965 0.31635638 0.4951144  0.54590032
 0.59494997 0.70629195 0.90245666 0.50286129 0.91766265 0.76770094
 0.51301845 0.89831692 0.38436267 0.41542311 0.22502459 0.43667832
 0.722736  ] 0.6305883620637974
1 [0.97023753 0.77007146 0.89925137 0.35616046 0.5148513  0.55845941
 0.61472075 0.71437694 0.90497071 0.52417123 0.9127491  0.77343046
 0.52640526 0.9032209  0.39940428 0.44990367 0.12553732 0.45378893
 0.72813937] 0.636834233427707
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.281, MeanIU:  0.6368, Best_mIoU:  0.6913
[0.97023753 0.77007146 0.89925137 0.35616046 0.5148513  0.55845941
 0.61472075 0.71437694 0.90497071 0.52417123 0.9127491  0.77343046
 0.52640526 0.9032209  0.39940428 0.44990367 0.12553732 0.45378893
 0.72813937]
Epoch: [196/484] Iter:[0/24], Time: 0.95, lr: [0.006267472159959184], Loss: 0.153299
Epoch: [196/484] Iter:[10/24], Time: 0.44, lr: [0.006259310798270619], Loss: 0.193560
Epoch: [196/484] Iter:[20/24], Time: 0.42, lr: [0.006251148254032658], Loss: 0.177310
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97165658 0.77633562 0.90195189 0.32298337 0.48277379 0.56735595
 0.60357559 0.71605541 0.90553999 0.4950762  0.92603034 0.77273205
 0.52292233 0.91121377 0.41555418 0.51370961 0.34625602 0.49647602
 0.72129886] 0.6510261880050818
1 [0.97140597 0.78085164 0.90282353 0.35967014 0.50443282 0.57926915
 0.6194943  0.72837383 0.90729957 0.51340548 0.92317849 0.77794449
 0.53106082 0.91969199 0.47080865 0.647567   0.34191987 0.52168971
 0.72712949] 0.6698956275604616
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.262, MeanIU:  0.6699, Best_mIoU:  0.6913
[0.97140597 0.78085164 0.90282353 0.35967014 0.50443282 0.57926915
 0.6194943  0.72837383 0.90729957 0.51340548 0.92317849 0.77794449
 0.53106082 0.91969199 0.47080865 0.647567   0.34191987 0.52168971
 0.72712949]
Epoch: [197/484] Iter:[0/24], Time: 0.94, lr: [0.006247882904801023], Loss: 0.166476
Epoch: [197/484] Iter:[10/24], Time: 0.44, lr: [0.006239718701821529], Loss: 0.160965
Epoch: [197/484] Iter:[20/24], Time: 0.42, lr: [0.0062315533117528335], Loss: 0.172184
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97353314 0.78883051 0.90499086 0.31518094 0.47497724 0.58409648
 0.60663469 0.72230883 0.90752202 0.52416772 0.93403008 0.77238608
 0.50378817 0.91794423 0.44897638 0.56259184 0.27044429 0.51353089
 0.72466136] 0.6552945129397493
1 [0.97351321 0.78947745 0.90614065 0.3553712  0.48476235 0.59310844
 0.62629287 0.73360494 0.91041322 0.55512967 0.93264018 0.77839731
 0.5271724  0.92458967 0.50639788 0.61549338 0.25952147 0.52812013
 0.72904653] 0.6699575237133616
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6700, Best_mIoU:  0.6913
[0.97351321 0.78947745 0.90614065 0.3553712  0.48476235 0.59310844
 0.62629287 0.73360494 0.91041322 0.55512967 0.93264018 0.77839731
 0.5271724  0.92458967 0.50639788 0.61549338 0.25952147 0.52812013
 0.72904653]
Epoch: [198/484] Iter:[0/24], Time: 0.98, lr: [0.006228286822914655], Loss: 0.173520
Epoch: [198/484] Iter:[10/24], Time: 0.44, lr: [0.006220119767725392], Loss: 0.171609
Epoch: [198/484] Iter:[20/24], Time: 0.42, lr: [0.0062119515208737324], Loss: 0.172883
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97304678 0.78534635 0.90406354 0.28007267 0.49514102 0.57666975
 0.61179204 0.71170888 0.90839315 0.5062785  0.93166755 0.77011862
 0.52517648 0.91041157 0.44782859 0.53475067 0.47012754 0.519708
 0.71857827] 0.6621515774689785
1 [0.97337185 0.78349492 0.90640914 0.31934294 0.51412018 0.58951176
 0.62777235 0.72967997 0.91114978 0.50491151 0.93217169 0.77663823
 0.53352515 0.91857092 0.46948932 0.63665311 0.54887372 0.53516035
 0.72418862] 0.6807913423087243
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.266, MeanIU:  0.6808, Best_mIoU:  0.6913
[0.97337185 0.78349492 0.90640914 0.31934294 0.51412018 0.58951176
 0.62777235 0.72967997 0.91114978 0.50491151 0.93217169 0.77663823
 0.53352515 0.91857092 0.46948932 0.63665311 0.54887372 0.53516035
 0.72418862]
Epoch: [199/484] Iter:[0/24], Time: 0.97, lr: [0.0062086838880387385], Loss: 0.231703
Epoch: [199/484] Iter:[10/24], Time: 0.44, lr: [0.006200513969640476], Loss: 0.173370
Epoch: [199/484] Iter:[20/24], Time: 0.42, lr: [0.006192342854972875], Loss: 0.152385
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97182956 0.78727364 0.90186235 0.31506586 0.48630327 0.57509977
 0.61027686 0.7208684  0.90977985 0.50901706 0.93213679 0.77573612
 0.5197089  0.91196634 0.44778466 0.55951638 0.47098754 0.50419487
 0.72354417] 0.664892230304429
1 [0.97158717 0.78635795 0.90151257 0.32682388 0.5083602  0.58441295
 0.62937986 0.73633438 0.91190918 0.51103028 0.93299717 0.78216094
 0.5389199  0.91870922 0.47198303 0.64386802 0.51826048 0.52673764
 0.73342327] 0.680777267807595
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.267, MeanIU:  0.6808, Best_mIoU:  0.6913
[0.97158717 0.78635795 0.90151257 0.32682388 0.5083602  0.58441295
 0.62937986 0.73633438 0.91190918 0.51103028 0.93299717 0.78216094
 0.5389199  0.91870922 0.47198303 0.64386802 0.51826048 0.52673764
 0.73342327]
Epoch: [200/484] Iter:[0/24], Time: 0.98, lr: [0.0061890740737183874], Loss: 0.186695
Epoch: [200/484] Iter:[10/24], Time: 0.44, lr: [0.0061809012810306345], Loss: 0.174227
Epoch: [200/484] Iter:[20/24], Time: 0.42, lr: [0.006172727287432478], Loss: 0.185381
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97044364 0.78394852 0.9037483  0.33166548 0.49552765 0.57666968
 0.60178738 0.72697791 0.90773667 0.52625532 0.93066361 0.77266151
 0.5137238  0.92147111 0.48533891 0.63445968 0.41363013 0.5255726
 0.72254662] 0.6707804491071622
1 [0.97091639 0.78694765 0.90335197 0.33142131 0.50823921 0.58298152
 0.62205724 0.73898516 0.90998525 0.54615666 0.9307297  0.77845823
 0.52953601 0.92667981 0.53984805 0.69453505 0.38473951 0.53465381
 0.72636595] 0.6813993934058018
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.257, MeanIU:  0.6814, Best_mIoU:  0.6913
[0.97091639 0.78694765 0.90335197 0.33142131 0.50823921 0.58298152
 0.62205724 0.73898516 0.90998525 0.54615666 0.9307297  0.77845823
 0.52953601 0.92667981 0.53984805 0.69453505 0.38473951 0.53465381
 0.72636595]
Epoch: [201/484] Iter:[0/24], Time: 0.97, lr: [0.006169457353303062], Loss: 0.114983
Epoch: [201/484] Iter:[10/24], Time: 0.44, lr: [0.006161281675163178], Loss: 0.165812
Epoch: [201/484] Iter:[20/24], Time: 0.42, lr: [0.006153104791437326], Loss: 0.158995
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97241217 0.78481329 0.90365497 0.37571709 0.49280782 0.5795424
 0.61314247 0.72013028 0.90747978 0.49591029 0.92208359 0.77716664
 0.52323039 0.91401123 0.34580235 0.39229872 0.42011118 0.51375546
 0.73060373] 0.6518249390247358
1 [0.97253067 0.78967574 0.90562    0.42857713 0.51593032 0.5857466
 0.62104242 0.72815182 0.90990203 0.5182653  0.92439942 0.785167
 0.54414202 0.91680769 0.32759522 0.31888442 0.3578552  0.5400378
 0.73620129] 0.6540280048551012
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.267, MeanIU:  0.6540, Best_mIoU:  0.6913
[0.97253067 0.78967574 0.90562    0.42857713 0.51593032 0.5857466
 0.62104242 0.72815182 0.90990203 0.5182653  0.92439942 0.785167
 0.54414202 0.91680769 0.32759522 0.31888442 0.3578552  0.5400378
 0.73620129]
Epoch: [202/484] Iter:[0/24], Time: 0.95, lr: [0.006149833699944432], Loss: 0.107239
Epoch: [202/484] Iter:[10/24], Time: 0.44, lr: [0.006141655125106717], Loss: 0.188328
Epoch: [202/484] Iter:[20/24], Time: 0.42, lr: [0.006133475339972595], Loss: 0.187961
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97104759 0.78172486 0.89711038 0.26039124 0.46490726 0.57758628
 0.5991981  0.71933455 0.90401599 0.51684937 0.92096829 0.77229219
 0.51981033 0.9072141  0.34641584 0.4966689  0.41534235 0.47098733
 0.71618767] 0.6451606636950682
1 [0.97114275 0.78050013 0.89515478 0.24101009 0.47110297 0.58157939
 0.6161773  0.72797521 0.90464844 0.52672255 0.91748108 0.77841465
 0.54056446 0.91205713 0.38016388 0.564499   0.39473564 0.49825444
 0.71867997] 0.6537296761064951
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.281, MeanIU:  0.6537, Best_mIoU:  0.6913
[0.97114275 0.78050013 0.89515478 0.24101009 0.47110297 0.58157939
 0.6161773  0.72797521 0.90464844 0.52672255 0.91748108 0.77841465
 0.54056446 0.91205713 0.38016388 0.564499   0.39473564 0.49825444
 0.71867997]
Epoch: [203/484] Iter:[0/24], Time: 0.97, lr: [0.006130203086594193], Loss: 0.138368
Epoch: [203/484] Iter:[10/24], Time: 0.44, lr: [0.006122021603728975], Loss: 0.166289
Epoch: [203/484] Iter:[20/24], Time: 0.42, lr: [0.006113838905821655], Loss: 0.175806
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96906589 0.77377878 0.89432494 0.36854444 0.46738771 0.56569614
 0.59162018 0.70342751 0.90568899 0.48572026 0.92624603 0.75631771
 0.49372145 0.90090731 0.33978601 0.55179335 0.28533812 0.48180719
 0.71499522] 0.6408509068321306
1 [0.97003223 0.78119313 0.8953568  0.42298365 0.48048491 0.57135882
 0.60907458 0.71127968 0.90775729 0.49393328 0.93030235 0.7609431
 0.50151163 0.89729538 0.35306439 0.60063921 0.24046167 0.49267476
 0.71143552] 0.649041177845097
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.286, MeanIU:  0.6490, Best_mIoU:  0.6913
[0.97003223 0.78119313 0.8953568  0.42298365 0.48048491 0.57135882
 0.60907458 0.71127968 0.90775729 0.49393328 0.93030235 0.7609431
 0.50151163 0.89729538 0.35306439 0.60063921 0.24046167 0.49267476
 0.71143552]
Epoch: [204/484] Iter:[0/24], Time: 0.97, lr: [0.006110565486001861], Loss: 0.161420
Epoch: [204/484] Iter:[10/24], Time: 0.44, lr: [0.006102381083694573], Loss: 0.162731
Epoch: [204/484] Iter:[20/24], Time: 0.42, lr: [0.00609419546156383], Loss: 0.152668
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9742736  0.79264604 0.90236674 0.33922657 0.49425882 0.58723493
 0.60847654 0.72513309 0.90736055 0.50491589 0.92039914 0.77708377
 0.52811769 0.9100422  0.40843002 0.50893892 0.43559543 0.53012065
 0.72492778] 0.6620814913651515
1 [0.97516491 0.79729007 0.90237925 0.39046686 0.50919811 0.5930331
 0.62647731 0.73684246 0.90952823 0.52547826 0.91721705 0.78080483
 0.53925972 0.91737224 0.45135846 0.60654757 0.40629913 0.54235846
 0.73071135] 0.6767256507606995
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.6767, Best_mIoU:  0.6913
[0.97516491 0.79729007 0.90237925 0.39046686 0.50919811 0.5930331
 0.62647731 0.73684246 0.90952823 0.52547826 0.91721705 0.78080483
 0.53925972 0.91737224 0.45135846 0.60654757 0.40629913 0.54235846
 0.73071135]
Epoch: [205/484] Iter:[0/24], Time: 0.96, lr: [0.006090920870712538], Loss: 0.131261
Epoch: [205/484] Iter:[10/24], Time: 0.44, lr: [0.006082733537462769], Loss: 0.187734
Epoch: [205/484] Iter:[20/24], Time: 0.42, lr: [0.006074544979572143], Loss: 0.174667
00

0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97288999 0.78254626 0.9048679  0.34973416 0.498996   0.58314418
 0.6097363  0.72516308 0.91001961 0.49732966 0.93129812 0.77790442
 0.53146285 0.90624499 0.37930156 0.47330257 0.48075502 0.52855684
 0.73007784] 0.6617542818868779
1 [0.97267927 0.78470561 0.90615419 0.40728297 0.51508162 0.59107362
 0.62411823 0.73288387 0.91127482 0.51065358 0.93235119 0.78278617
 0.54551063 0.91538047 0.39030742 0.58545277 0.56167632 0.5548875
 0.73466202] 0.6820485403971405
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.258, MeanIU:  0.6820, Best_mIoU:  0.6913
[0.97267927 0.78470561 0.90615419 0.40728297 0.51508162 0.59107362
 0.62411823 0.73288387 0.91127482 0.51065358 0.93235119 0.78278617
 0.54551063 0.91538047 0.39030742 0.58545277 0.56167632 0.5548875
 0.73466202]
Epoch: [206/484] Iter:[0/24], Time: 0.97, lr: [0.006071269213064638], Loss: 0.323329
Epoch: [206/484] Iter:[10/24], Time: 0.44, lr: [0.006063078937285177], Loss: 0.171966
Epoch: [206/484] Iter:[20/24], Time: 0.42, lr: [0.006054887432011008], Loss: 0.180558
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97469322 0.79912831 0.90143444 0.35700811 0.49900349 0.58622704
 0.60530733 0.72752247 0.90713277 0.5090141  0.92727149 0.77172577
 0.50621165 0.91279664 0.42685095 0.50924174 0.42382308 0.5326132
 0.73002018] 0.6635276842209192
1 [0.97445034 0.79976318 0.90324205 0.43426898 0.51944638 0.59368852
 0.62384221 0.73895163 0.90954147 0.52607204 0.92600051 0.77972559
 0.53222244 0.92024488 0.44413366 0.63162738 0.49553212 0.53879931
 0.73341965] 0.6855248601278416
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6855, Best_mIoU:  0.6913
[0.97445034 0.79976318 0.90324205 0.43426898 0.51944638 0.59368852
 0.62384221 0.73895163 0.90954147 0.52607204 0.92600051 0.77972559
 0.53222244 0.92024488 0.44413366 0.63162738 0.49553212 0.53879931
 0.73341965]
Epoch: [207/484] Iter:[0/24], Time: 0.96, lr: [0.006051610485187581], Loss: 0.154770
Epoch: [207/484] Iter:[10/24], Time: 0.44, lr: [0.006043417255203448], Loss: 0.148331
Epoch: [207/484] Iter:[20/24], Time: 0.42, lr: [0.006035222790833898], Loss: 0.157844
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97296205 0.7894156  0.89825779 0.30242234 0.47979042 0.58374574
 0.60777944 0.72436592 0.90882387 0.4986458  0.9328153  0.78168157
 0.52912628 0.91831036 0.42836906 0.55880719 0.38198715 0.55901386
 0.73402894] 0.6626499299733896
1 [0.97291329 0.79135016 0.89747553 0.33375771 0.50055062 0.59369422
 0.62976874 0.73795361 0.91037426 0.5167963  0.93234254 0.7887164
 0.54591644 0.92537422 0.46613899 0.68168523 0.40843143 0.57065448
 0.73594486] 0.6810441595867257
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.265, MeanIU:  0.6810, Best_mIoU:  0.6913
[0.97291329 0.79135016 0.89747553 0.33375771 0.50055062 0.59369422
 0.62976874 0.73795361 0.91037426 0.5167963  0.93234254 0.7887164
 0.54591644 0.92537422 0.46613899 0.68168523 0.40843143 0.57065448
 0.73594486]
Epoch: [208/484] Iter:[0/24], Time: 0.97, lr: [0.006031944658999453], Loss: 0.130967
Epoch: [208/484] Iter:[10/24], Time: 0.45, lr: [0.006023748463046918], Loss: 0.159684
Epoch: [208/484] Iter:[20/24], Time: 0.42, lr: [0.006015551027780978], Loss: 0.156822
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97162905 0.75929115 0.90307024 0.32238666 0.51041476 0.58287379
 0.60289214 0.71863251 0.90635158 0.50336692 0.92890983 0.77785153
 0.51887756 0.9040389  0.41469351 0.5119674  0.43310863 0.51653239
 0.73005639] 0.6587865757713413
1 [0.97269291 0.77242859 0.9056761  0.37860245 0.52796595 0.59404024
 0.62736985 0.72885997 0.90911975 0.5220183  0.92947262 0.784447
 0.5393582  0.91236396 0.44644013 0.60958518 0.49737023 0.53068623
 0.73601832] 0.6802376838617804
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.267, MeanIU:  0.6802, Best_mIoU:  0.6913
[0.97269291 0.77242859 0.9056761  0.37860245 0.52796595 0.59404024
 0.62736985 0.72885997 0.90911975 0.5220183  0.92947262 0.784447
 0.5393582  0.91236396 0.44644013 0.60958518 0.49737023 0.53068623
 0.73601832]
Epoch: [209/484] Iter:[0/24], Time: 0.99, lr: [0.006012271706204635], Loss: 0.088690
Epoch: [209/484] Iter:[10/24], Time: 0.45, lr: [0.006004072532430214], Loss: 0.160534
Epoch: [209/484] Iter:[20/24], Time: 0.42, lr: [0.005995872114376699], Loss: 0.159474
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97379975 0.78910697 0.90342018 0.3601176  0.46247149 0.59354071
 0.60990813 0.72582382 0.90832506 0.54019791 0.93212671 0.77143561
 0.50929598 0.90921915 0.44841025 0.50671103 0.45936898 0.50547991
 0.72860125] 0.6651242374402795
1 [0.97312895 0.7885113  0.90404795 0.39553246 0.47678129 0.59939051
 0.62731924 0.73603347 0.91008148 0.55702659 0.93141002 0.77906296
 0.53530639 0.91839192 0.47883801 0.61189276 0.50252941 0.53287685
 0.73565643] 0.6838851568399344
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.262, MeanIU:  0.6839, Best_mIoU:  0.6913
[0.97312895 0.7885113  0.90404795 0.39553246 0.47678129 0.59939051
 0.62731924 0.73603347 0.91008148 0.55702659 0.93141002 0.77906296
 0.53530639 0.91839192 0.47883801 0.61189276 0.50252941 0.53287685
 0.73565643]
Epoch: [210/484] Iter:[0/24], Time: 0.97, lr: [0.005992591598291394], Loss: 0.121176
Epoch: [210/484] Iter:[10/24], Time: 0.44, lr: [0.005984389434750825], Loss: 0.173566
Epoch: [210/484] Iter:[20/24], Time: 0.42, lr: [0.005976186021927364], Loss: 0.184544
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97063526 0.78231511 0.90681528 0.38968612 0.49621561 0.58318052
 0.61673415 0.7275931  0.91002965 0.53190436 0.93161588 0.77723577
 0.54058336 0.91117587 0.46334394 0.51750763 0.45322147 0.53919971
 0.72807681] 0.6724773469637962
1 [0.97100786 0.78640312 0.9084     0.4252343  0.5202006  0.59044524
 0.63368368 0.73902356 0.91167907 0.5510722  0.93083273 0.77971523
 0.546125   0.9197673  0.51640226 0.6212695  0.46778152 0.55793451
 0.73426655] 0.6900654857176771
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.6901, Best_mIoU:  0.6913
[0.97100786 0.78640312 0.9084     0.4252343  0.5202006  0.59044524
 0.63368368 0.73902356 0.91167907 0.5510722  0.93083273 0.77971523
 0.546125   0.9197673  0.51640226 0.6212695  0.46778152 0.55793451
 0.73426655]
Epoch: [211/484] Iter:[0/24], Time: 0.97, lr: [0.005972904306529429], Loss: 0.189891
Epoch: [211/484] Iter:[10/24], Time: 0.44, lr: [0.005964699141186664], Loss: 0.150890
Epoch: [211/484] Iter:[20/24], Time: 0.42, lr: [0.0059564927215186424], Loss: 0.144355
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97155836 0.78446628 0.90398153 0.34717475 0.47597821 0.59593432
 0.61261048 0.73615865 0.91149881 0.52947905 0.93144909 0.77403697
 0.45174288 0.91463573 0.47340971 0.56017736 0.4504064  0.52756576
 0.73693229] 0.6678524541809744
1 [0.97147281 0.78324809 0.90383328 0.37883406 0.49965689 0.60285653
 0.63375631 0.74845729 0.91331597 0.53005061 0.93121962 0.78510365
 0.49782811 0.92509097 0.52420756 0.68922626 0.44353978 0.55674641
 0.7436982 ] 0.6874811783521546
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6875, Best_mIoU:  0.6913
[0.97147281 0.78324809 0.90383328 0.37883406 0.49965689 0.60285653
 0.63375631 0.74845729 0.91331597 0.53005061 0.93121962 0.78510365
 0.49782811 0.92509097 0.52420756 0.68922626 0.44353978 0.55674641
 0.7436982 ]
Epoch: [212/484] Iter:[0/24], Time: 0.97, lr: [0.005953209801967397], Loss: 0.105845
Epoch: [212/484] Iter:[10/24], Time: 0.45, lr: [0.0059450016226935435], Loss: 0.140261
Epoch: [212/484] Iter:[20/24], Time: 0.42, lr: [0.005936792184013069], Loss: 0.154046
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97335238 0.79383036 0.90403398 0.42394463 0.45224215 0.58960871
 0.60520664 0.73312544 0.90950942 0.54538573 0.93482794 0.77053471
 0.50055339 0.91405008 0.47846729 0.54787657 0.45813594 0.53859473
 0.72801004] 0.6737521113578306
1 [0.97354161 0.79749235 0.90581143 0.48149996 0.46427244 0.59723826
 0.62463645 0.74666623 0.91098008 0.55835159 0.9341931  0.77756039
 0.52537525 0.92314678 0.52985229 0.65898611 0.45135746 0.54675123
 0.72771667] 0.6913384042402774
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.252, MeanIU:  0.6913, Best_mIoU:  0.6913
[0.97354161 0.79749235 0.90581143 0.48149996 0.46427244 0.59723826
 0.62463645 0.74666623 0.91098008 0.55835159 0.9341931  0.77756039
 0.52537525 0.92314678 0.52985229 0.65898611 0.45135746 0.54675123
 0.72771667]
Epoch: [213/484] Iter:[0/24], Time: 0.97, lr: [0.005933508055430394], Loss: 0.178306
Epoch: [213/484] Iter:[10/24], Time: 0.45, lr: [0.005925296850002649], Loss: 0.165030
Epoch: [213/484] Iter:[20/24], Time: 0.42, lr: [0.005917084380047481], Loss: 0.154704
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97148401 0.77959779 0.90698563 0.33469545 0.50245525 0.59044926
 0.61061005 0.72584494 0.9101855  0.49093464 0.93216898 0.77044249
 0.50547107 0.91447989 0.47051026 0.53632687 0.46904178 0.5143175
 0.73133489] 0.6667019077577994
1 [0.97211973 0.78466194 0.90868977 0.3782478  0.5230033  0.59696563
 0.62270356 0.74087816 0.91149043 0.47910772 0.93044549 0.77774915
 0.53439556 0.9235269  0.5347715  0.65690421 0.52283523 0.53485259
 0.73851418] 0.6879927801186985
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.257, MeanIU:  0.6880, Best_mIoU:  0.6913
[0.97211973 0.78466194 0.90868977 0.3782478  0.5230033  0.59696563
 0.62270356 0.74087816 0.91149043 0.47910772 0.93044549 0.77774915
 0.53439556 0.9235269  0.5347715  0.65690421 0.52283523 0.53485259
 0.73851418]
Epoch: [214/484] Iter:[0/24], Time: 0.98, lr: [0.0059137990375173945], Loss: 0.187121
Epoch: [214/484] Iter:[10/24], Time: 0.45, lr: [0.0059055847936179745], Loss: 0.173305
Epoch: [214/484] Iter:[20/24], Time: 0.42, lr: [0.005897369280030432], Loss: 0.156650
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97432874 0.79603056 0.9051176  0.33186344 0.47444294 0.57814393
 0.60776002 0.72634103 0.90794461 0.51605812 0.93695777 0.78349091
 0.53640022 0.92018273 0.46582885 0.54315775 0.42323218 0.52681086
 0.73257798] 0.667719487229762
1 [0.97484164 0.80154999 0.90724147 0.37903464 0.50199485 0.58494063
 0.6313179  0.74146349 0.90961988 0.53364418 0.93595591 0.79189012
 0.55063847 0.92844725 0.52869292 0.68894917 0.49562848 0.55452871
 0.73626801] 0.693507773913982
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6935, Best_mIoU:  0.6935
[0.97484164 0.80154999 0.90724147 0.37903464 0.50199485 0.58494063
 0.6313179  0.74146349 0.90961988 0.53364418 0.93595591 0.79189012
 0.55063847 0.92844725 0.52869292 0.68894917 0.49562848 0.55452871
 0.73626801]
Epoch: [215/484] Iter:[0/24], Time: 0.96, lr: [0.005894082718598648], Loss: 0.145098
Epoch: [215/484] Iter:[10/24], Time: 0.44, lr: [0.005885865423813699], Loss: 0.151095
Epoch: [215/484] Iter:[20/24], Time: 0.42, lr: [0.005877646854139566], Loss: 0.139356
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97442083 0.79517718 0.90740102 0.35933946 0.50168834 0.58319713
 0.60654183 0.73293132 0.90948732 0.51777047 0.93172405 0.78185985
 0.51765075 0.91990675 0.46914179 0.54669677 0.4356906  0.53404606
 0.73431378] 0.6715255421935995
1 [0.97461704 0.79939249 0.90860121 0.4218627  0.52254713 0.58336345
 0.6300778  0.74486605 0.91143694 0.52746568 0.92608214 0.79024817
 0.54173857 0.92896581 0.52525246 0.70688963 0.49677083 0.55851936
 0.74041325] 0.6967953010043477
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.6968, Best_mIoU:  0.6968
[0.97461704 0.79939249 0.90860121 0.4218627  0.52254713 0.58336345
 0.6300778  0.74486605 0.91143694 0.52746568 0.92608214 0.79024817
 0.54173857 0.92896581 0.52525246 0.70688963 0.49677083 0.55851936
 0.74041325]
Epoch: [216/484] Iter:[0/24], Time: 0.93, lr: [0.005874359068813055], Loss: 0.142336
Epoch: [216/484] Iter:[10/24], Time: 0.44, lr: [0.005866138710631534], Loss: 0.166793
Epoch: [216/484] Iter:[20/24], Time: 0.42, lr: [0.005857917072318945], Loss: 0.155385
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97436276 0.78920446 0.90532591 0.34851885 0.47860247 0.59276785
 0.61204629 0.73091054 0.9098247  0.54073762 0.93489397 0.78554497
 0.54425978 0.918365   0.44683509 0.52575902 0.42812806 0.49864973
 0.72663653] 0.6679670320815839
1 [0.97415966 0.79206918 0.90701356 0.41114153 0.5004066  0.59992759
 0.63212387 0.74431065 0.91206196 0.55410921 0.93422509 0.78914694
 0.55211103 0.92720992 0.50022023 0.66434051 0.42541993 0.51197311
 0.73322529] 0.6876418865159214
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.253, MeanIU:  0.6876, Best_mIoU:  0.6968
[0.97415966 0.79206918 0.90701356 0.41114153 0.5004066  0.59992759
 0.63212387 0.74431065 0.91206196 0.55410921 0.93422509 0.78914694
 0.55211103 0.92720992 0.50022023 0.66434051 0.42541993 0.51197311
 0.73322529]
Epoch: [217/484] Iter:[0/24], Time: 0.95, lr: [0.0058546280580654845], Loss: 0.163981
Epoch: [217/484] Iter:[10/24], Time: 0.44, lr: [0.0058464046238780355], Loss: 0.154138
Epoch: [217/484] Iter:[20/24], Time: 0.41, lr: [0.005838179904276334], Loss: 0.140577
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97615966 0.79942748 0.90821267 0.33086256 0.50155382 0.58836756
 0.61697692 0.732449   0.90932652 0.53160748 0.9302669  0.78101967
 0.53954781 0.91706733 0.44900031 0.55986269 0.46383139 0.5146365
 0.731124  ] 0.672700014638666
1 [0.97630044 0.80546024 0.90965755 0.37910169 0.53011258 0.59684015
 0.63471331 0.74482091 0.91124218 0.53305109 0.92760747 0.78563023
 0.5513645  0.92379448 0.48819595 0.68555449 0.54738426 0.53898144
 0.73813462] 0.695155135934576
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6952, Best_mIoU:  0.6968
[0.97630044 0.80546024 0.90965755 0.37910169 0.53011258 0.59684015
 0.63471331 0.74482091 0.91124218 0.53305109 0.92760747 0.78563023
 0.5513645  0.92379448 0.48819595 0.68555449 0.54738426 0.53898144
 0.73813462]
Epoch: [218/484] Iter:[0/24], Time: 0.97, lr: [0.005834889656024053], Loss: 0.221133
Epoch: [218/484] Iter:[10/24], Time: 0.44, lr: [0.0058266631331218635], Loss: 0.205131
Epoch: [218/484] Iter:[20/24], Time: 0.42, lr: [0.005818435319480452], Loss: 0.180748
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97280677 0.7827101  0.90459992 0.36366273 0.49612771 0.5895119
 0.61184492 0.73735905 0.90922853 0.51474768 0.93455317 0.77991139
 0.51431355 0.91749979 0.46466019 0.57527204 0.48574347 0.45551786
 0.72629968] 0.6703352875594046
1 [0.97265074 0.78935038 0.90745238 0.40347954 0.51879789 0.59458956
 0.63313143 0.74667252 0.91130432 0.52514661 0.93156749 0.78663764
 0.54276504 0.92494101 0.51387168 0.720415   0.61035104 0.48159673
 0.73417297] 0.6973102095111416
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6973, Best_mIoU:  0.6973
[0.97265074 0.78935038 0.90745238 0.40347954 0.51879789 0.59458956
 0.63313143 0.74667252 0.91130432 0.52514661 0.93156749 0.78663764
 0.54276504 0.92494101 0.51387168 0.720415   0.61035104 0.48159673
 0.73417297]
Epoch: [219/484] Iter:[0/24], Time: 0.95, lr: [0.005815143832117368], Loss: 0.129583
Epoch: [219/484] Iter:[10/24], Time: 0.44, lr: [0.005806914207691007], Loss: 0.183155
Epoch: [219/484] Iter:[20/24], Time: 0.42, lr: [0.005798683287158182], Loss: 0.169597
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97326808 0.78534169 0.90161981 0.38170394 0.49267551 0.58108891
 0.61648679 0.73410094 0.90683644 0.51134222 0.91757642 0.76805869
 0.48577231 0.91041279 0.42099114 0.56231574 0.44556794 0.49462942
 0.72501675] 0.6639371318688896
1 [0.97353181 0.79039148 0.9032602  0.41359856 0.51145673 0.58302465
 0.63053772 0.74352418 0.90944787 0.52895877 0.91615861 0.77349751
 0.50889043 0.92001456 0.45246445 0.67797621 0.47644895 0.52566314
 0.7334791 ] 0.6827539440712641
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.258, MeanIU:  0.6828, Best_mIoU:  0.6973
[0.97353181 0.79039148 0.9032602  0.41359856 0.51145673 0.58302465
 0.63053772 0.74352418 0.90944787 0.52895877 0.91615861 0.77349751
 0.50889043 0.92001456 0.45246445 0.67797621 0.47644895 0.52566314
 0.7334791 ]
Epoch: [220/484] Iter:[0/24], Time: 0.95, lr: [0.00579539055553173], Loss: 0.114513
Epoch: [220/484] Iter:[10/24], Time: 0.44, lr: [0.005787157816669962], Loss: 0.226832
Epoch: [220/484] Iter:[20/24], Time: 0.42, lr: [0.005778923776291723], Loss: 0.219219
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.964385   0.72216314 0.88763219 0.33296128 0.47548437 0.57064011
 0.59638929 0.7144617  0.90177091 0.42510268 0.90903593 0.76827961
 0.51717974 0.91804962 0.47532577 0.60310309 0.40080209 0.4605391
 0.7247904 ] 0.6509524224136309
1 [0.96400118 0.7189251  0.8868478  0.36913996 0.49267929 0.57656643
 0.61187488 0.7211333  0.90315363 0.41678094 0.90717154 0.77386957
 0.53588569 0.92475952 0.5316176  0.71078523 0.45947451 0.47238806
 0.73106247] 0.6688482481238329
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.313, MeanIU:  0.6688, Best_mIoU:  0.6973
[0.96400118 0.7189251  0.8868478  0.36913996 0.49267929 0.57656643
 0.61187488 0.7211333  0.90315363 0.41678094 0.90717154 0.77386957
 0.53588569 0.92475952 0.5316176  0.71078523 0.45947451 0.47238806
 0.73106247]
Epoch: [221/484] Iter:[0/24], Time: 0.95, lr: [0.0057756297952082794], Loss: 0.150670
Epoch: [221/484] Iter:[10/24], Time: 0.44, lr: [0.005767393928896873], Loss: 0.154795
Epoch: [221/484] Iter:[20/24], Time: 0.42, lr: [0.005759156755615711], Loss: 0.163101
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97016514 0.77471519 0.90325104 0.29561469 0.47609276 0.5731657
 0.60879254 0.73095657 0.90890934 0.49123166 0.92919972 0.77787994
 0.53328222 0.91365032 0.43582654 0.59880003 0.42834928 0.50744119
 0.72911091] 0.6624439352462635
1 [0.97113931 0.78048053 0.9051067  0.35002829 0.50712758 0.58088547
 0.61987783 0.73669308 0.91035297 0.50473782 0.92942521 0.78254084
 0.53767214 0.92083866 0.47561789 0.69020864 0.46158915 0.52934896
 0.7357459 ] 0.6804956308318035
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.263, MeanIU:  0.6805, Best_mIoU:  0.6973
[0.97113931 0.78048053 0.9051067  0.35002829 0.50712758 0.58088547
 0.61987783 0.73669308 0.91035297 0.50473782 0.92942521 0.78254084
 0.53767214 0.92083866 0.47561789 0.69020864 0.46158915 0.52934896
 0.7357459 ]
Epoch: [222/484] Iter:[0/24], Time: 0.97, lr: [0.005755861519840108], Loss: 0.122282
Epoch: [222/484] Iter:[10/24], Time: 0.44, lr: [0.005747622512960615], Loss: 0.143651
Epoch: [222/484] Iter:[20/24], Time: 0.42, lr: [0.005739382193614292], Loss: 0.160477
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96824483 0.76725241 0.89988805 0.29849673 0.46291894 0.57915223
 0.60902156 0.72063387 0.90937816 0.46409519 0.93610013 0.77763879
 0.51488246 0.8983717  0.35241536 0.58492053 0.28926549 0.51100504
 0.7208193 ] 0.6455000410961208
1 [0.96789515 0.76727374 0.90029016 0.32211111 0.48725319 0.58768291
 0.62814932 0.73380884 0.91022752 0.45998023 0.9359985  0.78503187
 0.5388213  0.89029538 0.36309406 0.65286245 0.27846711 0.53387182
 0.72767184] 0.6563571838913937
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.282, MeanIU:  0.6564, Best_mIoU:  0.6973
[0.96789515 0.76727374 0.90029016 0.32211111 0.48725319 0.58768291
 0.62814932 0.73380884 0.91022752 0.45998023 0.9359985  0.78503187
 0.5388213  0.89029538 0.36309406 0.65286245 0.27846711 0.53387182
 0.72767184]
Epoch: [223/484] Iter:[0/24], Time: 0.99, lr: [0.005736085697869327], Loss: 0.262752
Epoch: [223/484] Iter:[10/24], Time: 0.45, lr: [0.005727843537197845], Loss: 0.171783
Epoch: [223/484] Iter:[20/24], Time: 0.42, lr: [0.00571960005851815], Loss: 0.170548
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9679466  0.7685017  0.90090651 0.30342605 0.45989048 0.57560939
 0.59762586 0.72259411 0.90565812 0.497498   0.93149811 0.76503026
 0.4665318  0.92097597 0.4385172  0.62896184 0.3169377  0.42185518
 0.73368295] 0.6486130425151778
1 [0.96825892 0.7713042  0.90328856 0.3077342  0.47259221 0.57909971
 0.61695284 0.72259994 0.90791699 0.50013712 0.93230119 0.77118204
 0.49087736 0.92654647 0.52249856 0.71342589 0.39626279 0.46312927
 0.74233288] 0.6688653239901444
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.268, MeanIU:  0.6689, Best_mIoU:  0.6973
[0.96825892 0.7713042  0.90328856 0.3077342  0.47259221 0.57909971
 0.61695284 0.72259994 0.90791699 0.50013712 0.93230119 0.77118204
 0.49087736 0.92654647 0.52249856 0.71342589 0.39626279 0.46312927
 0.74233288]
Epoch: [224/484] Iter:[0/24], Time: 0.96, lr: [0.005716302297484084], Loss: 0.136109
Epoch: [224/484] Iter:[10/24], Time: 0.44, lr: [0.005708056969689999], Loss: 0.154811
Epoch: [224/484] Iter:[20/24], Time: 0.42, lr: [0.005699810318301484], Loss: 0.146178
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97027488 0.77622789 0.90288335 0.27507439 0.49191387 0.58556377
 0.60565636 0.7196294  0.90940371 0.48689884 0.93273866 0.77291914
 0.50261276 0.91993798 0.47582786 0.61223866 0.43035726 0.51979858
 0.73353149] 0.664394150247822
1 [0.97030795 0.77644091 0.90367334 0.27012669 0.51610514 0.59102414
 0.62491556 0.72861574 0.91044907 0.49017068 0.93446883 0.77923119
 0.51871095 0.92732688 0.53340127 0.73240073 0.53600073 0.54865981
 0.74022206] 0.6859079834930771
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.267, MeanIU:  0.6859, Best_mIoU:  0.6973
[0.97030795 0.77644091 0.90367334 0.27012669 0.51610514 0.59102414
 0.62491556 0.72861574 0.91044907 0.49017068 0.93446883 0.77923119
 0.51871095 0.92732688 0.53340127 0.73240073 0.53600073 0.54865981
 0.74022206]
Epoch: [225/484] Iter:[0/24], Time: 0.98, lr: [0.005696511286615531], Loss: 0.130300
Epoch: [225/484] Iter:[10/24], Time: 0.45, lr: [0.005688262778260244], Loss: 0.167952
Epoch: [225/484] Iter:[20/24], Time: 0.42, lr: [0.005680012940678934], Loss: 0.165033
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97307297 0.79088851 0.90137515 0.29519384 0.46965458 0.56971015
 0.61006635 0.72812617 0.90567061 0.52179052 0.93246524 0.78383952
 0.53150255 0.92311555 0.48385873 0.63664103 0.38872364 0.52577802
 0.73172087] 0.6685891582068728
1 [0.97299968 0.79398749 0.90255614 0.32885733 0.48924214 0.57352951
 0.63022615 0.73884269 0.90789801 0.54519821 0.93309019 0.79077069
 0.54483459 0.93189129 0.5943379  0.72640349 0.42337091 0.54536351
 0.73852566] 0.6901013457468139
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.6901, Best_mIoU:  0.6973
[0.97299968 0.79398749 0.90255614 0.32885733 0.48924214 0.57352951
 0.63022615 0.73884269 0.90789801 0.54519821 0.93309019 0.79077069
 0.54483459 0.93189129 0.5943379  0.72640349 0.42337091 0.54536351
 0.73852566]
Epoch: [226/484] Iter:[0/24], Time: 0.96, lr: [0.0056767126329347484], Loss: 0.207448
Epoch: [226/484] Iter:[10/24], Time: 0.45, lr: [0.0056684609304703815], Loss: 0.144151
Epoch: [226/484] Iter:[20/24], Time: 0.42, lr: [0.005660207893102474], Loss: 0.147144
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97347533 0.78950372 0.90359504 0.3110407  0.47460505 0.57234795
 0.61111934 0.7297637  0.9110349  0.52368153 0.93266813 0.77835963
 0.49815465 0.91423659 0.40631124 0.6093021  0.46263096 0.50315842
 0.73722217] 0.665379535176032
1 [0.97368933 0.79033104 0.90389934 0.32259949 0.49107439 0.57046316
 0.63325701 0.74041697 0.91362207 0.52179294 0.93169925 0.78651719
 0.51554133 0.92185896 0.41060472 0.6942511  0.46683668 0.53203473
 0.74532564] 0.6771481761750107
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.263, MeanIU:  0.6771, Best_mIoU:  0.6973
[0.97368933 0.79033104 0.90389934 0.32259949 0.49107439 0.57046316
 0.63325701 0.74041697 0.91362207 0.52179294 0.93169925 0.78651719
 0.51554133 0.92185896 0.41060472 0.6942511  0.46683668 0.53203473
 0.74532564]
Epoch: [227/484] Iter:[0/24], Time: 0.97, lr: [0.005656906303849626], Loss: 0.197405
Epoch: [227/484] Iter:[10/24], Time: 0.44, lr: [0.0056486513936176985], Loss: 0.179128
Epoch: [227/484] Iter:[20/24], Time: 0.42, lr: [0.005640395142758241], Loss: 0.171665
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9754047  0.80423631 0.90663378 0.38686041 0.4910951  0.59283026
 0.61808401 0.73145625 0.91085749 0.53614909 0.93764289 0.77383172
 0.51540341 0.90819304 0.43289169 0.48598603 0.483433   0.47345413
 0.72685937] 0.6679632991068701
1 [0.97560395 0.80844596 0.90716559 0.42224292 0.50931376 0.60218926
 0.62911674 0.7419147  0.91259993 0.54620388 0.93660369 0.78300742
 0.53145296 0.91766041 0.46588756 0.6220026  0.60049891 0.49060884
 0.73336389] 0.6913622612318577
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.253, MeanIU:  0.6914, Best_mIoU:  0.6973
[0.97560395 0.80844596 0.90716559 0.42224292 0.50931376 0.60218926
 0.62911674 0.7419147  0.91259993 0.54620388 0.93660369 0.78300742
 0.53145296 0.91766041 0.46588756 0.6220026  0.60049891 0.49060884
 0.73336389]
Epoch: [228/484] Iter:[0/24], Time: 0.97, lr: [0.0056370922665016825], Loss: 0.166289
Epoch: [228/484] Iter:[10/24], Time: 0.44, lr: [0.005628834134731777], Loss: 0.158954
Epoch: [228/484] Iter:[20/24], Time: 0.42, lr: [0.005620574656563312], Loss: 0.154835
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96879749 0.77407437 0.90523925 0.36274431 0.48527981 0.57877709
 0.61998674 0.73248962 0.90940081 0.53457731 0.92799173 0.77581906
 0.51840508 0.90661075 0.40204675 0.48331445 0.4874083  0.50896737
 0.72616554] 0.6635839918355001
1 [0.96829071 0.77181334 0.90640386 0.37467298 0.50407572 0.58460577
 0.6293182  0.74215867 0.91204823 0.55627846 0.93239433 0.78470169
 0.54042641 0.91377344 0.40465628 0.59599542 0.57333425 0.52699015
 0.7298804 ] 0.6816746475053115
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.271, MeanIU:  0.6817, Best_mIoU:  0.6973
[0.96829071 0.77181334 0.90640386 0.37467298 0.50407572 0.58460577
 0.6293182  0.74215867 0.91204823 0.55627846 0.93239433 0.78470169
 0.54042641 0.91377344 0.40465628 0.59599542 0.57333425 0.52699015
 0.7298804 ]
Epoch: [229/484] Iter:[0/24], Time: 0.97, lr: [0.005617270487762836], Loss: 0.201481
Epoch: [229/484] Iter:[10/24], Time: 0.44, lr: [0.005609009120571242], Loss: 0.187645
Epoch: [229/484] Iter:[20/24], Time: 0.42, lr: [0.005600746401162439], Loss: 0.175116
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97162468 0.78768653 0.90636001 0.38531339 0.50340299 0.59387665
 0.61889034 0.72710721 0.9092999  0.53763324 0.92917599 0.78451056
 0.53693529 0.91454711 0.42248954 0.56168547 0.47840221 0.51426427
 0.72457389] 0.6740936447282722
1 [0.97187152 0.79187677 0.90776887 0.42002197 0.52530264 0.6017926
 0.63273876 0.72588302 0.91139575 0.56086495 0.92727968 0.7908758
 0.55280478 0.92291192 0.447286   0.65940442 0.45342187 0.53307442
 0.72985831] 0.6877070557330532
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.6877, Best_mIoU:  0.6973
[0.97187152 0.79187677 0.90776887 0.42002197 0.52530264 0.6017926
 0.63273876 0.72588302 0.91139575 0.56086495 0.92727968 0.7908758
 0.55280478 0.92291192 0.447286   0.65940442 0.45342187 0.53307442
 0.72985831]
Epoch: [230/484] Iter:[0/24], Time: 0.97, lr: [0.005597440934232129], Loss: 0.087470
Epoch: [230/484] Iter:[10/24], Time: 0.44, lr: [0.005589176317620458], Loss: 0.144447
Epoch: [230/484] Iter:[20/24], Time: 0.42, lr: [0.0055809103429247334], Loss: 0.168667
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97160981 0.78994077 0.90604977 0.37126312 0.49271766 0.59620144
 0.61871691 0.71899765 0.90971925 0.50018761 0.92881982 0.78361293
 0.54289568 0.91401255 0.42336809 0.53783297 0.4732906  0.52993082
 0.72580092] 0.6702614923734289
1 [0.97137071 0.79171272 0.90492108 0.36716928 0.51008896 0.60175431
 0.63557188 0.7237763  0.91193619 0.5102332  0.92169712 0.7874452
 0.55060027 0.92221712 0.46462647 0.62208159 0.42735833 0.54730628
 0.73168631] 0.6791343859624933
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6791, Best_mIoU:  0.6973
[0.97137071 0.79171272 0.90492108 0.36716928 0.51008896 0.60175431
 0.63557188 0.7237763  0.91193619 0.5102332  0.92169712 0.7874452
 0.55060027 0.92221712 0.46462647 0.62208159 0.42735833 0.54730628
 0.73168631]
Epoch: [231/484] Iter:[0/24], Time: 0.98, lr: [0.005577603572232405], Loss: 0.191941
Epoch: [231/484] Iter:[10/24], Time: 0.44, lr: [0.005569335692086182], Loss: 0.171541
Epoch: [231/484] Iter:[20/24], Time: 0.42, lr: [0.005561066447940276], Loss: 0.168405
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9719321  0.78412826 0.90663039 0.34698465 0.49671065 0.59151282
 0.61995242 0.72064022 0.90628971 0.51846452 0.93290886 0.78003288
 0.53290141 0.91052842 0.44554937 0.55644489 0.46845691 0.54451286
 0.72702728] 0.6716636121895719
1 [0.9719596  0.78735438 0.90782756 0.37197675 0.51512573 0.59874004
 0.63274702 0.72835378 0.90871731 0.53324494 0.92865981 0.78499264
 0.53871071 0.91813475 0.50446705 0.64532451 0.49854657 0.55685937
 0.73172823] 0.6875510907445374
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.6876, Best_mIoU:  0.6973
[0.9719596  0.78735438 0.90782756 0.37197675 0.51512573 0.59874004
 0.63274702 0.72835378 0.90871731 0.53324494 0.92865981 0.78499264
 0.53871071 0.91813475 0.50446705 0.64532451 0.49854657 0.55685937
 0.73172823]
Epoch: [232/484] Iter:[0/24], Time: 0.97, lr: [0.005557758367806911], Loss: 0.106507
Epoch: [232/484] Iter:[10/24], Time: 0.44, lr: [0.005549487209894153], Loss: 0.136011
Epoch: [232/484] Iter:[20/24], Time: 0.42, lr: [0.0055412146820166975], Loss: 0.139162
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97299477 0.78409733 0.90698546 0.40130113 0.4880635  0.59030368
 0.619443   0.72119784 0.90986328 0.5223434  0.93520044 0.78248719
 0.53306984 0.91368383 0.40568509 0.53437301 0.44565153 0.55335411
 0.73159205] 0.6711416048109154
1 [0.9733023  0.78908417 0.90895928 0.43122856 0.50738765 0.59557064
 0.63586728 0.72730099 0.91157647 0.53271951 0.93634678 0.79069451
 0.55151759 0.92188915 0.44593417 0.65914726 0.49857415 0.56581614
 0.7361693 ] 0.6904782059272045
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6905, Best_mIoU:  0.6973
[0.9733023  0.78908417 0.90895928 0.43122856 0.50738765 0.59557064
 0.63586728 0.72730099 0.91157647 0.53271951 0.93634678 0.79069451
 0.55151759 0.92188915 0.44593417 0.65914726 0.49857415 0.56581614
 0.7361693 ]
Epoch: [233/484] Iter:[0/24], Time: 0.97, lr: [0.005537905286715861], Loss: 0.101422
Epoch: [233/484] Iter:[10/24], Time: 0.44, lr: [0.005529630836685632], Loss: 0.126862
Epoch: [233/484] Iter:[20/24], Time: 0.42, lr: [0.005521355010675687], Loss: 0.139712
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97308628 0.78707278 0.90358638 0.36248465 0.45621362 0.5996506
 0.62061883 0.73052188 0.91132105 0.53393339 0.93733602 0.78082943
 0.54203355 0.91147175 0.4030165  0.52706409 0.41856836 0.54009619
 0.73023054] 0.6667966261286297
1 [0.97343719 0.7905118  0.90387416 0.38397751 0.47107022 0.60702529
 0.63859106 0.7395668  0.91428951 0.5412215  0.93837636 0.78593748
 0.55012639 0.92047437 0.43811262 0.66627364 0.42223009 0.56299722
 0.73654738] 0.6834021363510004
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.258, MeanIU:  0.6834, Best_mIoU:  0.6973
[0.97343719 0.7905118  0.90387416 0.38397751 0.47107022 0.60702529
 0.63859106 0.7395668  0.91428951 0.5412215  0.93837636 0.78593748
 0.55012639 0.92047437 0.43811262 0.66627364 0.42223009 0.56299722
 0.73654738]
Epoch: [234/484] Iter:[0/24], Time: 0.97, lr: [0.005518044294432946], Loss: 0.178565
Epoch: [234/484] Iter:[10/24], Time: 0.45, lr: [0.005509766537813874], Loss: 0.166950
Epoch: [234/484] Iter:[20/24], Time: 0.42, lr: [0.005501487399149448], Loss: 0.155826
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9722444  0.78518569 0.90332413 0.44568562 0.46667475 0.58481363
 0.61488894 0.71507426 0.90889704 0.50942866 0.93790335 0.74838192
 0.35223153 0.91452317 0.41277193 0.5548116  0.40420802 0.53635969
 0.73663344] 0.6581074621411523
1 [0.97271675 0.79051445 0.90310931 0.47807244 0.47693285 0.58984448
 0.63265732 0.7217498  0.91123713 0.51837868 0.93842129 0.75139654
 0.35090959 0.92163238 0.43500171 0.64819744 0.39227754 0.5548557
 0.74032011] 0.6699066052320269
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.264, MeanIU:  0.6699, Best_mIoU:  0.6973
[0.97271675 0.79051445 0.90310931 0.47807244 0.47693285 0.58984448
 0.63265732 0.7217498  0.91123713 0.51837868 0.93842129 0.75139654
 0.35090959 0.92163238 0.43500171 0.64819744 0.39227754 0.5548557
 0.74032011]
Epoch: [235/484] Iter:[0/24], Time: 0.95, lr: [0.005498175356141773], Loss: 0.102239
Epoch: [235/484] Iter:[10/24], Time: 0.44, lr: [0.005489894278340555], Loss: 0.161934
Epoch: [235/484] Iter:[20/24], Time: 0.42, lr: [0.005481611812377098], Loss: 0.146771
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97353437 0.79725062 0.90680826 0.4080823  0.49886102 0.59402162
 0.61884173 0.72182703 0.90891588 0.52315212 0.93640346 0.7850979
 0.5457781  0.91092429 0.42456733 0.50577761 0.43038359 0.54517588
 0.73343008] 0.6720438513349032
1 [0.97424867 0.80271027 0.9069809  0.42329237 0.51580371 0.6023764
 0.63644307 0.73138736 0.91130004 0.54728027 0.93652795 0.78802452
 0.5493601  0.91423149 0.46572412 0.54375089 0.4274044  0.55515537
 0.74024453] 0.6827498123866922
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.251, MeanIU:  0.6827, Best_mIoU:  0.6973
[0.97424867 0.80271027 0.9069809  0.42329237 0.51580371 0.6023764
 0.63644307 0.73138736 0.91130004 0.54728027 0.93652795 0.78802452
 0.5493601  0.91423149 0.46572412 0.54375089 0.4274044  0.55515537
 0.74024453]
Epoch: [236/484] Iter:[0/24], Time: 0.97, lr: [0.0054782984367322575], Loss: 0.095581
Epoch: [236/484] Iter:[10/24], Time: 0.44, lr: [0.005470014023032142], Loss: 0.133238
Epoch: [236/484] Iter:[20/24], Time: 0.42, lr: [0.005461728215001], Loss: 0.155582
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97291006 0.79153287 0.90403658 0.41901161 0.4375676  0.59534251
 0.61382787 0.72743112 0.90873216 0.49837789 0.93162598 0.78016336
 0.52354866 0.91765563 0.43236997 0.54720484 0.43682635 0.51394587
 0.72817097] 0.6673832585745568
1 [0.97300747 0.79383625 0.90346643 0.45469753 0.44168321 0.60210578
 0.6321703  0.7358576  0.91020024 0.50851006 0.92766206 0.7889246
 0.54903112 0.92383461 0.46742993 0.62398654 0.45049476 0.54217935
 0.73470636] 0.6823044309886663
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.257, MeanIU:  0.6823, Best_mIoU:  0.6973
[0.97300747 0.79383625 0.90346643 0.45469753 0.44168321 0.60210578
 0.6321703  0.7358576  0.91020024 0.50851006 0.92766206 0.7889246
 0.54903112 0.92383461 0.46742993 0.62398654 0.45049476 0.54217935
 0.73470636]
Epoch: [237/484] Iter:[0/24], Time: 0.95, lr: [0.005458413500796946], Loss: 0.061297
Epoch: [237/484] Iter:[10/24], Time: 0.44, lr: [0.005450125736356178], Loss: 0.143511
Epoch: [237/484] Iter:[20/24], Time: 0.42, lr: [0.0054418365713630535], Loss: 0.150845
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97442514 0.80006905 0.90947754 0.40438389 0.50306016 0.59897678
 0.62060005 0.73054452 0.91089631 0.51358552 0.93062281 0.77524147
 0.53259579 0.91317968 0.44083687 0.52667814 0.50719157 0.54103027
 0.73737627] 0.6774090434474852
1 [0.97462611 0.8031293  0.91065849 0.4334359  0.51831666 0.60336457
 0.63503189 0.73502623 0.912807   0.52665677 0.92975782 0.78023577
 0.54077505 0.92081392 0.4733101  0.59084839 0.54357887 0.56068178
 0.74028547] 0.691228425554096
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6912, Best_mIoU:  0.6973
[0.97462611 0.8031293  0.91065849 0.4334359  0.51831666 0.60336457
 0.63503189 0.73502623 0.912807   0.52665677 0.92975782 0.78023577
 0.54077505 0.92081392 0.4733101  0.59084839 0.54357887 0.56068178
 0.74028547]
Epoch: [238/484] Iter:[0/24], Time: 0.96, lr: [0.005438520512627289], Loss: 0.105887
Epoch: [238/484] Iter:[10/24], Time: 0.44, lr: [0.005430229382477541], Loss: 0.113456
Epoch: [238/484] Iter:[20/24], Time: 0.42, lr: [0.005421936845500896], Loss: 0.126688
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97362409 0.79248874 0.90937341 0.41342512 0.49679426 0.59537533
 0.618156   0.72970291 0.91097972 0.48385687 0.93787963 0.78581789
 0.52425992 0.91784367 0.44782723 0.54783305 0.46187432 0.53303387
 0.73281931] 0.6743665972621032
1 [0.97395102 0.79401756 0.91024065 0.4467432  0.51337354 0.60514423
 0.63493731 0.74148913 0.91185803 0.49004219 0.93796605 0.79180558
 0.54043948 0.92560532 0.49888395 0.61548682 0.45094761 0.54782312
 0.7383694 ] 0.6878486415558224
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.6878, Best_mIoU:  0.6973
[0.97395102 0.79401756 0.91024065 0.4467432  0.51337354 0.60514423
 0.63493731 0.74148913 0.91185803 0.49004219 0.93796605 0.79180558
 0.54043948 0.92560532 0.49888395 0.61548682 0.45094761 0.54782312
 0.7383694 ]
Epoch: [239/484] Iter:[0/24], Time: 0.96, lr: [0.005418619436209846], Loss: 0.155922
Epoch: [239/484] Iter:[10/24], Time: 0.44, lr: [0.005410324925254614], Loss: 0.147982
Epoch: [239/484] Iter:[20/24], Time: 0.42, lr: [0.005402029001144062], Loss: 0.142175
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97422648 0.79597445 0.91047483 0.36984259 0.50888961 0.59965392
 0.62083009 0.73478504 0.90927711 0.5094886  0.93954478 0.78360472
 0.53886339 0.91765886 0.45394952 0.5535876  0.49128144 0.53632762
 0.72684165] 0.6776369633137566
1 [0.97424557 0.79976187 0.91126521 0.38179377 0.53366275 0.60814016
 0.6422741  0.74451743 0.9092555  0.52444629 0.93869026 0.7839716
 0.53577602 0.92452021 0.50053496 0.62603541 0.51512267 0.54345714
 0.73324575] 0.6910903517426119
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.6911, Best_mIoU:  0.6973
[0.97424557 0.79976187 0.91126521 0.38179377 0.53366275 0.60814016
 0.6422741  0.74451743 0.9092555  0.52444629 0.93869026 0.7839716
 0.53577602 0.92452021 0.50053496 0.62603541 0.51512267 0.54345714
 0.73324575]
Epoch: [240/484] Iter:[0/24], Time: 0.95, lr: [0.005398710235222419], Loss: 0.104718
Epoch: [240/484] Iter:[10/24], Time: 0.44, lr: [0.0053904123282354005], Loss: 0.135867
Epoch: [240/484] Iter:[20/24], Time: 0.42, lr: [0.005382113001710066], Loss: 0.131244
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97423206 0.7978687  0.91056967 0.39685585 0.50177633 0.59760676
 0.62586696 0.74160419 0.91179797 0.50844344 0.93768035 0.78786391
 0.54694255 0.91605677 0.44566591 0.53898918 0.49494096 0.54709341
 0.73883083] 0.680036094925676
1 [0.97448136 0.80117359 0.91112395 0.40919128 0.51878381 0.60342321
 0.64451023 0.75185562 0.91295538 0.51639071 0.93674977 0.79206891
 0.55738012 0.92162564 0.48694697 0.60407847 0.53029744 0.56799857
 0.74414557] 0.6939568743303606
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.6940, Best_mIoU:  0.6973
[0.97448136 0.80117359 0.91112395 0.40919128 0.51878381 0.60342321
 0.64451023 0.75185562 0.91295538 0.51639071 0.93674977 0.79206891
 0.55738012 0.92162564 0.48694697 0.60407847 0.53029744 0.56799857
 0.74414557]
Epoch: [241/484] Iter:[0/24], Time: 1.00, lr: [0.005378792873030137], Loss: 0.169432
Epoch: [241/484] Iter:[10/24], Time: 0.45, lr: [0.005370491554653574], Loss: 0.126016
Epoch: [241/484] Iter:[20/24], Time: 0.42, lr: [0.005362188810300432], Loss: 0.138431
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97381547 0.80107613 0.90711608 0.35243547 0.47751038 0.59320396
 0.62607188 0.74171478 0.90993115 0.51672396 0.93496747 0.78675597
 0.54032905 0.91785468 0.4413556  0.53749486 0.47398452 0.53907094
 0.73549885] 0.6740479585457867
1 [0.97431972 0.8049205  0.90726021 0.32906229 0.50157053 0.59968517
 0.64433691 0.75167509 0.91177122 0.53644826 0.93471997 0.79244078
 0.55610827 0.92429734 0.47714779 0.61336936 0.53065359 0.55714719
 0.74020578] 0.6887968415971645
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.6888, Best_mIoU:  0.6973
[0.97431972 0.8049205  0.90726021 0.32906229 0.50157053 0.59968517
 0.64433691 0.75167509 0.91177122 0.53644826 0.93471997 0.79244078
 0.55610827 0.92429734 0.47714779 0.61336936 0.53065359 0.55714719
 0.74020578]
Epoch: [242/484] Iter:[0/24], Time: 0.94, lr: [0.005358867312681466], Loss: 0.112834
Epoch: [242/484] Iter:[10/24], Time: 0.44, lr: [0.00535056256742446], Loss: 0.117140
Epoch: [242/484] Iter:[20/24], Time: 0.42, lr: [0.005342256389696638], Loss: 0.120887
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97438859 0.80102436 0.90844123 0.34781644 0.44309931 0.59708622
 0.62530798 0.74358228 0.90802431 0.51395944 0.93675244 0.7803292
 0.53439132 0.91890287 0.44666378 0.53163528 0.46165328 0.53040353
 0.72807247] 0.6700807550484662
1 [0.9747269  0.80537406 0.91048082 0.35941529 0.47167011 0.6007928
 0.63774885 0.75161742 0.90857654 0.52136902 0.93794784 0.78660717
 0.55289141 0.92656372 0.5003039  0.63221264 0.52627564 0.54519017
 0.73701506] 0.6887778611773517
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.6888, Best_mIoU:  0.6973
[0.9747269  0.80537406 0.91048082 0.35941529 0.47167011 0.6007928
 0.63774885 0.75161742 0.90857654 0.52136902 0.93794784 0.78660717
 0.55289141 0.92656372 0.5003039  0.63221264 0.52627564 0.54519017
 0.73701506]
Epoch: [243/484] Iter:[0/24], Time: 0.96, lr: [0.005338933516904143], Loss: 0.178071
Epoch: [243/484] Iter:[10/24], Time: 0.44, lr: [0.005330625329140951], Loss: 0.162072
Epoch: [243/484] Iter:[20/24], Time: 0.42, lr: [0.005322315702356005], Loss: 0.154639
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97389241 0.79611132 0.90761255 0.34080602 0.47978055 0.60110853
 0.62246898 0.73888177 0.90817447 0.52551273 0.93807099 0.78372429
 0.53274307 0.91618701 0.43560656 0.49984076 0.44418507 0.5355631
 0.73456899] 0.6692020612026323
1 [0.97396354 0.79918627 0.90859355 0.35221444 0.50343443 0.60809229
 0.63880799 0.74896891 0.90944247 0.54137886 0.93842574 0.78878344
 0.54554768 0.92380621 0.47587769 0.58285257 0.43267725 0.55112513
 0.74011896] 0.6822788122339597
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.253, MeanIU:  0.6823, Best_mIoU:  0.6973
[0.97396354 0.79918627 0.90859355 0.35221444 0.50343443 0.60809229
 0.63880799 0.74896891 0.90944247 0.54137886 0.93842574 0.78878344
 0.54554768 0.92380621 0.47587769 0.58285257 0.43267725 0.55112513
 0.74011896]
Epoch: [244/484] Iter:[0/24], Time: 0.96, lr: [0.005318991448101063], Loss: 0.157840
Epoch: [244/484] Iter:[10/24], Time: 0.44, lr: [0.005310679802069352], Loss: 0.146837
Epoch: [244/484] Iter:[20/24], Time: 0.42, lr: [0.005302366710407516], Loss: 0.145339
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97386964 0.79881281 0.9087585  0.37233032 0.49288844 0.60221526
 0.61731458 0.74568379 0.91009285 0.52847019 0.92554034 0.78715157
 0.54059228 0.91794596 0.43190512 0.58190547 0.47169071 0.54081796
 0.73384133] 0.6779909020829021
1 [0.97445971 0.80262895 0.91010659 0.39689841 0.51273156 0.61053902
 0.63981227 0.75461641 0.9118063  0.54349351 0.92256784 0.7927362
 0.55241938 0.92562974 0.46542934 0.65626474 0.50384748 0.5690258
 0.73977309] 0.6939361227676542
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.6939, Best_mIoU:  0.6973
[0.97445971 0.80262895 0.91010659 0.39689841 0.51273156 0.61053902
 0.63981227 0.75461641 0.9118063  0.54349351 0.92256784 0.7927362
 0.55241938 0.92562974 0.46542934 0.65626474 0.50384748 0.5690258
 0.73977309]
Epoch: [245/484] Iter:[0/24], Time: 0.96, lr: [0.005299041068346068], Loss: 0.159497
Epoch: [245/484] Iter:[10/24], Time: 0.44, lr: [0.005290725948145149], Loss: 0.135612
Epoch: [245/484] Iter:[20/24], Time: 0.42, lr: [0.005282409375647552], Loss: 0.136025
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97410027 0.79650838 0.90782115 0.31500924 0.4732432  0.59320161
 0.62701559 0.74227479 0.91085794 0.53375248 0.93208724 0.77136332
 0.43673159 0.91865895 0.4341558  0.58347623 0.51920648 0.52957542
 0.73731099] 0.6703342459645869
1 [0.97424835 0.79864189 0.90736425 0.29418637 0.49475607 0.60025707
 0.64456712 0.75150415 0.91268565 0.55125624 0.92969474 0.7816836
 0.47555239 0.92601461 0.4658629  0.65512598 0.54736523 0.56060727
 0.74471986] 0.6850575652553621
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6851, Best_mIoU:  0.6973
[0.97424835 0.79864189 0.90736425 0.29418637 0.49475607 0.60025707
 0.64456712 0.75150415 0.91268565 0.55125624 0.92969474 0.7816836
 0.47555239 0.92601461 0.4658629  0.65512598 0.54736523 0.56060727
 0.74471986]
Epoch: [246/484] Iter:[0/24], Time: 0.97, lr: [0.005279082339379693], Loss: 0.085675
Epoch: [246/484] Iter:[10/24], Time: 0.44, lr: [0.005270763728968713], Loss: 0.149597
Epoch: [246/484] Iter:[20/24], Time: 0.42, lr: [0.00526244365953557], Loss: 0.147118
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97322736 0.79253868 0.90629372 0.3056647  0.48846378 0.59975763
 0.6285989  0.74018445 0.90875056 0.53120572 0.93471894 0.75738928
 0.49524469 0.9160199  0.4591784  0.56624974 0.49466695 0.53905064
 0.71626082] 0.6712349921696233
1 [0.97381793 0.79614402 0.90643957 0.30412927 0.50723457 0.60528096
 0.64170651 0.74940581 0.91032132 0.5443947  0.93449903 0.74127774
 0.45983766 0.92245489 0.48646841 0.62430042 0.50845385 0.54633303
 0.72445762] 0.6782609107328877
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.262, MeanIU:  0.6783, Best_mIoU:  0.6973
[0.97381793 0.79614402 0.90643957 0.30412927 0.50723457 0.60528096
 0.64170651 0.74940581 0.91032132 0.5443947  0.93449903 0.74127774
 0.45983766 0.92245489 0.48646841 0.62430042 0.50845385 0.54633303
 0.72445762]
Epoch: [247/484] Iter:[0/24], Time: 0.95, lr: [0.005259115222604805], Loss: 0.099026
Epoch: [247/484] Iter:[10/24], Time: 0.44, lr: [0.005250793105800928], Loss: 0.135699
Epoch: [247/484] Iter:[20/24], Time: 0.42, lr: [0.005242469523189688], Loss: 0.139572
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97354133 0.78990302 0.90843238 0.38683622 0.48836612 0.59128935
 0.61909204 0.74055247 0.90969968 0.53032727 0.92875882 0.76358479
 0.41941602 0.92180306 0.48854295 0.58960431 0.44782365 0.5412521
 0.73460027] 0.6722855702502469
1 [0.97337683 0.79108854 0.90982195 0.44139833 0.51035352 0.59924647
 0.63583145 0.74921744 0.91137008 0.53941855 0.9253565  0.76493587
 0.43053391 0.92899314 0.542911   0.63205705 0.3719044  0.55711583
 0.74046645] 0.681863016391364
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.6819, Best_mIoU:  0.6973
[0.97337683 0.79108854 0.90982195 0.44139833 0.51035352 0.59924647
 0.63583145 0.74921744 0.91137008 0.53941855 0.9253565  0.76493587
 0.43053391 0.92899314 0.542911   0.63205705 0.3719044  0.55711583
 0.74046645]
Epoch: [248/484] Iter:[0/24], Time: 0.99, lr: [0.00523913967908221], Loss: 0.108729
Epoch: [248/484] Iter:[10/24], Time: 0.44, lr: [0.0052308140395587385], Loss: 0.131407
Epoch: [248/484] Iter:[20/24], Time: 0.42, lr: [0.00522248692738221], Loss: 0.131780
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97419871 0.79443802 0.90932067 0.39638866 0.4896292  0.59997041
 0.6286049  0.74448184 0.90995454 0.54040388 0.93328411 0.78208319
 0.53439717 0.91251432 0.46685155 0.57380726 0.48673625 0.54428297
 0.72789266] 0.6815389645156358
1 [0.97423335 0.79584763 0.91049448 0.44235405 0.51574636 0.60761523
 0.64640293 0.75361792 0.912943   0.55360762 0.93307881 0.78856293
 0.54993341 0.91963226 0.50951204 0.62586464 0.50821696 0.56375399
 0.73665252] 0.6972668492873679
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6973, Best_mIoU:  0.6973
[0.97423335 0.79584763 0.91049448 0.44235405 0.51574636 0.60761523
 0.64640293 0.75361792 0.912943   0.55360762 0.93307881 0.78856293
 0.54993341 0.91963226 0.50951204 0.62586464 0.50821696 0.56375399
 0.73665252]
Epoch: [249/484] Iter:[0/24], Time: 0.96, lr: [0.00521915566952613], Loss: 0.105024
Epoch: [249/484] Iter:[10/24], Time: 0.44, lr: [0.0052108264908106265], Loss: 0.116945
Epoch: [249/484] Iter:[20/24], Time: 0.42, lr: [0.0052024958325350736], Loss: 0.125429
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97322075 0.79179108 0.91010217 0.40811958 0.49548627 0.59900415
 0.62438789 0.74415229 0.91030254 0.54840517 0.93444936 0.78186474
 0.52484018 0.91617257 0.47360077 0.58052866 0.48872235 0.55136507
 0.73645328] 0.6838404667029706
1 [0.97327609 0.79557108 0.91234777 0.4585848  0.52260608 0.60653911
 0.64335868 0.7523881  0.91339344 0.57632634 0.93481148 0.78740331
 0.54780808 0.92461397 0.50561754 0.63741962 0.44202403 0.57374744
 0.74246381] 0.6973842501587793
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6974, Best_mIoU:  0.6974
[0.97327609 0.79557108 0.91234777 0.4585848  0.52260608 0.60653911
 0.64335868 0.7523881  0.91339344 0.57632634 0.93481148 0.78740331
 0.54780808 0.92461397 0.50561754 0.63741962 0.44202403 0.57374744
 0.74246381]
Epoch: [250/484] Iter:[0/24], Time: 0.94, lr: [0.005199163154299655], Loss: 0.152359
Epoch: [250/484] Iter:[10/24], Time: 0.44, lr: [0.005190830419771997], Loss: 0.160357
Epoch: [250/484] Iter:[20/24], Time: 0.42, lr: [0.005182496198715191], Loss: 0.149565
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97357297 0.79629344 0.90512763 0.34996526 0.49418274 0.60358373
 0.62986547 0.74070845 0.91135866 0.54386224 0.92058164 0.78207372
 0.52202489 0.92438578 0.47801199 0.63414037 0.44120744 0.54259185
 0.73533319] 0.6804669197372103
1 [0.97329146 0.80024852 0.90470543 0.34996049 0.52149746 0.61096542
 0.6453183  0.74984108 0.91354586 0.56109325 0.91369016 0.79004603
 0.54414307 0.92810167 0.53335829 0.68683546 0.4478277  0.56165472
 0.73963445] 0.6934609903614614
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.6935, Best_mIoU:  0.6974
[0.97329146 0.80024852 0.90470543 0.34996049 0.52149746 0.61096542
 0.6453183  0.74984108 0.91354586 0.56109325 0.91369016 0.79004603
 0.54414307 0.92810167 0.53335829 0.68683546 0.4478277  0.56165472
 0.73963445]
Epoch: [251/484] Iter:[0/24], Time: 0.96, lr: [0.005179162093410075], Loss: 0.121633
Epoch: [251/484] Iter:[10/24], Time: 0.44, lr: [0.005170825786300503], Loss: 0.135997
Epoch: [251/484] Iter:[20/24], Time: 0.42, lr: [0.005162487985629745], Loss: 0.135372
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9739511  0.79657612 0.90679685 0.2649417  0.5079975  0.60555888
 0.6349289  0.74466381 0.90988748 0.53638897 0.93238394 0.78497988
 0.53683585 0.91672453 0.45319073 0.55424733 0.4318578  0.54618266
 0.73270029] 0.6721470692435748
1 [0.9740103  0.7986073  0.90739657 0.25667408 0.52606305 0.61145251
 0.65120956 0.75337264 0.91236551 0.55612808 0.93349274 0.78976583
 0.55461651 0.92235507 0.4931276  0.61520866 0.42924102 0.56566291
 0.73617991] 0.6835226241476763
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.259, MeanIU:  0.6835, Best_mIoU:  0.6974
[0.9740103  0.7986073  0.90739657 0.25667408 0.52606305 0.61145251
 0.65120956 0.75337264 0.91236551 0.55612808 0.93349274 0.78976583
 0.55461651 0.92235507 0.4931276  0.61520866 0.42924102 0.56566291
 0.73617991]
Epoch: [252/484] Iter:[0/24], Time: 0.97, lr: [0.005159152446504152], Loss: 0.151819
Epoch: [252/484] Iter:[10/24], Time: 0.44, lr: [0.005150812549891269], Loss: 0.140487
Epoch: [252/484] Iter:[20/24], Time: 0.42, lr: [0.00514247115262138], Loss: 0.152062
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96673718 0.79356566 0.89346754 0.26799449 0.51632727 0.57073737
 0.62863848 0.7359285  0.90167118 0.52514405 0.92615908 0.77012706
 0.52589047 0.90922007 0.47539533 0.5588652  0.47771045 0.49457345
 0.72222411] 0.6663356287407464
1 [0.96823658 0.79662456 0.89480841 0.28531585 0.53842079 0.57855351
 0.63735491 0.73925803 0.90167367 0.52827376 0.92944172 0.77369477
 0.53777334 0.91575141 0.51855587 0.60467575 0.53381441 0.49895941
 0.72381285] 0.6792105052005522
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.278, MeanIU:  0.6792, Best_mIoU:  0.6974
[0.96823658 0.79662456 0.89480841 0.28531585 0.53842079 0.57855351
 0.63735491 0.73925803 0.90167367 0.52827376 0.92944172 0.77369477
 0.53777334 0.91575141 0.51855587 0.60467575 0.53381441 0.49895941
 0.72381285]
Epoch: [253/484] Iter:[0/24], Time: 0.98, lr: [0.005139134172863301], Loss: 0.146087
Epoch: [253/484] Iter:[10/24], Time: 0.44, lr: [0.005130790669672037], Loss: 0.186491
Epoch: [253/484] Iter:[20/24], Time: 0.42, lr: [0.005122445658663309], Loss: 0.184756
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97274553 0.78903698 0.90403459 0.36478899 0.49501591 0.59272963
 0.61579421 0.73793242 0.90677501 0.53767729 0.9341641  0.78302768
 0.53775629 0.92319862 0.45271315 0.55500852 0.31065367 0.50851433
 0.72001481] 0.6653464072342745
1 [0.97186696 0.79381085 0.90527608 0.41395842 0.50399686 0.60188895
 0.63206871 0.74561523 0.90924391 0.55810903 0.93609059 0.78681744
 0.55042833 0.92704796 0.52889925 0.63885031 0.26668812 0.53598275
 0.7270577 ] 0.6807209185129343
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6807, Best_mIoU:  0.6974
[0.97186696 0.79381085 0.90527608 0.41395842 0.50399686 0.60188895
 0.63206871 0.74561523 0.90924391 0.55810903 0.93609059 0.78681744
 0.55042833 0.92704796 0.52889925 0.63885031 0.26668812 0.53598275
 0.7270577 ]
Epoch: [254/484] Iter:[0/24], Time: 0.93, lr: [0.005119107231398681], Loss: 0.165910
Epoch: [254/484] Iter:[10/24], Time: 0.44, lr: [0.005110760104398225], Loss: 0.144612
Epoch: [254/484] Iter:[20/24], Time: 0.42, lr: [0.005102411462354338], Loss: 0.135087
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97413519 0.80009797 0.90618564 0.30482447 0.48357135 0.59012619
 0.62961871 0.74471654 0.91083905 0.5102996  0.9356859  0.7848715
 0.53614193 0.91762545 0.43215792 0.57862463 0.44264213 0.55070596
 0.73053612] 0.6717582222565249
1 [0.9745214  0.80417773 0.90752343 0.31068799 0.49861604 0.59187921
 0.64371334 0.75044526 0.9127944  0.52650985 0.93627598 0.78541876
 0.5388463  0.92502801 0.49111341 0.66876107 0.50440209 0.56165369
 0.73585028] 0.6878009598002433
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.6878, Best_mIoU:  0.6974
[0.9745214  0.80417773 0.90752343 0.31068799 0.49861604 0.59187921
 0.64371334 0.75044526 0.9127944  0.52650985 0.93627598 0.78541876
 0.5388463  0.92502801 0.49111341 0.66876107 0.50440209 0.56165369
 0.73585028]
Epoch: [255/484] Iter:[0/24], Time: 0.97, lr: [0.005099071580646209], Loss: 0.111581
Epoch: [255/484] Iter:[10/24], Time: 0.44, lr: [0.0050907208124479075], Loss: 0.146679
Epoch: [255/484] Iter:[20/24], Time: 0.42, lr: [0.005082368521913806], Loss: 0.140974
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97517818 0.80261158 0.90693217 0.36829402 0.49098105 0.59278665
 0.62280411 0.7432179  0.90888281 0.49252928 0.93603924 0.78435826
 0.53493757 0.92133726 0.4528644  0.59421905 0.41526539 0.56570181
 0.73031482] 0.6757502923463161
1 [0.97471236 0.80789325 0.90869422 0.39107349 0.51746464 0.60369863
 0.64291714 0.75127296 0.91083803 0.50371942 0.93547526 0.78869116
 0.54822331 0.92791951 0.51924341 0.67255207 0.39359364 0.58555964
 0.73533564] 0.6904672504610987
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.6905, Best_mIoU:  0.6974
[0.97471236 0.80789325 0.90869422 0.39107349 0.51746464 0.60369863
 0.64291714 0.75127296 0.91083803 0.50371942 0.93547526 0.78869116
 0.54822331 0.92791951 0.51924341 0.67255207 0.39359364 0.58555964
 0.73533564]
Epoch: [256/484] Iter:[0/24], Time: 0.95, lr: [0.005079027178761477], Loss: 0.163806
Epoch: [256/484] Iter:[10/24], Time: 0.44, lr: [0.005070672751816684], Loss: 0.139825
Epoch: [256/484] Iter:[20/24], Time: 0.42, lr: [0.005062316795176419], Loss: 0.135576
0
0
0
0
10
10
10
10
20
20
2020

30
30
30
30
40
40
40
40
0 [0.97500528 0.79704761 0.91057194 0.38768078 0.50107066 0.60098061
 0.62349008 0.73889032 0.91111269 0.50503779 0.93684751 0.78550363
 0.53582338 0.91582895 0.44220517 0.556739   0.49176733 0.5570398
 0.73131106] 0.6791554514455347
1 [0.97477456 0.80189018 0.91259119 0.4230391  0.52644672 0.60998346
 0.64474143 0.75172936 0.91312271 0.51841088 0.93768213 0.78952189
 0.54863997 0.9228684  0.46868231 0.67221207 0.63029836 0.58167972
 0.73649077] 0.7034108015555691
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7034, Best_mIoU:  0.7034
[0.97477456 0.80189018 0.91259119 0.4230391  0.52644672 0.60998346
 0.64474143 0.75172936 0.91312271 0.51841088 0.93768213 0.78952189
 0.54863997 0.9228684  0.46868231 0.67221207 0.63029836 0.58167972
 0.73649077]
Epoch: [257/484] Iter:[0/24], Time: 0.93, lr: [0.00505897398351458], Loss: 0.150013
Epoch: [257/484] Iter:[10/24], Time: 0.44, lr: [0.0050506158801124764], Loss: 0.147637
Epoch: [257/484] Iter:[20/24], Time: 0.42, lr: [0.005042256239587005], Loss: 0.145781
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97606758 0.80570433 0.90996271 0.38530909 0.49370899 0.60525821
 0.62436551 0.74323531 0.91146037 0.51224557 0.93879959 0.78445393
 0.54027242 0.9212143  0.46814507 0.57521886 0.49336724 0.56282312
 0.73374691] 0.6834399535663556
1 [0.97569463 0.80916024 0.91103479 0.42177303 0.51681044 0.61340164
 0.63609493 0.74856667 0.91277416 0.52583427 0.9394077  0.78980764
 0.55426123 0.92883601 0.51778137 0.69127964 0.59204559 0.58480451
 0.7388945 ] 0.705698052070004
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7057, Best_mIoU:  0.7057
[0.97569463 0.80916024 0.91103479 0.42177303 0.51681044 0.61340164
 0.63609493 0.74856667 0.91277416 0.52583427 0.9394077  0.78980764
 0.55426123 0.92883601 0.51778137 0.69127964 0.59204559 0.58480451
 0.7388945 ]
Epoch: [258/484] Iter:[0/24], Time: 0.96, lr: [0.005038911952284847], Loss: 0.151089
Epoch: [258/484] Iter:[10/24], Time: 0.44, lr: [0.005030550154550224], Loss: 0.130108
Epoch: [258/484] Iter:[20/24], Time: 0.42, lr: [0.005022186812195169], Loss: 0.130951
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97505936 0.80121035 0.9093804  0.36628036 0.48554744 0.60070496
 0.62665986 0.74689417 0.91074976 0.52213694 0.93635509 0.78310882
 0.5316574  0.91753547 0.4574369  0.55397443 0.4716051  0.56096777
 0.73548055] 0.6785655324685019
1 [0.97480821 0.80400826 0.90798113 0.34257494 0.49299029 0.60692898
 0.63558877 0.75239087 0.91238647 0.53534032 0.93715236 0.78731906
 0.54663878 0.9224485  0.47770431 0.60757933 0.53570842 0.57723898
 0.7402214 ] 0.6893162833678542
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.6893, Best_mIoU:  0.7057
[0.97480821 0.80400826 0.90798113 0.34257494 0.49299029 0.60692898
 0.63558877 0.75239087 0.91238647 0.53534032 0.93715236 0.78731906
 0.54663878 0.9224485  0.47770431 0.60757933 0.53570842 0.57723898
 0.7402214 ]
Epoch: [259/484] Iter:[0/24], Time: 0.97, lr: [0.005018841042055489], Loss: 0.100281
Epoch: [259/484] Iter:[10/24], Time: 0.44, lr: [0.005010475531946474], Loss: 0.122607
Epoch: [259/484] Iter:[20/24], Time: 0.42, lr: [0.00500210846964985], Loss: 0.131285
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97461301 0.79396404 0.91160569 0.37349293 0.48736504 0.60333456
 0.62919789 0.74623108 0.9126077  0.50771989 0.94026747 0.78575441
 0.53092432 0.92088052 0.48362429 0.5850473  0.49076537 0.56435365
 0.7352807 ] 0.6830015734034401
1 [0.97455387 0.79748551 0.91318924 0.38701067 0.51113889 0.61397476
 0.64626944 0.75610362 0.91469417 0.51546231 0.94191855 0.79030478
 0.54918807 0.92770981 0.52816236 0.66486322 0.55246227 0.58122456
 0.73982763] 0.7002917754991401
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7003, Best_mIoU:  0.7057
[0.97455387 0.79748551 0.91318924 0.38701067 0.51113889 0.61397476
 0.64626944 0.75610362 0.91469417 0.51546231 0.94191855 0.79030478
 0.54918807 0.92770981 0.52816236 0.66486322 0.55246227 0.58122456
 0.73982763]
Epoch: [260/484] Iter:[0/24], Time: 0.95, lr: [0.004998761209408129], Loss: 0.145896
Epoch: [260/484] Iter:[10/24], Time: 0.44, lr: [0.004990391968713892], Loss: 0.128009
Epoch: [260/484] Iter:[20/24], Time: 0.42, lr: [0.004982021168193775], Loss: 0.129766
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97606246 0.80625703 0.91105866 0.40925788 0.49697479 0.59941095
 0.63158127 0.74440816 0.91156781 0.53076586 0.93609665 0.78643881
 0.53171859 0.91870383 0.47439538 0.58160204 0.49270968 0.54952503
 0.73044416] 0.6852094228342314
1 [0.97615254 0.81140796 0.91341448 0.46865768 0.52415509 0.60860211
 0.64076571 0.74898973 0.91422802 0.55357011 0.93568889 0.79120975
 0.54754178 0.92654056 0.53065659 0.6491967  0.54132572 0.55303444
 0.73208738] 0.7035381720630689
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.236, MeanIU:  0.7035, Best_mIoU:  0.7057
[0.97615254 0.81140796 0.91341448 0.46865768 0.52415509 0.60860211
 0.64076571 0.74898973 0.91422802 0.55357011 0.93568889 0.79120975
 0.54754178 0.92654056 0.53065659 0.6491967  0.54132572 0.55303444
 0.73208738]
Epoch: [261/484] Iter:[0/24], Time: 0.94, lr: [0.004978672410517248], Loss: 0.214157
Epoch: [261/484] Iter:[10/24], Time: 0.44, lr: [0.004970299420855652], Loss: 0.163939
Epoch: [261/484] Iter:[20/24], Time: 0.42, lr: [0.004961924863657825], Loss: 0.159851
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97549289 0.80696249 0.91078631 0.35628953 0.49604421 0.59507692
 0.6196343  0.74257637 0.91092355 0.51905272 0.93615385 0.7878011
 0.54144541 0.9169477  0.45461585 0.54969724 0.53044313 0.55739809
 0.73742606] 0.6813035651937813
1 [0.97576554 0.81271016 0.91337523 0.38933943 0.52356578 0.60028351
 0.63764346 0.75031658 0.91165201 0.51820095 0.93644577 0.79241223
 0.55356894 0.92499548 0.50747925 0.63257989 0.58676146 0.56544924
 0.74246681] 0.6986848281955355
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.6987, Best_mIoU:  0.7057
[0.97576554 0.81271016 0.91337523 0.38933943 0.52356578 0.60028351
 0.63764346 0.75031658 0.91165201 0.51820095 0.93644577 0.79241223
 0.55356894 0.92499548 0.50747925 0.63257989 0.58676146 0.56544924
 0.74246681]
Epoch: [262/484] Iter:[0/24], Time: 0.95, lr: [0.004958574601144527], Loss: 0.054999
Epoch: [262/484] Iter:[10/24], Time: 0.44, lr: [0.004950197843959735], Loss: 0.154723
Epoch: [262/484] Iter:[20/24], Time: 0.42, lr: [0.004941819511455273], Loss: 0.147473
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9755457  0.80711937 0.90509844 0.34692263 0.49125178 0.59247382
 0.62745531 0.74684992 0.90872774 0.49873618 0.93473726 0.78388934
 0.51949914 0.92216873 0.45441999 0.54439217 0.47110601 0.55734435
 0.74282225] 0.6752926394021648
1 [0.97543543 0.80981263 0.90631772 0.36991345 0.51047059 0.60094761
 0.64503694 0.75656282 0.91031439 0.51183324 0.933294   0.79106596
 0.54376742 0.93035501 0.53388113 0.61969727 0.47197215 0.56940058
 0.7482615 ] 0.6914915695165359
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6915, Best_mIoU:  0.7057
[0.97543543 0.80981263 0.90631772 0.36991345 0.51047059 0.60094761
 0.64503694 0.75656282 0.91031439 0.51183324 0.933294   0.79106596
 0.54376742 0.93035501 0.53388113 0.61969727 0.47197215 0.56940058
 0.7482615 ]
Epoch: [263/484] Iter:[0/24], Time: 0.96, lr: [0.004938467736633074], Loss: 0.095057
Epoch: [263/484] Iter:[10/24], Time: 0.44, lr: [0.004930087193193121], Loss: 0.148901
Epoch: [263/484] Iter:[20/24], Time: 0.42, lr: [0.004921705066575941], Loss: 0.144478
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97511216 0.80282152 0.90903647 0.3140932  0.49582568 0.60183966
 0.62651316 0.74624018 0.91120345 0.52690558 0.93535824 0.77989549
 0.53963227 0.90784529 0.41336449 0.50545827 0.54019883 0.54264788
 0.74200299] 0.6745260423489792
1 [0.97490045 0.80671019 0.91014941 0.31816331 0.52251653 0.61073424
 0.64589519 0.75540261 0.91304373 0.53268213 0.93482872 0.78400992
 0.54770609 0.91623021 0.44445378 0.58212987 0.57931366 0.56756786
 0.74903093] 0.6892352010865231
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.252, MeanIU:  0.6892, Best_mIoU:  0.7057
[0.97490045 0.80671019 0.91014941 0.31816331 0.52251653 0.61073424
 0.64589519 0.75540261 0.91304373 0.53268213 0.93482872 0.78400992
 0.54770609 0.91623021 0.44445378 0.58212987 0.57931366 0.56756786
 0.74903093]
Epoch: [264/484] Iter:[0/24], Time: 0.97, lr: [0.004918351771901559], Loss: 0.146765
Epoch: [264/484] Iter:[10/24], Time: 0.44, lr: [0.0049099674232958666], Loss: 0.153230
Epoch: [264/484] Iter:[20/24], Time: 0.42, lr: [0.00490158148358023], Loss: 0.140173
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9745303  0.80613182 0.90941001 0.34196046 0.51052523 0.60152592
 0.62635891 0.74204425 0.91122132 0.52810601 0.9347876  0.78318277
 0.53614624 0.91550871 0.42737772 0.57900159 0.47665984 0.55729024
 0.74061244] 0.6790727037112461
1 [0.97447234 0.81108219 0.9113799  0.36755584 0.54141418 0.6112308
 0.64550558 0.75500304 0.91356675 0.54253606 0.93398339 0.7894272
 0.55605299 0.92416016 0.46762189 0.66426832 0.44810683 0.585706
 0.74817628] 0.6942763022725762
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.6943, Best_mIoU:  0.7057
[0.97447234 0.81108219 0.9113799  0.36755584 0.54141418 0.6112308
 0.64550558 0.75500304 0.91356675 0.54253606 0.93398339 0.7894272
 0.55605299 0.92416016 0.46762189 0.66426832 0.44810683 0.585706
 0.74817628]
Epoch: [265/484] Iter:[0/24], Time: 0.99, lr: [0.004898226661438224], Loss: 0.099225
Epoch: [265/484] Iter:[10/24], Time: 0.45, lr: [0.0048898384885750846], Loss: 0.150465
Epoch: [265/484] Iter:[20/24], Time: 0.42, lr: [0.004881448716593049], Loss: 0.144028
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97431084 0.8025077  0.90382054 0.28151505 0.48802961 0.59358835
 0.63004902 0.74672515 0.90783633 0.52409693 0.92179616 0.78495522
 0.52707717 0.91878652 0.46281877 0.6041217  0.47529529 0.57055867
 0.73770988] 0.6766104671160954
1 [0.97429258 0.80830345 0.90370444 0.29402922 0.51376099 0.60030983
 0.64678622 0.75686206 0.91056851 0.54827088 0.91736618 0.79194577
 0.55136181 0.92772321 0.51488545 0.69834592 0.43306531 0.58042387
 0.74409858] 0.690321277633376
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.251, MeanIU:  0.6903, Best_mIoU:  0.7057
[0.97429258 0.80830345 0.90370444 0.29402922 0.51376099 0.60030983
 0.64678622 0.75686206 0.91056851 0.54827088 0.91736618 0.79194577
 0.55136181 0.92772321 0.51488545 0.69834592 0.43306531 0.58042387
 0.74409858]
Epoch: [266/484] Iter:[0/24], Time: 0.98, lr: [0.004878092359294797], Loss: 0.177436
Epoch: [266/484] Iter:[10/24], Time: 0.44, lr: [0.004869700342898788], Loss: 0.150166
Epoch: [266/484] Iter:[20/24], Time: 0.42, lr: [0.004861306719297618], Loss: 0.139302
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97010337 0.7782408  0.90858945 0.34377285 0.49742661 0.5864367
 0.62770785 0.74728098 0.90918827 0.52830811 0.93590716 0.78241182
 0.51936416 0.91829976 0.44121477 0.58506287 0.49065387 0.56799823
 0.74143477] 0.6778632841659166
1 [0.97026264 0.78319155 0.91129823 0.40552589 0.53095461 0.59576479
 0.64538592 0.75720935 0.91202216 0.55538428 0.93440957 0.79020544
 0.54221375 0.92557376 0.47854812 0.67467554 0.4889913  0.58640185
 0.74813305] 0.6966395685647156
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.259, MeanIU:  0.6966, Best_mIoU:  0.7057
[0.97026264 0.78319155 0.91129823 0.40552589 0.53095461 0.59576479
 0.64538592 0.75720935 0.91202216 0.55538428 0.93440957 0.79020544
 0.54221375 0.92557376 0.47854812 0.67467554 0.4889913  0.58640185
 0.74813305]
Epoch: [267/484] Iter:[0/24], Time: 0.96, lr: [0.004857948819080275], Loss: 0.234141
Epoch: [267/484] Iter:[10/24], Time: 0.44, lr: [0.004849552939689649], Loss: 0.127677
Epoch: [267/484] Iter:[20/24], Time: 0.42, lr: [0.00484115544492917], Loss: 0.122040
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97042095 0.77771207 0.90891013 0.32955587 0.49311798 0.59975762
 0.63330219 0.74527297 0.90829759 0.49813579 0.93348563 0.78416188
 0.54241918 0.9170637  0.43974917 0.57982937 0.49100045 0.56237121
 0.73175834] 0.6761222148078949
1 [0.97042058 0.7824274  0.90932764 0.37352129 0.52265349 0.60259826
 0.64128258 0.75205942 0.90881276 0.51192629 0.93164025 0.78750849
 0.55014267 0.92365044 0.47169316 0.68096542 0.58254359 0.57766354
 0.73514345] 0.6955779334685154
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.260, MeanIU:  0.6956, Best_mIoU:  0.7057
[0.97042058 0.7824274  0.90932764 0.37352129 0.52265349 0.60259826
 0.64128258 0.75205942 0.90881276 0.51192629 0.93164025 0.78750849
 0.55014267 0.92365044 0.47169316 0.68096542 0.58254359 0.57766354
 0.73514345]
Epoch: [268/484] Iter:[0/24], Time: 0.98, lr: [0.004837795993954604], Loss: 0.142013
Epoch: [268/484] Iter:[10/24], Time: 0.44, lr: [0.004829396231918609], Loss: 0.133590
Epoch: [268/484] Iter:[20/24], Time: 0.42, lr: [0.004820994846268523], Loss: 0.140232
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97311826 0.79218938 0.90980399 0.36392669 0.48389128 0.60741174
 0.6326026  0.74850687 0.91020844 0.49171354 0.93645625 0.78126482
 0.54463916 0.92379524 0.46145214 0.5907377  0.43735336 0.55823123
 0.73770558] 0.6781583286385416
1 [0.97294606 0.79426682 0.91027014 0.38571051 0.4970921  0.61288624
 0.64398243 0.75733262 0.91150414 0.49334644 0.93585958 0.78348801
 0.55109723 0.92954006 0.4942659  0.65203253 0.3942578  0.57294403
 0.74415231] 0.6861565763434612
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.6862, Best_mIoU:  0.7057
[0.97294606 0.79426682 0.91027014 0.38571051 0.4970921  0.61288624
 0.64398243 0.75733262 0.91150414 0.49334644 0.93585958 0.78348801
 0.55109723 0.92954006 0.4942659  0.65203253 0.3942578  0.57294403
 0.74415231]
Epoch: [269/484] Iter:[0/24], Time: 0.99, lr: [0.004817633836622232], Loss: 0.089595
Epoch: [269/484] Iter:[10/24], Time: 0.45, lr: [0.004809230172098395], Loss: 0.131789
Epoch: [269/484] Iter:[20/24], Time: 0.42, lr: [0.004800824875635524], Loss: 0.133025
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97293317 0.78301595 0.91098176 0.34895068 0.49808188 0.60207223
 0.63244691 0.74768916 0.9107602  0.51601769 0.93660665 0.78663916
 0.53086091 0.92031535 0.45922272 0.59438336 0.4585572  0.55372815
 0.73075061] 0.6786323009517807
1 [0.97295799 0.78891584 0.9129111  0.38732808 0.52990377 0.61189797
 0.64323962 0.75648162 0.91327831 0.53830985 0.93748538 0.79063739
 0.54329905 0.92614648 0.49486191 0.67702501 0.4648769  0.58036279
 0.73926143] 0.6952200265023796
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6952, Best_mIoU:  0.7057
[0.97295799 0.78891584 0.9129111  0.38732808 0.52990377 0.61189797
 0.64323962 0.75648162 0.91327831 0.53830985 0.93748538 0.79063739
 0.54329905 0.92614648 0.49486191 0.67702501 0.4648769  0.58036279
 0.73926143]
Epoch: [270/484] Iter:[0/24], Time: 0.98, lr: [0.004797462299325534], Loss: 0.291386
Epoch: [270/484] Iter:[10/24], Time: 0.45, lr: [0.004789054712276882], Loss: 0.124693
Epoch: [270/484] Iter:[20/24], Time: 0.42, lr: [0.004780645484882384], Loss: 0.130817
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97381987 0.7899402  0.9095962  0.35055424 0.48577057 0.60861513
 0.6249844  0.74656218 0.91079414 0.49579865 0.93632587 0.78634144
 0.54529238 0.92359713 0.45813259 0.5957037  0.44872853 0.50732958
 0.73101838] 0.675205535572092
1 [0.97399694 0.79574928 0.91069511 0.37729476 0.50908854 0.61755308
 0.64496626 0.75826259 0.91207538 0.51072235 0.93625716 0.79031452
 0.55546121 0.92806354 0.48660031 0.66614123 0.46625149 0.52899701
 0.73834224] 0.6898333163573619
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6898, Best_mIoU:  0.7057
[0.97399694 0.79574928 0.91069511 0.37729476 0.50908854 0.61755308
 0.64496626 0.75826259 0.91207538 0.51072235 0.93625716 0.79031452
 0.55546121 0.92806354 0.48660031 0.66614123 0.46625149 0.52899701
 0.73834224]
Epoch: [271/484] Iter:[0/24], Time: 0.95, lr: [0.004777281333838124], Loss: 0.160670
Epoch: [271/484] Iter:[10/24], Time: 0.44, lr: [0.004768869804030359], Loss: 0.146066
Epoch: [271/484] Iter:[20/24], Time: 0.42, lr: [0.004760456625386873], Loss: 0.141473
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97377251 0.78957264 0.90998082 0.35544658 0.48910981 0.60506247
 0.63215002 0.74694862 0.90998024 0.48907334 0.93866907 0.78653016
 0.5245541  0.92135132 0.45318037 0.57737636 0.45367502 0.54089885
 0.73490338] 0.6753808248961486
1 [0.97375014 0.79442693 0.91136929 0.39323883 0.5132765  0.61497244
 0.64912847 0.75741741 0.91153221 0.50010749 0.93878642 0.79312491
 0.54842922 0.92634191 0.48249673 0.64149922 0.47389595 0.56200003
 0.74069462] 0.6908678272917649
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6909, Best_mIoU:  0.7057
[0.97375014 0.79442693 0.91136929 0.39323883 0.5132765  0.61497244
 0.64912847 0.75741741 0.91153221 0.50010749 0.93878642 0.79312491
 0.54842922 0.92634191 0.48249673 0.64149922 0.47389595 0.56200003
 0.74069462]
Epoch: [272/484] Iter:[0/24], Time: 0.96, lr: [0.004757090891458025], Loss: 0.143807
Epoch: [272/484] Iter:[10/24], Time: 0.44, lr: [0.004748675398456647], Loss: 0.118270
Epoch: [272/484] Iter:[20/24], Time: 0.42, lr: [0.004740258248045384], Loss: 0.116649
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97396422 0.79141589 0.90862658 0.35592741 0.47691245 0.59999472
 0.63325061 0.74736936 0.90981024 0.49108937 0.93845568 0.78203001
 0.53288409 0.92260712 0.47629701 0.58839378 0.41051615 0.53171465
 0.72850933] 0.6736720353152432
1 [0.97379333 0.79448177 0.90991434 0.40149672 0.49627354 0.60832799
 0.64329435 0.75742251 0.91215222 0.5148839  0.93937515 0.78747261
 0.54232605 0.92961982 0.52802941 0.65421769 0.41831541 0.53549336
 0.73353666] 0.6884435177076283
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.252, MeanIU:  0.6884, Best_mIoU:  0.7057
[0.97379333 0.79448177 0.90991434 0.40149672 0.49627354 0.60832799
 0.64329435 0.75742251 0.91215222 0.5148839  0.93937515 0.78747261
 0.54232605 0.92961982 0.52802941 0.65421769 0.41831541 0.53549336
 0.73353666]
Epoch: [273/484] Iter:[0/24], Time: 0.98, lr: [0.00473689092300072], Loss: 0.153804
Epoch: [273/484] Iter:[10/24], Time: 0.45, lr: [0.004728471446168077], Loss: 0.125912
Epoch: [273/484] Iter:[20/24], Time: 0.42, lr: [0.0047200503032658655], Loss: 0.129646
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96465096 0.75462154 0.90821815 0.28676904 0.48545443 0.60003079
 0.63048232 0.74672508 0.90882949 0.47202654 0.93469628 0.77839006
 0.52213498 0.92400119 0.47141988 0.59669591 0.42568937 0.50398343
 0.72354407] 0.6651770268840301
1 [0.96569402 0.759579   0.9089891  0.30490219 0.50954239 0.60867079
 0.64282151 0.752839   0.9094163  0.48709439 0.93077947 0.78416203
 0.53452622 0.92997131 0.51450372 0.65224763 0.41430998 0.51625603
 0.72896443] 0.6765931317488123
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.271, MeanIU:  0.6766, Best_mIoU:  0.7057
[0.96569402 0.759579   0.9089891  0.30490219 0.50954239 0.60867079
 0.64282151 0.752839   0.9094163  0.48709439 0.93077947 0.78416203
 0.53452622 0.92997131 0.51450372 0.65224763 0.41430998 0.51625603
 0.72896443]
Epoch: [274/484] Iter:[0/24], Time: 0.97, lr: [0.004716681378792051], Loss: 0.112488
Epoch: [274/484] Iter:[10/24], Time: 0.44, lr: [0.0047082578972843575], Loss: 0.119276
Epoch: [274/484] Iter:[20/24], Time: 0.42, lr: [0.004699832740960618], Loss: 0.120392
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97110086 0.78592362 0.90867043 0.35049647 0.49343247 0.59578739
 0.63200918 0.74709412 0.91105306 0.48979236 0.93889826 0.78503495
 0.53236057 0.92378237 0.47937791 0.59568356 0.42053505 0.54620166
 0.73221109] 0.6757602827478454
1 [0.97212392 0.79076615 0.90922249 0.37032382 0.51741162 0.6069745
 0.64959157 0.75543405 0.91289527 0.50029257 0.93903575 0.78970725
 0.54578275 0.93046402 0.55204515 0.65292093 0.41909994 0.5654536
 0.74064749] 0.6905364654519103
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.6905, Best_mIoU:  0.7057
[0.97212392 0.79076615 0.90922249 0.37032382 0.51741162 0.6069745
 0.64959157 0.75543405 0.91289527 0.50029257 0.93903575 0.78970725
 0.54578275 0.93046402 0.55204515 0.65292093 0.41909994 0.5654536
 0.74064749]
Epoch: [275/484] Iter:[0/24], Time: 0.94, lr: [0.004696462208660998], Loss: 0.133015
Epoch: [275/484] Iter:[10/24], Time: 0.44, lr: [0.004688034701425269], Loss: 0.126939
Epoch: [275/484] Iter:[20/24], Time: 0.42, lr: [0.0046796055105389365], Loss: 0.117249
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97057598 0.78448243 0.90902655 0.34671143 0.4967561  0.59667507
 0.63406927 0.74687558 0.91146126 0.51726072 0.93222899 0.78660936
 0.53576702 0.92575572 0.48972027 0.60502195 0.43619302 0.54215563
 0.73118973] 0.6788703211415538
1 [0.97121151 0.78820108 0.90984986 0.36946894 0.51747618 0.60615458
 0.64791877 0.75598651 0.91396933 0.54382877 0.92698182 0.79106631
 0.54755493 0.93186854 0.582857   0.66622245 0.44304946 0.55395224
 0.73979243] 0.6951268794243721
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.253, MeanIU:  0.6951, Best_mIoU:  0.7057
[0.97121151 0.78820108 0.90984986 0.36946894 0.51747618 0.60615458
 0.64791877 0.75598651 0.91396933 0.54382877 0.92698182 0.79106631
 0.54755493 0.93186854 0.582857   0.66622245 0.44304946 0.55395224
 0.73979243]
Epoch: [276/484] Iter:[0/24], Time: 0.96, lr: [0.004676233361932307], Loss: 0.067896
Epoch: [276/484] Iter:[10/24], Time: 0.44, lr: [0.004667801807703249], Loss: 0.118211
Epoch: [276/484] Iter:[20/24], Time: 0.42, lr: [0.0046593685608996354], Loss: 0.125773
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9723385  0.79273856 0.90939097 0.36381996 0.50521457 0.60599463
 0.63570089 0.75038223 0.91246648 0.51962057 0.9364531  0.7887316
 0.53467721 0.92350171 0.48020923 0.60530473 0.45892661 0.55952698
 0.73337978] 0.6835988583444809
1 [0.97276271 0.7952941  0.90977586 0.38765824 0.52991127 0.61338676
 0.65302052 0.75843807 0.91455453 0.52938911 0.93580171 0.79467084
 0.55226829 0.93067409 0.546593   0.67019131 0.45013396 0.57186528
 0.74073966] 0.6977436474869997
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.6977, Best_mIoU:  0.7057
[0.97276271 0.7952941  0.90977586 0.38765824 0.52991127 0.61338676
 0.65302052 0.75843807 0.91455453 0.52938911 0.93580171 0.79467084
 0.55226829 0.93067409 0.546593   0.67019131 0.45013396 0.57186528
 0.74073966]
Epoch: [277/484] Iter:[0/24], Time: 0.98, lr: [0.004655994787418963], Loss: 0.078678
Epoch: [277/484] Iter:[10/24], Time: 0.44, lr: [0.004647559164715799], Loss: 0.121178
Epoch: [277/484] Iter:[20/24], Time: 0.42, lr: [0.004639121840423391], Loss: 0.131494
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97062814 0.78323657 0.90617273 0.35837662 0.49650886 0.60437776
 0.63305765 0.74748908 0.91093959 0.5370806  0.93542521 0.78346121
 0.54162991 0.92349392 0.48488446 0.60719172 0.42813188 0.54852159
 0.72864469] 0.6804869577230422
1 [0.97066544 0.78487603 0.90577884 0.39087108 0.51616518 0.60857462
 0.64160043 0.75339887 0.91210971 0.55077024 0.93334497 0.78865408
 0.55056294 0.92998228 0.56620784 0.6600674  0.32232012 0.5646645
 0.73462251] 0.6886966892331061
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.257, MeanIU:  0.6887, Best_mIoU:  0.7057
[0.97066544 0.78487603 0.90577884 0.39087108 0.51616518 0.60857462
 0.64160043 0.75339887 0.91210971 0.55077024 0.93334497 0.78865408
 0.55056294 0.92998228 0.56620784 0.6600674  0.32232012 0.5646645
 0.73462251]
Epoch: [278/484] Iter:[0/24], Time: 0.97, lr: [0.004635746433414536], Loss: 0.154235
Epoch: [278/484] Iter:[10/24], Time: 0.45, lr: [0.004627306720537765], Loss: 0.154491
Epoch: [278/484] Iter:[20/24], Time: 0.42, lr: [0.004618865296964954], Loss: 0.137944
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97395667 0.80329068 0.90957891 0.36625901 0.51489927 0.60817515
 0.63362305 0.7446917  0.91191653 0.53128406 0.93581728 0.78031226
 0.53084348 0.9193734  0.48083092 0.59227477 0.44555573 0.50064604
 0.72690477] 0.679485983118639
1 [0.97418142 0.80458489 0.91052735 0.38944332 0.53707621 0.6158012
 0.64686225 0.75247168 0.91400137 0.54577621 0.93575723 0.78748596
 0.54468914 0.92720213 0.56118892 0.65579881 0.39080336 0.51197237
 0.73460454] 0.6915909662570459
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.6916, Best_mIoU:  0.7057
[0.97418142 0.80458489 0.91052735 0.38944332 0.53707621 0.6158012
 0.64686225 0.75247168 0.91400137 0.54577621 0.93575723 0.78748596
 0.54468914 0.92720213 0.56118892 0.65579881 0.39080336 0.51197237
 0.73460454]
Epoch: [279/484] Iter:[0/24], Time: 0.98, lr: [0.004615488247685355], Loss: 0.119150
Epoch: [279/484] Iter:[10/24], Time: 0.44, lr: [0.00460704442271345], Loss: 0.112556
Epoch: [279/484] Iter:[20/24], Time: 0.42, lr: [0.004598598877845207], Loss: 0.126491
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97483715 0.80806217 0.90752588 0.41202656 0.49316448 0.59535284
 0.62367779 0.74744721 0.91202803 0.53843622 0.93526306 0.7896493
 0.54834333 0.91966896 0.47087459 0.58332539 0.42255039 0.54862974
 0.72994616] 0.6821478549204426
1 [0.97541536 0.81194746 0.90715812 0.43190754 0.50699032 0.60090188
 0.64401224 0.75642832 0.91458877 0.55953056 0.93354225 0.79449421
 0.56211599 0.92681479 0.51769447 0.64389808 0.35389237 0.56714317
 0.73664561] 0.6918485006360947
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6918, Best_mIoU:  0.7057
[0.97541536 0.81194746 0.90715812 0.43190754 0.50699032 0.60090188
 0.64401224 0.75642832 0.91458877 0.55953056 0.93354225 0.79449421
 0.56211599 0.92681479 0.51769447 0.64389808 0.35389237 0.56714317
 0.73664561]
Epoch: [280/484] Iter:[0/24], Time: 0.95, lr: [0.004595220177462539], Loss: 0.172762
Epoch: [280/484] Iter:[10/24], Time: 0.44, lr: [0.004586772218248572], Loss: 0.141634
Epoch: [280/484] Iter:[20/24], Time: 0.42, lr: [0.0045783225298430446], Loss: 0.134566
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97386515 0.8028672  0.90944177 0.43829791 0.51268728 0.60739782
 0.6293667  0.74692348 0.91327363 0.52675447 0.93414195 0.78974882
 0.54549582 0.92627685 0.50997405 0.61268667 0.44531685 0.53517514
 0.73214183] 0.6890438630132248
1 [0.97434734 0.80640397 0.91072312 0.48202241 0.53586938 0.61666809
 0.64513647 0.75763425 0.91534777 0.53711517 0.93424527 0.79508955
 0.55835617 0.93204806 0.6055254  0.6542997  0.37355996 0.54494022
 0.74023048] 0.701029619524961
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.238, MeanIU:  0.7010, Best_mIoU:  0.7057
[0.97434734 0.80640397 0.91072312 0.48202241 0.53586938 0.61666809
 0.64513647 0.75763425 0.91534777 0.53711517 0.93424527 0.79508955
 0.55835617 0.93204806 0.6055254  0.6542997  0.37355996 0.54494022
 0.74023048]
Epoch: [281/484] Iter:[0/24], Time: 1.00, lr: [0.0045749421694338535], Loss: 0.173792
Epoch: [281/484] Iter:[10/24], Time: 0.45, lr: [0.004566490053602067], Loss: 0.153316
Epoch: [281/484] Iter:[20/24], Time: 0.42, lr: [0.004558036199187115], Loss: 0.150445
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96044366 0.7442913  0.90554783 0.35905942 0.50568501 0.59462445
 0.61933804 0.74366229 0.90584057 0.50992453 0.93045174 0.78292002
 0.52388685 0.91038123 0.46050433 0.51020146 0.42766193 0.55490241
 0.72803643] 0.667229658825214
1 [0.95852037 0.73547049 0.90558273 0.39782499 0.51713544 0.60074207
 0.63524803 0.75416359 0.90628682 0.52221745 0.92874081 0.78742154
 0.53900398 0.91603294 0.52630138 0.58033725 0.37338414 0.57222248
 0.73307516] 0.6784058761954622
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.283, MeanIU:  0.6784, Best_mIoU:  0.7057
[0.95852037 0.73547049 0.90558273 0.39782499 0.51713544 0.60074207
 0.63524803 0.75416359 0.90628682 0.52221745 0.92874081 0.78742154
 0.53900398 0.91603294 0.52630138 0.58033725 0.37338414 0.57222248
 0.73307516]
Epoch: [282/484] Iter:[0/24], Time: 0.97, lr: [0.004554654169735421], Loss: 0.115677
Epoch: [282/484] Iter:[10/24], Time: 0.44, lr: [0.004546197874677709], Loss: 0.167244
Epoch: [282/484] Iter:[20/24], Time: 0.42, lr: [0.004537739831547366], Loss: 0.162713
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97255006 0.80190412 0.90949996 0.35882762 0.51189719 0.60250121
 0.62961645 0.74625491 0.90919936 0.53367623 0.93759769 0.78520938
 0.52683797 0.9156514  0.50002534 0.56965463 0.4399988  0.55195204
 0.72615294] 0.6804740681152686
1 [0.97258233 0.80283342 0.91063489 0.38623463 0.53476445 0.61070806
 0.64606067 0.75438867 0.91065075 0.5513416  0.93979061 0.79233051
 0.54454227 0.92281727 0.5732665  0.62600231 0.38424963 0.57776782
 0.7344171 ] 0.6934412353386739
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6934, Best_mIoU:  0.7057
[0.97258233 0.80283342 0.91063489 0.38623463 0.53476445 0.61070806
 0.64606067 0.75438867 0.91065075 0.5513416  0.93979061 0.79233051
 0.54454227 0.92281727 0.5732665  0.62600231 0.38424963 0.57776782
 0.7344171 ]
Epoch: [283/484] Iter:[0/24], Time: 0.96, lr: [0.004534356123943243], Loss: 0.111130
Epoch: [283/484] Iter:[10/24], Time: 0.44, lr: [0.004525895626815581], Loss: 0.118109
Epoch: [283/484] Iter:[20/24], Time: 0.42, lr: [0.004517433372026449], Loss: 0.127087
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97429457 0.80572232 0.91155702 0.40136505 0.52456575 0.61066301
 0.62967793 0.74987189 0.91240134 0.53761023 0.93840127 0.78709816
 0.52472651 0.92310256 0.50960228 0.5800912  0.45936951 0.55527003
 0.7318055 ] 0.6877471655928102
1 [0.97489102 0.80780694 0.91312083 0.46137676 0.54726002 0.61806984
 0.6444797  0.76011145 0.91477217 0.55506074 0.93970406 0.79234988
 0.54595942 0.93098397 0.60713991 0.6547923  0.43114886 0.58030214
 0.74170489] 0.7063702582529011
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.239, MeanIU:  0.7064, Best_mIoU:  0.7064
[0.97489102 0.80780694 0.91312083 0.46137676 0.54726002 0.61806984
 0.6444797  0.76011145 0.91477217 0.55506074 0.93970406 0.79234988
 0.54595942 0.93098397 0.60713991 0.6547923  0.43114886 0.58030214
 0.74170489]
Epoch: [284/484] Iter:[0/24], Time: 0.94, lr: [0.004514047977064566], Loss: 0.143021
Epoch: [284/484] Iter:[10/24], Time: 0.44, lr: [0.004505583254783354], Loss: 0.146083
Epoch: [284/484] Iter:[20/24], Time: 0.42, lr: [0.004497116765150915], Loss: 0.131718
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97398842 0.80310832 0.9104603  0.38374033 0.52443652 0.59884276
 0.62794533 0.74883681 0.90976742 0.5100873  0.93744042 0.78932845
 0.53092632 0.92138665 0.50477071 0.58597389 0.44877769 0.55703784
 0.73257724] 0.6841806687662239
1 [0.97455644 0.80572716 0.91269898 0.44610985 0.54792902 0.60668212
 0.64950149 0.7579642  0.91262114 0.53070624 0.93847333 0.79497406
 0.54934085 0.92841924 0.57238342 0.64298221 0.38728878 0.56941836
 0.73987134] 0.6982972752175032
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.6983, Best_mIoU:  0.7064
[0.97455644 0.80572716 0.91269898 0.44610985 0.54792902 0.60668212
 0.64950149 0.7579642  0.91262114 0.53070624 0.93847333 0.79497406
 0.54934085 0.92841924 0.57238342 0.64298221 0.38728878 0.56941836
 0.73987134]
Epoch: [285/484] Iter:[0/24], Time: 1.00, lr: [0.004493729673529062], Loss: 0.120774
Epoch: [285/484] Iter:[10/24], Time: 0.45, lr: [0.004485260702767398], Loss: 0.134944
Epoch: [285/484] Iter:[20/24], Time: 0.42, lr: [0.004476789954862253], Loss: 0.138809
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97454552 0.80384691 0.90930945 0.39940253 0.51523834 0.60259497
 0.63029146 0.74960212 0.90936044 0.47127685 0.93676272 0.78475463
 0.50467636 0.92531717 0.49137225 0.57376013 0.47719397 0.55533286
 0.73943317] 0.6817932544137247
1 [0.97511862 0.80743963 0.91119721 0.46401824 0.53904425 0.613331
 0.64896408 0.75928429 0.91119112 0.48722225 0.93686235 0.79211221
 0.52740116 0.93187625 0.58236393 0.65485333 0.45700445 0.57644943
 0.74637934] 0.7011638482239287
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7012, Best_mIoU:  0.7064
[0.97511862 0.80743963 0.91119721 0.46401824 0.53904425 0.613331
 0.64896408 0.75928429 0.91119112 0.48722225 0.93686235 0.79211221
 0.52740116 0.93187625 0.58236393 0.65485333 0.45700445 0.57644943
 0.74637934]
Epoch: [286/484] Iter:[0/24], Time: 0.98, lr: [0.004473401157179826], Loss: 0.150999
Epoch: [286/484] Iter:[10/24], Time: 0.45, lr: [0.004464927914363701], Loss: 0.126297
Epoch: [286/484] Iter:[20/24], Time: 0.42, lr: [0.004456452884507741], Loss: 0.140239
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97454738 0.8061237  0.91218029 0.39692557 0.52287313 0.60429107
 0.63241895 0.74538818 0.91059148 0.51019817 0.93845403 0.78646214
 0.51979165 0.92064308 0.47866428 0.57354938 0.51878918 0.55516231
 0.73138693] 0.6862337311551695
1 [0.97468064 0.80752389 0.91458754 0.45379494 0.54488603 0.61252172
 0.64960238 0.7569171  0.91313693 0.53064713 0.93919772 0.79342402
 0.54336873 0.92769552 0.5589195  0.66362505 0.5344614  0.57073449
 0.73670885] 0.7066543977457397
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7067, Best_mIoU:  0.7067
[0.97468064 0.80752389 0.91458754 0.45379494 0.54488603 0.61252172
 0.64960238 0.7569171  0.91313693 0.53064713 0.93919772 0.79342402
 0.54336873 0.92769552 0.5589195  0.66362505 0.5344614  0.57073449
 0.73670885]
Epoch: [287/484] Iter:[0/24], Time: 0.95, lr: [0.004453062371264193], Loss: 0.105557
Epoch: [287/484] Iter:[10/24], Time: 0.44, lr: [0.004444584832568607], Loss: 0.118628
Epoch: [287/484] Iter:[20/24], Time: 0.42, lr: [0.004436105496831087], Loss: 0.129579
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97491159 0.80573744 0.91020742 0.41572964 0.52029424 0.6050517
 0.63393066 0.74685702 0.91040355 0.51402442 0.93834603 0.78537352
 0.53631237 0.92120715 0.47055668 0.59815796 0.50660463 0.558797
 0.72864464] 0.6884814566594395
1 [0.97492726 0.80827529 0.91261618 0.4722554  0.5428081  0.61289873
 0.65081574 0.75804615 0.91331809 0.53461686 0.93865216 0.78944357
 0.55063997 0.927656   0.52775456 0.68667099 0.56308036 0.57750941
 0.73367853] 0.7092454393842181
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7092, Best_mIoU:  0.7092
[0.97492726 0.80827529 0.91261618 0.4722554  0.5428081  0.61289873
 0.65081574 0.75804615 0.91331809 0.53461686 0.93865216 0.78944357
 0.55063997 0.927656   0.52775456 0.68667099 0.56308036 0.57750941
 0.73367853]
Epoch: [288/484] Iter:[0/24], Time: 0.96, lr: [0.004432713258424357], Loss: 0.146225
Epoch: [288/484] Iter:[10/24], Time: 0.44, lr: [0.004424231399769357], Loss: 0.141137
Epoch: [288/484] Iter:[20/24], Time: 0.42, lr: [0.004415747733962901], Loss: 0.135830
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9741872  0.80302843 0.90826026 0.3284709  0.51055849 0.60836318
 0.63322408 0.75139945 0.91036817 0.51454463 0.93861485 0.78704058
 0.54112531 0.92356242 0.49330666 0.59935176 0.48059012 0.55553028
 0.73778927] 0.684174527875973
1 [0.97484188 0.80614205 0.90925146 0.36267688 0.53322882 0.6145703
 0.63840714 0.75928157 0.91296278 0.53778419 0.939877   0.79149621
 0.55538284 0.9294308  0.56342093 0.67667261 0.4721845  0.57490185
 0.74469435] 0.6998530605516121
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.6999, Best_mIoU:  0.7092
[0.97484188 0.80614205 0.90925146 0.36267688 0.53322882 0.6145703
 0.63840714 0.75928157 0.91296278 0.53778419 0.939877   0.79149621
 0.55538284 0.9294308  0.56342093 0.67667261 0.4721845  0.57490185
 0.74469435]
Epoch: [289/484] Iter:[0/24], Time: 0.98, lr: [0.004412353760687803], Loss: 0.169837
Epoch: [289/484] Iter:[10/24], Time: 0.44, lr: [0.0044038675577344295], Loss: 0.137189
Epoch: [289/484] Iter:[20/24], Time: 0.42, lr: [0.004395379537410947], Loss: 0.125585
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97472624 0.80476543 0.90903636 0.35908397 0.51853357 0.60254095
 0.63493394 0.74748173 0.91135096 0.51784395 0.93939351 0.78256104
 0.50439713 0.92453884 0.49287163 0.60269282 0.4751551  0.5589453
 0.73694993] 0.684094863491463
1 [0.97523181 0.80662003 0.90965107 0.39853511 0.54285366 0.60637747
 0.64929415 0.75653558 0.91361732 0.53192004 0.94035733 0.78941672
 0.53030596 0.93081165 0.55907806 0.68275846 0.48791241 0.58363297
 0.74462261] 0.7020806534815548
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7021, Best_mIoU:  0.7092
[0.97523181 0.80662003 0.90965107 0.39853511 0.54285366 0.60637747
 0.64929415 0.75653558 0.91361732 0.53192004 0.94035733 0.78941672
 0.53030596 0.93081165 0.55907806 0.68275846 0.48791241 0.58363297
 0.74462261]
Epoch: [290/484] Iter:[0/24], Time: 0.96, lr: [0.004391983819457524], Loss: 0.127198
Epoch: [290/484] Iter:[10/24], Time: 0.44, lr: [0.004383493247603684], Loss: 0.145514
Epoch: [290/484] Iter:[20/24], Time: 0.42, lr: [0.004375000848050198], Loss: 0.138684
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97035643 0.77372953 0.91146985 0.35271175 0.53592514 0.58723213
 0.63355497 0.73583446 0.90821206 0.48643453 0.93776606 0.78119641
 0.52543319 0.92186681 0.48057063 0.58543695 0.49292114 0.54332955
 0.73742588] 0.6790214463046553
1 [0.96988201 0.77501672 0.91368468 0.39780634 0.55854413 0.58849965
 0.6391286  0.74628741 0.90938174 0.50034777 0.93882426 0.78749729
 0.53592342 0.92656216 0.53707306 0.66425223 0.55456246 0.55275633
 0.7394251 ] 0.6966029140956692
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.258, MeanIU:  0.6966, Best_mIoU:  0.7092
[0.96988201 0.77501672 0.91368468 0.39780634 0.55854413 0.58849965
 0.6391286  0.74628741 0.90938174 0.50034777 0.93882426 0.78749729
 0.53592342 0.92656216 0.53707306 0.66425223 0.55456246 0.55275633
 0.7394251 ]
Epoch: [291/484] Iter:[0/24], Time: 0.97, lr: [0.004371603375502047], Loss: 0.144465
Epoch: [291/484] Iter:[10/24], Time: 0.44, lr: [0.004363108409878287], Loss: 0.141329
Epoch: [291/484] Iter:[20/24], Time: 0.42, lr: [0.004354611606112678], Loss: 0.153144
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97448052 0.80161023 0.9070504  0.30551705 0.4569211  0.58793207
 0.63073504 0.73361336 0.83700175 0.2742095  0.93895541 0.78320699
 0.51191643 0.92149063 0.48697734 0.58052631 0.50020338 0.5528307
 0.72447515] 0.6584028088188452
1 [0.97540581 0.80830153 0.90928991 0.356374   0.46846895 0.59408959
 0.6419578  0.7461522  0.83122595 0.26662652 0.94081956 0.7900994
 0.538384   0.92734888 0.55435102 0.66236285 0.55413508 0.56611935
 0.72736651] 0.6767831006344109
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.314, MeanIU:  0.6768, Best_mIoU:  0.7092
[0.97540581 0.80830153 0.90928991 0.356374   0.46846895 0.59408959
 0.6419578  0.7461522  0.83122595 0.26662652 0.94081956 0.7900994
 0.538384   0.92734888 0.55435102 0.66236285 0.55413508 0.56611935
 0.72736651]
Epoch: [292/484] Iter:[0/24], Time: 0.95, lr: [0.004351212368945237], Loss: 0.101791
Epoch: [292/484] Iter:[10/24], Time: 0.44, lr: [0.004342712984410433], Loss: 0.168677
Epoch: [292/484] Iter:[20/24], Time: 0.42, lr: [0.004334211751177087], Loss: 0.150856
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97304042 0.79504355 0.90267355 0.28365771 0.4710482  0.59626332
 0.63331168 0.74310226 0.90640205 0.43427837 0.9336851  0.78451413
 0.51542187 0.91857494 0.47433788 0.54366865 0.49616972 0.51398247
 0.73300302] 0.6659041525190897
1 [0.97430037 0.79728352 0.90125691 0.31456099 0.48090455 0.60808489
 0.64436068 0.75385462 0.90544271 0.41717601 0.93474923 0.79116181
 0.53367321 0.92590942 0.53308344 0.63061351 0.50941979 0.52224727
 0.73780269] 0.6797834536111336
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.272, MeanIU:  0.6798, Best_mIoU:  0.7092
[0.97430037 0.79728352 0.90125691 0.31456099 0.48090455 0.60808489
 0.64436068 0.75385462 0.90544271 0.41717601 0.93474923 0.79116181
 0.53367321 0.92590942 0.53308344 0.63061351 0.50941979 0.52224727
 0.73780269]
Epoch: [293/484] Iter:[0/24], Time: 0.98, lr: [0.00433081073925588], Loss: 0.087056
Epoch: [293/484] Iter:[10/24], Time: 0.44, lr: [0.004322306910392835], Loss: 0.119534
Epoch: [293/484] Iter:[20/24], Time: 0.42, lr: [0.004313801222158205], Loss: 0.144695
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97523837 0.80293212 0.90615068 0.36833842 0.48968073 0.60317991
 0.62903093 0.74304604 0.90878314 0.47864331 0.93689667 0.78740317
 0.54257857 0.92290873 0.5285839  0.59270223 0.45110074 0.51936719
 0.72785228] 0.6797061655809253
1 [0.97558001 0.80642913 0.90887446 0.41278906 0.51159851 0.61272984
 0.64918392 0.75438836 0.9097845  0.48839536 0.93687419 0.79361027
 0.56076097 0.92931503 0.60891516 0.6705695  0.3855837  0.53310625
 0.73215023] 0.6937178128479491
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6937, Best_mIoU:  0.7092
[0.97558001 0.80642913 0.90887446 0.41278906 0.51159851 0.61272984
 0.64918392 0.75438836 0.9097845  0.48839536 0.93687419 0.79361027
 0.56076097 0.92931503 0.60891516 0.6705695  0.3855837  0.53310625
 0.73215023]
Epoch: [294/484] Iter:[0/24], Time: 0.98, lr: [0.004310398425237057], Loss: 0.132365
Epoch: [294/484] Iter:[10/24], Time: 0.45, lr: [0.004301890126348002], Loss: 0.142953
Epoch: [294/484] Iter:[20/24], Time: 0.42, lr: [0.004293379957296061], Loss: 0.138356
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97337026 0.79043637 0.90614251 0.35540384 0.48782513 0.59600766
 0.63188325 0.73885591 0.90837142 0.48469536 0.93804666 0.78570509
 0.54667099 0.92377479 0.51343662 0.58038487 0.44437907 0.5118762
 0.72619791] 0.6759717849464955
1 [0.97376091 0.79185794 0.90878098 0.40434298 0.50885328 0.59770088
 0.6456978  0.74875513 0.90909367 0.48352739 0.94115728 0.79211528
 0.55981142 0.93085567 0.59589425 0.6514553  0.42628025 0.52423397
 0.73176367] 0.6908388440989995
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.258, MeanIU:  0.6908, Best_mIoU:  0.7092
[0.97376091 0.79185794 0.90878098 0.40434298 0.50885328 0.59770088
 0.6456978  0.74875513 0.90909367 0.48352739 0.94115728 0.79211528
 0.55981142 0.93085567 0.59589425 0.6514553  0.42628025 0.52423397
 0.73176367]
Epoch: [295/484] Iter:[0/24], Time: 1.00, lr: [0.004289975365015272], Loss: 0.105904
Epoch: [295/484] Iter:[10/24], Time: 0.45, lr: [0.00428146257011727], Loss: 0.138297
Epoch: [295/484] Iter:[20/24], Time: 0.42, lr: [0.004272947894144881], Loss: 0.137898
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97362323 0.79959674 0.90497415 0.34621146 0.45719505 0.60502457
 0.63004467 0.74348476 0.90721832 0.50689243 0.93911306 0.78247229
 0.52617534 0.92234822 0.50395317 0.5914027  0.41587687 0.51965689
 0.73290646] 0.674114230312625
1 [0.973839   0.80131263 0.90448021 0.37241044 0.45427242 0.61389392
 0.64818956 0.7526892  0.90846971 0.51216122 0.93959215 0.78959409
 0.54536622 0.93105478 0.58254579 0.66904699 0.35692663 0.55991184
 0.73867197] 0.687075198843576
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.262, MeanIU:  0.6871, Best_mIoU:  0.7092
[0.973839   0.80131263 0.90448021 0.37241044 0.45427242 0.61389392
 0.64818956 0.7526892  0.90846971 0.51216122 0.93959215 0.78959409
 0.54536622 0.93105478 0.58254579 0.66904699 0.35692663 0.55991184
 0.73867197]
Epoch: [296/484] Iter:[0/24], Time: 0.97, lr: [0.004269541496029355], Loss: 0.127422
Epoch: [296/484] Iter:[10/24], Time: 0.44, lr: [0.004261024178849604], Loss: 0.154088
Epoch: [296/484] Iter:[20/24], Time: 0.42, lr: [0.004252504969561779], Loss: 0.142291
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97447874 0.80700037 0.91133881 0.41786058 0.51599911 0.59894874
 0.62433284 0.74171945 0.90956171 0.52902499 0.94181868 0.78993487
 0.53312217 0.92102217 0.49414579 0.59732439 0.47293258 0.53152762
 0.73566876] 0.6867243352359305
1 [0.97484173 0.80836301 0.91332302 0.47182425 0.54255043 0.60748368
 0.64157549 0.74975292 0.9127401  0.5430078  0.94417185 0.79535005
 0.54893382 0.92822102 0.55506019 0.68753553 0.47262693 0.56073264
 0.74373456] 0.7053594213907194
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.241, MeanIU:  0.7054, Best_mIoU:  0.7092
[0.97484173 0.80836301 0.91332302 0.47182425 0.54255043 0.60748368
 0.64157549 0.74975292 0.9127401  0.5430078  0.94417185 0.79535005
 0.54893382 0.92822102 0.55506019 0.68753553 0.47262693 0.56073264
 0.74373456]
Epoch: [297/484] Iter:[0/24], Time: 0.96, lr: [0.004249096755019119], Loss: 0.116335
Epoch: [297/484] Iter:[10/24], Time: 0.45, lr: [0.0042405748889901545], Loss: 0.123017
Epoch: [297/484] Iter:[20/24], Time: 0.42, lr: [0.004232051119695209], Loss: 0.129045
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97340002 0.79394175 0.91065509 0.3808647  0.50880525 0.60774381
 0.62968421 0.74809011 0.90943737 0.49451611 0.9416312  0.78852898
 0.53736061 0.92451432 0.51577795 0.61490511 0.44564776 0.5392516
 0.7372449 ] 0.6843158325640496
1 [0.97331323 0.79355854 0.91203146 0.41498429 0.52738057 0.61585971
 0.64096911 0.75615198 0.91136752 0.50267777 0.94280491 0.79546111
 0.5538632  0.93153724 0.60939507 0.69096319 0.39866333 0.56031674
 0.74382344] 0.6986906536228276
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.6987, Best_mIoU:  0.7092
[0.97331323 0.79355854 0.91203146 0.41498429 0.52738057 0.61585971
 0.64096911 0.75615198 0.91136752 0.50267777 0.94280491 0.79546111
 0.5538632  0.93153724 0.60939507 0.69096319 0.39866333 0.56031674
 0.74382344]
Epoch: [298/484] Iter:[0/24], Time: 0.98, lr: [0.00422864107801377], Loss: 0.167909
Epoch: [298/484] Iter:[10/24], Time: 0.44, lr: [0.004220114636268557], Loss: 0.128145
Epoch: [298/484] Iter:[20/24], Time: 0.42, lr: [0.004211586279973163], Loss: 0.141973
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97458078 0.80288507 0.90854582 0.35508429 0.49771951 0.60332623
 0.62769606 0.74823207 0.90351285 0.51994767 0.94007876 0.77932894
 0.51698078 0.91797779 0.48029008 0.58768712 0.50021037 0.52463712
 0.73728182] 0.6803159544869323
1 [0.97447089 0.80293634 0.90982443 0.38161762 0.51215976 0.61184895
 0.64365329 0.75467006 0.90375276 0.53458398 0.94193592 0.78502195
 0.53539479 0.92447865 0.53035529 0.68566266 0.53503726 0.54885184
 0.74098435] 0.6977495156132519
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.251, MeanIU:  0.6977, Best_mIoU:  0.7092
[0.97447089 0.80293634 0.90982443 0.38161762 0.51215976 0.61184895
 0.64365329 0.75467006 0.90375276 0.53458398 0.94193592 0.78502195
 0.53539479 0.92447865 0.53035529 0.68566266 0.53503726 0.54885184
 0.74098435]
Epoch: [299/484] Iter:[0/24], Time: 0.95, lr: [0.004208174400320054], Loss: 0.099470
Epoch: [299/484] Iter:[10/24], Time: 0.44, lr: [0.004199643355686979], Loss: 0.121783
Epoch: [299/484] Iter:[20/24], Time: 0.42, lr: [0.004191110385091107], Loss: 0.145184
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97398013 0.79666398 0.90889156 0.31749744 0.51504421 0.6013759
 0.62518588 0.74690162 0.90943902 0.50770705 0.93864534 0.78396405
 0.51358289 0.91783957 0.4752084  0.55991382 0.51389177 0.5317209
 0.73629743] 0.6775658397194263
1 [0.97407242 0.80021515 0.90888471 0.31301765 0.53196062 0.61005843
 0.64409193 0.75630739 0.91075787 0.52820364 0.9379206  0.78896902
 0.52559308 0.92419721 0.52589058 0.67260319 0.57809354 0.55926081
 0.74170777] 0.6964108205358479
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.6964, Best_mIoU:  0.7092
[0.97407242 0.80021515 0.90888471 0.31301765 0.53196062 0.61005843
 0.64409193 0.75630739 0.91075787 0.52820364 0.9379206  0.78896902
 0.52559308 0.92419721 0.52589058 0.67260319 0.57809354 0.55926081
 0.74170777]
Epoch: [300/484] Iter:[0/24], Time: 0.94, lr: [0.004187696656510159], Loss: 0.121509
Epoch: [300/484] Iter:[10/24], Time: 0.44, lr: [0.004179160981507899], Loss: 0.174701
Epoch: [300/484] Iter:[20/24], Time: 0.42, lr: [0.0041706233689996425], Loss: 0.195910
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96757027 0.76355063 0.90589186 0.29909407 0.50228698 0.60061675
 0.61965374 0.74275059 0.90339142 0.42508304 0.937664   0.77232092
 0.50956142 0.91660659 0.50203747 0.54259282 0.41228875 0.49446258
 0.72914452] 0.6603457050790403
1 [0.96697767 0.75457107 0.90566798 0.29293636 0.5179636  0.6081397
 0.63429195 0.75031529 0.90606334 0.36460063 0.93995036 0.77758586
 0.52263489 0.92421498 0.57443044 0.6272049  0.45013411 0.51738947
 0.73562946] 0.6721422148023457
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.280, MeanIU:  0.6721, Best_mIoU:  0.7092
[0.96697767 0.75457107 0.90566798 0.29293636 0.5179636  0.6081397
 0.63429195 0.75031529 0.90606334 0.36460063 0.93995036 0.77758586
 0.52263489 0.92421498 0.57443044 0.6272049  0.45013411 0.51738947
 0.73562946]
Epoch: [301/484] Iter:[0/24], Time: 0.98, lr: [0.004167207780409325], Loss: 0.203346
Epoch: [301/484] Iter:[10/24], Time: 0.44, lr: [0.0041586674472416105], Loss: 0.147945
Epoch: [301/484] Iter:[20/24], Time: 0.42, lr: [0.004150125164891899], Loss: 0.137978
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97066475 0.7847473  0.90932314 0.3030465  0.50546775 0.60529328
 0.6389258  0.74240688 0.90994526 0.53045576 0.93791469 0.78545729
 0.50237884 0.92149804 0.50891312 0.54827207 0.47653597 0.55871713
 0.73051561] 0.6773936419261105
1 [0.96994522 0.78456991 0.91050098 0.32916953 0.52480862 0.61464347
 0.65274139 0.75454757 0.91196599 0.54618142 0.93857218 0.79302854
 0.52755057 0.92817485 0.5990343  0.63961943 0.51607961 0.58094309
 0.73244588] 0.6976064504447322
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.257, MeanIU:  0.6976, Best_mIoU:  0.7092
[0.96994522 0.78456991 0.91050098 0.32916953 0.52480862 0.61464347
 0.65274139 0.75454757 0.91196599 0.54618142 0.93857218 0.79302854
 0.52755057 0.92817485 0.5990343  0.63961943 0.51607961 0.58094309
 0.73244588]
Epoch: [302/484] Iter:[0/24], Time: 0.98, lr: [0.004146707705083188], Loss: 0.105101
Epoch: [302/484] Iter:[10/24], Time: 0.45, lr: [0.004138162685633448], Loss: 0.134803
Epoch: [302/484] Iter:[20/24], Time: 0.42, lr: [0.0041296157051906425], Loss: 0.127517
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97024927 0.78411329 0.91010723 0.29603405 0.51770347 0.60989778
 0.63749494 0.7476131  0.91064319 0.54456275 0.940185   0.78731688
 0.52899715 0.91969374 0.50694144 0.51968333 0.47384872 0.56293083
 0.73854364] 0.6792926210156108
1 [0.96962076 0.78331427 0.91179441 0.28961624 0.53023013 0.61751285
 0.65342336 0.75780793 0.91149881 0.54625492 0.94301125 0.79537722
 0.54780761 0.92512991 0.57529766 0.59405887 0.47822699 0.58935386
 0.74393021] 0.6928035393262061
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.259, MeanIU:  0.6928, Best_mIoU:  0.7092
[0.96962076 0.78331427 0.91179441 0.28961624 0.53023013 0.61751285
 0.65342336 0.75780793 0.91149881 0.54625492 0.94301125 0.79537722
 0.54780761 0.92512991 0.57529766 0.59405887 0.47822699 0.58935386
 0.74393021]
Epoch: [303/484] Iter:[0/24], Time: 0.96, lr: [0.004126196362824848], Loss: 0.183488
Epoch: [303/484] Iter:[10/24], Time: 0.44, lr: [0.004117646628650728], Loss: 0.138568
Epoch: [303/484] Iter:[20/24], Time: 0.42, lr: [0.004109094921535083], Loss: 0.142820
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96713341 0.76874089 0.90337526 0.33098173 0.50079432 0.60299467
 0.63306443 0.74733132 0.9100421  0.53516435 0.89563316 0.78676485
 0.52775471 0.91766068 0.50339474 0.53279138 0.44291191 0.52965846
 0.72642356] 0.6717166268624044
1 [0.96610668 0.76990245 0.90377988 0.34990865 0.52539424 0.61088413
 0.65165107 0.75745522 0.91180179 0.55244453 0.88626542 0.79221388
 0.54559255 0.92362581 0.57250696 0.6016605  0.42172669 0.56069914
 0.72844866] 0.685898329244489
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.274, MeanIU:  0.6859, Best_mIoU:  0.7092
[0.96610668 0.76990245 0.90377988 0.34990865 0.52539424 0.61088413
 0.65165107 0.75745522 0.91180179 0.55244453 0.88626542 0.79221388
 0.54559255 0.92362581 0.57250696 0.6016605  0.42172669 0.56069914
 0.72844866]
Epoch: [304/484] Iter:[0/24], Time: 0.96, lr: [0.004105673685141619], Loss: 0.193205
Epoch: [304/484] Iter:[10/24], Time: 0.44, lr: [0.004097119207469382], Loss: 0.132927
Epoch: [304/484] Iter:[20/24], Time: 0.42, lr: [0.0040885627447673975], Loss: 0.134402
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97212623 0.78526771 0.91160987 0.37799572 0.50786004 0.61319337
 0.6348393  0.7463947  0.91236016 0.52954993 0.93483907 0.79040269
 0.54678846 0.92164974 0.50759155 0.55508779 0.45156035 0.56529593
 0.73477484] 0.6841677601157088
1 [0.97181018 0.78669108 0.9136465  0.40355533 0.52649896 0.61933047
 0.65032802 0.75613898 0.91350564 0.53762546 0.93617602 0.79393565
 0.55673158 0.92825483 0.57744829 0.63758808 0.47301521 0.58475812
 0.73970034] 0.7003546707160863
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.7004, Best_mIoU:  0.7092
[0.97181018 0.78669108 0.9136465  0.40355533 0.52649896 0.61933047
 0.65032802 0.75613898 0.91350564 0.53762546 0.93617602 0.79393565
 0.55673158 0.92825483 0.57744829 0.63758808 0.47301521 0.58475812
 0.73970034]
Epoch: [305/484] Iter:[0/24], Time: 0.98, lr: [0.004085139602741506], Loss: 0.074959
Epoch: [305/484] Iter:[10/24], Time: 0.44, lr: [0.0040765803524602925], Loss: 0.138451
Epoch: [305/484] Iter:[20/24], Time: 0.42, lr: [0.004068019104918925], Loss: 0.148209
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97163003 0.78569209 0.90760084 0.38076317 0.50779426 0.60660786
 0.6301167  0.74483473 0.91018147 0.53846513 0.91220602 0.79053619
 0.54472761 0.92275068 0.51139026 0.56555413 0.42086709 0.55166134
 0.73193169] 0.6808058577540169
1 [0.97065394 0.78527518 0.90849871 0.3871655  0.52739157 0.61333795
 0.64673794 0.75489102 0.91288512 0.56285817 0.90705879 0.79623071
 0.55770088 0.92752922 0.56208438 0.63489844 0.42511286 0.56772022
 0.7373422 ] 0.6939669903086026
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.6940, Best_mIoU:  0.7092
[0.97065394 0.78527518 0.90849871 0.3871655  0.52739157 0.61333795
 0.64673794 0.75489102 0.91288512 0.56285817 0.90705879 0.79623071
 0.55770088 0.92752922 0.56208438 0.63489844 0.42511286 0.56772022
 0.7373422 ]
Epoch: [306/484] Iter:[0/24], Time: 0.99, lr: [0.004064594045519346], Loss: 0.110870
Epoch: [306/484] Iter:[10/24], Time: 0.45, lr: [0.004056029993175323], Loss: 0.127929
Epoch: [306/484] Iter:[20/24], Time: 0.42, lr: [0.004047463931196065], Loss: 0.133794
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9750182  0.80354549 0.91031938 0.35411818 0.50434128 0.60247288
 0.6339071  0.74718843 0.91175214 0.53614153 0.93383841 0.78124949
 0.53644531 0.91553012 0.51111618 0.53899071 0.42829214 0.50743862
 0.72731311] 0.6767904582526297
1 [0.97500544 0.80423987 0.91272579 0.3760898  0.52533802 0.60990179
 0.64620858 0.75486625 0.91396192 0.55855424 0.93749888 0.78517893
 0.54906844 0.92101852 0.56645026 0.59806671 0.47126328 0.50861979
 0.73169486] 0.6918816501013291
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.6919, Best_mIoU:  0.7092
[0.97500544 0.80423987 0.91272579 0.3760898  0.52533802 0.60990179
 0.64620858 0.75486625 0.91396192 0.55855424 0.93749888 0.78517893
 0.54906844 0.92101852 0.56645026 0.59806671 0.47126328 0.50861979
 0.73169486]
Epoch: [307/484] Iter:[0/24], Time: 0.96, lr: [0.004044036942542651], Loss: 0.130158
Epoch: [307/484] Iter:[10/24], Time: 0.44, lr: [0.004035468058333003], Loss: 0.132913
Epoch: [307/484] Iter:[20/24], Time: 0.42, lr: [0.004026897151965829], Loss: 0.128662
00

0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97472908 0.79524851 0.90948599 0.36610769 0.50166159 0.60893935
 0.62961928 0.7493067  0.91007367 0.48012886 0.93903674 0.78484768
 0.53558092 0.91675889 0.4962615  0.52849844 0.39042803 0.55795886
 0.72917029] 0.6738864248488605
1 [0.97494291 0.79764847 0.91104814 0.37842875 0.52309535 0.61645618
 0.64065762 0.75683609 0.91001628 0.47379728 0.93896209 0.7916404
 0.55365349 0.92140421 0.54483131 0.5565745  0.35934757 0.58390727
 0.73351372] 0.6824611374071026
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.258, MeanIU:  0.6825, Best_mIoU:  0.7092
[0.97494291 0.79764847 0.91104814 0.37842875 0.52309535 0.61645618
 0.64065762 0.75683609 0.91001628 0.47379728 0.93896209 0.7916404
 0.55365349 0.92140421 0.54483131 0.5565745  0.35934757 0.58390727
 0.73351372]
Epoch: [308/484] Iter:[0/24], Time: 0.99, lr: [0.0040234682220371114], Loss: 0.135006
Epoch: [308/484] Iter:[10/24], Time: 0.45, lr: [0.004014894475803906], Loss: 0.147388
Epoch: [308/484] Iter:[20/24], Time: 0.42, lr: [0.004006318694741069], Loss: 0.141804
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97535286 0.80366297 0.91091464 0.41236577 0.50933097 0.6025597
 0.63228207 0.74455911 0.91094478 0.52152612 0.9385834  0.7858924
 0.53945383 0.91609744 0.49399986 0.54359115 0.39562731 0.53910714
 0.73590677] 0.6795662250929811
1 [0.97531241 0.80654441 0.91290504 0.43060846 0.53164449 0.61551183
 0.64740832 0.75573748 0.91173201 0.53477919 0.93830554 0.79193986
 0.55099473 0.92051831 0.54266057 0.57510153 0.31157604 0.56324248
 0.7385096 ] 0.6871069636876864
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6871, Best_mIoU:  0.7092
[0.97531241 0.80654441 0.91290504 0.43060846 0.53164449 0.61551183
 0.64740832 0.75573748 0.91173201 0.53477919 0.93830554 0.79193986
 0.55099473 0.92051831 0.54266057 0.57510153 0.31157604 0.56324248
 0.7385096 ]
Epoch: [309/484] Iter:[0/24], Time: 0.98, lr: [0.004002887811371758], Loss: 0.177425
Epoch: [309/484] Iter:[10/24], Time: 0.45, lr: [0.003994309172595663], Loss: 0.131061
Epoch: [309/484] Iter:[20/24], Time: 0.42, lr: [0.003985728486165364], Loss: 0.127291
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97524598 0.80405825 0.90999188 0.38847737 0.51372414 0.60527291
 0.63568792 0.74470218 0.90980939 0.52755272 0.93713403 0.78967956
 0.54682791 0.91731282 0.50349316 0.54935812 0.43639632 0.54638661
 0.72191348] 0.6822644609297019
1 [0.9752438  0.80512228 0.9104077  0.37625762 0.52970747 0.61644532
 0.64856177 0.75653982 0.91091081 0.55116973 0.93554756 0.79314833
 0.56235239 0.92055806 0.54715288 0.57517981 0.3709259  0.55867123
 0.72355218] 0.6877607711464162
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.6878, Best_mIoU:  0.7092
[0.9752438  0.80512228 0.9104077  0.37625762 0.52970747 0.61644532
 0.64856177 0.75653982 0.91091081 0.55116973 0.93554756 0.79314833
 0.56235239 0.92055806 0.54715288 0.57517981 0.3709259  0.55867123
 0.72355218]
Epoch: [310/484] Iter:[0/24], Time: 0.96, lr: [0.003982295637043792], Loss: 0.100302
Epoch: [310/484] Iter:[10/24], Time: 0.44, lr: [0.003973712074837647], Loss: 0.132769
Epoch: [310/484] Iter:[20/24], Time: 0.42, lr: [0.0039651264519975265], Loss: 0.131507
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.973896   0.79579252 0.9114511  0.3902461  0.50832522 0.60929646
 0.63840487 0.74890569 0.90994052 0.54404301 0.93752438 0.77799291
 0.48108337 0.92074501 0.50509002 0.58630162 0.41443823 0.54587252
 0.7362246 ] 0.6808196928989353
1 [0.97413514 0.80019761 0.9126227  0.4148177  0.52650884 0.61872172
 0.65302579 0.75688267 0.91272215 0.57089628 0.93605721 0.78558246
 0.512739   0.92663217 0.55694534 0.6407424  0.34332114 0.56654767
 0.74238509] 0.6921833195561385
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6922, Best_mIoU:  0.7092
[0.97413514 0.80019761 0.9126227  0.4148177  0.52650884 0.61872172
 0.65302579 0.75688267 0.91272215 0.57089628 0.93605721 0.78558246
 0.512739   0.92663217 0.55694534 0.6407424  0.34332114 0.56654767
 0.74238509]
Epoch: [311/484] Iter:[0/24], Time: 0.95, lr: [0.003961691624663038], Loss: 0.195186
Epoch: [311/484] Iter:[10/24], Time: 0.45, lr: [0.003953103107765271], Loss: 0.160380
Epoch: [311/484] Iter:[20/24], Time: 0.42, lr: [0.003944512517095775], Loss: 0.144859
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97479418 0.79649693 0.9119616  0.36747579 0.52724426 0.61407101
 0.63680397 0.74706923 0.91217067 0.53325031 0.94036332 0.78375703
 0.52847826 0.92099203 0.51350448 0.58191456 0.41682734 0.53299532
 0.72583234] 0.6824211907647108
1 [0.97461275 0.79707036 0.91219162 0.34129456 0.54159774 0.62242298
 0.65082865 0.75445417 0.91388943 0.55381848 0.94113866 0.79115428
 0.54859453 0.92826515 0.56749903 0.65314494 0.3405819  0.55384139
 0.73265914] 0.6904768299692309
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.251, MeanIU:  0.6905, Best_mIoU:  0.7092
[0.97461275 0.79707036 0.91219162 0.34129456 0.54159774 0.62242298
 0.65082865 0.75445417 0.91388943 0.55381848 0.94113866 0.79115428
 0.54859453 0.92826515 0.56749903 0.65314494 0.3405819  0.55384139
 0.73265914]
Epoch: [312/484] Iter:[0/24], Time: 0.97, lr: [0.003941075698936048], Loss: 0.072938
Epoch: [312/484] Iter:[10/24], Time: 0.44, lr: [0.003932482195703935], Loss: 0.125730
Epoch: [312/484] Iter:[20/24], Time: 0.42, lr: [0.0039238866054014995], Loss: 0.123027
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97539159 0.80385051 0.91138402 0.34953784 0.52444443 0.61035579
 0.63642923 0.74492777 0.91140975 0.54283342 0.9375115  0.78676633
 0.50948126 0.91982802 0.48963288 0.55425139 0.46545103 0.54734566
 0.73357453] 0.6818108920878749
1 [0.97559339 0.80619901 0.91183192 0.3449684  0.54244419 0.61786353
 0.65475879 0.75634881 0.91309118 0.57286801 0.93515289 0.79325732
 0.5382527  0.92338138 0.52463505 0.60043333 0.51392886 0.56542093
 0.74068788] 0.696374608997597
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.6964, Best_mIoU:  0.7092
[0.97559339 0.80619901 0.91183192 0.3449684  0.54244419 0.61786353
 0.65475879 0.75634881 0.91309118 0.57286801 0.93515289 0.79325732
 0.5382527  0.92338138 0.52463505 0.60043333 0.51392886 0.56542093
 0.74068788]
Epoch: [313/484] Iter:[0/24], Time: 0.97, lr: [0.003920447783649804], Loss: 0.059211
Epoch: [313/484] Iter:[10/24], Time: 0.44, lr: [0.003911849262052576], Loss: 0.108349
Epoch: [313/484] Iter:[20/24], Time: 0.42, lr: [0.003903248639922666], Loss: 0.137741
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97196552 0.79509949 0.90219258 0.39246566 0.50998875 0.56749274
 0.59816508 0.72561733 0.90793468 0.51660527 0.93307852 0.78455148
 0.53064395 0.91659529 0.47036364 0.54833769 0.39787121 0.49361535
 0.72518831] 0.6677775016117591
1 [0.97192905 0.79872736 0.90412357 0.42994433 0.52357564 0.58084512
 0.62308211 0.7406459  0.90977834 0.54986913 0.933532   0.79042166
 0.55118285 0.9210411  0.51587176 0.60038716 0.37052192 0.51177307
 0.73169121] 0.6820496449115271
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.267, MeanIU:  0.6820, Best_mIoU:  0.7092
[0.97192905 0.79872736 0.90412357 0.42994433 0.52357564 0.58084512
 0.62308211 0.7406459  0.90977834 0.54986913 0.933532   0.79042166
 0.55118285 0.9210411  0.51587176 0.60038716 0.37052192 0.51177307
 0.73169121]
Epoch: [314/484] Iter:[0/24], Time: 0.97, lr: [0.0038998078016550565], Loss: 0.227042
Epoch: [314/484] Iter:[10/24], Time: 0.44, lr: [0.003891204229266822], Loss: 0.169697
Epoch: [314/484] Iter:[20/24], Time: 0.42, lr: [0.0038825985427167913], Loss: 0.168956
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97222528 0.8029545  0.90824599 0.31052435 0.49926241 0.5949675
 0.6124158  0.73405578 0.9053351  0.45437875 0.92993158 0.77952645
 0.49856842 0.9238735  0.48238812 0.60867012 0.44938945 0.52968393
 0.72889502] 0.6697522126738992
1 [0.97112502 0.80450908 0.90963446 0.32633419 0.51618394 0.60116761
 0.63144609 0.74082459 0.90364244 0.47735778 0.93041964 0.78613253
 0.52149929 0.92848434 0.53860824 0.6906282  0.471122   0.55626032
 0.73562518] 0.686368681069586
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.265, MeanIU:  0.6864, Best_mIoU:  0.7092
[0.97112502 0.80450908 0.90963446 0.32633419 0.51618394 0.60116761
 0.63144609 0.74082459 0.90364244 0.47735778 0.93041964 0.78613253
 0.52149929 0.92848434 0.53860824 0.6906282  0.471122   0.55626032
 0.73562518]
Epoch: [315/484] Iter:[0/24], Time: 0.96, lr: [0.0038791556748492334], Loss: 0.107008
Epoch: [315/484] Iter:[10/24], Time: 0.44, lr: [0.0038705470188417535], Loss: 0.130518
Epoch: [315/484] Iter:[20/24], Time: 0.42, lr: [0.003861936234873534], Loss: 0.135228
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97171303 0.80321091 0.9099264  0.29016825 0.51763414 0.61500313
 0.63544189 0.7460191  0.91127235 0.54063074 0.93770254 0.78681545
 0.54200714 0.92205823 0.4847514  0.57145695 0.46226388 0.54418663
 0.73342322] 0.6802992312355726
1 [0.97294887 0.8056633  0.91092317 0.28512648 0.53255042 0.62073429
 0.65000973 0.75578835 0.91157328 0.57086227 0.93787458 0.79190875
 0.55608011 0.92551156 0.54491681 0.65785744 0.46682726 0.56354833
 0.74002916] 0.6947754827998813
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.252, MeanIU:  0.6948, Best_mIoU:  0.7092
[0.97294887 0.8056633  0.91092317 0.28512648 0.53255042 0.62073429
 0.65000973 0.75578835 0.91157328 0.57086227 0.93787458 0.79190875
 0.55608011 0.92551156 0.54491681 0.65785744 0.46682726 0.56354833
 0.74002916]
Epoch: [316/484] Iter:[0/24], Time: 0.95, lr: [0.0038584913241589583], Loss: 0.105741
Epoch: [316/484] Iter:[10/24], Time: 0.44, lr: [0.0038498775512942206], Loss: 0.130356
Epoch: [316/484] Iter:[20/24], Time: 0.42, lr: [0.003841261636496836], Loss: 0.136619
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97111506 0.79095439 0.90798003 0.39471657 0.52384493 0.59540147
 0.62632187 0.74436158 0.91021652 0.56280728 0.93172087 0.77213668
 0.53206893 0.91916809 0.4794577  0.56790429 0.4464028  0.54674565
 0.73101873] 0.6818075495295429
1 [0.9697017  0.79401115 0.90772087 0.43035364 0.5440092  0.60666712
 0.64714405 0.75466766 0.9114928  0.58608845 0.93188951 0.77496178
 0.5477563  0.92475684 0.54356924 0.62923801 0.41460208 0.56210364
 0.73780017] 0.6957123254281053
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6957, Best_mIoU:  0.7092
[0.9697017  0.79401115 0.90772087 0.43035364 0.5440092  0.60666712
 0.64714405 0.75466766 0.9114928  0.58608845 0.93188951 0.77496178
 0.5477563  0.92475684 0.54356924 0.62923801 0.41460208 0.56210364
 0.73780017]
Epoch: [317/484] Iter:[0/24], Time: 0.97, lr: [0.003837814669522124], Loss: 0.113528
Epoch: [317/484] Iter:[10/24], Time: 0.44, lr: [0.0038291957461447563], Loss: 0.139552
Epoch: [317/484] Iter:[20/24], Time: 0.41, lr: [0.0038205746666866453], Loss: 0.153624
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96747473 0.7691619  0.90988747 0.35248543 0.50874081 0.59762318
 0.62834526 0.74548932 0.91127271 0.55805892 0.93770697 0.77861326
 0.53196241 0.92133573 0.4904061  0.58573254 0.42945792 0.5457863
 0.73429398] 0.679149207546914
1 [0.96742519 0.77539733 0.91035795 0.40251514 0.53156883 0.60851951
 0.64515963 0.75081979 0.91330447 0.58246164 0.93774004 0.78433408
 0.55305636 0.92758332 0.56663378 0.66023161 0.38787725 0.5698917
 0.73864498] 0.695448557314527
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.259, MeanIU:  0.6954, Best_mIoU:  0.7092
[0.96742519 0.77539733 0.91035795 0.40251514 0.53156883 0.60851951
 0.64515963 0.75081979 0.91330447 0.58246164 0.93774004 0.78433408
 0.55305636 0.92758332 0.56663378 0.66023161 0.38787725 0.5698917
 0.73864498]
Epoch: [318/484] Iter:[0/24], Time: 0.96, lr: [0.0038171256298695335], Loss: 0.159792
Epoch: [318/484] Iter:[10/24], Time: 0.44, lr: [0.0038085015218990172], Loss: 0.140413
Epoch: [318/484] Iter:[20/24], Time: 0.42, lr: [0.003799875243520175], Loss: 0.132447
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97290062 0.79175241 0.90988493 0.34917131 0.52388526 0.60722196
 0.63599102 0.7461577  0.91186408 0.55188034 0.93718724 0.78381679
 0.51476486 0.92158915 0.50104926 0.61458392 0.48835552 0.57972446
 0.73434152] 0.688216965870662
1 [0.97266724 0.79546118 0.90913035 0.37090068 0.54327819 0.61538654
 0.64858862 0.75413297 0.91264393 0.57497116 0.93726925 0.78980406
 0.53825953 0.92850915 0.57053101 0.69763415 0.51853709 0.59013319
 0.73899246] 0.7056226707429856
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.7056, Best_mIoU:  0.7092
[0.97266724 0.79546118 0.90913035 0.37090068 0.54327819 0.61538654
 0.64858862 0.75413297 0.91264393 0.57497116 0.93726925 0.78980406
 0.53825953 0.92850915 0.57053101 0.69763415 0.51853709 0.59013319
 0.73899246]
Epoch: [319/484] Iter:[0/24], Time: 0.96, lr: [0.0037964241231060927], Loss: 0.153169
Epoch: [319/484] Iter:[10/24], Time: 0.44, lr: [0.003787794796028785], Loss: 0.122052
Epoch: [319/484] Iter:[20/24], Time: 0.42, lr: [0.0037791632840327013], Loss: 0.130571
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9732067  0.7975805  0.9112195  0.39324565 0.53181606 0.59907552
 0.6356729  0.74743928 0.91262814 0.56744923 0.93828696 0.78355805
 0.52459283 0.9196111  0.48891035 0.59471647 0.47558234 0.55835466
 0.73344498] 0.6887574322144472
1 [0.97254032 0.79885977 0.91194447 0.43169565 0.55427209 0.60759903
 0.65060926 0.75631148 0.91407462 0.59440593 0.93716125 0.78878914
 0.54677943 0.92541422 0.54582939 0.65591501 0.51841456 0.58052116
 0.74021936] 0.7069134820226868
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7069, Best_mIoU:  0.7092
[0.97254032 0.79885977 0.91194447 0.43169565 0.55427209 0.60759903
 0.65060926 0.75631148 0.91407462 0.59440593 0.93716125 0.78878914
 0.54677943 0.92541422 0.54582939 0.65591501 0.51841456 0.58052116
 0.74021936]
Epoch: [320/484] Iter:[0/24], Time: 0.95, lr: [0.0037757100660915205], Loss: 0.128911
Epoch: [320/484] Iter:[10/24], Time: 0.44, lr: [0.0037670754849524857], Loss: 0.126047
Epoch: [320/484] Iter:[20/24], Time: 0.42, lr: [0.0037584387041978864], Loss: 0.125937
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97472229 0.80657448 0.91216343 0.40803111 0.52592204 0.60993052
 0.63558551 0.74856874 0.91297987 0.56829662 0.93935772 0.77902119
 0.49781805 0.9210324  0.48759931 0.61485601 0.47275682 0.55918742
 0.73885037] 0.6901712582737847
1 [0.97461126 0.8091278  0.91391053 0.44555521 0.54823402 0.61729341
 0.65037635 0.75763252 0.91481075 0.59433762 0.93870295 0.78692684
 0.52612234 0.92646115 0.53833891 0.68442556 0.50126078 0.58374445
 0.74584177] 0.7083007487490712
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7083, Best_mIoU:  0.7092
[0.97461126 0.8091278  0.91391053 0.44555521 0.54823402 0.61729341
 0.65037635 0.75763252 0.91481075 0.59433762 0.93870295 0.78692684
 0.52612234 0.92646115 0.53833891 0.68442556 0.50126078 0.58374445
 0.74584177]
Epoch: [321/484] Iter:[0/24], Time: 0.98, lr: [0.0037549833746205953], Loss: 0.119819
Epoch: [321/484] Iter:[10/24], Time: 0.45, lr: [0.003746343504015214], Loss: 0.140956
Epoch: [321/484] Iter:[20/24], Time: 0.42, lr: [0.003737701418907595], Loss: 0.130349
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97505107 0.80661819 0.9122072  0.36452096 0.52164682 0.60870523
 0.63890339 0.74448087 0.91211207 0.55628555 0.93983001 0.7894227
 0.54097625 0.92263313 0.49482615 0.60709915 0.45343375 0.56120912
 0.73819304] 0.6888502442330102
1 [0.97516224 0.80978979 0.91274376 0.37365742 0.54228791 0.61699499
 0.65232535 0.75090047 0.91381371 0.5821169  0.94052144 0.79440542
 0.55415571 0.92775598 0.55574734 0.66656501 0.44600257 0.57857604
 0.74374753] 0.7019615574751299
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7020, Best_mIoU:  0.7092
[0.97516224 0.80978979 0.91274376 0.37365742 0.54228791 0.61699499
 0.65232535 0.75090047 0.91381371 0.5821169  0.94052144 0.79440542
 0.55415571 0.92775598 0.55574734 0.66656501 0.44600257 0.57857604
 0.74374753]
Epoch: [322/484] Iter:[0/24], Time: 0.95, lr: [0.003734243963402885], Loss: 0.165255
Epoch: [322/484] Iter:[10/24], Time: 0.44, lr: [0.0037255987674682757], Loss: 0.118507
Epoch: [322/484] Iter:[20/24], Time: 0.42, lr: [0.003716951341951217], Loss: 0.122863
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97499108 0.80504659 0.9119621  0.41747113 0.51380194 0.61184501
 0.63567606 0.74652395 0.91283637 0.55211608 0.93595694 0.78912889
 0.54095236 0.9217554  0.48177671 0.60170992 0.49876761 0.5654665
 0.74095416] 0.6925651998070009
1 [0.97450635 0.80680859 0.91383001 0.46636157 0.54120892 0.6190832
 0.65028772 0.75649453 0.91401312 0.56709983 0.93382724 0.79450325
 0.55605767 0.92643704 0.53593956 0.67606692 0.58309206 0.58472103
 0.74684107] 0.7130094569919524
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7130, Best_mIoU:  0.7130
[0.97450635 0.80680859 0.91383001 0.46636157 0.54120892 0.6190832
 0.65028772 0.75649453 0.91401312 0.56709983 0.93382724 0.79450325
 0.55605767 0.92643704 0.53593956 0.67606692 0.58309206 0.58472103
 0.74684107]
Epoch: [323/484] Iter:[0/24], Time: 0.94, lr: [0.0037134917460419836], Loss: 0.146153
Epoch: [323/484] Iter:[10/24], Time: 0.44, lr: [0.003704841188448178], Loss: 0.135034
Epoch: [323/484] Iter:[20/24], Time: 0.42, lr: [0.003696188385994442], Loss: 0.142790
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97520233 0.80440811 0.91160263 0.36365177 0.52496887 0.61257329
 0.64447025 0.74484487 0.91258442 0.55412781 0.94112505 0.79032261
 0.52824046 0.91817822 0.48752337 0.58951413 0.5030882  0.54239717
 0.73756634] 0.688757363637428
1 [0.97495515 0.80853635 0.913382   0.39372102 0.54242698 0.61883829
 0.65873305 0.75304891 0.91446922 0.57648692 0.94246033 0.79493401
 0.54889399 0.92381681 0.55545881 0.66713499 0.60434199 0.56746034
 0.74445677] 0.7107134693844456
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7107, Best_mIoU:  0.7130
[0.97495515 0.80853635 0.913382   0.39372102 0.54242698 0.61883829
 0.65873305 0.75304891 0.91446922 0.57648692 0.94246033 0.79493401
 0.54889399 0.92381681 0.55545881 0.66713499 0.60434199 0.56746034
 0.74445677]
Epoch: [324/484] Iter:[0/24], Time: 0.94, lr: [0.0036927266350142043], Loss: 0.195281
Epoch: [324/484] Iter:[10/24], Time: 0.44, lr: [0.003684070678955118], Loss: 0.129228
Epoch: [324/484] Iter:[20/24], Time: 0.42, lr: [0.0036754124625575173], Loss: 0.127388
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97511969 0.80727498 0.91140837 0.37706169 0.52459899 0.60658722
 0.64011783 0.74967861 0.91083563 0.54651351 0.94112628 0.77563864
 0.49376588 0.922956   0.51345622 0.62480995 0.49457088 0.551539
 0.73888726] 0.6897866646060455
1 [0.97494074 0.81119892 0.91310893 0.41454391 0.54922291 0.60993676
 0.65376951 0.75592338 0.91155055 0.56338117 0.94279416 0.78308248
 0.52359521 0.92988439 0.5930001  0.70254836 0.53647509 0.57620438
 0.74465063] 0.709990083524547
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7100, Best_mIoU:  0.7130
[0.97494074 0.81119892 0.91310893 0.41454391 0.54922291 0.60993676
 0.65376951 0.75592338 0.91155055 0.56338117 0.94279416 0.78308248
 0.52359521 0.92988439 0.5930001  0.70254836 0.53647509 0.57620438
 0.74465063]
Epoch: [325/484] Iter:[0/24], Time: 0.95, lr: [0.003671948541646725], Loss: 0.083803
Epoch: [325/484] Iter:[10/24], Time: 0.44, lr: [0.00366328714983089], Loss: 0.128356
Epoch: [325/484] Iter:[20/24], Time: 0.42, lr: [0.003654623481992921], Loss: 0.119106
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97488088 0.80483172 0.91101894 0.37279674 0.51750439 0.5929372
 0.64115898 0.74949827 0.91098414 0.54756604 0.94141501 0.7894799
 0.53991527 0.92266143 0.48915653 0.6301986  0.47379289 0.55500599
 0.73224077] 0.6893180902582688
1 [0.97480201 0.80728234 0.91125877 0.36792437 0.53665802 0.59575275
 0.65545796 0.75811452 0.9121089  0.56988262 0.94264549 0.79370422
 0.55342015 0.92700777 0.54123776 0.67837959 0.50126638 0.57913194
 0.73814754] 0.7023254257721434
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7023, Best_mIoU:  0.7130
[0.97480201 0.80728234 0.91125877 0.36792437 0.53665802 0.59575275
 0.65545796 0.75811452 0.9121089  0.56988262 0.94264549 0.79370422
 0.55342015 0.92700777 0.54123776 0.67837959 0.50126638 0.57913194
 0.73814754]
Epoch: [326/484] Iter:[0/24], Time: 0.95, lr: [0.0036511573760951883], Loss: 0.153145
Epoch: [326/484] Iter:[10/24], Time: 0.44, lr: [0.0036424905107362415], Loss: 0.141032
Epoch: [326/484] Iter:[20/24], Time: 0.42, lr: [0.0036338213534624674], Loss: 0.147305
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97462584 0.80336927 0.91068318 0.37378    0.51155349 0.59766146
 0.64142391 0.74637238 0.91205069 0.53587708 0.94220283 0.78776169
 0.52588836 0.92605537 0.49731102 0.65130084 0.48226885 0.56077726
 0.74117689] 0.6906389687638473
1 [0.97457843 0.80651065 0.91269821 0.39683095 0.53596529 0.60665928
 0.6580525  0.75548609 0.91361919 0.56449408 0.9431733  0.79419422
 0.54771546 0.93099074 0.54888972 0.7141822  0.51687614 0.58434683
 0.74876456] 0.7081067298114767
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7081, Best_mIoU:  0.7130
[0.97457843 0.80651065 0.91269821 0.39683095 0.53596529 0.60665928
 0.6580525  0.75548609 0.91361919 0.56449408 0.9431733  0.79419422
 0.54771546 0.93099074 0.54888972 0.7141822  0.51687614 0.58434683
 0.74876456]
Epoch: [327/484] Iter:[0/24], Time: 0.98, lr: [0.003630353047320699], Loss: 0.104882
Epoch: [327/484] Iter:[10/24], Time: 0.44, lr: [0.0036216806701276243], Loss: 0.124342
Epoch: [327/484] Iter:[20/24], Time: 0.42, lr: [0.003613005984913821], Loss: 0.125829
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97481052 0.80918002 0.90803094 0.35371189 0.49394209 0.60350251
 0.64428401 0.75117388 0.91127924 0.56598442 0.94019686 0.78569494
 0.52826276 0.9261183  0.47018129 0.6440195  0.40675755 0.55899324
 0.74249613] 0.685190531700703
1 [0.97505648 0.81147157 0.90736837 0.34328247 0.51069338 0.61065974
 0.65705182 0.76083486 0.91246647 0.58910063 0.93968555 0.79230802
 0.54801293 0.92955345 0.51047476 0.69036414 0.3312418  0.57758424
 0.74910656] 0.6919114340011078
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6919, Best_mIoU:  0.7130
[0.97505648 0.81147157 0.90736837 0.34328247 0.51069338 0.61065974
 0.65705182 0.76083486 0.91246647 0.58910063 0.93968555 0.79230802
 0.54801293 0.92955345 0.51047476 0.69036414 0.3312418  0.57758424
 0.74910656]
Epoch: [328/484] Iter:[0/24], Time: 0.97, lr: [0.0036095354630662326], Loss: 0.074503
Epoch: [328/484] Iter:[10/24], Time: 0.44, lr: [0.00360085753523335], Loss: 0.135700
Epoch: [328/484] Iter:[20/24], Time: 0.42, lr: [0.0035921772830563754], Loss: 0.129239
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97459757 0.80662146 0.90901215 0.34887242 0.50153493 0.60841888
 0.63154722 0.75259172 0.91051021 0.56235091 0.94129848 0.7869411
 0.51763923 0.92663401 0.50054746 0.6485278  0.47628041 0.5520257
 0.73722341] 0.68911447804237
1 [0.97453813 0.80838137 0.90933894 0.35999819 0.52379992 0.61323719
 0.64957781 0.76176931 0.91217912 0.5922083  0.94089892 0.79238126
 0.53997345 0.93105386 0.57143291 0.70984662 0.52693404 0.57286939
 0.74139791] 0.7069377180063731
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7069, Best_mIoU:  0.7130
[0.97453813 0.80838137 0.90933894 0.35999819 0.52379992 0.61323719
 0.64957781 0.76176931 0.91217912 0.5922083  0.94089892 0.79238126
 0.53997345 0.93105386 0.57143291 0.70984662 0.52693404 0.57286939
 0.74139791]
Epoch: [329/484] Iter:[0/24], Time: 0.96, lr: [0.0035887045298324182], Loss: 0.081237
Epoch: [329/484] Iter:[10/24], Time: 0.45, lr: [0.0035800210120291096], Loss: 0.120920
Epoch: [329/484] Iter:[20/24], Time: 0.42, lr: [0.003571335153336518], Loss: 0.124735
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97464604 0.80820627 0.9118637  0.37663608 0.51342675 0.61434441
 0.64060476 0.7512745  0.91136994 0.55025093 0.94025681 0.78063315
 0.49747875 0.92822271 0.4452598  0.62215533 0.40664918 0.52467137
 0.73784077] 0.6808311186505951
1 [0.97464116 0.80905992 0.91357242 0.40380337 0.53447474 0.61920272
 0.65191214 0.76101428 0.91234997 0.55468907 0.9388491  0.78972885
 0.52909769 0.93385771 0.4636676  0.62827712 0.37843913 0.55089102
 0.74397666] 0.6890265617206065
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6890, Best_mIoU:  0.7130
[0.97464116 0.80905992 0.91357242 0.40380337 0.53447474 0.61920272
 0.65191214 0.76101428 0.91234997 0.55468907 0.9388491  0.78972885
 0.52909769 0.93385771 0.4636676  0.62827712 0.37843913 0.55089102
 0.74397666]
Epoch: [330/484] Iter:[0/24], Time: 0.97, lr: [0.0035678601528526892], Loss: 0.120895
Epoch: [330/484] Iter:[10/24], Time: 0.44, lr: [0.0035591710052128374], Loss: 0.118632
Epoch: [330/484] Iter:[20/24], Time: 0.42, lr: [0.0035504794999122124], Loss: 0.118608
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97500365 0.80826113 0.9134383  0.38260184 0.53311754 0.6092234
 0.64482689 0.75242761 0.91274468 0.56138759 0.94149958 0.78544326
 0.5217294  0.92642576 0.48126554 0.62688504 0.4184401  0.5323949
 0.73678596] 0.6875737981574939
1 [0.97496042 0.80969275 0.91534076 0.41118072 0.55658256 0.61646154
 0.65845915 0.76313173 0.91428702 0.58244943 0.93984149 0.79215986
 0.54356488 0.932847   0.51326773 0.65015609 0.36376399 0.56037691
 0.74263689] 0.69690320612599
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.6969, Best_mIoU:  0.7130
[0.97496042 0.80969275 0.91534076 0.41118072 0.55658256 0.61646154
 0.65845915 0.76313173 0.91428702 0.58244943 0.93984149 0.79215986
 0.54356488 0.932847   0.51326773 0.65015609 0.36376399 0.56037691
 0.74263689]
Epoch: [331/484] Iter:[0/24], Time: 0.99, lr: [0.0035470022360677565], Loss: 0.080469
Epoch: [331/484] Iter:[10/24], Time: 0.45, lr: [0.003538307418178924], Loss: 0.128657
Epoch: [331/484] Iter:[20/24], Time: 0.42, lr: [0.0035296102256269254], Loss: 0.127735
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97348618 0.7998343  0.91088685 0.35371663 0.52145916 0.60648146
 0.64099801 0.75231456 0.91237382 0.56535655 0.94155381 0.78264211
 0.51185369 0.92633297 0.52597699 0.6240756  0.44943731 0.54248588
 0.74004759] 0.6884901814501719
1 [0.9729651  0.79888831 0.91140599 0.34809303 0.54384857 0.61090333
 0.65826618 0.76249663 0.91416482 0.59192746 0.94277772 0.78761052
 0.53605922 0.93147882 0.60538977 0.67528714 0.46102995 0.56343572
 0.74471222] 0.7031968682311098
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7032, Best_mIoU:  0.7130
[0.9729651  0.79888831 0.91140599 0.34809303 0.54384857 0.61090333
 0.65826618 0.76249663 0.91416482 0.59192746 0.94277772 0.78761052
 0.53605922 0.93147882 0.60538977 0.67528714 0.46102995 0.56343572
 0.74471222]
Epoch: [332/484] Iter:[0/24], Time: 0.98, lr: [0.0035261306820994144], Loss: 0.165121
Epoch: [332/484] Iter:[10/24], Time: 0.45, lr: [0.0035174301529917107], Loss: 0.127035
Epoch: [332/484] Iter:[20/24], Time: 0.42, lr: [0.0035087272319828224], Loss: 0.127448
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97422145 0.79839989 0.91101166 0.34556928 0.50890143 0.60990993
 0.64080369 0.74955313 0.91269308 0.55842254 0.94111357 0.79026634
 0.54476771 0.92386706 0.52623492 0.61041359 0.44563605 0.532989
 0.73842599] 0.6875368586873072
1 [0.97366346 0.79774938 0.91181308 0.34749789 0.52943051 0.61526856
 0.65577977 0.75919551 0.91461871 0.57535307 0.94332726 0.79474272
 0.55562846 0.92846145 0.59747131 0.64696776 0.42423244 0.55050667
 0.74271239] 0.6981273891668204
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6981, Best_mIoU:  0.7130
[0.97366346 0.79774938 0.91181308 0.34749789 0.52943051 0.61526856
 0.65577977 0.75919551 0.91461871 0.57535307 0.94332726 0.79474272
 0.55562846 0.92846145 0.59747131 0.64696776 0.42423244 0.55050667
 0.74271239]
Epoch: [333/484] Iter:[0/24], Time: 1.00, lr: [0.003505245392223629], Loss: 0.081640
Epoch: [333/484] Iter:[10/24], Time: 0.45, lr: [0.003496539110358289], Loss: 0.111827
Epoch: [333/484] Iter:[20/24], Time: 0.42, lr: [0.0034878304191132666], Loss: 0.119171
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97381899 0.79419319 0.91125564 0.35605507 0.51119122 0.61317784
 0.64142674 0.75388176 0.91152258 0.55087653 0.9428682  0.78584605
 0.52592118 0.92493586 0.51470873 0.61534286 0.44609636 0.54013561
 0.74284394] 0.6871630709508304
1 [0.97331739 0.79507646 0.91197698 0.37819774 0.52808475 0.62155617
 0.65702496 0.76449034 0.91363877 0.56645609 0.94337517 0.79209922
 0.54532241 0.92893929 0.57216183 0.65235288 0.43200105 0.56484857
 0.74925742] 0.6994830261673803
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6995, Best_mIoU:  0.7130
[0.97331739 0.79507646 0.91197698 0.37819774 0.52808475 0.62155617
 0.65702496 0.76449034 0.91363877 0.56645609 0.94337517 0.79209922
 0.54532241 0.92893929 0.57216183 0.65235288 0.43200105 0.56484857
 0.74925742]
Epoch: [334/484] Iter:[0/24], Time: 0.94, lr: [0.0034843462663428984], Loss: 0.116665
Epoch: [334/484] Iter:[10/24], Time: 0.44, lr: [0.0034756341896005406], Loss: 0.122534
Epoch: [334/484] Iter:[20/24], Time: 0.42, lr: [0.0034669196857545427], Loss: 0.136410
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9736633  0.79605104 0.91115913 0.33443049 0.50077426 0.60587408
 0.63756627 0.75418615 0.91234405 0.55772061 0.94297776 0.78894145
 0.54200746 0.92192993 0.52435661 0.60029775 0.47280865 0.53345289
 0.73685253] 0.686704970232109
1 [0.97331536 0.79785035 0.91124495 0.33471766 0.51762797 0.6126316
 0.6509007  0.76459831 0.91408308 0.57927159 0.94331918 0.79395083
 0.55409775 0.9256264  0.56956954 0.63076016 0.4682839  0.56053629
 0.7425155 ] 0.6971000590884499
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.253, MeanIU:  0.6971, Best_mIoU:  0.7130
[0.97331536 0.79785035 0.91124495 0.33471766 0.51762797 0.6126316
 0.6509007  0.76459831 0.91408308 0.57927159 0.94331918 0.79395083
 0.55409775 0.9256264  0.56956954 0.63076016 0.4682839  0.56053629
 0.7425155 ]
Epoch: [335/484] Iter:[0/24], Time: 0.94, lr: [0.0034634332029578586], Loss: 0.157282
Epoch: [335/484] Iter:[10/24], Time: 0.44, lr: [0.003454715288626429], Loss: 0.129283
Epoch: [335/484] Iter:[20/24], Time: 0.42, lr: [0.0034459949292168115], Loss: 0.124645
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97495586 0.80663974 0.9108736  0.35074934 0.49737628 0.61094888
 0.63935059 0.75691475 0.9112433  0.52464829 0.94126074 0.78623795
 0.54144267 0.92340483 0.53175449 0.61717737 0.47715443 0.56087956
 0.73811272] 0.6895329151226577
1 [0.97441641 0.80719754 0.91168636 0.35535836 0.51844386 0.61930492
 0.64907021 0.76457508 0.91152198 0.51517729 0.9409618  0.79026702
 0.55300635 0.92727167 0.58141762 0.65072077 0.47481512 0.57865455
 0.74337555] 0.6982759197281713
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.6983, Best_mIoU:  0.7130
[0.97441641 0.80719754 0.91168636 0.35535836 0.51844386 0.61930492
 0.64907021 0.76457508 0.91152198 0.51517729 0.9409618  0.79026702
 0.55300635 0.92727167 0.58141762 0.65072077 0.47481512 0.57865455
 0.74337555]
Epoch: [336/484] Iter:[0/24], Time: 0.95, lr: [0.003442506099138104], Loss: 0.142100
Epoch: [336/484] Iter:[10/24], Time: 0.45, lr: [0.003433782303900478], Loss: 0.123119
Epoch: [336/484] Iter:[20/24], Time: 0.42, lr: [0.0034250560453542678], Loss: 0.120279
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97485366 0.80448858 0.91232329 0.36375301 0.50264997 0.60708585
 0.64097872 0.75667379 0.91109731 0.54079809 0.94195026 0.7856974
 0.52827714 0.92202151 0.50590223 0.59420049 0.4462894  0.5571812
 0.73690014] 0.6859537928373043
1 [0.974408   0.80521095 0.91219398 0.34578022 0.51040884 0.61224313
 0.65589873 0.76620143 0.91131512 0.54685031 0.94217363 0.78947361
 0.54826454 0.92381997 0.53643011 0.61263456 0.45186038 0.58297704
 0.74061812] 0.6930927719543871
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.6931, Best_mIoU:  0.7130
[0.974408   0.80521095 0.91219398 0.34578022 0.51040884 0.61224313
 0.65589873 0.76620143 0.91131512 0.54685031 0.94217363 0.78947361
 0.54826454 0.92381997 0.53643011 0.61263456 0.45186038 0.58297704
 0.74061812]
Epoch: [337/484] Iter:[0/24], Time: 0.97, lr: [0.003421564850492208], Loss: 0.144359
Epoch: [337/484] Iter:[10/24], Time: 0.44, lr: [0.003412835130413452], Loss: 0.132294
Epoch: [337/484] Iter:[20/24], Time: 0.42, lr: [0.003404102928534451], Loss: 0.137240
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97462915 0.80448135 0.91279752 0.36160585 0.5112126  0.60793733
 0.63642627 0.75653999 0.91201137 0.55062266 0.94285829 0.78514466
 0.54390938 0.92058008 0.52644422 0.60222875 0.4339104  0.5664059
 0.73251816] 0.6885402076345679
1 [0.97421118 0.80647724 0.91411246 0.3697177  0.52373385 0.61522941
 0.64120374 0.76399066 0.91328306 0.56862748 0.94533196 0.78819324
 0.55392897 0.92470478 0.58473358 0.6154629  0.35979478 0.58246697
 0.73645589] 0.6937715713450473
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6938, Best_mIoU:  0.7130
[0.97421118 0.80647724 0.91411246 0.3697177  0.52373385 0.61522941
 0.64120374 0.76399066 0.91328306 0.56862748 0.94533196 0.78819324
 0.55392897 0.92470478 0.58473358 0.6154629  0.35979478 0.58246697
 0.73645589]
Epoch: [338/484] Iter:[0/24], Time: 0.97, lr: [0.0034006093511368914], Loss: 0.149225
Epoch: [338/484] Iter:[10/24], Time: 0.45, lr: [0.003391873661651171], Loss: 0.135286
Epoch: [338/484] Iter:[20/24], Time: 0.42, lr: [0.0033831354716067127], Loss: 0.132276
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97501428 0.80453241 0.91321599 0.38786635 0.51049311 0.60376837
 0.63409583 0.75823059 0.91125943 0.52191219 0.94160591 0.78718606
 0.54535164 0.92693298 0.52667414 0.62333993 0.42168045 0.56805487
 0.74316465] 0.6897041679567661
1 [0.97461942 0.80712844 0.91498738 0.42280818 0.53038591 0.6093951
 0.64319078 0.76514794 0.91239006 0.53068151 0.94305589 0.79156687
 0.55541198 0.93211292 0.60400307 0.64529512 0.32428543 0.58456716
 0.7497622 ] 0.6968839667859604
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6969, Best_mIoU:  0.7130
[0.97461942 0.80712844 0.91498738 0.42280818 0.53038591 0.6093951
 0.64319078 0.76514794 0.91239006 0.53068151 0.94305589 0.79156687
 0.55541198 0.93211292 0.60400307 0.64529512 0.32428543 0.58456716
 0.7497622 ]
Epoch: [339/484] Iter:[0/24], Time: 0.96, lr: [0.0033796394936653445], Loss: 0.096518
Epoch: [339/484] Iter:[10/24], Time: 0.44, lr: [0.003370897789562451], Loss: 0.124272
Epoch: [339/484] Iter:[20/24], Time: 0.42, lr: [0.0033621535658697743], Loss: 0.120098
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97376525 0.79797162 0.91349751 0.40091513 0.51651597 0.61537131
 0.64613115 0.7577237  0.91136921 0.53936815 0.94218697 0.78962232
 0.53800244 0.92010073 0.52069598 0.59613767 0.48743103 0.54735804
 0.73818543] 0.6922289276355184
1 [0.97371731 0.8009131  0.91503446 0.43070079 0.53569598 0.62141061
 0.65888515 0.7662774  0.91240624 0.54905101 0.94264092 0.79397623
 0.55306479 0.92617681 0.58217856 0.63705334 0.4935157  0.56858505
 0.74238618] 0.7054562960490336
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7055, Best_mIoU:  0.7130
[0.97371731 0.8009131  0.91503446 0.43070079 0.53569598 0.62141061
 0.65888515 0.7662774  0.91240624 0.54905101 0.94264092 0.79397623
 0.55306479 0.92617681 0.58217856 0.63705334 0.4935157  0.56858505
 0.74238618]
Epoch: [340/484] Iter:[0/24], Time: 0.96, lr: [0.003358655169114637], Loss: 0.230217
Epoch: [340/484] Iter:[10/24], Time: 0.44, lr: [0.0033499074045261422], Loss: 0.155573
Epoch: [340/484] Iter:[20/24], Time: 0.42, lr: [0.0033411571010383808], Loss: 0.147179
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9712471  0.78231262 0.91199244 0.37412851 0.51280234 0.60872472
 0.64096655 0.75292677 0.9108237  0.51182442 0.94124027 0.78729996
 0.53493203 0.9229028  0.51705088 0.63011409 0.45465435 0.54878013
 0.7401634 ] 0.6870993205180325
1 [0.97082702 0.78228072 0.91327099 0.37696433 0.52904647 0.61696081
 0.65147674 0.75949077 0.91128432 0.52699498 0.94237701 0.79218116
 0.54771081 0.92878596 0.59234097 0.67119425 0.40010715 0.56256432
 0.74586717] 0.695880312665366
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.252, MeanIU:  0.6959, Best_mIoU:  0.7130
[0.97082702 0.78228072 0.91327099 0.37696433 0.52904647 0.61696081
 0.65147674 0.75949077 0.91128432 0.52699498 0.94237701 0.79218116
 0.54771081 0.92878596 0.59234097 0.67119425 0.40010715 0.56256432
 0.74586717]
Epoch: [341/484] Iter:[0/24], Time: 0.96, lr: [0.0033376562669322042], Loss: 0.152491
Epoch: [341/484] Iter:[10/24], Time: 0.44, lr: [0.003328902395317216], Loss: 0.133367
Epoch: [341/484] Iter:[20/24], Time: 0.42, lr: [0.003320145965208984], Loss: 0.124252
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97122056 0.78660438 0.9112498  0.40695018 0.50472705 0.60939739
 0.63745886 0.75039284 0.91015678 0.52783533 0.94137668 0.77664166
 0.54034658 0.92333311 0.51179454 0.62684107 0.44446089 0.50268411
 0.73323668] 0.6850899213301423
1 [0.97092169 0.78898316 0.91333383 0.44830899 0.5298935  0.61773112
 0.65051009 0.75989224 0.91169808 0.53767771 0.94136981 0.77869272
 0.5439803  0.93049198 0.57118792 0.67935303 0.44138564 0.50908507
 0.73789992] 0.6980208850781606
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.6980, Best_mIoU:  0.7130
[0.97092169 0.78898316 0.91333383 0.44830899 0.5298935  0.61773112
 0.65051009 0.75989224 0.91169808 0.53767771 0.94136981 0.77869272
 0.5439803  0.93049198 0.57118792 0.67935303 0.44138564 0.50908507
 0.73789992]
Epoch: [342/484] Iter:[0/24], Time: 0.97, lr: [0.0033166426749413782], Loss: 0.138462
Epoch: [342/484] Iter:[10/24], Time: 0.44, lr: [0.0033078826490718846], Loss: 0.135916
Epoch: [342/484] Iter:[20/24], Time: 0.42, lr: [0.003299120044824439], Loss: 0.130241
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97184799 0.7884147  0.91177889 0.3937193  0.50851086 0.6149858
 0.63967683 0.75449708 0.91102917 0.54136698 0.94261606 0.78399781
 0.49620612 0.92309552 0.5136314  0.61844136 0.43345455 0.52728985
 0.73493522] 0.684710288579542
1 [0.97128187 0.79202593 0.91245825 0.41422848 0.5330145  0.62081474
 0.65084433 0.76338893 0.91195985 0.5542355  0.94300605 0.79071174
 0.52043904 0.92954742 0.58289851 0.67208729 0.41149667 0.54725458
 0.74042004] 0.6980059858938835
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.253, MeanIU:  0.6980, Best_mIoU:  0.7130
[0.97128187 0.79202593 0.91245825 0.41422848 0.5330145  0.62081474
 0.65084433 0.76338893 0.91195985 0.5542355  0.94300605 0.79071174
 0.52043904 0.92954742 0.58289851 0.67208729 0.41149667 0.54725458
 0.74042004]
Epoch: [343/484] Iter:[0/24], Time: 1.00, lr: [0.0032956142793059136], Loss: 0.085151
Epoch: [343/484] Iter:[10/24], Time: 0.45, lr: [0.003286848051251698], Loss: 0.120391
Epoch: [343/484] Iter:[20/24], Time: 0.42, lr: [0.003278079224637688], Loss: 0.129309
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97188909 0.78848645 0.9124017  0.40588605 0.51559949 0.60740952
 0.63796722 0.75601362 0.91307029 0.53659473 0.94073428 0.78534096
 0.53029325 0.92699218 0.52507061 0.6318323  0.43920902 0.54969527
 0.74010853] 0.6902418187358951
1 [0.97189625 0.79227846 0.91333199 0.42059287 0.53600981 0.61311171
 0.64991422 0.76446878 0.91395153 0.5469592  0.94091818 0.78971459
 0.53939519 0.9327475  0.60665399 0.6893608  0.43680636 0.56122436
 0.74404053] 0.7033355957012881
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.7033, Best_mIoU:  0.7130
[0.97189625 0.79227846 0.91333199 0.42059287 0.53600981 0.61311171
 0.64991422 0.76446878 0.91395153 0.5469592  0.94091818 0.78971459
 0.53939519 0.9327475  0.60665399 0.6893608  0.43680636 0.56122436
 0.74404053]
Epoch: [344/484] Iter:[0/24], Time: 0.97, lr: [0.0032745709644934912], Loss: 0.092504
Epoch: [344/484] Iter:[10/24], Time: 0.44, lr: [0.003265798485606614], Loss: 0.100260
Epoch: [344/484] Iter:[20/24], Time: 0.42, lr: [0.003257023387674357], Loss: 0.112724
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97305141 0.792332   0.91082301 0.36261629 0.49241273 0.61716955
 0.6398683  0.7526479  0.913035   0.53512989 0.9423205  0.79129438
 0.53068481 0.92450036 0.52274615 0.62333634 0.41531558 0.54543021
 0.73620566] 0.6853115815614066
1 [0.97306992 0.79583106 0.91152812 0.37568353 0.51119859 0.62368572
 0.65527152 0.76124943 0.91396425 0.5485721  0.94312113 0.79662729
 0.54357252 0.93182434 0.60883597 0.68346601 0.39247344 0.56360783
 0.74324452] 0.698780383901547
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.6988, Best_mIoU:  0.7130
[0.97306992 0.79583106 0.91152812 0.37568353 0.51119859 0.62368572
 0.65527152 0.76124943 0.91396425 0.5485721  0.94312113 0.79662729
 0.54357252 0.93182434 0.60883597 0.68346601 0.39247344 0.56360783
 0.74324452]
Epoch: [345/484] Iter:[0/24], Time: 0.95, lr: [0.0032535126132381413], Loss: 0.135770
Epoch: [345/484] Iter:[10/24], Time: 0.44, lr: [0.0032447338341369614], Loss: 0.125475
Epoch: [345/484] Iter:[20/24], Time: 0.42, lr: [0.0032359524151942764], Loss: 0.128900
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97347522 0.7967248  0.9124166  0.41349654 0.50834116 0.61738676
 0.63584494 0.75844112 0.91219878 0.54815679 0.94294867 0.79200557
 0.53430577 0.92332205 0.51430758 0.62114451 0.45085197 0.54592846
 0.73643188] 0.6914594306068244
1 [0.97279888 0.79752862 0.91398649 0.45507503 0.53172171 0.62467516
 0.65324214 0.76746939 0.91426393 0.56996696 0.94406421 0.79760524
 0.54805152 0.9306191  0.58885568 0.68226713 0.41968195 0.56043386
 0.74278516] 0.7060574816841507
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7061, Best_mIoU:  0.7130
[0.97279888 0.79752862 0.91398649 0.45507503 0.53172171 0.62467516
 0.65324214 0.76746939 0.91426393 0.56996696 0.94406421 0.79760524
 0.54805152 0.9306191  0.58885568 0.68226713 0.41968195 0.56043386
 0.74278516]
Epoch: [346/484] Iter:[0/24], Time: 0.96, lr: [0.0032324391065015725], Loss: 0.085945
Epoch: [346/484] Iter:[10/24], Time: 0.44, lr: [0.0032236539770542877], Loss: 0.102888
Epoch: [346/484] Iter:[20/24], Time: 0.42, lr: [0.003214866186651837], Loss: 0.118453
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97314841 0.79707658 0.91195166 0.36973537 0.50486309 0.61545796
 0.63999379 0.75734426 0.91172306 0.54003171 0.94308399 0.78821957
 0.54364486 0.92300579 0.50970692 0.619031   0.43194501 0.52306062
 0.7357551 ] 0.6862515117930501
1 [0.97315214 0.80167723 0.91262404 0.38350835 0.52220052 0.62320736
 0.6511594  0.76547674 0.91319468 0.55462679 0.9439494  0.79092314
 0.55270984 0.92918051 0.58114892 0.67756131 0.40716497 0.5349187
 0.74082374] 0.6978530407915228
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.6979, Best_mIoU:  0.7130
[0.97315214 0.80167723 0.91262404 0.38350835 0.52220052 0.62320736
 0.6511594  0.76547674 0.91319468 0.55462679 0.9439494  0.79092314
 0.55270984 0.92918051 0.58114892 0.67756131 0.40716497 0.5349187
 0.74082374]
Epoch: [347/484] Iter:[0/24], Time: 0.95, lr: [0.003211350323433333], Loss: 0.168967
Epoch: [347/484] Iter:[10/24], Time: 0.44, lr: [0.0032025587927410433], Loss: 0.131417
Epoch: [347/484] Iter:[20/24], Time: 0.42, lr: [0.003193764579655178], Loss: 0.127667
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97289277 0.79422775 0.91181516 0.3936507  0.51400667 0.6115564
 0.6380585  0.75423691 0.9118346  0.54611763 0.94254244 0.78303221
 0.5170935  0.92301317 0.51423805 0.60643568 0.43856805 0.53797293
 0.73798435] 0.6868040767623846
1 [0.97248375 0.7988796  0.91254691 0.43743844 0.53645156 0.62188354
 0.65091179 0.7641554  0.91407669 0.56523413 0.94218168 0.78882812
 0.53154968 0.92972816 0.57458536 0.66867884 0.45007887 0.55477411
 0.74380884] 0.7030671303020576
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.7031, Best_mIoU:  0.7130
[0.97248375 0.7988796  0.91254691 0.43743844 0.53645156 0.62188354
 0.65091179 0.7641554  0.91407669 0.56523413 0.94218168 0.78882812
 0.53154968 0.92972816 0.57458536 0.66867884 0.45007887 0.55477411
 0.74380884]
Epoch: [348/484] Iter:[0/24], Time: 0.95, lr: [0.003190246141329798], Loss: 0.129714
Epoch: [348/484] Iter:[10/24], Time: 0.44, lr: [0.003181448157709044], Loss: 0.112087
Epoch: [348/484] Iter:[20/24], Time: 0.41, lr: [0.0031726474699241302], Loss: 0.118152
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9740116  0.79931567 0.91213888 0.381008   0.50851119 0.61982531
 0.64062118 0.75661349 0.91253198 0.55689828 0.941777   0.78833603
 0.53035816 0.9221071  0.51035725 0.59411015 0.44799211 0.53798336
 0.7332812 ] 0.6877777868239523
1 [0.9735343  0.80199867 0.91247588 0.41680073 0.52381956 0.62629755
 0.65258185 0.76548971 0.91435096 0.56795506 0.94260493 0.79400871
 0.54670376 0.92808081 0.55745691 0.65672305 0.46665577 0.55977355
 0.73897618] 0.7024362070959466
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.7024, Best_mIoU:  0.7130
[0.9735343  0.80199867 0.91247588 0.41680073 0.52381956 0.62629755
 0.65258185 0.76548971 0.91435096 0.56795506 0.94260493 0.79400871
 0.54670376 0.92808081 0.55745691 0.65672305 0.46665577 0.55977355
 0.73897618]
Epoch: [349/484] Iter:[0/24], Time: 0.96, lr: [0.003169126435591908], Loss: 0.144752
Epoch: [349/484] Iter:[10/24], Time: 0.44, lr: [0.0031603219465566913], Loss: 0.129884
Epoch: [349/484] Iter:[20/24], Time: 0.41, lr: [0.0031515147312469], Loss: 0.126330
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97378039 0.79314153 0.91355667 0.38323103 0.52156017 0.61228086
 0.63959082 0.75571308 0.9125426  0.51137756 0.94248632 0.78789526
 0.53109994 0.92319272 0.5157675  0.59518497 0.4272983  0.52128238
 0.73360564] 0.6839256715185558
1 [0.97357342 0.79589603 0.91467605 0.40791768 0.54485734 0.61959639
 0.65230537 0.76440563 0.91393223 0.51071272 0.94251466 0.7934797
 0.54875675 0.92961887 0.57585138 0.66381492 0.44030894 0.54301638
 0.73795493] 0.6985889155597502
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.251, MeanIU:  0.6986, Best_mIoU:  0.7130
[0.97357342 0.79589603 0.91467605 0.40791768 0.54485734 0.61959639
 0.65230537 0.76440563 0.91393223 0.51071272 0.94251466 0.7934797
 0.54875675 0.92961887 0.57585138 0.66381492 0.44030894 0.54301638
 0.73795493]
Epoch: [350/484] Iter:[0/24], Time: 0.95, lr: [0.0031479910796816213], Loss: 0.100020
Epoch: [350/484] Iter:[10/24], Time: 0.44, lr: [0.0031391800319248704], Loss: 0.118921
Epoch: [350/484] Iter:[20/24], Time: 0.42, lr: [0.003130366235435413], Loss: 0.119252
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97471138 0.80167399 0.91298625 0.37317088 0.519731   0.60966274
 0.64060022 0.75695839 0.91249703 0.52554338 0.94259378 0.78826927
 0.52853122 0.92170316 0.50633558 0.59395583 0.44756632 0.52448169
 0.7319576 ] 0.6848910372055733
1 [0.97468422 0.80525809 0.91251221 0.36246583 0.53653769 0.61697581
 0.65513696 0.76652424 0.91333478 0.53307636 0.94341537 0.79346543
 0.55052632 0.92577025 0.55391596 0.65003654 0.48843266 0.54303313
 0.73599519] 0.6979524769000619
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.6980, Best_mIoU:  0.7130
[0.97468422 0.80525809 0.91251221 0.36246583 0.53653769 0.61697581
 0.65513696 0.76652424 0.91333478 0.53307636 0.94341537 0.79346543
 0.55052632 0.92577025 0.55391596 0.65003654 0.48843266 0.54303313
 0.73599519]
Epoch: [351/484] Iter:[0/24], Time: 0.95, lr: [0.00312683994507705], Loss: 0.073684
Epoch: [351/484] Iter:[10/24], Time: 0.44, lr: [0.0031180222844515142], Loss: 0.117708
Epoch: [351/484] Iter:[20/24], Time: 0.42, lr: [0.003109201852279287], Loss: 0.115254
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97409714 0.79699773 0.91354709 0.37788029 0.53125016 0.61385083
 0.64486826 0.75676137 0.91251179 0.52294631 0.94243397 0.78433503
 0.521576   0.92303753 0.5086757  0.60992062 0.46752785 0.54111492
 0.74100287] 0.6886492354798088
1 [0.97409743 0.80064222 0.91491198 0.40477945 0.55340075 0.62072179
 0.65707435 0.76558416 0.91434519 0.53750103 0.94302752 0.79078344
 0.54014844 0.92930242 0.56062585 0.6804724  0.49098963 0.56329203
 0.74546931] 0.70458786156065
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.7046, Best_mIoU:  0.7130
[0.97409743 0.80064222 0.91491198 0.40477945 0.55340075 0.62072179
 0.65707435 0.76558416 0.91434519 0.53750103 0.94302752 0.79078344
 0.54014844 0.92930242 0.56062585 0.6804724  0.49098963 0.56329203
 0.74546931]
Epoch: [352/484] Iter:[0/24], Time: 0.96, lr: [0.0031056729012261864], Loss: 0.105213
Epoch: [352/484] Iter:[10/24], Time: 0.44, lr: [0.0030968485727247454], Loss: 0.098545
Epoch: [352/484] Iter:[20/24], Time: 0.42, lr: [0.0030880214494983897], Loss: 0.109809
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97365416 0.79420144 0.9116723  0.32387654 0.51434672 0.6101249
 0.63673031 0.75698536 0.91056458 0.47922266 0.94188754 0.78429858
 0.54030414 0.92475526 0.5074273  0.60407192 0.44207252 0.53091803
 0.73821324] 0.6802803948283818
1 [0.97447757 0.79808206 0.91232942 0.3329589  0.53308927 0.61846826
 0.65166776 0.76577541 0.91145062 0.48771771 0.94201175 0.78542998
 0.54065315 0.93029418 0.5638302  0.67081282 0.46126224 0.55685093
 0.74290656] 0.6936878314316002
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.6937, Best_mIoU:  0.7130
[0.97447757 0.79808206 0.91232942 0.3329589  0.53308927 0.61846826
 0.65166776 0.76577541 0.91145062 0.48771771 0.94201175 0.78542998
 0.54065315 0.93029418 0.5638302  0.67081282 0.46126224 0.55685093
 0.74290656]
Epoch: [353/484] Iter:[0/24], Time: 1.02, lr: [0.0030844898154992217], Loss: 0.093973
Epoch: [353/484] Iter:[10/24], Time: 0.45, lr: [0.0030756587632345673], Loss: 0.140267
Epoch: [353/484] Iter:[20/24], Time: 0.42, lr: [0.003066824892693888], Loss: 0.129237
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97224596 0.7801346  0.91130097 0.30016267 0.5123664  0.61531831
 0.64164012 0.7570534  0.91168276 0.50249294 0.94268714 0.77995817
 0.46187516 0.92202486 0.51229668 0.60292221 0.4730099  0.51064396
 0.73605208] 0.6760983318385009
1 [0.97225907 0.78559968 0.91264302 0.31486853 0.53548919 0.6230932
 0.65386572 0.76487752 0.91307315 0.52264013 0.94350011 0.78669676
 0.4850108  0.92785184 0.5728919  0.68172593 0.51876116 0.5403494
 0.74213951] 0.6945966642964682
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.258, MeanIU:  0.6946, Best_mIoU:  0.7130
[0.97225907 0.78559968 0.91264302 0.31486853 0.53548919 0.6230932
 0.65386572 0.76487752 0.91307315 0.52264013 0.94350011 0.78669676
 0.4850108  0.92785184 0.5728919  0.68172593 0.51876116 0.5403494
 0.74213951]
Epoch: [354/484] Iter:[0/24], Time: 0.97, lr: [0.0030632905531393423], Loss: 0.191822
Epoch: [354/484] Iter:[10/24], Time: 0.44, lr: [0.0030544527203230365], Loss: 0.148103
Epoch: [354/484] Iter:[20/24], Time: 0.42, lr: [0.003045612045297779], Loss: 0.138966
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97465394 0.79526925 0.91240641 0.40493429 0.51965222 0.61815471
 0.63776456 0.75624816 0.91318661 0.52384376 0.94218771 0.78987862
 0.53102645 0.92734956 0.52337129 0.60485887 0.49853671 0.53255032
 0.73383519] 0.6915636123383117
1 [0.97462292 0.79988643 0.91424784 0.45421569 0.54537305 0.62569339
 0.65007501 0.76472627 0.91417659 0.538014   0.94336688 0.79663282
 0.54362445 0.933594   0.60476243 0.69623281 0.53693611 0.55426821
 0.73983847] 0.7121203881858432
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7121, Best_mIoU:  0.7130
[0.97462292 0.79988643 0.91424784 0.45421569 0.54537305 0.62569339
 0.65007501 0.76472627 0.91417659 0.538014   0.94336688 0.79663282
 0.54362445 0.933594   0.60476243 0.69623281 0.53693611 0.55426821
 0.73983847]
Epoch: [355/484] Iter:[0/24], Time: 0.95, lr: [0.0030420749772119954], Loss: 0.105647
Epoch: [355/484] Iter:[10/24], Time: 0.44, lr: [0.0030332303061328474], Loss: 0.129956
Epoch: [355/484] Iter:[20/24], Time: 0.42, lr: [0.0030243827685208016], Loss: 0.128682
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97429826 0.79971937 0.91091799 0.36530149 0.5096153  0.60748988
 0.63975549 0.75518563 0.91283543 0.53122975 0.94103402 0.79114555
 0.52777077 0.92555003 0.53252614 0.60254523 0.47141458 0.51842871
 0.73410981] 0.686888075225331
1 [0.97490388 0.80484545 0.91250838 0.3979966  0.53201792 0.61294255
 0.65435789 0.76209689 0.91433239 0.55314145 0.94084162 0.79735712
 0.54614025 0.93210623 0.62651662 0.6850337  0.51511353 0.53734164
 0.74060014] 0.7073786453525642
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7074, Best_mIoU:  0.7130
[0.97490388 0.80484545 0.91250838 0.3979966  0.53201792 0.61294255
 0.65435789 0.76209689 0.91433239 0.55314145 0.94084162 0.79735712
 0.54614025 0.93210623 0.62651662 0.6850337  0.51511353 0.53734164
 0.74060014]
Epoch: [356/484] Iter:[0/24], Time: 0.96, lr: [0.003020842948552534], Loss: 0.143921
Epoch: [356/484] Iter:[10/24], Time: 0.44, lr: [0.0030119913805542985], Loss: 0.116984
Epoch: [356/484] Iter:[20/24], Time: 0.42, lr: [0.0030031369212986897], Loss: 0.124045
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97366154 0.80142452 0.91096493 0.34503269 0.49829863 0.60886476
 0.63555705 0.75440457 0.91287109 0.52648275 0.94176348 0.78922389
 0.53346243 0.9231105  0.52317027 0.58844681 0.4749347  0.52086212
 0.72931205] 0.683781513903218
1 [0.97430369 0.80571219 0.91254259 0.3582429  0.52050553 0.61612632
 0.64584665 0.7622059  0.91425502 0.54690388 0.94284848 0.79463977
 0.54861241 0.92933555 0.60942435 0.67030355 0.54803727 0.54648293
 0.73627791] 0.7043477311513906
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7043, Best_mIoU:  0.7130
[0.97430369 0.80571219 0.91254259 0.3582429  0.52050553 0.61612632
 0.64584665 0.7622059  0.91425502 0.54690388 0.94284848 0.79463977
 0.54861241 0.92933555 0.60942435 0.67030355 0.54803727 0.54648293
 0.73627791]
Epoch: [357/484] Iter:[0/24], Time: 0.94, lr: [0.002999594325712176], Loss: 0.141849
Epoch: [357/484] Iter:[10/24], Time: 0.44, lr: [0.002990735801170526], Loss: 0.137941
Epoch: [357/484] Iter:[20/24], Time: 0.42, lr: [0.0029818743602366914], Loss: 0.129323
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97387734 0.79985791 0.90833044 0.33977345 0.49469796 0.60690059
 0.64127978 0.75702861 0.91130627 0.53209517 0.94062474 0.78955305
 0.55067759 0.92196151 0.5217295  0.56001116 0.48302907 0.52855016
 0.72969436] 0.6837357195043634
1 [0.97384317 0.80191349 0.90849518 0.34069835 0.50877234 0.61501555
 0.65335267 0.76408237 0.91258194 0.55213994 0.9413956  0.79210711
 0.55468437 0.92770598 0.58685751 0.63911217 0.53422455 0.55084843
 0.73563971] 0.6996563384531457
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.6997, Best_mIoU:  0.7130
[0.97384317 0.80191349 0.90849518 0.34069835 0.50877234 0.61501555
 0.65335267 0.76408237 0.91258194 0.55213994 0.9413956  0.79210711
 0.55468437 0.92770598 0.58685751 0.63911217 0.53422455 0.55084843
 0.73563971]
Epoch: [358/484] Iter:[0/24], Time: 0.96, lr: [0.002978328964902235], Loss: 0.116717
Epoch: [358/484] Iter:[10/24], Time: 0.44, lr: [0.002969463423200991], Loss: 0.121009
Epoch: [358/484] Iter:[20/24], Time: 0.42, lr: [0.002960594939552276], Loss: 0.138169
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9726593  0.79037757 0.91284058 0.38010758 0.51006466 0.61359615
 0.64428545 0.75897903 0.91327434 0.5430218  0.94139715 0.78638076
 0.52858185 0.92232278 0.53232861 0.56487155 0.49091747 0.51891233
 0.73136086] 0.6871726228677415
1 [0.97279324 0.79464243 0.91419152 0.39615004 0.53611847 0.62116163
 0.65694818 0.7665317  0.91511525 0.56908737 0.94325724 0.79252392
 0.54304464 0.93005807 0.61811338 0.67489863 0.55555466 0.51775281
 0.73635392] 0.708120899002115
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7081, Best_mIoU:  0.7130
[0.97279324 0.79464243 0.91419152 0.39615004 0.53611847 0.62116163
 0.65694818 0.7665317  0.91511525 0.56908737 0.94325724 0.79252392
 0.54304464 0.93005807 0.61811338 0.67489863 0.55555466 0.51775281
 0.73635392]
Epoch: [359/484] Iter:[0/24], Time: 0.98, lr: [0.0029570467199365176], Loss: 0.123729
Epoch: [359/484] Iter:[10/24], Time: 0.44, lr: [0.002948174099443095], Loss: 0.104098
Epoch: [359/484] Iter:[20/24], Time: 0.42, lr: [0.0029392985110159777], Loss: 0.112272
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97155334 0.78598477 0.91125619 0.37123555 0.48790235 0.61504917
 0.63871075 0.75179273 0.91393734 0.54744447 0.94122266 0.79075119
 0.52945032 0.92025183 0.52659668 0.53416298 0.48109723 0.55812458
 0.73273928] 0.6846980746274073
1 [0.97149607 0.78792217 0.91318716 0.40093289 0.51363072 0.62310607
 0.6537035  0.76177781 0.9159509  0.56342764 0.94229271 0.79686681
 0.5461806  0.92761665 0.60920592 0.63950326 0.56433076 0.56820975
 0.73981284] 0.7073239062302396
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.257, MeanIU:  0.7073, Best_mIoU:  0.7130
[0.97149607 0.78792217 0.91318716 0.40093289 0.51363072 0.62310607
 0.6537035  0.76177781 0.9159509  0.56342764 0.94229271 0.79686681
 0.5461806  0.92761665 0.60920592 0.63950326 0.56433076 0.56820975
 0.73981284]
Epoch: [360/484] Iter:[0/24], Time: 0.99, lr: [0.0029357474421718467], Loss: 0.100895
Epoch: [360/484] Iter:[10/24], Time: 0.45, lr: [0.0029268676802118943], Loss: 0.125881
Epoch: [360/484] Iter:[20/24], Time: 0.42, lr: [0.0029179849238902696], Loss: 0.123253
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97329555 0.79389945 0.91065099 0.34516925 0.47937204 0.61846903
 0.64387384 0.75766836 0.91416369 0.54847871 0.93989765 0.792272
 0.54114392 0.91936819 0.52018475 0.53730003 0.47247833 0.55542488
 0.73881295] 0.6843117676774427
1 [0.97314893 0.79536825 0.91213898 0.37071345 0.50167131 0.6246915
 0.65607724 0.76604526 0.91570278 0.56513829 0.94214242 0.79862803
 0.55434803 0.92541402 0.58903117 0.63269546 0.56035318 0.57179802
 0.74635546] 0.7053400943309667
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.252, MeanIU:  0.7053, Best_mIoU:  0.7130
[0.97314893 0.79536825 0.91213898 0.37071345 0.50167131 0.6246915
 0.65607724 0.76604526 0.91570278 0.56513829 0.94214242 0.79862803
 0.55434803 0.92541402 0.58903117 0.63269546 0.56035318 0.57179802
 0.74635546]
Epoch: [361/484] Iter:[0/24], Time: 0.97, lr: [0.002914430980446603], Loss: 0.135165
Epoch: [361/484] Iter:[10/24], Time: 0.44, lr: [0.0029055440132778015], Loss: 0.124333
Epoch: [361/484] Iter:[20/24], Time: 0.42, lr: [0.002896654024866418], Loss: 0.126802
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97277933 0.78873891 0.91091313 0.3787918  0.48639117 0.60908074
 0.63955204 0.75643598 0.91319471 0.52026116 0.94006406 0.78553791
 0.53876155 0.92217869 0.52776156 0.55470241 0.47008315 0.55902848
 0.73653294] 0.6847784058224496
1 [0.97272521 0.79131644 0.91240611 0.40854383 0.5068285  0.61523859
 0.65303623 0.7641309  0.91468545 0.53665182 0.94088198 0.78926266
 0.54892527 0.92819532 0.5963574  0.65469268 0.53803859 0.57823235
 0.74214336] 0.7048575097102783
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.7049, Best_mIoU:  0.7130
[0.97272521 0.79131644 0.91240611 0.40854383 0.5068285  0.61523859
 0.65303623 0.7641309  0.91468545 0.53665182 0.94088198 0.78926266
 0.54892527 0.92819532 0.5963574  0.65469268 0.53803859 0.57823235
 0.74214336]
Epoch: [362/484] Iter:[0/24], Time: 0.96, lr: [0.0028930971810172297], Loss: 0.183178
Epoch: [362/484] Iter:[10/24], Time: 0.44, lr: [0.0028842029438022], Loss: 0.126640
Epoch: [362/484] Iter:[20/24], Time: 0.42, lr: [0.002875305657999198], Loss: 0.130279
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97394418 0.80004858 0.91093681 0.36879802 0.48618436 0.61199064
 0.6321217  0.75373415 0.91163902 0.55257227 0.93786026 0.79143024
 0.53627549 0.92573786 0.52390551 0.60488814 0.45855408 0.53174646
 0.73094167] 0.6864899699436267
1 [0.97404201 0.80374059 0.91231971 0.39305089 0.50451395 0.61800792
 0.64873738 0.76107977 0.91317176 0.57791747 0.93755924 0.7965272
 0.55345668 0.93176603 0.58038416 0.66620881 0.52994347 0.55415236
 0.73532493] 0.7048370700703731
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.251, MeanIU:  0.7048, Best_mIoU:  0.7130
[0.97404201 0.80374059 0.91231971 0.39305089 0.50451395 0.61800792
 0.64873738 0.76107977 0.91317176 0.57791747 0.93755924 0.7965272
 0.55345668 0.93176603 0.58038416 0.66620881 0.52994347 0.55415236
 0.73532493]
Epoch: [363/484] Iter:[0/24], Time: 0.94, lr: [0.0028717458874925874], Loss: 0.152277
Epoch: [363/484] Iter:[10/24], Time: 0.44, lr: [0.0028628443142709008], Loss: 0.119199
Epoch: [363/484] Iter:[20/24], Time: 0.42, lr: [0.002853939664639422], Loss: 0.116104
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9746509  0.80222718 0.91146639 0.37086946 0.47989154 0.61419612
 0.63966817 0.75728954 0.91371882 0.53958562 0.94129762 0.79029943
 0.54439094 0.92359423 0.51061996 0.60422028 0.44502687 0.54397387
 0.73348043] 0.6863403876492836
1 [0.97469906 0.8053463  0.91213801 0.38334262 0.49395088 0.62094956
 0.64880739 0.76540994 0.91467453 0.55300319 0.94189177 0.79422074
 0.55270186 0.92951394 0.56713114 0.66658936 0.44747239 0.56966473
 0.73840021] 0.6989425074248682
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.249, MeanIU:  0.6989, Best_mIoU:  0.7130
[0.97469906 0.8053463  0.91213801 0.38334262 0.49395088 0.62094956
 0.64880739 0.76540994 0.91467453 0.55300319 0.94189177 0.79422074
 0.55270186 0.92951394 0.56713114 0.66658936 0.44747239 0.56966473
 0.73840021]
Epoch: [364/484] Iter:[0/24], Time: 0.96, lr: [0.0028503769407661034], Loss: 0.119224
Epoch: [364/484] Iter:[10/24], Time: 0.44, lr: [0.002841467964425318], Loss: 0.112474
Epoch: [364/484] Iter:[20/24], Time: 0.42, lr: [0.002832555883364136], Loss: 0.116875
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97416089 0.80040619 0.91170331 0.37306811 0.49026284 0.6164256
 0.64335528 0.75794078 0.91296516 0.53639345 0.94212806 0.7884568
 0.533092   0.92208772 0.5220514  0.60394603 0.47476068 0.54688463
 0.73419119] 0.6886463210165311
1 [0.97440401 0.80443774 0.91323549 0.39812247 0.51111228 0.62347653
 0.65583928 0.76647978 0.91422788 0.54850887 0.94305745 0.79453679
 0.54733808 0.92906802 0.58937546 0.68126843 0.51973311 0.57369053
 0.7378857 ] 0.7066209404022008
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.7066, Best_mIoU:  0.7130
[0.97440401 0.80443774 0.91323549 0.39812247 0.51111228 0.62347653
 0.65583928 0.76647978 0.91422788 0.54850887 0.94305745 0.79453679
 0.54733808 0.92906802 0.58937546 0.68126843 0.51973311 0.57369053
 0.7378857 ]
Epoch: [365/484] Iter:[0/24], Time: 0.98, lr: [0.0028289901789455836], Loss: 0.190716
Epoch: [365/484] Iter:[10/24], Time: 0.44, lr: [0.0028200737311913], Loss: 0.120601
Epoch: [365/484] Iter:[20/24], Time: 0.42, lr: [0.002811154149904444], Loss: 0.126672
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97391716 0.79646564 0.91278004 0.40323418 0.49191925 0.61946046
 0.64452062 0.75922482 0.913691   0.53865032 0.94211847 0.78979237
 0.52877669 0.92219536 0.51110121 0.60032039 0.47840804 0.5432494
 0.73782326] 0.689876247444324
1 [0.97403252 0.79988836 0.91347521 0.42810261 0.51407118 0.62576202
 0.65832506 0.76728792 0.91501404 0.54854924 0.94245807 0.79638278
 0.54583771 0.92838867 0.55994117 0.67462388 0.51778391 0.57087527
 0.74237402] 0.7064828225282321
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7065, Best_mIoU:  0.7130
[0.97403252 0.79988836 0.91347521 0.42810261 0.51407118 0.62576202
 0.65832506 0.76728792 0.91501404 0.54854924 0.94245807 0.79638278
 0.54583771 0.92838867 0.55994117 0.67462388 0.51778391 0.57087527
 0.74237402]
Epoch: [366/484] Iter:[0/24], Time: 0.97, lr: [0.002807585437280611], Loss: 0.118302
Epoch: [366/484] Iter:[10/24], Time: 0.44, lr: [0.0027986614486054796], Loss: 0.114813
Epoch: [366/484] Iter:[20/24], Time: 0.42, lr: [0.0027897342970708066], Loss: 0.123700
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96827142 0.77368718 0.91006252 0.36741767 0.4830212  0.6194626
 0.63950595 0.75624817 0.91320764 0.54560529 0.94160114 0.78756559
 0.52995205 0.92116366 0.52074798 0.59037142 0.45114041 0.52454469
 0.7379713 ] 0.6832393606654925
1 [0.96778594 0.77398098 0.9100378  0.35874884 0.50125212 0.62756982
 0.65337369 0.76559607 0.91382783 0.55580874 0.94032971 0.79322916
 0.54848263 0.92763282 0.58223272 0.66645164 0.48111733 0.54772651
 0.74389098] 0.6978460703787368
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.265, MeanIU:  0.6978, Best_mIoU:  0.7130
[0.96778594 0.77398098 0.9100378  0.35874884 0.50125212 0.62756982
 0.65337369 0.76559607 0.91382783 0.55580874 0.94032971 0.79322916
 0.54848263 0.92763282 0.58223272 0.66645164 0.48111733 0.54772651
 0.74389098]
Epoch: [367/484] Iter:[0/24], Time: 0.98, lr: [0.0027861625480874288], Loss: 0.142231
Epoch: [367/484] Iter:[10/24], Time: 0.44, lr: [0.002777230947739079], Loss: 0.135589
Epoch: [367/484] Iter:[20/24], Time: 0.42, lr: [0.0027682961546757244], Loss: 0.125023
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97025806 0.7817234  0.91007706 0.34320573 0.47767214 0.62092105
 0.64257224 0.7600156  0.91272652 0.54283893 0.94334427 0.79111563
 0.54050366 0.92170777 0.51883613 0.59136215 0.44281848 0.54505035
 0.73500013] 0.6837762795509468
1 [0.96987231 0.78141828 0.90878309 0.29012648 0.48532031 0.62788423
 0.65366795 0.76818865 0.91246122 0.55135706 0.94396014 0.79696548
 0.55861347 0.92465733 0.56129527 0.63772678 0.43779499 0.57251736
 0.74067593] 0.6906992805870453
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.259, MeanIU:  0.6907, Best_mIoU:  0.7130
[0.96987231 0.78141828 0.90878309 0.29012648 0.48532031 0.62788423
 0.65366795 0.76818865 0.91246122 0.55135706 0.94396014 0.79696548
 0.55861347 0.92465733 0.56129527 0.63772678 0.43779499 0.57251736
 0.74067593]
Epoch: [368/484] Iter:[0/24], Time: 0.95, lr: [0.0027647213406711718], Loss: 0.077701
Epoch: [368/484] Iter:[10/24], Time: 0.45, lr: [0.0027557820566190094], Loss: 0.111242
Epoch: [368/484] Iter:[20/24], Time: 0.42, lr: [0.0027468395494537087], Loss: 0.112792
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97065835 0.78201806 0.91252851 0.37207936 0.49673117 0.61708229
 0.64413057 0.75934555 0.91139911 0.51560698 0.94070287 0.79055588
 0.53243774 0.92272741 0.52240787 0.57726376 0.46436545 0.55325658
 0.73589485] 0.6853259138258813
1 [0.97061275 0.7837743  0.91317252 0.3606122  0.51885985 0.62619893
 0.6580487  0.76645761 0.91217319 0.53478815 0.94033147 0.79569148
 0.5518567  0.93004211 0.5948002  0.65387457 0.46491596 0.58279949
 0.74058847] 0.6999788748823115
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.256, MeanIU:  0.7000, Best_mIoU:  0.7130
[0.97061275 0.7837743  0.91317252 0.3606122  0.51885985 0.62619893
 0.6580487  0.76645761 0.91217319 0.53478815 0.94033147 0.79569148
 0.5518567  0.93004211 0.5948002  0.65387457 0.46491596 0.58279949
 0.74058847]
Epoch: [369/484] Iter:[0/24], Time: 0.99, lr: [0.0027432616412453713], Loss: 0.107040
Epoch: [369/484] Iter:[10/24], Time: 0.45, lr: [0.00273431460014619], Loss: 0.126410
Epoch: [369/484] Iter:[20/24], Time: 0.42, lr: [0.0027253643049783755], Loss: 0.124191
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97295226 0.78847272 0.91099824 0.35851102 0.50345235 0.59898581
 0.63833025 0.75531366 0.91015254 0.50221668 0.94095825 0.78964561
 0.54388079 0.92123681 0.5212308  0.57919792 0.4906773  0.53902395
 0.73677304] 0.6843163166256092
1 [0.97284722 0.79167085 0.91159078 0.34032218 0.52375247 0.60527784
 0.64618315 0.76317302 0.91080775 0.52062255 0.94116441 0.79491814
 0.55709913 0.92652733 0.58130784 0.66086703 0.56499535 0.56412439
 0.74300802] 0.7010662868838009
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.255, MeanIU:  0.7011, Best_mIoU:  0.7130
[0.97284722 0.79167085 0.91159078 0.34032218 0.52375247 0.60527784
 0.64618315 0.76317302 0.91080775 0.52062255 0.94116441 0.79491814
 0.55709913 0.92652733 0.58130784 0.66086703 0.56499535 0.56412439
 0.74300802]
Epoch: [370/484] Iter:[0/24], Time: 0.99, lr: [0.0027217832728485642], Loss: 0.113405
Epoch: [370/484] Iter:[10/24], Time: 0.45, lr: [0.002712828400010931], Loss: 0.120402
Epoch: [370/484] Iter:[20/24], Time: 0.42, lr: [0.0027038702415765887], Loss: 0.121633
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97345349 0.79055399 0.91450126 0.39468985 0.53509685 0.62016815
 0.64280613 0.75666937 0.91160191 0.49729242 0.94191988 0.79112319
 0.5434532  0.92079207 0.51622524 0.56981753 0.48385349 0.56440593
 0.7408783 ] 0.6899632761159576
1 [0.97363299 0.79526345 0.9164609  0.43034853 0.5583736  0.62698722
 0.65459322 0.76470316 0.91292101 0.51402274 0.94255502 0.79722457
 0.55829648 0.92678315 0.55881572 0.64817584 0.561677   0.58804685
 0.74626952] 0.7092184728914344
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7092, Best_mIoU:  0.7130
[0.97363299 0.79526345 0.9164609  0.43034853 0.5583736  0.62698722
 0.65459322 0.76470316 0.91292101 0.51402274 0.94255502 0.79722457
 0.55829648 0.92678315 0.55881572 0.64817584 0.561677   0.58804685
 0.74626952]
Epoch: [371/484] Iter:[0/24], Time: 0.98, lr: [0.0027002860552579225], Loss: 0.138708
Epoch: [371/484] Iter:[10/24], Time: 0.44, lr: [0.0026913232746052624], Loss: 0.112531
Epoch: [371/484] Iter:[20/24], Time: 0.42, lr: [0.002682357176239467], Loss: 0.116052
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97435177 0.79668061 0.91302355 0.37966367 0.52577764 0.62096996
 0.64392601 0.75612215 0.91238759 0.50729021 0.93959001 0.78832718
 0.53153769 0.91893    0.5139502  0.55804305 0.4785321  0.55680962
 0.74092459] 0.6872019797851143
1 [0.97455862 0.80082053 0.91412071 0.39245933 0.55103143 0.62795152
 0.65414339 0.76219167 0.91351012 0.52940501 0.93868962 0.79477443
 0.55154494 0.92352577 0.54823928 0.62004206 0.57150938 0.58402592
 0.74722157] 0.7052508051387641
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7053, Best_mIoU:  0.7130
[0.97455862 0.80082053 0.91412071 0.39245933 0.55103143 0.62795152
 0.65414339 0.76219167 0.91351012 0.52940501 0.93868962 0.79477443
 0.55154494 0.92352577 0.54823928 0.62004206 0.57150938 0.58402592
 0.74722157]
Epoch: [372/484] Iter:[0/24], Time: 0.97, lr: [0.002678769804899729], Loss: 0.085248
Epoch: [372/484] Iter:[10/24], Time: 0.44, lr: [0.002669799038932076], Loss: 0.107611
Epoch: [372/484] Iter:[20/24], Time: 0.42, lr: [0.0026608249225301517], Loss: 0.114981
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97369588 0.79510988 0.91486052 0.37681589 0.51916181 0.61580092
 0.64649773 0.76105139 0.91352403 0.5287665  0.94147146 0.78856929
 0.53376462 0.92019325 0.51814587 0.55378289 0.41217908 0.56334464
 0.73464015] 0.6848092529506516
1 [0.97367473 0.79812028 0.91554276 0.37206336 0.53619763 0.62388808
 0.6586487  0.76912086 0.91452507 0.55583464 0.94352328 0.79280334
 0.54751695 0.92429176 0.56454369 0.58658776 0.39012342 0.58657949
 0.73856932] 0.6943239548279687
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.6943, Best_mIoU:  0.7130
[0.97367473 0.79812028 0.91554276 0.37206336 0.53619763 0.62388808
 0.6586487  0.76912086 0.91452507 0.55583464 0.94352328 0.79280334
 0.54751695 0.92429176 0.56454369 0.58658776 0.39012342 0.58657949
 0.73856932]
Epoch: [373/484] Iter:[0/24], Time: 0.95, lr: [0.0026572343347565936], Loss: 0.088805
Epoch: [373/484] Iter:[10/24], Time: 0.44, lr: [0.002648255504510915], Loss: 0.124984
Epoch: [373/484] Iter:[20/24], Time: 0.42, lr: [0.0026392732904881684], Loss: 0.137450
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97352855 0.79575877 0.90962532 0.36587033 0.50617883 0.60658067
 0.64387726 0.76026803 0.9121493  0.5339813  0.93874707 0.79291318
 0.54806153 0.92270355 0.52169853 0.57351459 0.42241106 0.55008133
 0.73932099] 0.6851194841761798
1 [0.97373027 0.80038296 0.91065101 0.37779291 0.52807518 0.61394273
 0.65720214 0.76740628 0.9133846  0.56092381 0.94056117 0.79831719
 0.56147427 0.92806213 0.56781815 0.63460964 0.451403   0.56266401
 0.74320058] 0.699558000648
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.6996, Best_mIoU:  0.7130
[0.97373027 0.80038296 0.91065101 0.37779291 0.52807518 0.61394273
 0.65720214 0.76740628 0.9133846  0.56092381 0.94056117 0.79831719
 0.56147427 0.92806213 0.56781815 0.63460964 0.451403   0.56266401
 0.74320058]
Epoch: [374/484] Iter:[0/24], Time: 0.99, lr: [0.002635679454271223], Loss: 0.197877
Epoch: [374/484] Iter:[10/24], Time: 0.44, lr: [0.0026266924792802825], Loss: 0.132716
Epoch: [374/484] Iter:[20/24], Time: 0.42, lr: [0.0026177020865302135], Loss: 0.128768
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9746931  0.80361856 0.91047627 0.36296332 0.50754705 0.61677819
 0.64532564 0.75328955 0.91152865 0.548156   0.94175767 0.78660407
 0.522659   0.92360624 0.51457812 0.58194126 0.41973183 0.56375363
 0.73902967] 0.6856862004908575
1 [0.97464465 0.80671822 0.91125827 0.36045472 0.52682733 0.62492694
 0.6598498  0.76233237 0.91207267 0.56740788 0.94262422 0.79292742
 0.54617288 0.93006187 0.61014444 0.65527179 0.44503889 0.58394674
 0.74268171] 0.702913832402794
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7029, Best_mIoU:  0.7130
[0.97464465 0.80671822 0.91125827 0.36045472 0.52682733 0.62492694
 0.6598498  0.76233237 0.91207267 0.56740788 0.94262422 0.79292742
 0.54617288 0.93006187 0.61014444 0.65527179 0.44503889 0.58394674
 0.74268171]
Epoch: [375/484] Iter:[0/24], Time: 0.96, lr: [0.0026141049692466095], Loss: 0.148443
Epoch: [375/484] Iter:[10/24], Time: 0.44, lr: [0.0026051097674962775], Loss: 0.113123
Epoch: [375/484] Iter:[20/24], Time: 0.42, lr: [0.002596111113347242], Loss: 0.118270
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9738699  0.79584438 0.91377725 0.36505941 0.52009558 0.6213873
 0.64544499 0.7595279  0.91338228 0.5455692  0.94149762 0.79099924
 0.54205975 0.92307202 0.51843323 0.57799769 0.43189839 0.56184818
 0.73603408] 0.6883051776408682
1 [0.97388267 0.79934431 0.91493362 0.36580735 0.53985703 0.62735047
 0.65499213 0.76600485 0.9147757  0.56766082 0.94193681 0.79722609
 0.55782257 0.93039681 0.60380043 0.65675059 0.46633168 0.58445221
 0.74067478] 0.7054737333640482
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7055, Best_mIoU:  0.7130
[0.97388267 0.79934431 0.91493362 0.36580735 0.53985703 0.62735047
 0.65499213 0.76600485 0.9147757  0.56766082 0.94193681 0.79722609
 0.55782257 0.93039681 0.60380043 0.65675059 0.46633168 0.58445221
 0.74067478]
Epoch: [376/484] Iter:[0/24], Time: 0.99, lr: [0.0025925106817424677], Loss: 0.091299
Epoch: [376/484] Iter:[10/24], Time: 0.45, lr: [0.0025835071696274185], Loss: 0.102631
Epoch: [376/484] Iter:[20/24], Time: 0.42, lr: [0.0025745001697976156], Loss: 0.115531
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97248971 0.78669432 0.91306096 0.37212624 0.51509912 0.62146907
 0.64547088 0.75572112 0.9140848  0.54192215 0.94302648 0.79035505
 0.53395431 0.92604113 0.5210176  0.5875143  0.41348553 0.5651105
 0.74182183] 0.6873928992397231
1 [0.97267481 0.79230387 0.91460327 0.39271527 0.53821175 0.62959353
 0.65601968 0.76437151 0.91546171 0.56506178 0.9440027  0.79622237
 0.55166736 0.93224492 0.62647365 0.6569132  0.42735492 0.58764304
 0.746807  ] 0.7058077024321443
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7058, Best_mIoU:  0.7130
[0.97267481 0.79230387 0.91460327 0.39271527 0.53821175 0.62959353
 0.65601968 0.76437151 0.91546171 0.56506178 0.9440027  0.79622237
 0.55166736 0.93224492 0.62647365 0.6569132  0.42735492 0.58764304
 0.746807  ]
Epoch: [377/484] Iter:[0/24], Time: 0.98, lr: [0.002570896389967727], Loss: 0.200487
Epoch: [377/484] Iter:[10/24], Time: 0.45, lr: [0.0025618844822454445], Loss: 0.135942
Epoch: [377/484] Iter:[20/24], Time: 0.42, lr: [0.0025528690507961964], Loss: 0.127464
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97409682 0.79411974 0.91308647 0.36694564 0.50672859 0.61878438
 0.64578723 0.75906136 0.91350153 0.54557974 0.94130989 0.78900869
 0.53904863 0.92306856 0.51557634 0.58489463 0.43049341 0.56097113
 0.73992282] 0.6874729260626499
1 [0.97402961 0.79751705 0.91410248 0.3759698  0.52251504 0.62602554
 0.65764574 0.76754184 0.91490586 0.56582399 0.94156282 0.79535862
 0.55431382 0.92854872 0.59158691 0.65358776 0.45720821 0.5847454
 0.74534915] 0.7035967549575478
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7036, Best_mIoU:  0.7130
[0.97402961 0.79751705 0.91410248 0.3759698  0.52251504 0.62602554
 0.65764574 0.76754184 0.91490586 0.56582399 0.94156282 0.79535862
 0.55431382 0.92854872 0.59158691 0.65358776 0.45720821 0.5847454
 0.74534915]
Epoch: [378/484] Iter:[0/24], Time: 0.98, lr: [0.002549261888168915], Loss: 0.152757
Epoch: [378/484] Iter:[10/24], Time: 0.45, lr: [0.0025402414979119113], Loss: 0.122775
Epoch: [378/484] Iter:[20/24], Time: 0.42, lr: [0.002531217547199125], Loss: 0.123494
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97387599 0.79269378 0.90923293 0.34738469 0.49483578 0.61193833
 0.64283301 0.76309161 0.91252868 0.53623809 0.94283896 0.79244671
 0.531437   0.92322928 0.51543707 0.58201079 0.38715199 0.56476159
 0.73981484] 0.6823042697599091
1 [0.97400326 0.79599775 0.9092063  0.34179911 0.50838847 0.62022325
 0.65724285 0.7709574  0.91359478 0.55161554 0.94356999 0.79899864
 0.55264421 0.9301587  0.60316521 0.63490328 0.3542847  0.59400679
 0.74696569] 0.6948276800038439
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.253, MeanIU:  0.6948, Best_mIoU:  0.7130
[0.97400326 0.79599775 0.9092063  0.34179911 0.50838847 0.62022325
 0.65724285 0.7709574  0.91359478 0.55161554 0.94356999 0.79899864
 0.55264421 0.9301587  0.60316521 0.63490328 0.3542847  0.59400679
 0.74696569]
Epoch: [379/484] Iter:[0/24], Time: 0.94, lr: [0.0025276069665142067], Loss: 0.137086
Epoch: [379/484] Iter:[10/24], Time: 0.44, lr: [0.002518578005060399], Loss: 0.120506
Epoch: [379/484] Iter:[20/24], Time: 0.42, lr: [0.0025095454456841287], Loss: 0.123042
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97449297 0.79258525 0.91291685 0.34621785 0.50606865 0.61701149
 0.65074253 0.7614336  0.91400343 0.53787096 0.94134338 0.79446024
 0.54749644 0.9220121  0.51468447 0.57975921 0.44525602 0.55897597
 0.73095314] 0.6867518197751947
1 [0.97490953 0.79692949 0.91365202 0.35609215 0.52135761 0.62297292
 0.65773437 0.76801191 0.9155401  0.55590601 0.94130502 0.79785352
 0.56031769 0.92781779 0.58869373 0.64646284 0.4768     0.57811663
 0.73622186] 0.701931326094482
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.7019, Best_mIoU:  0.7130
[0.97490953 0.79692949 0.91365202 0.35609215 0.52135761 0.62297292
 0.65773437 0.76801191 0.9155401  0.55590601 0.94130502 0.79785352
 0.56031769 0.92781779 0.58869373 0.64646284 0.4768     0.57811663
 0.73622186]
Epoch: [380/484] Iter:[0/24], Time: 0.97, lr: [0.002505931410972959], Loss: 0.072914
Epoch: [380/484] Iter:[10/24], Time: 0.44, lr: [0.0024968937878740737], Loss: 0.117741
Epoch: [380/484] Iter:[20/24], Time: 0.42, lr: [0.002487852528626094], Loss: 0.121804
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97407645 0.79545685 0.91241579 0.33833776 0.51419649 0.62332097
 0.65108972 0.7597318  0.91455953 0.54566526 0.94170756 0.78909597
 0.53322042 0.92272207 0.50228562 0.57780975 0.42055696 0.56445547
 0.74044513] 0.6853236621410405
1 [0.97449021 0.79905191 0.91239278 0.32041013 0.53034232 0.62838602
 0.65905437 0.76711859 0.91504951 0.56363144 0.94204458 0.79390807
 0.55628185 0.92637546 0.55236979 0.63168765 0.47290593 0.58646628
 0.74618514] 0.6988501068672298
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.6989, Best_mIoU:  0.7130
[0.97449021 0.79905191 0.91239278 0.32041013 0.53034232 0.62838602
 0.65905437 0.76711859 0.91504951 0.56363144 0.94204458 0.79390807
 0.55628185 0.92637546 0.55236979 0.63168765 0.47290593 0.58646628
 0.74618514]
Epoch: [381/484] Iter:[0/24], Time: 0.99, lr: [0.0024842350031904843], Loss: 0.127507
Epoch: [381/484] Iter:[10/24], Time: 0.45, lr: [0.002475188626158427], Loss: 0.132937
Epoch: [381/484] Iter:[20/24], Time: 0.42, lr: [0.0024661385739677124], Loss: 0.129552
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97365175 0.79200569 0.91279112 0.38243354 0.50248498 0.6203251
 0.6430634  0.76099241 0.91412343 0.53917141 0.94251509 0.79177602
 0.55173867 0.92293077 0.52006156 0.57226428 0.42175628 0.56544465
 0.73993973] 0.6878668355937421
1 [0.97338369 0.79504105 0.91381592 0.40814608 0.52356472 0.62647242
 0.65413156 0.76737152 0.91521757 0.55625172 0.94395573 0.79704421
 0.5640516  0.92754788 0.58636876 0.63332403 0.4371491  0.58634293
 0.74727078] 0.7029711196570739
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7030, Best_mIoU:  0.7130
[0.97338369 0.79504105 0.91381592 0.40814608 0.52356472 0.62647242
 0.65413156 0.76737152 0.91521757 0.55625172 0.94395573 0.79704421
 0.5640516  0.92754788 0.58636876 0.63332403 0.4371491  0.58634293
 0.74727078]
Epoch: [382/484] Iter:[0/24], Time: 0.97, lr: [0.002462517520357832], Loss: 0.130052
Epoch: [382/484] Iter:[10/24], Time: 0.45, lr: [0.0024534622952089007], Loss: 0.112342
Epoch: [382/484] Iter:[20/24], Time: 0.42, lr: [0.0024444033550849176], Loss: 0.116984
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97462957 0.79526947 0.91273622 0.37478977 0.50939948 0.61755715
 0.64684347 0.76214626 0.91361661 0.54299375 0.94048118 0.7893251
 0.53267116 0.9230416  0.52406671 0.57322471 0.42272545 0.56393839
 0.73622285] 0.6871409946884723
1 [0.97470404 0.79951831 0.91369222 0.39525475 0.52834801 0.62388477
 0.65774151 0.76861477 0.91479027 0.56269922 0.94122511 0.79565573
 0.5535073  0.92817193 0.60138555 0.63496658 0.45529372 0.58873663
 0.74316963] 0.7042821070686361
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7043, Best_mIoU:  0.7130
[0.97470404 0.79951831 0.91369222 0.39525475 0.52834801 0.62388477
 0.65774151 0.76861477 0.91479027 0.56269922 0.94122511 0.79565573
 0.5535073  0.92817193 0.60138555 0.63496658 0.45529372 0.58873663
 0.74316963]
Epoch: [383/484] Iter:[0/24], Time: 0.96, lr: [0.002440778735076349], Loss: 0.098872
Epoch: [383/484] Iter:[10/24], Time: 0.44, lr: [0.002431714565673189], Loss: 0.125739
Epoch: [383/484] Iter:[20/24], Time: 0.42, lr: [0.0024226466406468672], Loss: 0.131047
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97443944 0.79029258 0.91496833 0.39130511 0.53000164 0.60930987
 0.64755849 0.75954413 0.91307929 0.52299709 0.94029941 0.7926268
 0.53090775 0.92215572 0.51528054 0.56622837 0.43355457 0.55956859
 0.73131142] 0.6866015346325088
1 [0.97467674 0.79564248 0.91673552 0.42269184 0.5468741  0.61816264
 0.66001382 0.76733149 0.91405091 0.53672667 0.94104173 0.79883052
 0.55474691 0.92571109 0.58500138 0.61444471 0.46180215 0.58147746
 0.73751373] 0.7028145202489088
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7028, Best_mIoU:  0.7130
[0.97467674 0.79564248 0.91673552 0.42269184 0.5468741  0.61816264
 0.66001382 0.76733149 0.91405091 0.53672667 0.94104173 0.79883052
 0.55474691 0.92571109 0.58500138 0.61444471 0.46180215 0.58147746
 0.73751373]
Epoch: [384/484] Iter:[0/24], Time: 0.97, lr: [0.0024190184152167193], Loss: 0.137366
Epoch: [384/484] Iter:[10/24], Time: 0.45, lr: [0.0024099452034079044], Loss: 0.146277
Epoch: [384/484] Iter:[20/24], Time: 0.42, lr: [0.002400868194470201], Loss: 0.130827
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
Traceback (most recent call last):
  File "/usr/local/easybuild-2019/easybuild/software/compiler/gcccore/10.2.0/python/3.8.6/lib/python3.8/multiprocessing/queues.py", line 245, in _feed
    send_bytes(obj)
  File "/usr/local/easybuild-2019/easybuild/software/compiler/gcccore/10.2.0/python/3.8.6/lib/python3.8/multiprocessing/connection.py", line 200, in send_bytes
    self._send_bytes(m[offset:offset + size])
  File "/usr/local/easybuild-2019/easybuild/software/compiler/gcccore/10.2.0/python/3.8.6/lib/python3.8/multiprocessing/connection.py", line 411, in _send_bytes
    self._send(header + buf)
  File "/usr/local/easybuild-2019/easybuild/software/compiler/gcccore/10.2.0/python/3.8.6/lib/python3.8/multiprocessing/connection.py", line 368, in _send
    n = write(self._handle, buf)
BrokenPipeError: [Errno 32] Broken pipe
0 [0.97364598 0.79165557 0.91377979 0.37149501 0.5159412  0.61379485
 0.64324199 0.76063464 0.91263983 0.51447954 0.94082549 0.79186602
 0.53945933 0.92242818 0.50217954 0.56426973 0.44280582 0.56727555
 0.73533921] 0.68514511876814
1 [0.97363153 0.79569396 0.91526329 0.39967275 0.53430074 0.62168376
 0.65632796 0.76856188 0.912898   0.52500932 0.94156062 0.79754338
 0.56127845 0.92654359 0.56181295 0.62502288 0.49594812 0.59132435
 0.74098103] 0.7023715037361927
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7024, Best_mIoU:  0.7130
[0.97363153 0.79569396 0.91526329 0.39967275 0.53430074 0.62168376
 0.65632796 0.76856188 0.912898   0.52500932 0.94156062 0.79754338
 0.56127845 0.92654359 0.56181295 0.62502288 0.49594812 0.59132435
 0.74098103]
Epoch: [385/484] Iter:[0/24], Time: 0.97, lr: [0.0023972363237722417], Loss: 0.100643
Epoch: [385/484] Iter:[10/24], Time: 0.44, lr: [0.0023881539693293556], Loss: 0.124955
Epoch: [385/484] Iter:[20/24], Time: 0.42, lr: [0.0023790677753672542], Loss: 0.119144
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97515773 0.79995257 0.91350062 0.38863564 0.51994872 0.61956441
 0.64499156 0.7617486  0.9127699  0.5238177  0.9400953  0.79353317
 0.55377278 0.9221247  0.50760982 0.57333282 0.45759842 0.56175432
 0.74087637] 0.6900413236023131
1 [0.97523437 0.80591292 0.91557295 0.44237903 0.54085351 0.62760113
 0.65612769 0.76920091 0.91396793 0.54852504 0.94099401 0.79942673
 0.56747632 0.92725193 0.56100593 0.63846387 0.51085132 0.58795884
 0.74708383] 0.709257275677923
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7093, Best_mIoU:  0.7130
[0.97523437 0.80591292 0.91557295 0.44237903 0.54085351 0.62760113
 0.65612769 0.76920091 0.91396793 0.54852504 0.94099401 0.79942673
 0.56747632 0.92725193 0.56100593 0.63846387 0.51085132 0.58795884
 0.74708383]
Epoch: [386/484] Iter:[0/24], Time: 0.97, lr: [0.0023754322187060036], Loss: 0.131782
Epoch: [386/484] Iter:[10/24], Time: 0.44, lr: [0.002366340619258104], Loss: 0.106795
Epoch: [386/484] Iter:[20/24], Time: 0.42, lr: [0.002357245136987945], Loss: 0.115540
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97483937 0.799112   0.91447985 0.40249013 0.51844686 0.61155647
 0.64520335 0.76064453 0.91360409 0.53499493 0.94208218 0.78825237
 0.55417538 0.91966414 0.50866198 0.56126441 0.45216514 0.5469311
 0.73119778] 0.6884087401417318
1 [0.97486499 0.80408607 0.91696211 0.45011595 0.54018987 0.62035229
 0.65500839 0.7684029  0.91467907 0.55431967 0.94299018 0.78961576
 0.55425467 0.92534629 0.56702045 0.61794316 0.5004147  0.56889538
 0.73670372] 0.7053771377864958
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7054, Best_mIoU:  0.7130
[0.97486499 0.80408607 0.91696211 0.45011595 0.54018987 0.62035229
 0.65500839 0.7684029  0.91467907 0.55431967 0.94299018 0.78961576
 0.55425467 0.92534629 0.56702045 0.61794316 0.5004147  0.56889538
 0.73670372]
Epoch: [387/484] Iter:[0/24], Time: 0.98, lr: [0.0023536058527916827], Loss: 0.070521
Epoch: [387/484] Iter:[10/24], Time: 0.45, lr: [0.0023445049037569892], Loss: 0.119006
Epoch: [387/484] Iter:[20/24], Time: 0.42, lr: [0.0023354000276549656], Loss: 0.117220
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97379176 0.79375806 0.91170799 0.38995638 0.5073212  0.61513812
 0.64428121 0.75817666 0.91328692 0.50972196 0.94107686 0.78759624
 0.53345353 0.91784245 0.49548203 0.56542737 0.47612922 0.54911692
 0.73400323] 0.685119374563574
1 [0.97399637 0.79733238 0.91255224 0.42479835 0.52940124 0.62169423
 0.65763043 0.76574703 0.91483115 0.52293035 0.9411987  0.79412794
 0.55329226 0.9238595  0.54795259 0.61603621 0.51420042 0.57611669
 0.74156175] 0.70153999160671
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.252, MeanIU:  0.7015, Best_mIoU:  0.7130
[0.97399637 0.79733238 0.91255224 0.42479835 0.52940124 0.62169423
 0.65763043 0.76574703 0.91483115 0.52293035 0.9411987  0.79412794
 0.55329226 0.9238595  0.54795259 0.61603621 0.51420042 0.57611669
 0.74156175]
Epoch: [388/484] Iter:[0/24], Time: 1.01, lr: [0.0023317569734475916], Loss: 0.122223
Epoch: [388/484] Iter:[10/24], Time: 0.45, lr: [0.0023226465679622786], Loss: 0.128380
Epoch: [388/484] Iter:[20/24], Time: 0.42, lr: [0.0023135321901919664], Loss: 0.127384
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96542243 0.75273699 0.91038467 0.36040143 0.49789849 0.61503652
 0.64138089 0.75827569 0.91332041 0.5295316  0.94274645 0.78652767
 0.5199903  0.9205979  0.51814699 0.57696394 0.45616093 0.55702785
 0.73119757] 0.6817762477173397
1 [0.96624984 0.75854505 0.91197627 0.39185849 0.51474298 0.62226397
 0.65673387 0.76586252 0.91490211 0.54675667 0.94388209 0.79432915
 0.54028282 0.92507264 0.58100143 0.64371394 0.52203107 0.5764087
 0.73674362] 0.7007030130871652
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.271, MeanIU:  0.7007, Best_mIoU:  0.7130
[0.96624984 0.75854505 0.91197627 0.39185849 0.51474298 0.62226397
 0.65673387 0.76586252 0.91490211 0.54675667 0.94388209 0.79432915
 0.54028282 0.92507264 0.58100143 0.64371394 0.52203107 0.5764087
 0.73674362]
Epoch: [389/484] Iter:[0/24], Time: 0.99, lr: [0.0023098853225636527], Loss: 0.162260
Epoch: [389/484] Iter:[10/24], Time: 0.44, lr: [0.002300765351407561], Loss: 0.143705
Epoch: [389/484] Iter:[20/24], Time: 0.42, lr: [0.002291641361744314], Loss: 0.127490
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96992753 0.77314386 0.91079085 0.35619156 0.49836011 0.61726245
 0.64512605 0.76030838 0.91182595 0.51975839 0.93965325 0.7893234
 0.52822046 0.91636088 0.50688953 0.53049236 0.45527759 0.55281296
 0.73497617] 0.6798264073428816
1 [0.96954787 0.77072844 0.91178528 0.37107735 0.51648186 0.62314124
 0.65807    0.76793334 0.91306844 0.52848334 0.94020394 0.79696459
 0.54966824 0.92203346 0.55920335 0.57838002 0.51409627 0.58177282
 0.74022192] 0.6954137773269619
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.267, MeanIU:  0.6954, Best_mIoU:  0.7130
[0.96954787 0.77072844 0.91178528 0.37107735 0.51648186 0.62314124
 0.65807    0.76793334 0.91306844 0.52848334 0.94020394 0.79696459
 0.54966824 0.92203346 0.55920335 0.57838002 0.51409627 0.58177282
 0.74022192]
Epoch: [390/484] Iter:[0/24], Time: 0.95, lr: [0.0022879906363208837], Loss: 0.045782
Epoch: [390/484] Iter:[10/24], Time: 0.44, lr: [0.0022788609878400114], Loss: 0.109879
Epoch: [390/484] Iter:[20/24], Time: 0.41, lr: [0.002269727273592044], Loss: 0.108622
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97207157 0.78923878 0.90967367 0.34099891 0.48075247 0.61442914
 0.64443509 0.76104392 0.91289893 0.52290554 0.94245868 0.79228958
 0.54397398 0.91970698 0.51430364 0.55595511 0.43044136 0.55717253
 0.73677849] 0.6811330721545735
1 [0.97190929 0.79039551 0.91063821 0.35710952 0.4987435  0.62141413
 0.64817116 0.76869253 0.91442495 0.533186   0.94390885 0.79986796
 0.56099768 0.92604878 0.58877311 0.62121934 0.46772509 0.58470243
 0.7437385 ] 0.6974561325175354
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.260, MeanIU:  0.6975, Best_mIoU:  0.7130
[0.97190929 0.79039551 0.91063821 0.35710952 0.4987435  0.62141413
 0.64817116 0.76869253 0.91442495 0.533186   0.94390885 0.79986796
 0.56099768 0.92604878 0.58877311 0.62121934 0.46772509 0.58470243
 0.7437385 ]
Epoch: [391/484] Iter:[0/24], Time: 1.01, lr: [0.00226607264500301], Loss: 0.162764
Epoch: [391/484] Iter:[10/24], Time: 0.45, lr: [0.002256933205028592], Loss: 0.128488
Epoch: [391/484] Iter:[20/24], Time: 0.42, lr: [0.0022477896509545915], Loss: 0.121629
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97302525 0.79554002 0.90958039 0.36624994 0.48288085 0.61081188
 0.64497703 0.76158672 0.91337674 0.53992502 0.9387258  0.78681108
 0.53054738 0.92144558 0.52241443 0.56688451 0.42071478 0.5665703
 0.73813479] 0.6836948694648322
1 [0.9728776  0.79745303 0.91049631 0.38460547 0.50239874 0.6192431
 0.65803735 0.76896289 0.91477999 0.55893075 0.94049008 0.79277142
 0.54914227 0.92835771 0.6175689  0.64090417 0.44187729 0.59104615
 0.74290378] 0.7017287885598121
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.254, MeanIU:  0.7017, Best_mIoU:  0.7130
[0.9728776  0.79745303 0.91049631 0.38460547 0.50239874 0.6192431
 0.65803735 0.76896289 0.91477999 0.55893075 0.94049008 0.79277142
 0.54914227 0.92835771 0.6175689  0.64090417 0.44187729 0.59104615
 0.74290378]
Epoch: [392/484] Iter:[0/24], Time: 0.96, lr: [0.002244131072799776], Loss: 0.053142
Epoch: [392/484] Iter:[10/24], Time: 0.44, lr: [0.002234981724563763], Loss: 0.107178
Epoch: [392/484] Iter:[20/24], Time: 0.42, lr: [0.002225828212786806], Loss: 0.114113
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97394731 0.7979921  0.91081666 0.36533786 0.49852507 0.61185152
 0.64630844 0.75815111 0.91360701 0.53317951 0.93846686 0.79189574
 0.52163541 0.91968234 0.51199502 0.55406041 0.45929581 0.54782954
 0.73594315] 0.6837116249012694
1 [0.97395198 0.80028252 0.91088728 0.33088291 0.51479085 0.61929794
 0.65886481 0.76675902 0.91436451 0.54887097 0.94013459 0.79849661
 0.54707064 0.92361863 0.57284314 0.60066471 0.50194705 0.56216326
 0.74116519] 0.6961608736555527
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.251, MeanIU:  0.6962, Best_mIoU:  0.7130
[0.97395198 0.80028252 0.91088728 0.33088291 0.51479085 0.61929794
 0.65886481 0.76675902 0.91436451 0.54887097 0.94013459 0.79849661
 0.54707064 0.92361863 0.57284314 0.60066471 0.50194705 0.56216326
 0.74116519]
Epoch: [393/484] Iter:[0/24], Time: 0.97, lr: [0.002222165637601467], Loss: 0.146293
Epoch: [393/484] Iter:[10/24], Time: 0.44, lr: [0.0022130062616482157], Loss: 0.117153
Epoch: [393/484] Iter:[20/24], Time: 0.42, lr: [0.002203842671565811], Loss: 0.117375
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97342112 0.80024286 0.91086857 0.392935   0.49263157 0.61587286
 0.64580725 0.75863668 0.91352911 0.52561883 0.93989745 0.79114565
 0.53170494 0.92019866 0.50249546 0.58067709 0.42397044 0.5477975
 0.73998029] 0.6846016491158652
1 [0.97392688 0.8033185  0.91165831 0.39833524 0.51160455 0.62295779
 0.65934995 0.7672338  0.91441886 0.53890177 0.9398113  0.79763271
 0.54780309 0.92789934 0.57205757 0.66201785 0.45032753 0.57577943
 0.74774508] 0.7011989240934732
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.253, MeanIU:  0.7012, Best_mIoU:  0.7130
[0.97392688 0.8033185  0.91165831 0.39833524 0.51160455 0.62295779
 0.65934995 0.7672338  0.91441886 0.53890177 0.9398113  0.79763271
 0.54780309 0.92789934 0.57205757 0.66201785 0.45032753 0.57577943
 0.74774508]
Epoch: [394/484] Iter:[0/24], Time: 0.99, lr: [0.002200176050784188], Loss: 0.113430
Epoch: [394/484] Iter:[10/24], Time: 0.44, lr: [0.0021910065248781216], Loss: 0.108954
Epoch: [394/484] Iter:[20/24], Time: 0.42, lr: [0.0021818327330681416], Loss: 0.113301
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97367651 0.80457134 0.91176113 0.39920827 0.49704744 0.61611071
 0.64854336 0.76174497 0.91455948 0.57339535 0.94081746 0.7944484
 0.53894807 0.92124391 0.50722951 0.57320736 0.4099741  0.5572284
 0.73541209] 0.6883751508134923
1 [0.97395262 0.80683796 0.91231678 0.39437804 0.51472011 0.62335359
 0.66239401 0.77059007 0.91576473 0.59704217 0.94175665 0.79923664
 0.55185167 0.92719926 0.57091097 0.64358295 0.41576559 0.57674018
 0.74179551] 0.7021152362724744
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7021, Best_mIoU:  0.7130
[0.97395262 0.80683796 0.91231678 0.39437804 0.51472011 0.62335359
 0.66239401 0.77059007 0.91576473 0.59704217 0.94175665 0.79923664
 0.55185167 0.92719926 0.57091097 0.64358295 0.41576559 0.57674018
 0.74179551]
Epoch: [395/484] Iter:[0/24], Time: 0.96, lr: [0.002178162016985334], Loss: 0.118285
Epoch: [395/484] Iter:[10/24], Time: 0.44, lr: [0.002168982216014382], Loss: 0.127113
Epoch: [395/484] Iter:[20/24], Time: 0.42, lr: [0.002159798096136653], Loss: 0.126096
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9741775  0.80493226 0.91422223 0.40828491 0.50698823 0.61609816
 0.649709   0.7620995  0.91369878 0.56371665 0.94002052 0.79254904
 0.53945611 0.92109148 0.50710222 0.56627789 0.42170627 0.55592999
 0.73292016] 0.6889989934459484
1 [0.97446615 0.80774949 0.91547108 0.41343128 0.52915677 0.6236686
 0.66026851 0.76860852 0.91470184 0.58005289 0.94019937 0.79823895
 0.55308443 0.9280816  0.59731918 0.64944489 0.4745579  0.57977978
 0.73954777] 0.7077804743324062
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7078, Best_mIoU:  0.7130
[0.97446615 0.80774949 0.91547108 0.41343128 0.52915677 0.6236686
 0.66026851 0.76860852 0.91470184 0.58005289 0.94019937 0.79823895
 0.55308443 0.9280816  0.59731918 0.64944489 0.4745579  0.57977978
 0.73954777]
Epoch: [396/484] Iter:[0/24], Time: 0.98, lr: [0.0021561232338687284], Loss: 0.084055
Epoch: [396/484] Iter:[10/24], Time: 0.44, lr: [0.0021469330297432592], Loss: 0.118658
Epoch: [396/484] Iter:[20/24], Time: 0.42, lr: [0.002137738452436557], Loss: 0.118482
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97463408 0.80596574 0.91423941 0.42112494 0.50376033 0.61370818
 0.64748186 0.7621106  0.91470358 0.5631101  0.94178053 0.79371093
 0.54425695 0.92324286 0.51453464 0.58517861 0.4106707  0.55046987
 0.73260629] 0.6903836948567786
1 [0.9748588  0.80864219 0.91578071 0.44432696 0.52625141 0.62072232
 0.66094612 0.7702213  0.91610449 0.57952547 0.94273982 0.79883175
 0.55584932 0.93088752 0.61210109 0.66884551 0.44171641 0.57137544
 0.73817862] 0.7093634339968534
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.241, MeanIU:  0.7094, Best_mIoU:  0.7130
[0.9748588  0.80864219 0.91578071 0.44432696 0.52625141 0.62072232
 0.66094612 0.7702213  0.91610449 0.57952547 0.94273982 0.79883175
 0.55584932 0.93088752 0.61210109 0.66884551 0.44171641 0.57137544
 0.73817862]
Epoch: [397/484] Iter:[0/24], Time: 0.97, lr: [0.002134059391878799], Loss: 0.113558
Epoch: [397/484] Iter:[10/24], Time: 0.44, lr: [0.0021248586534258185], Loss: 0.130864
Epoch: [397/484] Iter:[20/24], Time: 0.42, lr: [0.0021156534861999977], Loss: 0.122582
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97444661 0.80512926 0.91377854 0.42675681 0.50614318 0.61846165
 0.64609313 0.76291929 0.91451477 0.56307449 0.94167463 0.79087558
 0.53774663 0.92298754 0.51164071 0.58161689 0.40068404 0.56065197
 0.73367429] 0.6901510529968631
1 [0.9747325  0.80856617 0.91504947 0.45869661 0.5259437  0.62534907
 0.66113859 0.7706923  0.91602698 0.58073796 0.94179069 0.79584715
 0.55155252 0.93052515 0.61018905 0.66520094 0.41886293 0.57782271
 0.73949009] 0.7088533988640179
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.238, MeanIU:  0.7089, Best_mIoU:  0.7130
[0.9747325  0.80856617 0.91504947 0.45869661 0.5259437  0.62534907
 0.66113859 0.7706923  0.91602698 0.58073796 0.94179069 0.79584715
 0.55155252 0.93052515 0.61018905 0.66520094 0.41886293 0.57782271
 0.73949009]
Epoch: [398/484] Iter:[0/24], Time: 0.97, lr: [0.002111970173983164], Loss: 0.090206
Epoch: [398/484] Iter:[10/24], Time: 0.44, lr: [0.0021027587668354733], Loss: 0.121228
Epoch: [398/484] Iter:[20/24], Time: 0.42, lr: [0.0020935428739584498], Loss: 0.129179
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97430849 0.80341821 0.91250287 0.38232438 0.50321669 0.6197034
 0.64948493 0.76271669 0.91474526 0.5666892  0.94125559 0.79275214
 0.53377608 0.92187717 0.51346246 0.58599375 0.41526624 0.56048162
 0.73654672] 0.6889748374000333
1 [0.97447917 0.80615152 0.91335821 0.3892013  0.51823297 0.62631217
 0.66408715 0.77028504 0.91631391 0.58583187 0.94145086 0.79875268
 0.54962855 0.92873724 0.59933613 0.66615194 0.45468532 0.58245453
 0.74250966] 0.7067347484596694
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7067, Best_mIoU:  0.7130
[0.97447917 0.80615152 0.91335821 0.3892013  0.51823297 0.62631217
 0.66408715 0.77028504 0.91631391 0.58583187 0.94145086 0.79875268
 0.54962855 0.92873724 0.59933613 0.66615194 0.45468532 0.58245453
 0.74250966]
Epoch: [399/484] Iter:[0/24], Time: 0.96, lr: [0.0020898552554029447], Loss: 0.080067
Epoch: [399/484] Iter:[10/24], Time: 0.44, lr: [0.002080633041882971], Loss: 0.128833
Epoch: [399/484] Iter:[20/24], Time: 0.42, lr: [0.0020714062842622297], Loss: 0.116369
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97460243 0.80407316 0.91324257 0.39289016 0.50999382 0.61789816
 0.64904586 0.76087821 0.91450521 0.55816101 0.94143252 0.78982804
 0.52375425 0.92274817 0.51121442 0.59042211 0.41944802 0.55630807
 0.73764721] 0.6888470205025504
1 [0.9747312  0.80671179 0.91478154 0.40717473 0.52868678 0.62547068
 0.66339058 0.76910588 0.9162132  0.57461504 0.9415343  0.79680779
 0.54233498 0.92999441 0.59983979 0.67101859 0.45967797 0.57930216
 0.74493146] 0.7077012033991599
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7077, Best_mIoU:  0.7130
[0.9747312  0.80671179 0.91478154 0.40717473 0.52868678 0.62547068
 0.66339058 0.76910588 0.9162132  0.57461504 0.9415343  0.79680779
 0.54233498 0.92999441 0.59983979 0.67101859 0.45967797 0.57930216
 0.74493146]
Epoch: [400/484] Iter:[0/24], Time: 0.98, lr: [0.0020677143033300464], Loss: 0.106769
Epoch: [400/484] Iter:[10/24], Time: 0.45, lr: [0.002058481142328021], Loss: 0.116776
Epoch: [400/484] Iter:[20/24], Time: 0.42, lr: [0.0020492433773863423], Loss: 0.114694
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97480431 0.80587871 0.9128529  0.3901921  0.50264984 0.61212252
 0.64819351 0.76152722 0.91379388 0.55962596 0.93981589 0.79342594
 0.52914153 0.9233702  0.51400432 0.59108126 0.39735473 0.56086353
 0.73486772] 0.6876613722249193
1 [0.97506486 0.80852761 0.913876   0.39759292 0.52022986 0.61797428
 0.65715975 0.76888088 0.91480067 0.57121571 0.93931539 0.79958744
 0.5459488  0.93054182 0.61180098 0.66386075 0.40544702 0.58178853
 0.74267541] 0.7034888777130052
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7035, Best_mIoU:  0.7130
[0.97506486 0.80852761 0.913876   0.39759292 0.52022986 0.61797428
 0.65715975 0.76888088 0.91480067 0.57121571 0.93931539 0.79958744
 0.5459488  0.93054182 0.61180098 0.66386075 0.40544702 0.58178853
 0.74267541]
Epoch: [401/484] Iter:[0/24], Time: 0.96, lr: [0.00204554697663065], Loss: 0.090467
Epoch: [401/484] Iter:[10/24], Time: 0.44, lr: [0.0020363027234767783], Loss: 0.106439
Epoch: [401/484] Iter:[20/24], Time: 0.42, lr: [0.0020270538050217975], Loss: 0.114992
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97476917 0.80588626 0.91139162 0.38702358 0.49446612 0.61722636
 0.64992944 0.76181183 0.91392188 0.55281431 0.94090133 0.79086408
 0.53516531 0.92443768 0.51675777 0.58866089 0.36649194 0.5666454
 0.73572613] 0.6860469007078102
1 [0.97495572 0.80997414 0.91260139 0.41689428 0.51315546 0.62431253
 0.66108569 0.76912324 0.91512504 0.56540927 0.94286877 0.79677085
 0.54886099 0.93148425 0.63615687 0.66124001 0.33241595 0.59008268
 0.74171422] 0.7023279656619837
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7023, Best_mIoU:  0.7130
[0.97495572 0.80997414 0.91260139 0.41689428 0.51315546 0.62431253
 0.66108569 0.76912324 0.91512504 0.56540927 0.94286877 0.79677085
 0.54886099 0.93148425 0.63615687 0.66124001 0.33241595 0.59008268
 0.74171422]
Epoch: [402/484] Iter:[0/24], Time: 0.97, lr: [0.0020233529255340226], Loss: 0.103892
Epoch: [402/484] Iter:[10/24], Time: 0.44, lr: [0.0020140974318642937], Loss: 0.095157
Epoch: [402/484] Iter:[20/24], Time: 0.42, lr: [0.002004837209951537], Loss: 0.109199
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97468305 0.8055107  0.91035552 0.34754782 0.49276606 0.62140715
 0.64918579 0.76230458 0.91373056 0.56022762 0.93980912 0.79430329
 0.53959646 0.92300812 0.50956843 0.58369115 0.38034623 0.56094849
 0.72986534] 0.6841502875367064
1 [0.97492695 0.80815284 0.91065278 0.33229702 0.50424704 0.62757349
 0.66079833 0.76956346 0.91473007 0.56891039 0.94018233 0.79997201
 0.55592073 0.9300991  0.60333573 0.65917442 0.38392314 0.57777024
 0.73634291] 0.6978196302731452
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.250, MeanIU:  0.6978, Best_mIoU:  0.7130
[0.97492695 0.80815284 0.91065278 0.33229702 0.50424704 0.62757349
 0.66079833 0.76956346 0.91473007 0.56891039 0.94018233 0.79997201
 0.55592073 0.9300991  0.60333573 0.65917442 0.38392314 0.57777024
 0.73634291]
Epoch: [403/484] Iter:[0/24], Time: 0.99, lr: [0.002001131791305781], Loss: 0.089052
Epoch: [403/484] Iter:[10/24], Time: 0.44, lr: [0.001991864904920988], Loss: 0.110845
Epoch: [403/484] Iter:[20/24], Time: 0.42, lr: [0.00198259322570995], Loss: 0.112465
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97480696 0.80499993 0.9112279  0.38219164 0.49424442 0.61638336
 0.64882237 0.76043413 0.91457137 0.55615694 0.94078598 0.79265702
 0.53446495 0.9236515  0.5147004  0.59398948 0.39691269 0.56252593
 0.73672445] 0.6873816538848058
1 [0.97479344 0.80799316 0.91239029 0.40764606 0.512534   0.6228909
 0.66346929 0.76831367 0.91611128 0.57115045 0.94184471 0.79844841
 0.55182871 0.93072561 0.62214868 0.66869789 0.35559741 0.58756905
 0.74581394] 0.7031561550638535
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7032, Best_mIoU:  0.7130
[0.97479344 0.80799316 0.91239029 0.40764606 0.512534   0.6228909
 0.66346929 0.76831367 0.91611128 0.57115045 0.94184471 0.79844841
 0.55182871 0.93072561 0.62214868 0.66869789 0.35559741 0.58756905
 0.74581394]
Epoch: [404/484] Iter:[0/24], Time: 0.96, lr: [0.001978883205904584], Loss: 0.195297
Epoch: [404/484] Iter:[10/24], Time: 0.44, lr: [0.0019696047706221575], Loss: 0.127855
Epoch: [404/484] Iter:[20/24], Time: 0.42, lr: [0.0019603214762249927], Loss: 0.120732
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97427458 0.80204164 0.91207716 0.36795589 0.49426654 0.6161183
 0.64676411 0.76111539 0.91427458 0.55362669 0.93995439 0.78877328
 0.53100621 0.92637524 0.52827708 0.60888934 0.40756681 0.55977016
 0.73305283] 0.6876936958161871
1 [0.97446573 0.80570459 0.91341631 0.38164715 0.5109805  0.62318243
 0.66075167 0.76818625 0.91530337 0.56952061 0.93967029 0.79440681
 0.54881709 0.93208236 0.63230239 0.67581593 0.38614385 0.57920197
 0.74092731] 0.7027645571512273
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7028, Best_mIoU:  0.7130
[0.97446573 0.80570459 0.91341631 0.38164715 0.5109805  0.62318243
 0.66075167 0.76818625 0.91530337 0.56952061 0.93967029 0.79440681
 0.54881709 0.93208236 0.63230239 0.67581593 0.38614385 0.57920197
 0.74092731]
Epoch: [405/484] Iter:[0/24], Time: 0.95, lr: [0.001956606791621233], Loss: 0.133935
Epoch: [405/484] Iter:[10/24], Time: 0.44, lr: [0.001947316647119394], Loss: 0.112013
Epoch: [405/484] Iter:[20/24], Time: 0.42, lr: [0.0019380215754417382], Loss: 0.111636
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97424964 0.80127456 0.9124892  0.37931387 0.49255392 0.61834803
 0.6491877  0.76178992 0.9140904  0.55514705 0.94234725 0.78845751
 0.5294276  0.9256452  0.53029932 0.61211709 0.41030981 0.55707001
 0.73670209] 0.6889905352583254
1 [0.97445972 0.80609999 0.91422582 0.41257212 0.51148962 0.62559858
 0.66399122 0.7678345  0.91549164 0.57400049 0.94313905 0.79426806
 0.5482125  0.93212051 0.62856247 0.6706882  0.39613531 0.57340902
 0.74354372] 0.7050443438858487
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7050, Best_mIoU:  0.7130
[0.97445972 0.80609999 0.91422582 0.41257212 0.51148962 0.62559858
 0.66399122 0.7678345  0.91549164 0.57400049 0.94313905 0.79426806
 0.5482125  0.93212051 0.62856247 0.6706882  0.39613531 0.57340902
 0.74354372]
Epoch: [406/484] Iter:[0/24], Time: 0.98, lr: [0.0019343021606990191], Loss: 0.073616
Epoch: [406/484] Iter:[10/24], Time: 0.44, lr: [0.0019250001423527733], Loss: 0.121014
Epoch: [406/484] Iter:[20/24], Time: 0.42, lr: [0.0019156931269261677], Loss: 0.112924
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97402235 0.8001424  0.91115605 0.35824132 0.48806058 0.62220482
 0.64932699 0.75975631 0.91448691 0.54809453 0.9417498  0.79196219
 0.53733346 0.92276318 0.51285587 0.61598303 0.39775125 0.55848035
 0.73548497] 0.6863082304196526
1 [0.97427174 0.80433146 0.91255083 0.37721171 0.501547   0.62814095
 0.65974684 0.76698182 0.91604257 0.56375362 0.94267195 0.79707846
 0.55166151 0.93005596 0.59635982 0.67076088 0.35153006 0.58002748
 0.74274855] 0.6982880644174757
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.251, MeanIU:  0.6983, Best_mIoU:  0.7130
[0.97427174 0.80433146 0.91255083 0.37721171 0.501547   0.62814095
 0.65974684 0.76698182 0.91604257 0.56375362 0.94267195 0.79707846
 0.55166151 0.93005596 0.59635982 0.67076088 0.35153006 0.58002748
 0.74274855]
Epoch: [407/484] Iter:[0/24], Time: 1.00, lr: [0.0019119689149340972], Loss: 0.071537
Epoch: [407/484] Iter:[10/24], Time: 0.45, lr: [0.0019026548536425192], Loss: 0.102398
Epoch: [407/484] Iter:[20/24], Time: 0.42, lr: [0.0018933357234479082], Loss: 0.101181
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97413044 0.79912716 0.91369064 0.434012   0.50949765 0.61656814
 0.64868341 0.75944385 0.91466641 0.55020725 0.93900948 0.79382086
 0.54101697 0.92502688 0.50891176 0.61004127 0.41955189 0.55483683
 0.7376019 ] 0.6920970938889399
1 [0.97413689 0.803126   0.91552548 0.47495212 0.52847282 0.62362753
 0.662034   0.76722436 0.91657243 0.56841332 0.94132333 0.79874314
 0.55501718 0.93152029 0.5925222  0.67416718 0.41799577 0.57038834
 0.74551546] 0.7084883076609477
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.240, MeanIU:  0.7085, Best_mIoU:  0.7130
[0.97413689 0.803126   0.91552548 0.47495212 0.52847282 0.62362753
 0.662034   0.76722436 0.91657243 0.56841332 0.94132333 0.79874314
 0.55501718 0.93152029 0.5925222  0.67416718 0.41799577 0.57038834
 0.74551546]
Epoch: [408/484] Iter:[0/24], Time: 0.98, lr: [0.001889606645254576], Loss: 0.144985
Epoch: [408/484] Iter:[10/24], Time: 0.44, lr: [0.0018802803672588058], Loss: 0.137905
Epoch: [408/484] Iter:[20/24], Time: 0.42, lr: [0.0018709489465404713], Loss: 0.125324
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97457532 0.80179734 0.91327168 0.41599603 0.49860689 0.62431351
 0.65020256 0.76232537 0.91424431 0.54652747 0.94005387 0.78929882
 0.52933562 0.92490179 0.51745531 0.61848521 0.40935599 0.55631744
 0.73405646] 0.6905853144997555
1 [0.97432354 0.80447431 0.91478025 0.45445581 0.51583474 0.63060052
 0.66223988 0.76902823 0.91577235 0.5630286  0.94049288 0.79513627
 0.54700538 0.93118286 0.60295749 0.67643487 0.39171862 0.57438228
 0.74063281] 0.7054990364504694
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7055, Best_mIoU:  0.7130
[0.97432354 0.80447431 0.91478025 0.45445581 0.51583474 0.63060052
 0.66223988 0.76902823 0.91577235 0.5630286  0.94049288 0.79513627
 0.54700538 0.93118286 0.60295749 0.67643487 0.39171862 0.57438228
 0.74063281]
Epoch: [409/484] Iter:[0/24], Time: 0.97, lr: [0.0018672149312768667], Loss: 0.096899
Epoch: [409/484] Iter:[10/24], Time: 0.44, lr: [0.0018578762579681918], Loss: 0.101182
Epoch: [409/484] Iter:[20/24], Time: 0.42, lr: [0.0018485323660375073], Loss: 0.103725
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97472606 0.80215666 0.91447169 0.42991656 0.50567918 0.62244909
 0.65019877 0.76195445 0.91458568 0.55283044 0.94121795 0.79250663
 0.52987571 0.92565035 0.51667891 0.61319487 0.42141008 0.55760551
 0.736433  ] 0.6928179783397772
1 [0.97470581 0.80548906 0.91627795 0.47277108 0.52474924 0.63017633
 0.66396025 0.76871299 0.9164497  0.57200708 0.94147495 0.79826997
 0.54622264 0.93214896 0.60712434 0.67864162 0.4144042  0.57851367
 0.74524411] 0.709860208004293
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.239, MeanIU:  0.7099, Best_mIoU:  0.7130
[0.97470581 0.80548906 0.91627795 0.47277108 0.52474924 0.63017633
 0.66396025 0.76871299 0.9164497  0.57200708 0.94147495 0.79826997
 0.54622264 0.93214896 0.60712434 0.67864162 0.4144042  0.57851367
 0.74524411]
Epoch: [410/484] Iter:[0/24], Time: 1.01, lr: [0.0018447933408377772], Loss: 0.153342
Epoch: [410/484] Iter:[10/24], Time: 0.44, lr: [0.0018354420885551026], Loss: 0.122714
Epoch: [410/484] Iter:[20/24], Time: 0.42, lr: [0.001826085539583378], Loss: 0.113860
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97480855 0.80393729 0.9131141  0.40771157 0.4950821  0.6183754
 0.6467303  0.76000182 0.91466442 0.55042001 0.94186745 0.78860647
 0.53757066 0.92351894 0.50853742 0.62020132 0.42468935 0.55302606
 0.73088538] 0.6901972959133482
1 [0.97488078 0.80748638 0.9144536  0.43222374 0.51121137 0.62560541
 0.66080612 0.76748459 0.91603747 0.5662988  0.94238929 0.79439814
 0.55379865 0.93001744 0.58924549 0.67838469 0.38226157 0.56813656
 0.7378189 ] 0.7027862623580208
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7028, Best_mIoU:  0.7130
[0.97488078 0.80748638 0.9144536  0.43222374 0.51121137 0.62560541
 0.66080612 0.76748459 0.91603747 0.5662988  0.94238929 0.79439814
 0.55379865 0.93001744 0.58924549 0.67838469 0.38226157 0.56813656
 0.7378189 ]
Epoch: [411/484] Iter:[0/24], Time: 0.96, lr: [0.0018223414295006415], Loss: 0.099928
Epoch: [411/484] Iter:[10/24], Time: 0.44, lr: [0.0018129774093166406], Loss: 0.134880
Epoch: [411/484] Iter:[20/24], Time: 0.42, lr: [0.0018036080121163068], Loss: 0.132644
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.96611493 0.76082515 0.91196126 0.4120938  0.4939465  0.61765243
 0.64689769 0.76147405 0.91357128 0.55585388 0.93713345 0.79303052
 0.54572561 0.92359373 0.51927663 0.61100473 0.41931857 0.55434223
 0.73535574] 0.6883774829793576
1 [0.96588687 0.76340936 0.91252299 0.43480563 0.51165826 0.62463973
 0.65832377 0.76752153 0.91537864 0.57462446 0.93514027 0.79899839
 0.56169032 0.92995932 0.60721334 0.67124324 0.39719782 0.57344506
 0.74220914] 0.7024141124972129
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.259, MeanIU:  0.7024, Best_mIoU:  0.7130
[0.96588687 0.76340936 0.91252299 0.43480563 0.51165826 0.62463973
 0.65832377 0.76752153 0.91537864 0.57462446 0.93514027 0.79899839
 0.56169032 0.92995932 0.60721334 0.67124324 0.39719782 0.57344506
 0.74220914]
Epoch: [412/484] Iter:[0/24], Time: 1.00, lr: [0.0017998587400337065], Loss: 0.065237
Epoch: [412/484] Iter:[10/24], Time: 0.44, lr: [0.0017904817575288142], Loss: 0.106163
Epoch: [412/484] Iter:[20/24], Time: 0.42, lr: [0.0017810993153221147], Loss: 0.110594
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97286015 0.79266045 0.91281203 0.41228229 0.49583745 0.6237756
 0.64555343 0.76163881 0.91459334 0.56017791 0.94236143 0.79439946
 0.54419109 0.92315211 0.50705708 0.60128755 0.42584116 0.55979156
 0.73760859] 0.6909411293265628
1 [0.97267999 0.79668928 0.91333775 0.43289382 0.51367535 0.63021419
 0.66095312 0.76930576 0.91651517 0.58131513 0.94237458 0.80019878
 0.56054132 0.92873041 0.58520489 0.65903796 0.42746948 0.58577174
 0.74490708] 0.706411356939824
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7064, Best_mIoU:  0.7130
[0.97267999 0.79668928 0.91333775 0.43289382 0.51367535 0.63021419
 0.66095312 0.76930576 0.91651517 0.58131513 0.94237458 0.80019878
 0.56054132 0.92873041 0.58520489 0.65903796 0.42746948 0.58577174
 0.74490708]
Epoch: [413/484] Iter:[0/24], Time: 0.96, lr: [0.001777344801858777], Loss: 0.096301
Epoch: [413/484] Iter:[10/24], Time: 0.45, lr: [0.0017679546568821905], Loss: 0.118518
Epoch: [413/484] Iter:[20/24], Time: 0.42, lr: [0.001758558967056482], Loss: 0.116744
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9762251  0.80595819 0.91373252 0.41069237 0.50243446 0.62128422
 0.64975011 0.76383269 0.91551742 0.55436509 0.94115262 0.79351073
 0.52682829 0.92312597 0.50066495 0.61397122 0.44862425 0.56873108
 0.73826637] 0.6930877704733012
1 [0.97641374 0.80854163 0.91415478 0.42371741 0.51836199 0.62711696
 0.66237573 0.77134158 0.91711198 0.5688594  0.94052235 0.80003809
 0.54739714 0.92916412 0.54480939 0.6710078  0.46608223 0.59083739
 0.74429804] 0.7064290392634532
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7064, Best_mIoU:  0.7130
[0.97641374 0.80854163 0.91415478 0.42371741 0.51836199 0.62711696
 0.66237573 0.77134158 0.91711198 0.5688594  0.94052235 0.80003809
 0.54739714 0.92916412 0.54480939 0.6710078  0.46608223 0.59083739
 0.74429804]
Epoch: [414/484] Iter:[0/24], Time: 1.00, lr: [0.0017547991304679984], Loss: 0.061286
Epoch: [414/484] Iter:[10/24], Time: 0.45, lr: [0.0017453956168847165], Loss: 0.108651
Epoch: [414/484] Iter:[20/24], Time: 0.42, lr: [0.0017359864707334044], Loss: 0.119860
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97606446 0.80566643 0.91287017 0.39781819 0.49680727 0.61812838
 0.64718724 0.76082778 0.91445367 0.53832594 0.94150298 0.7925094
 0.52725177 0.92343327 0.49849861 0.61248461 0.43819699 0.55546644
 0.73391518] 0.6890215150505636
1 [0.97631205 0.80976482 0.91358508 0.41071745 0.51467439 0.62403101
 0.66171159 0.76801814 0.91539606 0.55107554 0.94175358 0.79813423
 0.54617068 0.92966644 0.55934575 0.67131419 0.44597359 0.57861903
 0.74097159] 0.7030123800697629
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7030, Best_mIoU:  0.7130
[0.97631205 0.80976482 0.91358508 0.41071745 0.51467439 0.62403101
 0.66171159 0.76801814 0.91539606 0.55107554 0.94175358 0.79813423
 0.54617068 0.92966644 0.55934575 0.67131419 0.44597359 0.57861903
 0.74097159]
Epoch: [415/484] Iter:[0/24], Time: 0.93, lr: [0.0017322212268064561], Loss: 0.102702
Epoch: [415/484] Iter:[10/24], Time: 0.44, lr: [0.0017228041322293448], Loss: 0.108221
Epoch: [415/484] Iter:[20/24], Time: 0.42, lr: [0.0017133813146773713], Loss: 0.114042
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9763068  0.81012883 0.91335075 0.40433308 0.49594578 0.61692084
 0.64857157 0.76231423 0.9148465  0.55432239 0.93948378 0.79090502
 0.53098793 0.92398199 0.50597877 0.6179461  0.43843518 0.5670046
 0.73496945] 0.6919333473885128
1 [0.97649912 0.81361807 0.91450958 0.43229898 0.51452263 0.62392819
 0.66049648 0.77008674 0.91596931 0.56781367 0.93879859 0.79700536
 0.54924075 0.93011394 0.56997454 0.67419965 0.44401996 0.5925741
 0.74294384] 0.7067691320600961
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.241, MeanIU:  0.7068, Best_mIoU:  0.7130
[0.97649912 0.81361807 0.91450958 0.43229898 0.51452263 0.62392819
 0.66049648 0.77008674 0.91596931 0.56781367 0.93879859 0.79700536
 0.54924075 0.93011394 0.56997454 0.67419965 0.44401996 0.5925741
 0.74294384]
Epoch: [416/484] Iter:[0/24], Time: 0.98, lr: [0.001709610576618043], Loss: 0.124697
Epoch: [416/484] Iter:[10/24], Time: 0.44, lr: [0.0017001796821237978], Loss: 0.114187
Epoch: [416/484] Iter:[20/24], Time: 0.42, lr: [0.0016907429714365534], Loss: 0.115869
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97683856 0.81300577 0.91459377 0.40483122 0.51432345 0.62090658
 0.64915886 0.7621423  0.9140798  0.55386535 0.94191206 0.79194094
 0.53195349 0.92480025 0.47970172 0.54316424 0.44860572 0.56673332
 0.73740227] 0.6889452461069688
1 [0.97724711 0.81738486 0.91583696 0.42337129 0.5344921  0.62852569
 0.66263962 0.76996982 0.91508141 0.56548066 0.94355303 0.79741288
 0.54812617 0.93141195 0.54662029 0.58860155 0.44856326 0.59118614
 0.74559681] 0.7026895588130713
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7027, Best_mIoU:  0.7130
[0.97724711 0.81738486 0.91583696 0.42337129 0.5344921  0.62852569
 0.66263962 0.76996982 0.91508141 0.56548066 0.94355303 0.79741288
 0.54812617 0.93141195 0.54662029 0.58860155 0.44856326 0.59118614
 0.74559681]
Epoch: [417/484] Iter:[0/24], Time: 0.98, lr: [0.0016869666497518682], Loss: 0.145355
Epoch: [417/484] Iter:[10/24], Time: 0.44, lr: [0.001677521729579656], Loss: 0.137564
Epoch: [417/484] Iter:[20/24], Time: 0.42, lr: [0.0016680708970540103], Loss: 0.124069
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97663705 0.8124093  0.91287826 0.38531486 0.50954979 0.6182159
 0.64591152 0.76227614 0.91349339 0.55731372 0.9414535  0.79347933
 0.53633562 0.9238968  0.47801428 0.54589493 0.44946703 0.57068149
 0.73490532] 0.6877962239493725
1 [0.97703728 0.81672872 0.91390128 0.39542804 0.52986552 0.6248178
 0.65917165 0.76963327 0.91495558 0.57527467 0.94244494 0.7990541
 0.5531937  0.92969249 0.54077404 0.57852028 0.47118781 0.58945909
 0.74220629] 0.7012287647879624
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7012, Best_mIoU:  0.7130
[0.97703728 0.81672872 0.91390128 0.39542804 0.52986552 0.6248178
 0.65917165 0.76963327 0.91495558 0.57527467 0.94244494 0.7990541
 0.5531937  0.92969249 0.54077404 0.57852028 0.47118781 0.58945909
 0.74220629]
Epoch: [418/484] Iter:[0/24], Time: 0.95, lr: [0.0016642888994261626], Loss: 0.117298
Epoch: [418/484] Iter:[10/24], Time: 0.44, lr: [0.0016548297206576163], Loss: 0.103469
Epoch: [418/484] Iter:[20/24], Time: 0.42, lr: [0.0016453645302937161], Loss: 0.108655
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97632056 0.811088   0.91274026 0.41107166 0.50168851 0.61312241
 0.64632808 0.76171413 0.91385267 0.55888538 0.9402425  0.7906138
 0.53647237 0.92306587 0.47784148 0.54778476 0.44291003 0.5654804
 0.73726408] 0.6878151031465229
1 [0.9768709  0.81611948 0.91395962 0.43544168 0.5221243  0.62029758
 0.65803461 0.76879204 0.9156142  0.58013854 0.94136674 0.79716939
 0.55441171 0.93039036 0.53797212 0.59593261 0.46698808 0.58775893
 0.74577703] 0.7034294691875086
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7034, Best_mIoU:  0.7130
[0.9768709  0.81611948 0.91395962 0.43544168 0.5221243  0.62029758
 0.65803461 0.76879204 0.9156142  0.58013854 0.94136674 0.79716939
 0.55441171 0.93039036 0.53797212 0.59593261 0.46698808 0.58775893
 0.74577703]
Epoch: [419/484] Iter:[0/24], Time: 0.96, lr: [0.0016415767614464353], Loss: 0.157701
Epoch: [419/484] Iter:[10/24], Time: 0.44, lr: [0.0016321030836655207], Loss: 0.131102
Epoch: [419/484] Iter:[20/24], Time: 0.42, lr: [0.001622623291817823], Loss: 0.118191
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97630245 0.81218702 0.91325064 0.40098557 0.51406716 0.61999822
 0.64535084 0.76086753 0.91464308 0.55678833 0.94103024 0.79287604
 0.54044509 0.92406378 0.4827425  0.55631233 0.45901524 0.56974058
 0.74051365] 0.6905884362463893
1 [0.97682156 0.81673539 0.91455388 0.42083025 0.53494497 0.62637543
 0.65844105 0.76821893 0.91598132 0.57205822 0.941842   0.79917107
 0.55775936 0.92996121 0.5426729  0.59779465 0.48755884 0.59168555
 0.74863014] 0.705370354295936
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7054, Best_mIoU:  0.7130
[0.97682156 0.81673539 0.91455388 0.42083025 0.53494497 0.62637543
 0.65844105 0.76821893 0.91598132 0.57205822 0.941842   0.79917107
 0.55775936 0.92996121 0.5426729  0.59779465 0.48755884 0.59168555
 0.74863014]
Epoch: [420/484] Iter:[0/24], Time: 0.97, lr: [0.0016188296533742475], Loss: 0.067461
Epoch: [420/484] Iter:[10/24], Time: 0.44, lr: [0.0016093412283054408], Loss: 0.094625
Epoch: [420/484] Iter:[20/24], Time: 0.42, lr: [0.0015998465833113235], Loss: 0.102144
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97613179 0.80733357 0.91100003 0.36537898 0.49475661 0.62182657
 0.64541832 0.76221076 0.91394325 0.53520144 0.94141755 0.79639383
 0.54866816 0.92360933 0.49881809 0.57603255 0.44610686 0.54753365
 0.73376187] 0.6866075368209816
1 [0.97659662 0.81251451 0.91198717 0.37426075 0.51472491 0.62924744
 0.65922995 0.76924386 0.91471776 0.54730665 0.94249308 0.80182532
 0.56423282 0.93058389 0.58488641 0.62565974 0.45466113 0.56529173
 0.74158049] 0.701107591367075
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.7011, Best_mIoU:  0.7130
[0.97659662 0.81251451 0.91198717 0.37426075 0.51472491 0.62924744
 0.65922995 0.76924386 0.91471776 0.54730665 0.94249308 0.80182532
 0.56423282 0.93058389 0.58488641 0.62565974 0.45466113 0.56529173
 0.74158049]
Epoch: [421/484] Iter:[0/24], Time: 0.95, lr: [0.0015960469736427006], Loss: 0.117903
Epoch: [421/484] Iter:[10/24], Time: 0.44, lr: [0.001586543544765685], Loss: 0.126146
Epoch: [421/484] Iter:[20/24], Time: 0.42, lr: [0.0015770337865498283], Loss: 0.115477
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97577781 0.80762779 0.91331    0.42841336 0.51017429 0.61570746
 0.64551171 0.76176482 0.91414958 0.54445728 0.94161724 0.79468908
 0.5319319  0.92423205 0.48996845 0.58950269 0.44469548 0.55099182
 0.73336141] 0.6904149588929824
1 [0.97617634 0.81293076 0.91549606 0.47084857 0.53102322 0.62284758
 0.65674338 0.76905741 0.9154726  0.55638266 0.94238792 0.80129645
 0.55383833 0.93106224 0.56812029 0.65151222 0.45909898 0.56855125
 0.74126447] 0.7075847744965348
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7076, Best_mIoU:  0.7130
[0.97617634 0.81293076 0.91549606 0.47084857 0.53102322 0.62284758
 0.65674338 0.76905741 0.9154726  0.55638266 0.94238792 0.80129645
 0.55383833 0.93106224 0.56812029 0.65151222 0.45909898 0.56855125
 0.74126447]
Epoch: [422/484] Iter:[0/24], Time: 0.99, lr: [0.0015732281006142932], Loss: 0.094961
Epoch: [422/484] Iter:[10/24], Time: 0.44, lr: [0.0015637094027532886], Loss: 0.114695
Epoch: [422/484] Iter:[20/24], Time: 0.42, lr: [0.0015541842624057988], Loss: 0.115283
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9756948  0.80800551 0.91135632 0.39810173 0.49474803 0.6144334
 0.64654284 0.76086526 0.91405232 0.54486682 0.94104598 0.79111453
 0.51691295 0.92451697 0.49137308 0.58017376 0.44616267 0.55776048
 0.73387366] 0.6869263753580317
1 [0.97603127 0.81295878 0.91173016 0.39819199 0.51082753 0.62189621
 0.66061532 0.76856597 0.91465033 0.5589909  0.94268261 0.79716854
 0.53821572 0.92957725 0.56866033 0.62799361 0.46640945 0.57826096
 0.74101562] 0.7012864491359635
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7013, Best_mIoU:  0.7130
[0.97603127 0.81295878 0.91173016 0.39819199 0.51082753 0.62189621
 0.66061532 0.76856597 0.91465033 0.5589909  0.94268261 0.79716854
 0.53821572 0.92957725 0.56866033 0.62799361 0.46640945 0.57826096
 0.74101562]
Epoch: [423/484] Iter:[0/24], Time: 0.96, lr: [0.0015503723915764117], Loss: 0.138224
Epoch: [423/484] Iter:[10/24], Time: 0.44, lr: [0.0015408381504619978], Loss: 0.112606
Epoch: [423/484] Iter:[20/24], Time: 0.42, lr: [0.0015312973497881143], Loss: 0.110276
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9756161  0.80811661 0.91280256 0.42356992 0.49977049 0.61610495
 0.64982266 0.76333742 0.91436527 0.54307724 0.94159273 0.78987016
 0.5182099  0.92279449 0.49690972 0.58178077 0.45643371 0.56259712
 0.7337011 ] 0.6900248902841108
1 [0.97584467 0.81290809 0.9139137  0.45071875 0.51940769 0.62315516
 0.66247752 0.76972134 0.91552058 0.55584416 0.94318175 0.79692972
 0.53851138 0.9289663  0.58393742 0.63835379 0.47983061 0.58485998
 0.74098315] 0.7071087237313768
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7071, Best_mIoU:  0.7130
[0.97584467 0.81290809 0.9139137  0.45071875 0.51940769 0.62315516
 0.66247752 0.76972134 0.91552058 0.55584416 0.94318175 0.79692972
 0.53851138 0.9289663  0.58393742 0.63835379 0.47983061 0.58485998
 0.74098315]
Epoch: [424/484] Iter:[0/24], Time: 0.99, lr: [0.0015274791816692471], Loss: 0.095006
Epoch: [424/484] Iter:[10/24], Time: 0.44, lr: [0.0015179291134703634], Loss: 0.135803
Epoch: [424/484] Iter:[20/24], Time: 0.42, lr: [0.0015083723645092787], Loss: 0.122186
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97523693 0.8087161  0.91211068 0.40852814 0.50641222 0.61694794
 0.6465354  0.76279523 0.91524812 0.54511562 0.94182822 0.78540372
 0.50003856 0.92179878 0.49034942 0.57391975 0.43932906 0.56788271
 0.73933626] 0.6872385715052113
1 [0.97566982 0.81286428 0.91284346 0.42848335 0.52539337 0.62429532
 0.65961335 0.77018752 0.91657647 0.55632427 0.94225854 0.79278107
 0.5210979  0.92801182 0.56680477 0.63119123 0.45080762 0.59075606
 0.74789249] 0.7028343542502166
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7028, Best_mIoU:  0.7130
[0.97566982 0.81286428 0.91284346 0.42848335 0.52539337 0.62429532
 0.65961335 0.77018752 0.91657647 0.55632427 0.94225854 0.79278107
 0.5210979  0.92801182 0.56680477 0.63119123 0.45080762 0.59075606
 0.74789249]
Epoch: [425/484] Iter:[0/24], Time: 0.96, lr: [0.0015045477827403566], Loss: 0.154701
Epoch: [425/484] Iter:[10/24], Time: 0.44, lr: [0.0014949815935639153], Loss: 0.112789
Epoch: [425/484] Iter:[20/24], Time: 0.42, lr: [0.0014854085980740691], Loss: 0.115193
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97553517 0.80750921 0.91254679 0.39848638 0.5103397  0.61577941
 0.64811795 0.76312454 0.91452168 0.5380295  0.94196789 0.79451254
 0.52902645 0.92413031 0.49993995 0.57662085 0.41096016 0.57034569
 0.73429039] 0.6876728720237328
1 [0.97596701 0.81340447 0.91392804 0.42725477 0.53282744 0.62288101
 0.65982698 0.76984343 0.91606989 0.55820719 0.94312679 0.80044031
 0.54721817 0.93059141 0.59032327 0.64004468 0.41548808 0.59428859
 0.74292939] 0.7049821530973107
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7050, Best_mIoU:  0.7130
[0.97596701 0.81340447 0.91392804 0.42725477 0.53282744 0.62288101
 0.65982698 0.76984343 0.91606989 0.55820719 0.94312679 0.80044031
 0.54721817 0.93059141 0.59032327 0.64004468 0.41548808 0.59428859
 0.74292939]
Epoch: [426/484] Iter:[0/24], Time: 1.00, lr: [0.0014815774821195626], Loss: 0.132260
Epoch: [426/484] Iter:[10/24], Time: 0.45, lr: [0.0014719948674748156], Loss: 0.104804
Epoch: [426/484] Iter:[20/24], Time: 0.42, lr: [0.001462405316382685], Loss: 0.112234
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97564242 0.8084488  0.91292521 0.4323709  0.51157318 0.61745372
 0.6487009  0.76305181 0.91503457 0.54889601 0.94293926 0.79353429
 0.5377023  0.92187453 0.48526734 0.56696371 0.44438693 0.56961398
 0.73451611] 0.6910997878529588
1 [0.97633393 0.81441995 0.91417389 0.45776405 0.53044989 0.62406719
 0.66274533 0.77047458 0.91626634 0.56493731 0.94361516 0.79974663
 0.55579831 0.92672025 0.55540168 0.61880612 0.4845707  0.58883313
 0.74253706] 0.7077716575620481
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.241, MeanIU:  0.7078, Best_mIoU:  0.7130
[0.97633393 0.81441995 0.91417389 0.45776405 0.53044989 0.62406719
 0.66274533 0.77047458 0.91626634 0.56493731 0.94361516 0.79974663
 0.55579831 0.92672025 0.55540168 0.61880612 0.4845707  0.58883313
 0.74253706]
Epoch: [427/484] Iter:[0/24], Time: 1.00, lr: [0.001458567541307135], Loss: 0.120796
Epoch: [427/484] Iter:[10/24], Time: 0.44, lr: [0.001448968185531692], Loss: 0.101620
Epoch: [427/484] Iter:[20/24], Time: 0.42, lr: [0.0014393617583408022], Loss: 0.103769
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97516575 0.80624149 0.91368795 0.44016847 0.51900032 0.61286367
 0.64783981 0.76268769 0.91442893 0.55510038 0.94147566 0.79153287
 0.53036012 0.92500012 0.49901627 0.59037898 0.43406336 0.56969389
 0.73375934] 0.6927613197902627
1 [0.97566339 0.81134437 0.91611142 0.48353046 0.54121361 0.61775969
 0.6610495  0.76979432 0.91599669 0.57128372 0.94233207 0.7979922
 0.54902954 0.93160466 0.57988048 0.66163274 0.451406   0.59006493
 0.74307052] 0.7110926483058229
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.241, MeanIU:  0.7111, Best_mIoU:  0.7130
[0.97566339 0.81134437 0.91611142 0.48353046 0.54121361 0.61775969
 0.6610495  0.76979432 0.91599669 0.57128372 0.94233207 0.7979922
 0.54902954 0.93160466 0.57988048 0.66163274 0.451406   0.59006493
 0.74307052]
Epoch: [428/484] Iter:[0/24], Time: 0.97, lr: [0.001435517194567526], Loss: 0.083709
Epoch: [428/484] Iter:[10/24], Time: 0.44, lr: [0.0014259007702115026], Loss: 0.101718
Epoch: [428/484] Iter:[20/24], Time: 0.42, lr: [0.0014162771343680322], Loss: 0.106483
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97505789 0.80363226 0.91254963 0.42519185 0.50240124 0.61647852
 0.64560126 0.76111931 0.91426002 0.54194341 0.94079932 0.79217082
 0.53090332 0.92343516 0.49337681 0.58656688 0.42893113 0.55850731
 0.73289688] 0.6887275279731966
1 [0.97549693 0.80772797 0.91417077 0.45365223 0.5235203  0.62237546
 0.65915886 0.76851171 0.91566954 0.55585917 0.94046105 0.79787395
 0.54984499 0.92993295 0.56791345 0.65148532 0.45532876 0.57900203
 0.74073933] 0.7057223558855364
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7057, Best_mIoU:  0.7130
[0.97549693 0.80772797 0.91417077 0.45365223 0.5235203  0.62237546
 0.65915886 0.76851171 0.91566954 0.55585917 0.94046105 0.79787395
 0.54984499 0.92993295 0.56791345 0.65148532 0.45532876 0.57900203
 0.74073933]
Epoch: [429/484] Iter:[0/24], Time: 0.99, lr: [0.0014124256474200181], Loss: 0.117097
Epoch: [429/484] Iter:[10/24], Time: 0.45, lr: [0.0014027918145844839], Loss: 0.113388
Epoch: [429/484] Iter:[20/24], Time: 0.42, lr: [0.0013931506247954173], Loss: 0.123048
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97447239 0.7986871  0.91216095 0.43940972 0.49029845 0.61944154
 0.64820077 0.76026966 0.9150238  0.54363196 0.94111429 0.79325472
 0.53901187 0.92518008 0.49944338 0.58830483 0.41772314 0.56867715
 0.7395815 ] 0.6902045952888605
1 [0.97476186 0.80336364 0.91365219 0.46718982 0.51315569 0.62622106
 0.65928442 0.76888388 0.91624188 0.56137993 0.94158789 0.79964978
 0.55405439 0.9310258  0.57323529 0.65048144 0.42007945 0.59191049
 0.74769752] 0.7059924435657116
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7060, Best_mIoU:  0.7130
[0.97476186 0.80336364 0.91365219 0.46718982 0.51315569 0.62622106
 0.65928442 0.76888388 0.91624188 0.56137993 0.94158789 0.79964978
 0.55405439 0.9310258  0.57323529 0.65048144 0.42007945 0.59191049
 0.74769752]
Epoch: [430/484] Iter:[0/24], Time: 0.97, lr: [0.0013892920750167245], Loss: 0.184110
Epoch: [430/484] Iter:[10/24], Time: 0.44, lr: [0.0013796404806421326], Loss: 0.112427
Epoch: [430/484] Iter:[20/24], Time: 0.42, lr: [0.001369981378141472], Loss: 0.117014
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97472865 0.80052721 0.91278855 0.43759389 0.50036876 0.61922023
 0.64972713 0.76200254 0.91491131 0.53864308 0.94214951 0.78993804
 0.53343606 0.92420274 0.49030162 0.58075015 0.42742828 0.57166856
 0.73950413] 0.6899942328838402
1 [0.97508972 0.80514176 0.91479858 0.46880677 0.51812358 0.62580619
 0.66046604 0.76996047 0.91599556 0.5544683  0.94229657 0.79584787
 0.5509448  0.92861381 0.55938622 0.62904466 0.43916731 0.59314128
 0.74706926] 0.7049562497373209
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7050, Best_mIoU:  0.7130
[0.97508972 0.80514176 0.91479858 0.46880677 0.51812358 0.62580619
 0.66046604 0.76996047 0.91599556 0.5544683  0.94229657 0.79584787
 0.5509448  0.92861381 0.55938622 0.62904466 0.43916731 0.59314128
 0.74706926]
Epoch: [431/484] Iter:[0/24], Time: 0.96, lr: [0.0013661156203973032], Loss: 0.066692
Epoch: [431/484] Iter:[10/24], Time: 0.44, lr: [0.0013564458974971152], Loss: 0.095385
Epoch: [431/484] Iter:[20/24], Time: 0.42, lr: [0.0013467685092551187], Loss: 0.103963
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97502861 0.80223597 0.91371113 0.44325907 0.51412706 0.61453699
 0.65009129 0.75937152 0.91409934 0.53176718 0.94200387 0.79135335
 0.52762538 0.92373103 0.4953343  0.59553359 0.44102933 0.56742872
 0.7339152 ] 0.6913780496694045
1 [0.97534196 0.80671465 0.91623835 0.47768118 0.53574786 0.62145627
 0.66321772 0.76802834 0.91523279 0.5483418  0.9421222  0.79701345
 0.54258794 0.92985984 0.57456008 0.6589199  0.44478278 0.58821652
 0.74086077] 0.7077328640990901
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7077, Best_mIoU:  0.7130
[0.97534196 0.80671465 0.91623835 0.47768118 0.53574786 0.62145627
 0.66321772 0.76802834 0.91523279 0.5483418  0.9421222  0.79701345
 0.54258794 0.92985984 0.57456008 0.6589199  0.44478278 0.58821652
 0.74086077]
Epoch: [432/484] Iter:[0/24], Time: 0.96, lr: [0.0013428953926084735], Loss: 0.087583
Epoch: [432/484] Iter:[10/24], Time: 0.44, lr: [0.0013332071594426306], Loss: 0.124566
Epoch: [432/484] Iter:[20/24], Time: 0.42, lr: [0.0013235110973124978], Loss: 0.134631
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97417424 0.79990912 0.91045078 0.43994868 0.48549838 0.6136227
 0.64595731 0.75692294 0.91407618 0.54784612 0.94248146 0.78968217
 0.52119862 0.92517113 0.4925283  0.58849766 0.40890911 0.5500712
 0.73012474] 0.6861616230495592
1 [0.97447778 0.80385224 0.91198113 0.46685117 0.50345284 0.62022645
 0.65882732 0.76537277 0.91520009 0.5636142  0.94247454 0.79576938
 0.53859978 0.93071276 0.57602806 0.64960537 0.40791567 0.56871049
 0.73624499] 0.7015745809068173
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.247, MeanIU:  0.7016, Best_mIoU:  0.7130
[0.97447778 0.80385224 0.91198113 0.46685117 0.50345284 0.62022645
 0.65882732 0.76537277 0.91520009 0.5636142  0.94247454 0.79576938
 0.53859978 0.93071276 0.57602806 0.64960537 0.40791567 0.56871049
 0.73624499]
Epoch: [433/484] Iter:[0/24], Time: 0.95, lr: [0.0013196304646751007], Loss: 0.078643
Epoch: [433/484] Iter:[10/24], Time: 0.44, lr: [0.0013099233238573465], Loss: 0.103062
Epoch: [433/484] Iter:[20/24], Time: 0.41, lr: [0.00130020818365304], Loss: 0.108288
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97380326 0.79568761 0.91352162 0.43227719 0.50476098 0.61625206
 0.64579903 0.75977964 0.91440765 0.53930192 0.94096175 0.79017166
 0.52158091 0.92525785 0.50357678 0.58947638 0.43147183 0.54295573
 0.73377097] 0.6881481493240208
1 [0.97417586 0.79966195 0.91586409 0.45958454 0.5261804  0.62349622
 0.66161978 0.7678097  0.9160157  0.55448497 0.94140148 0.79639446
 0.53966821 0.93220399 0.60050706 0.66409588 0.43895483 0.56435947
 0.74156561] 0.7062128523777753
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7062, Best_mIoU:  0.7130
[0.97417586 0.79966195 0.91586409 0.45958454 0.5261804  0.62349622
 0.66161978 0.7678097  0.9160157  0.55448497 0.94140148 0.79639446
 0.53966821 0.93220399 0.60050706 0.66409588 0.43895483 0.56435947
 0.74156561]
Epoch: [434/484] Iter:[0/24], Time: 0.94, lr: [0.00129631987140794], Loss: 0.092657
Epoch: [434/484] Iter:[10/24], Time: 0.44, lr: [0.0012865934089402928], Loss: 0.111997
Epoch: [434/484] Iter:[20/24], Time: 0.42, lr: [0.0012768587694384737], Loss: 0.111295
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97523299 0.80559011 0.9131048  0.41315556 0.51008745 0.61393994
 0.64742025 0.75859458 0.91454663 0.53856182 0.93885516 0.79259894
 0.53576943 0.92178998 0.50848626 0.58367254 0.45942361 0.55332185
 0.7378018 ] 0.690629141151453
1 [0.97566213 0.80938982 0.9149241  0.43673837 0.52922479 0.62024043
 0.65999593 0.7664844  0.91627812 0.55321721 0.9392224  0.79875684
 0.54796908 0.92806491 0.58319976 0.64577663 0.47188096 0.57378844
 0.7454638 ] 0.706119901494572
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7061, Best_mIoU:  0.7130
[0.97566213 0.80938982 0.9149241  0.43673837 0.52922479 0.62024043
 0.65999593 0.7664844  0.91627812 0.55321721 0.9392224  0.79875684
 0.54796908 0.92806491 0.58319976 0.64577663 0.47188096 0.57378844
 0.7454638 ]
Epoch: [435/484] Iter:[0/24], Time: 0.94, lr: [0.0012729626070314018], Loss: 0.141547
Epoch: [435/484] Iter:[10/24], Time: 0.44, lr: [0.0012632163912582035], Loss: 0.120853
Epoch: [435/484] Iter:[20/24], Time: 0.42, lr: [0.0012534618131163362], Loss: 0.116665
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97476572 0.80303477 0.91325873 0.40816653 0.50113649 0.61809617
 0.64703345 0.75938504 0.91491148 0.53708316 0.94132485 0.7950108
 0.5441506  0.92398367 0.50608374 0.58471079 0.44800181 0.55284223
 0.7369756 ] 0.6899976655668842
1 [0.9751544  0.80698313 0.91551108 0.4438665  0.519996   0.62475988
 0.66005079 0.76702439 0.91645364 0.5516375  0.94278228 0.80079352
 0.55593079 0.93028558 0.58877047 0.6634551  0.49351421 0.57256586
 0.74425618] 0.7091469103439922
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7091, Best_mIoU:  0.7130
[0.9751544  0.80698313 0.91551108 0.4438665  0.519996   0.62475988
 0.66005079 0.76702439 0.91645364 0.5516375  0.94278228 0.80079352
 0.55593079 0.93028558 0.58877047 0.6634551  0.49351421 0.57256586
 0.74425618]
Epoch: [436/484] Iter:[0/24], Time: 0.96, lr: [0.0012495576226125363], Loss: 0.144354
Epoch: [436/484] Iter:[10/24], Time: 0.44, lr: [0.0012397912030856096], Loss: 0.109969
Epoch: [436/484] Iter:[20/24], Time: 0.42, lr: [0.0012300162276672982], Loss: 0.110320
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97499092 0.80516055 0.91309569 0.39577393 0.50289838 0.61479508
 0.64608286 0.76202896 0.91454931 0.54581602 0.94139141 0.79188325
 0.53766741 0.92334897 0.51028961 0.57343257 0.41408485 0.55447965
 0.73800682] 0.6873566447156849
1 [0.97548339 0.8096447  0.91445262 0.39550795 0.52260301 0.62272469
 0.66007173 0.76965234 0.91593449 0.56250711 0.94249546 0.79751578
 0.55295208 0.92765072 0.5921635  0.61244816 0.41396231 0.57198863
 0.7454249 ] 0.7002728193300217
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7003, Best_mIoU:  0.7130
[0.97548339 0.8096447  0.91445262 0.39550795 0.52260301 0.62272469
 0.66007173 0.76965234 0.91593449 0.56250711 0.94249546 0.79751578
 0.55295208 0.92765072 0.5921635  0.61244816 0.41396231 0.57198863
 0.7454249 ]
Epoch: [437/484] Iter:[0/24], Time: 0.97, lr: [0.0012261038232701245], Loss: 0.128655
Epoch: [437/484] Iter:[10/24], Time: 0.44, lr: [0.0012163167295154026], Loss: 0.112703
Epoch: [437/484] Iter:[20/24], Time: 0.42, lr: [0.0012065208776128552], Loss: 0.111959
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97489534 0.80568803 0.91412109 0.42778238 0.51270474 0.62196976
 0.64631039 0.76106948 0.91507364 0.537816   0.94053411 0.79489806
 0.54596769 0.92416923 0.5115943  0.57710266 0.45520475 0.55094867
 0.73449696] 0.6922288053735763
1 [0.97543643 0.81063854 0.91606283 0.45179734 0.53440405 0.62889319
 0.65927743 0.76957173 0.91642023 0.5506695  0.94152043 0.80131582
 0.56104291 0.93055992 0.60754823 0.64531256 0.4635372  0.56539389
 0.74165089] 0.70900279476615
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.240, MeanIU:  0.7090, Best_mIoU:  0.7130
[0.97543643 0.81063854 0.91606283 0.45179734 0.53440405 0.62889319
 0.65927743 0.76957173 0.91642023 0.5506695  0.94152043 0.80131582
 0.56104291 0.93055992 0.60754823 0.64531256 0.4635372  0.56539389
 0.74165089]
Epoch: [438/484] Iter:[0/24], Time: 0.96, lr: [0.001202600065139951], Loss: 0.138414
Epoch: [438/484] Iter:[10/24], Time: 0.44, lr: [0.0011927918053147427], Loss: 0.116496
Epoch: [438/484] Iter:[20/24], Time: 0.42, lr: [0.001182974575756906], Loss: 0.108050
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97541765 0.80582887 0.91309149 0.43757052 0.50638364 0.61992005
 0.64599064 0.75936842 0.91498412 0.54179985 0.94173779 0.79433649
 0.54837809 0.92491153 0.50120774 0.58729438 0.42018484 0.55820013
 0.73793491] 0.6912916393090917
1 [0.97576334 0.80988644 0.91536347 0.4694324  0.52939297 0.62724083
 0.65969723 0.76759703 0.91669062 0.5588801  0.94295401 0.80047996
 0.56194022 0.93078502 0.59197606 0.65735164 0.43030703 0.57282729
 0.74534361] 0.7086268041847952
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.241, MeanIU:  0.7086, Best_mIoU:  0.7130
[0.97576334 0.80988644 0.91536347 0.4694324  0.52939297 0.62724083
 0.65969723 0.76759703 0.91669062 0.5588801  0.94295401 0.80047996
 0.56194022 0.93078502 0.59197606 0.65735164 0.43030703 0.57282729
 0.74534361]
Epoch: [439/484] Iter:[0/24], Time: 0.99, lr: [0.0011790451520691976], Loss: 0.069983
Epoch: [439/484] Iter:[10/24], Time: 0.45, lr: [0.0011692152114977329], Loss: 0.100296
Epoch: [439/484] Iter:[20/24], Time: 0.42, lr: [0.0011593760796310987], Loss: 0.105655
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97504382 0.80308938 0.91317787 0.43906131 0.5018455  0.62131641
 0.64640783 0.76112961 0.91522069 0.5315016  0.94154919 0.79429087
 0.54594781 0.92503795 0.50172309 0.58315227 0.42541528 0.55894337
 0.73837166] 0.6906434480459648
1 [0.97528276 0.80588252 0.91548681 0.47479548 0.52495878 0.62784554
 0.65721266 0.76805925 0.91669345 0.54070361 0.94316228 0.80056487
 0.56116857 0.93167305 0.58591834 0.66484472 0.46213828 0.57616945
 0.74613835] 0.7094051985456498
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7094, Best_mIoU:  0.7130
[0.97528276 0.80588252 0.91548681 0.47479548 0.52495878 0.62784554
 0.65721266 0.76805925 0.91669345 0.54070361 0.94316228 0.80056487
 0.56116857 0.93167305 0.58591834 0.66484472 0.46213828 0.57616945
 0.74613835]
Epoch: [440/484] Iter:[0/24], Time: 0.97, lr: [0.001155437832009219], Loss: 0.072470
Epoch: [440/484] Iter:[10/24], Time: 0.44, lr: [0.0011455856715824524], Loss: 0.103653
Epoch: [440/484] Iter:[20/24], Time: 0.42, lr: [0.0011357240876096906], Loss: 0.105932
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97507808 0.80497987 0.91359443 0.42563051 0.50629948 0.61559384
 0.64569292 0.76082682 0.91446127 0.54178228 0.94215226 0.79257804
 0.53830024 0.92281325 0.50157616 0.57348691 0.41781291 0.54940229
 0.73688294] 0.6883655004841108
1 [0.97540605 0.80855767 0.91540247 0.45472041 0.52720253 0.62219127
 0.65760814 0.76770723 0.91627148 0.55976099 0.94414238 0.79848705
 0.55479883 0.92944225 0.59088797 0.63620552 0.44631649 0.56911597
 0.74376595] 0.706210034127393
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7062, Best_mIoU:  0.7130
[0.97540605 0.80855767 0.91540247 0.45472041 0.52720253 0.62219127
 0.65760814 0.76770723 0.91627148 0.55976099 0.94414238 0.79848705
 0.55479883 0.92944225 0.59088797 0.63620552 0.44631649 0.56911597
 0.74376595]
Epoch: [441/484] Iter:[0/24], Time: 0.96, lr: [0.0011317767930716567], Loss: 0.133720
Epoch: [441/484] Iter:[10/24], Time: 0.44, lr: [0.001121901847495315], Loss: 0.119592
Epoch: [441/484] Iter:[20/24], Time: 0.42, lr: [0.0011120172346548254], Loss: 0.116505
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97555243 0.80784716 0.91332255 0.41860144 0.50189699 0.62000522
 0.64699774 0.76022467 0.91438004 0.54097593 0.94218116 0.79407315
 0.54029341 0.92213291 0.49683986 0.58222702 0.39554798 0.55357638
 0.73693024] 0.6875582248091013
1 [0.97590086 0.8128691  0.91533357 0.4466298  0.52410663 0.62727283
 0.66145032 0.7684879  0.9160058  0.56333912 0.94414494 0.79907773
 0.55518243 0.92738753 0.59845582 0.62986852 0.36691123 0.57136509
 0.74458823] 0.702546181833911
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7025, Best_mIoU:  0.7130
[0.97590086 0.8128691  0.91533357 0.4466298  0.52410663 0.62727283
 0.66145032 0.7684879  0.9160058  0.56333912 0.94414494 0.79907773
 0.55518243 0.92738753 0.59845582 0.62986852 0.36691123 0.57136509
 0.74458823]
Epoch: [442/484] Iter:[0/24], Time: 0.96, lr: [0.001108060659207932], Loss: 0.081184
Epoch: [442/484] Iter:[10/24], Time: 0.44, lr: [0.0010981623350804631], Loss: 0.122287
Epoch: [442/484] Iter:[20/24], Time: 0.42, lr: [0.0010882540876474226], Loss: 0.113287
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97538158 0.80674168 0.91397337 0.41556414 0.50629706 0.62209195
 0.65033358 0.76132322 0.91530614 0.53757018 0.94251911 0.79595953
 0.54065391 0.92479021 0.5099699  0.59243291 0.40326921 0.55466713
 0.73476841] 0.6896638533052377
1 [0.97566912 0.80993978 0.91598027 0.44217874 0.52779706 0.6286359
 0.66242523 0.76956339 0.91703218 0.55542555 0.94439231 0.80063504
 0.55498488 0.93062537 0.60623099 0.65117334 0.37699801 0.57109668
 0.74064715] 0.7042858425823715
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7043, Best_mIoU:  0.7130
[0.97566912 0.80993978 0.91598027 0.44217874 0.52779706 0.6286359
 0.66242523 0.76956339 0.91703218 0.55542555 0.94439231 0.80063504
 0.55498488 0.93062537 0.60623099 0.65117334 0.37699801 0.57109668
 0.74064715]
Epoch: [443/484] Iter:[0/24], Time: 0.98, lr: [0.0010842879854662689], Loss: 0.148983
Epoch: [443/484] Iter:[10/24], Time: 0.45, lr: [0.0010743656591656798], Loss: 0.109653
Epoch: [443/484] Iter:[20/24], Time: 0.42, lr: [0.0010644331402522862], Loss: 0.116973
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97595213 0.80970921 0.91390069 0.39862631 0.51075576 0.62128853
 0.64810248 0.76142352 0.91512118 0.53800201 0.94379915 0.78974832
 0.53740397 0.92476672 0.50745331 0.58562284 0.41930338 0.5636443
 0.73620124] 0.6895171074331349
1 [0.9765207  0.81464629 0.91511222 0.40903763 0.52733959 0.6280977
 0.65861501 0.77012377 0.91634973 0.55477748 0.94537387 0.79522132
 0.55169547 0.92937328 0.59748886 0.64101486 0.38824329 0.57858773
 0.74280684] 0.7021276650627281
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7021, Best_mIoU:  0.7130
[0.9765207  0.81464629 0.91511222 0.40903763 0.52733959 0.6280977
 0.65861501 0.77012377 0.91634973 0.55477748 0.94537387 0.79522132
 0.55169547 0.92937328 0.59748886 0.64101486 0.38824329 0.57858773
 0.74280684]
Epoch: [444/484] Iter:[0/24], Time: 0.98, lr: [0.0010604572527736376], Loss: 0.111495
Epoch: [444/484] Iter:[10/24], Time: 0.44, lr: [0.001050510268128959], Loss: 0.114957
Epoch: [444/484] Iter:[20/24], Time: 0.42, lr: [0.001040552807258147], Loss: 0.118554
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9759323  0.8097162  0.91293035 0.3979397  0.50836922 0.62082647
 0.64712603 0.76234904 0.91521009 0.53760649 0.94281033 0.79481333
 0.54064476 0.92490408 0.5061696  0.59046859 0.44534669 0.55972179
 0.73773427] 0.6910852282469845
1 [0.97632975 0.81429792 0.91434379 0.41110834 0.53024245 0.62783212
 0.65974687 0.77022294 0.91668338 0.55749038 0.94427181 0.80082925
 0.5569216  0.93009107 0.59501171 0.65921642 0.45949225 0.57466932
 0.74522954] 0.7075805737006404
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7076, Best_mIoU:  0.7130
[0.97632975 0.81429792 0.91434379 0.41110834 0.53024245 0.62783212
 0.65974687 0.77022294 0.91668338 0.55749038 0.94427181 0.80082925
 0.5569216  0.93009107 0.59501171 0.65921642 0.45949225 0.57466932
 0.74522954]
Epoch: [445/484] Iter:[0/24], Time: 0.96, lr: [0.0010365668621819106], Loss: 0.104795
Epoch: [445/484] Iter:[10/24], Time: 0.44, lr: [0.0010265945279013193], Loss: 0.111816
Epoch: [445/484] Iter:[20/24], Time: 0.42, lr: [0.0010166114183241806], Loss: 0.113701
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97594691 0.81056367 0.9142993  0.41624152 0.51362948 0.62349614
 0.6484356  0.76152664 0.91549882 0.53714832 0.94212765 0.79367977
 0.54499656 0.92484071 0.49933886 0.57502888 0.45070465 0.55714951
 0.7377854 ] 0.6917072840524707
1 [0.97641328 0.81507675 0.91555652 0.43418464 0.53245579 0.62977657
 0.65909368 0.7696427  0.91666716 0.55493605 0.94331997 0.79966094
 0.56158403 0.92997067 0.59287313 0.64577969 0.4720757  0.57155033
 0.74549242] 0.7087426342860956
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.239, MeanIU:  0.7087, Best_mIoU:  0.7130
[0.97641328 0.81507675 0.91555652 0.43418464 0.53245579 0.62977657
 0.65909368 0.7696427  0.91666716 0.55493605 0.94331997 0.79966094
 0.56158403 0.92997067 0.59287313 0.64577969 0.4720757  0.57155033
 0.74549242]
Epoch: [446/484] Iter:[0/24], Time: 0.96, lr: [0.0010126151285080422], Loss: 0.093233
Epoch: [446/484] Iter:[10/24], Time: 0.44, lr: [0.0010026167153311556], Loss: 0.119511
Epoch: [446/484] Iter:[20/24], Time: 0.42, lr: [0.0009926072110535028], Loss: 0.107054
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97621599 0.81100298 0.91317276 0.41418721 0.50628162 0.62006238
 0.64888234 0.76079462 0.91524079 0.53691859 0.93955102 0.79709147
 0.54664774 0.9229264  0.48783906 0.57114934 0.41729312 0.57001328
 0.74185506] 0.6893224091845243
1 [0.97658234 0.81474742 0.91452638 0.43288749 0.52576712 0.62719857
 0.66101331 0.76824925 0.91651496 0.55306171 0.94054916 0.80353854
 0.56451973 0.92835857 0.5701532  0.63378708 0.43401289 0.59174358
 0.75095594] 0.7056930126893536
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7057, Best_mIoU:  0.7130
[0.97658234 0.81474742 0.91452638 0.43288749 0.52576712 0.62719857
 0.66101331 0.76824925 0.91651496 0.55306171 0.94054916 0.80353854
 0.56451973 0.92835857 0.5701532  0.63378708 0.43401289 0.59174358
 0.75095594]
Epoch: [447/484] Iter:[0/24], Time: 1.00, lr: [0.0009886002732868004], Loss: 0.094838
Epoch: [447/484] Iter:[10/24], Time: 0.45, lr: [0.0009785750108233667], Loss: 0.106591
Epoch: [447/484] Iter:[20/24], Time: 0.42, lr: [0.000968538323301121], Loss: 0.109816
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97549085 0.80793774 0.91398093 0.42216136 0.50817727 0.61790996
 0.64661633 0.76047621 0.91488724 0.54130624 0.94132785 0.79516047
 0.5461418  0.92443925 0.49451287 0.58398503 0.43472074 0.56493853
 0.73755376] 0.6911433913784933
1 [0.97590356 0.81231761 0.91600507 0.45346313 0.52825609 0.62410427
 0.65897086 0.76793368 0.91639882 0.55952602 0.94255933 0.80117192
 0.56359706 0.93086881 0.58644883 0.66590283 0.44722005 0.58427146
 0.7460907 ] 0.7095268465029145
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7095, Best_mIoU:  0.7130
[0.97590356 0.81231761 0.91600507 0.45346313 0.52825609 0.62410427
 0.65897086 0.76793368 0.91639882 0.55952602 0.94255933 0.80117192
 0.56359706 0.93086881 0.58644883 0.66590283 0.44722005 0.58427146
 0.7460907 ]
Epoch: [448/484] Iter:[0/24], Time: 0.96, lr: [0.0009645204169410678], Loss: 0.115066
Epoch: [448/484] Iter:[10/24], Time: 0.44, lr: [0.000954467490151902], Loss: 0.109821
Epoch: [448/484] Iter:[20/24], Time: 0.42, lr: [0.0009444027846081778], Loss: 0.101080
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97585269 0.80914597 0.91448462 0.42610243 0.51468888 0.62357712
 0.64857134 0.76339067 0.91507173 0.53924756 0.94056152 0.7950363
 0.53888454 0.92468667 0.49425564 0.57911815 0.43289091 0.56336956
 0.73846087] 0.6914419555247351
1 [0.97624602 0.81360983 0.91618355 0.45683382 0.53544389 0.62998752
 0.66036747 0.77060609 0.91657115 0.55506719 0.94125733 0.8015008
 0.55986062 0.93041021 0.58388205 0.65359965 0.44749533 0.58089091
 0.74724802] 0.7093190250069248
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.239, MeanIU:  0.7093, Best_mIoU:  0.7130
[0.97624602 0.81360983 0.91618355 0.45683382 0.53544389 0.62998752
 0.66036747 0.77060609 0.91657115 0.55506719 0.94125733 0.8015008
 0.55986062 0.93041021 0.58388205 0.65359965 0.44749533 0.58089091
 0.74724802]
Epoch: [449/484] Iter:[0/24], Time: 0.98, lr: [0.0009403735700586821], Loss: 0.160764
Epoch: [449/484] Iter:[10/24], Time: 0.45, lr: [0.0009302921153270493], Loss: 0.126306
Epoch: [449/484] Iter:[20/24], Time: 0.42, lr: [0.0009201985066354681], Loss: 0.115885
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97620872 0.80939077 0.91450979 0.41049512 0.51212631 0.62226996
 0.65112753 0.76375554 0.9151129  0.53498699 0.94312475 0.79568878
 0.54591555 0.92412976 0.49658353 0.57994572 0.45429754 0.57071189
 0.73850587] 0.6925730009588641
1 [0.97648081 0.81339344 0.91608567 0.43121881 0.5324593  0.62931918
 0.66252445 0.77089373 0.91672523 0.55238042 0.94397687 0.80216221
 0.56375866 0.93051341 0.59041526 0.66102509 0.47163117 0.58881237
 0.74715559] 0.7105753504977358
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7106, Best_mIoU:  0.7130
[0.97648081 0.81339344 0.91608567 0.43121881 0.5324593  0.62931918
 0.66252445 0.77089373 0.91672523 0.55238042 0.94397687 0.80216221
 0.56375866 0.93051341 0.59041526 0.66102509 0.47163117 0.58881237
 0.74715559]
Epoch: [450/484] Iter:[0/24], Time: 0.97, lr: [0.0009161576236452944], Loss: 0.088183
Epoch: [450/484] Iter:[10/24], Time: 0.44, lr: [0.0009060467243777279], Loss: 0.083405
Epoch: [450/484] Iter:[20/24], Time: 0.42, lr: [0.0008959232724465671], Loss: 0.099188
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97592977 0.80975554 0.91371593 0.40251668 0.50745817 0.62276714
 0.64893712 0.76245667 0.91501079 0.53876953 0.94293803 0.7950959
 0.54625812 0.92530194 0.49731846 0.5859227  0.42518997 0.56319375
 0.7388994 ] 0.6903913475911889
1 [0.97633088 0.81361991 0.91503334 0.4136858  0.52551527 0.62910494
 0.6609813  0.76965369 0.91630842 0.55334299 0.94378895 0.80153369
 0.56352364 0.93131424 0.58718128 0.65687307 0.43350394 0.58622627
 0.74750285] 0.7065802355202342
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7066, Best_mIoU:  0.7130
[0.97633088 0.81361991 0.91503334 0.4136858  0.52551527 0.62910494
 0.6609813  0.76965369 0.91630842 0.55334299 0.94378895 0.80153369
 0.56352364 0.93131424 0.58718128 0.65687307 0.43350394 0.58622627
 0.74750285]
Epoch: [451/484] Iter:[0/24], Time: 0.95, lr: [0.000891870338199348], Loss: 0.119811
Epoch: [451/484] Iter:[10/24], Time: 0.44, lr: [0.0008817290198836482], Loss: 0.112384
Epoch: [451/484] Iter:[20/24], Time: 0.42, lr: [0.0008715747244632119], Loss: 0.108619
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97572725 0.80770483 0.91409519 0.4083891  0.51249533 0.6206718
 0.647894   0.76136976 0.91452441 0.54207549 0.94221397 0.79439965
 0.54604916 0.92385241 0.49246773 0.57991241 0.42626024 0.55478207
 0.73782661] 0.6896163904923045
1 [0.97617047 0.81232387 0.91571337 0.42553161 0.53202537 0.62691947
 0.6594081  0.76786462 0.91620259 0.56034888 0.94314377 0.80050471
 0.56273262 0.92995855 0.57533195 0.6541113  0.4467719  0.57507023
 0.74610309] 0.7066440250057697
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7066, Best_mIoU:  0.7130
[0.97617047 0.81232387 0.91571337 0.42553161 0.53202537 0.62691947
 0.6594081  0.76786462 0.91620259 0.56034888 0.94314377 0.80050471
 0.56273262 0.92995855 0.57533195 0.6541113  0.4467719  0.57507023
 0.74610309]
Epoch: [452/484] Iter:[0/24], Time: 0.96, lr: [0.0008675093314266722], Loss: 0.097032
Epoch: [452/484] Iter:[10/24], Time: 0.45, lr: [0.0008573365560612182], Loss: 0.108073
Epoch: [452/484] Iter:[20/24], Time: 0.42, lr: [0.0008471503508819597], Loss: 0.106993
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97594732 0.80953164 0.91408926 0.41775033 0.5107289  0.62156184
 0.64723127 0.76069762 0.91490656 0.53839881 0.94296063 0.79394322
 0.54390785 0.92441197 0.49620699 0.57937297 0.40958647 0.56273538
 0.73778169] 0.6895658262957131
1 [0.9763067  0.81368916 0.91557675 0.43568617 0.52926737 0.62789057
 0.65924244 0.76765292 0.91624954 0.55582324 0.94416391 0.80085962
 0.56231151 0.93033012 0.58006177 0.64616416 0.41257466 0.58578669
 0.74571507] 0.70554486196835
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7055, Best_mIoU:  0.7130
[0.9763067  0.81368916 0.91557675 0.43568617 0.52926737 0.62789057
 0.65924244 0.76765292 0.91624954 0.55582324 0.94416391 0.80085962
 0.56231151 0.93033012 0.58006177 0.64616416 0.41257466 0.58578669
 0.74571507]
Epoch: [453/484] Iter:[0/24], Time: 0.97, lr: [0.0008430720643773877], Loss: 0.100388
Epoch: [453/484] Iter:[10/24], Time: 0.45, lr: [0.0008328667241690389], Loss: 0.101117
Epoch: [453/484] Iter:[20/24], Time: 0.42, lr: [0.0008226474702996507], Loss: 0.103287
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.976123   0.81157007 0.91372581 0.40907505 0.50799389 0.62262686
 0.6467834  0.76124973 0.91464334 0.54322149 0.94313783 0.79491312
 0.53764445 0.92232992 0.49377681 0.56721611 0.42730007 0.55017346
 0.7326109 ] 0.688216596415686
1 [0.97652097 0.81600293 0.91551441 0.42859641 0.52781071 0.62860106
 0.65947685 0.76820481 0.91591812 0.56120031 0.94441187 0.80129695
 0.55662122 0.92838737 0.57523918 0.63280313 0.45261057 0.56785097
 0.73890277] 0.7050510852705133
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7051, Best_mIoU:  0.7130
[0.97652097 0.81600293 0.91551441 0.42859641 0.52781071 0.62860106
 0.65947685 0.76820481 0.91591812 0.56120031 0.94441187 0.80129695
 0.55662122 0.92838737 0.57523918 0.63280313 0.45261057 0.56785097
 0.73890277]
Epoch: [454/484] Iter:[0/24], Time: 0.95, lr: [0.0008185558257448763], Loss: 0.148383
Epoch: [454/484] Iter:[10/24], Time: 0.44, lr: [0.0008083167359520834], Loss: 0.110878
Epoch: [454/484] Iter:[20/24], Time: 0.42, lr: [0.0007980632142440923], Loss: 0.110134
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97607779 0.80994249 0.9145946  0.42600321 0.51179527 0.62214849
 0.64932843 0.76280931 0.91475386 0.54144037 0.94292462 0.79418979
 0.53671426 0.92501816 0.50023429 0.58614291 0.42372593 0.55327474
 0.73798932] 0.6910056752361827
1 [0.97633954 0.81399813 0.91635925 0.46300961 0.533132   0.62874912
 0.66239159 0.77014875 0.91620701 0.56033927 0.9437535  0.80013241
 0.55610657 0.93177323 0.59869861 0.66394265 0.42567692 0.57270208
 0.74598818] 0.7094446535611629
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.240, MeanIU:  0.7094, Best_mIoU:  0.7130
[0.97633954 0.81399813 0.91635925 0.46300961 0.533132   0.62874912
 0.66239159 0.77014875 0.91620701 0.56033927 0.9437535  0.80013241
 0.55610657 0.93177323 0.59869861 0.66394265 0.42567692 0.57270208
 0.74598818]
Epoch: [455/484] Iter:[0/24], Time: 1.01, lr: [0.0007939577140135427], Loss: 0.172403
Epoch: [455/484] Iter:[10/24], Time: 0.45, lr: [0.0007836836047854768], Loss: 0.119112
Epoch: [455/484] Iter:[20/24], Time: 0.42, lr: [0.0007733945072426306], Loss: 0.124973
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97600757 0.8100255  0.91259443 0.41467651 0.49835766 0.61832062
 0.64541675 0.76224686 0.91432334 0.54249767 0.9412628  0.79324019
 0.5315035  0.92221082 0.48441165 0.57076105 0.40563778 0.54970503
 0.73676979] 0.6857878696551218
1 [0.97632524 0.81310924 0.91321581 0.4196579  0.51375927 0.62274114
 0.65726783 0.76921189 0.91528893 0.55790772 0.94112701 0.79886189
 0.55255377 0.92687056 0.54808512 0.62217731 0.41729676 0.56703624
 0.74399567] 0.6987625944638189
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.6988, Best_mIoU:  0.7130
[0.97632524 0.81310924 0.91321581 0.4196579  0.51375927 0.62274114
 0.65726783 0.76921189 0.91528893 0.55790772 0.94112701 0.79886189
 0.55255377 0.92687056 0.54808512 0.62217731 0.41729676 0.56703624
 0.74399567]
Epoch: [456/484] Iter:[0/24], Time: 0.95, lr: [0.0007692746170760122], Loss: 0.091437
Epoch: [456/484] Iter:[10/24], Time: 0.44, lr: [0.000758964124106303], Loss: 0.118649
Epoch: [456/484] Iter:[20/24], Time: 0.42, lr: [0.000748638043981525], Loss: 0.119362
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97607504 0.81161767 0.91381306 0.42017355 0.50589789 0.61678412
 0.64904066 0.76266985 0.91420518 0.54796874 0.94223588 0.79201728
 0.52832424 0.92208659 0.48854538 0.55314651 0.43822572 0.5537641
 0.73808419] 0.6881408228254795
1 [0.9763685  0.81453886 0.9139976  0.40936863 0.51686865 0.6214911
 0.66060017 0.77005473 0.91485662 0.56392456 0.94205897 0.79757126
 0.54925282 0.92450862 0.54408793 0.58461903 0.43961971 0.56899803
 0.74505632] 0.6977811632597479
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.6978, Best_mIoU:  0.7130
[0.9763685  0.81453886 0.9139976  0.40936863 0.51686865 0.6214911
 0.66060017 0.77005473 0.91485662 0.56392456 0.94205897 0.79757126
 0.54925282 0.92450862 0.54408793 0.58461903 0.43961971 0.56899803
 0.74505632]
Epoch: [457/484] Iter:[0/24], Time: 0.98, lr: [0.0007445031888574533], Loss: 0.238184
Epoch: [457/484] Iter:[10/24], Time: 0.44, lr: [0.0007341548426304642], Loss: 0.124589
Epoch: [457/484] Iter:[20/24], Time: 0.42, lr: [0.000723790263008328], Loss: 0.123839
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9756554  0.80880547 0.91414391 0.42188001 0.50968506 0.61975931
 0.65027332 0.75978176 0.91439068 0.54881957 0.94246222 0.79109174
 0.52820892 0.92449536 0.49654735 0.57098651 0.46192408 0.55461027
 0.74041087] 0.6912595683315075
1 [0.97590322 0.81262621 0.91486279 0.41885899 0.5240688  0.62634147
 0.66241517 0.768268   0.91526867 0.56558773 0.94251922 0.7972126
 0.54676554 0.9283938  0.56927546 0.63366896 0.5004629  0.56771604
 0.7471094 ] 0.7061749980823123
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7062, Best_mIoU:  0.7130
[0.97590322 0.81262621 0.91486279 0.41885899 0.5240688  0.62634147
 0.66241517 0.768268   0.91526867 0.56558773 0.94251922 0.7972126
 0.54676554 0.9283938  0.56927546 0.63366896 0.5004629  0.56771604
 0.7471094 ]
Epoch: [458/484] Iter:[0/24], Time: 0.98, lr: [0.0007196398223800099], Loss: 0.180157
Epoch: [458/484] Iter:[10/24], Time: 0.45, lr: [0.0007092520357359116], Loss: 0.114754
Epoch: [458/484] Iter:[20/24], Time: 0.42, lr: [0.0006988473163012868], Loss: 0.128005
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97581961 0.80930522 0.91352005 0.41928189 0.4965655  0.62115116
 0.64773457 0.76047169 0.91463822 0.54795542 0.94221904 0.79439768
 0.54072263 0.92544925 0.50695639 0.58691993 0.43004071 0.56181212
 0.7390792 ] 0.6912652775491125
1 [0.97608604 0.81296055 0.91475898 0.43346517 0.51350783 0.62850403
 0.66127987 0.76910888 0.91559422 0.56690819 0.94217223 0.8000467
 0.55739806 0.93063464 0.58882002 0.65565864 0.42302159 0.57877094
 0.74557196] 0.7060141326616852
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.241, MeanIU:  0.7060, Best_mIoU:  0.7130
[0.97608604 0.81296055 0.91475898 0.43346517 0.51350783 0.62850403
 0.66127987 0.76910888 0.91559422 0.56690819 0.94217223 0.8000467
 0.55739806 0.93063464 0.58882002 0.65565864 0.42302159 0.57877094
 0.74557196]
Epoch: [459/484] Iter:[0/24], Time: 1.00, lr: [0.0006946806185667451], Loss: 0.116521
Epoch: [459/484] Iter:[10/24], Time: 0.45, lr: [0.0006842516722454517], Loss: 0.109085
Epoch: [459/484] Iter:[20/24], Time: 0.42, lr: [0.0006738050338654253], Loss: 0.117745
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97604568 0.81024726 0.91359639 0.42018826 0.4998992  0.622987
 0.6489152  0.76194098 0.91471655 0.53800089 0.94288502 0.79349024
 0.54104192 0.92450468 0.49618376 0.58151542 0.45653993 0.56305542
 0.73883462] 0.6918204430466348
1 [0.97630782 0.81380817 0.91526092 0.43720036 0.52005254 0.62922217
 0.66039947 0.76967327 0.91578698 0.55119328 0.94342171 0.79987775
 0.55781092 0.93022872 0.5681864  0.66242921 0.48994179 0.58268688
 0.74621752] 0.7089318890848487
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7089, Best_mIoU:  0.7130
[0.97630782 0.81380817 0.91526092 0.43720036 0.52005254 0.62922217
 0.66039947 0.76967327 0.91578698 0.55119328 0.94342171 0.79987775
 0.55781092 0.93022872 0.5681864  0.66242921 0.48994179 0.58268688
 0.74621752]
Epoch: [460/484] Iter:[0/24], Time: 0.97, lr: [0.0006696213499130277], Loss: 0.124542
Epoch: [460/484] Iter:[10/24], Time: 0.44, lr: [0.0006591493756514041], Loss: 0.114667
Epoch: [460/484] Iter:[20/24], Time: 0.42, lr: [0.0006486588823019564], Loss: 0.110513
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97592309 0.80917599 0.91410292 0.42296997 0.50474242 0.62157798
 0.64965858 0.76116958 0.91429526 0.54318028 0.94245444 0.79295961
 0.53650375 0.92385011 0.49573373 0.5752177  0.4387897  0.56469444
 0.73855981] 0.6908189131230564
1 [0.97605661 0.81255479 0.9158118  0.44407901 0.52571358 0.62805815
 0.66236801 0.76881174 0.91556408 0.56099693 0.94295089 0.79858691
 0.55334793 0.92929747 0.57355883 0.63873006 0.4307619  0.58367763
 0.74532507] 0.7055921769160894
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7056, Best_mIoU:  0.7130
[0.97605661 0.81255479 0.9158118  0.44407901 0.52571358 0.62805815
 0.66236801 0.76881174 0.91556408 0.56099693 0.94295089 0.79858691
 0.55334793 0.92929747 0.57355883 0.63873006 0.4307619  0.58367763
 0.74532507]
Epoch: [461/484] Iter:[0/24], Time: 1.01, lr: [0.0006444574179307085], Loss: 0.166193
Epoch: [461/484] Iter:[10/24], Time: 0.45, lr: [0.0006339403785756915], Loss: 0.127457
Epoch: [461/484] Iter:[20/24], Time: 0.42, lr: [0.0006234039160192827], Loss: 0.117435
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97618615 0.81118755 0.91386422 0.4281844  0.5031153  0.62096209
 0.65062601 0.75968505 0.91454126 0.54800736 0.94282034 0.79477064
 0.54278926 0.92398061 0.49820968 0.58436326 0.46445653 0.56169143
 0.73857502] 0.6935797978082018
1 [0.97647116 0.81521819 0.9158226  0.45666362 0.52478941 0.62675603
 0.66270205 0.76692585 0.91605237 0.56701732 0.94352694 0.80056566
 0.55778816 0.93070195 0.588088   0.67051085 0.47800165 0.58187535
 0.74572298] 0.7118526381976296
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7119, Best_mIoU:  0.7130
[0.97647116 0.81521819 0.9158226  0.45666362 0.52478941 0.62675603
 0.66270205 0.76692585 0.91605237 0.56701732 0.94352694 0.80056566
 0.55778816 0.93070195 0.588088   0.67051085 0.47800165 0.58187535
 0.74572298]
Epoch: [462/484] Iter:[0/24], Time: 0.94, lr: [0.0006191838029789736], Loss: 0.129502
Epoch: [462/484] Iter:[10/24], Time: 0.44, lr: [0.0006086194689316551], Loss: 0.104551
Epoch: [462/484] Iter:[20/24], Time: 0.42, lr: [0.0005980347193857134], Loss: 0.111659
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97589208 0.81148676 0.91436248 0.40055343 0.50419415 0.62421566
 0.65149016 0.76296997 0.9142042  0.55366306 0.94385675 0.79456987
 0.53792479 0.92438863 0.50175451 0.57375144 0.44839783 0.56368487
 0.7339051 ] 0.69132977633189
1 [0.97615762 0.81533488 0.91577369 0.39867953 0.52284877 0.62971912
 0.66364917 0.77030302 0.91528234 0.57053647 0.94479802 0.80010679
 0.55604697 0.93030519 0.59338874 0.65246359 0.46908842 0.57924703
 0.7392814 ] 0.7075268812522995
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7075, Best_mIoU:  0.7130
[0.97615762 0.81533488 0.91577369 0.39867953 0.52284877 0.62971912
 0.66364917 0.77030302 0.91528234 0.57053647 0.94479802 0.80010679
 0.55604697 0.93030519 0.59338874 0.65246359 0.46908842 0.57924703
 0.7392814 ]
Epoch: [463/484] Iter:[0/24], Time: 0.96, lr: [0.0005937950047097665], Loss: 0.095807
Epoch: [463/484] Iter:[10/24], Time: 0.44, lr: [0.0005831809258185771], Loss: 0.101288
Epoch: [463/484] Iter:[20/24], Time: 0.42, lr: [0.0005725453376319317], Loss: 0.108074
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97613258 0.81121984 0.91408196 0.4041768  0.50494447 0.6227788
 0.65207992 0.7626401  0.91455985 0.55202382 0.94235078 0.79543652
 0.53806696 0.92448315 0.50151025 0.58455377 0.45517594 0.56013356
 0.73817783] 0.6923435210813488
1 [0.97630135 0.81440952 0.91562589 0.41361014 0.52596886 0.62843784
 0.6648694  0.77020359 0.91592104 0.56901683 0.94281047 0.80091764
 0.55586487 0.93129575 0.59343551 0.6734401  0.47005472 0.58045531
 0.7449667 ] 0.7098739754197189
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.240, MeanIU:  0.7099, Best_mIoU:  0.7130
[0.97630135 0.81440952 0.91562589 0.41361014 0.52596886 0.62843784
 0.6648694  0.77020359 0.91592104 0.56901683 0.94281047 0.80091764
 0.55586487 0.93129575 0.59343551 0.6734401  0.47005472 0.58045531
 0.7449667 ]
Epoch: [464/484] Iter:[0/24], Time: 0.97, lr: [0.0005682849708384633], Loss: 0.115399
Epoch: [464/484] Iter:[10/24], Time: 0.44, lr: [0.0005576184425934621], Loss: 0.109309
Epoch: [464/484] Iter:[20/24], Time: 0.42, lr: [0.0005469291936449633], Loss: 0.109505
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97600476 0.80985691 0.9144246  0.40529191 0.5020725  0.61951699
 0.65277346 0.7625532  0.91462534 0.54643562 0.94259631 0.79420847
 0.53449765 0.92240597 0.50123987 0.57308002 0.44655212 0.56455396
 0.73850788] 0.6905893435604408
1 [0.97612767 0.81330342 0.91546132 0.40470826 0.52003587 0.62550346
 0.66516617 0.76992428 0.91566462 0.56526453 0.9435719  0.7995982
 0.55279847 0.92825211 0.58870666 0.64279086 0.45143244 0.58615516
 0.74539068] 0.705781899043354
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7058, Best_mIoU:  0.7130
[0.97612767 0.81330342 0.91546132 0.40470826 0.52003587 0.62550346
 0.66516617 0.76992428 0.91566462 0.56526453 0.9435719  0.7995982
 0.55279847 0.92825211 0.58870666 0.64279086 0.45143244 0.58615516
 0.74539068]
Epoch: [465/484] Iter:[0/24], Time: 0.98, lr: [0.0005426470112488497], Loss: 0.166585
Epoch: [465/484] Iter:[10/24], Time: 0.45, lr: [0.0005319250337646522], Loss: 0.118583
Epoch: [465/484] Iter:[20/24], Time: 0.42, lr: [0.0005211789868808419], Loss: 0.110341
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97609158 0.81075254 0.91472197 0.42130756 0.50582327 0.61868386
 0.65145479 0.76208422 0.91462077 0.54950924 0.94208896 0.79400811
 0.53616263 0.92447642 0.50806767 0.5838716  0.43707491 0.56407619
 0.73922071] 0.6923208940501429
1 [0.97638487 0.81495235 0.9163771  0.43942528 0.5249361  0.62543103
 0.66413751 0.76948221 0.9159164  0.56722864 0.94233262 0.79979006
 0.55209266 0.9304386  0.59668378 0.6616271  0.43236539 0.5855816
 0.74695061] 0.7085333636418832
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7085, Best_mIoU:  0.7130
[0.97638487 0.81495235 0.9163771  0.43942528 0.5249361  0.62543103
 0.66413751 0.76948221 0.9159164  0.56722864 0.94233262 0.79979006
 0.55209266 0.9304386  0.59668378 0.6616271  0.43236539 0.5855816
 0.74695061]
Epoch: [466/484] Iter:[0/24], Time: 0.96, lr: [0.0005168736934759387], Loss: 0.088584
Epoch: [466/484] Iter:[10/24], Time: 0.44, lr: [0.0005060929212442119], Loss: 0.107489
Epoch: [466/484] Iter:[20/24], Time: 0.42, lr: [0.0004952865693491596], Loss: 0.112494
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97598186 0.80952409 0.91432492 0.40086872 0.50382507 0.62277132
 0.65086488 0.76198435 0.91456311 0.54165756 0.94318342 0.79271235
 0.52984856 0.92345614 0.50133365 0.57512752 0.44522851 0.56284516
 0.7342211 ] 0.6897011730406379
1 [0.97633332 0.81316538 0.91558726 0.39391715 0.5232948  0.62855539
 0.66267264 0.7693947  0.91559245 0.55635921 0.94359156 0.79878897
 0.54633278 0.92862157 0.57823153 0.64446303 0.45243297 0.58019079
 0.74082091] 0.7035971789276992
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7036, Best_mIoU:  0.7130
[0.97633332 0.81316538 0.91558726 0.39391715 0.5232948  0.62855539
 0.66267264 0.7693947  0.91559245 0.55635921 0.94359156 0.79878897
 0.54633278 0.92862157 0.57823153 0.64446303 0.45243297 0.58019079
 0.74082091]
Epoch: [467/484] Iter:[0/24], Time: 0.98, lr: [0.0004909567142616703], Loss: 0.144967
Epoch: [467/484] Iter:[10/24], Time: 0.44, lr: [0.0004801133939384513], Loss: 0.103174
Epoch: [467/484] Iter:[20/24], Time: 0.42, lr: [0.00046924279181807766], Loss: 0.108878
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97598887 0.81084332 0.9136821  0.39965689 0.49770544 0.62011662
 0.64843115 0.7615571  0.91473006 0.55353879 0.94269835 0.79446221
 0.534645   0.92382977 0.49675266 0.57266546 0.4331944  0.56674187
 0.73824841] 0.6894467621143153
1 [0.97631521 0.81437615 0.91475417 0.39209139 0.51417052 0.62662907
 0.66083704 0.76944362 0.91589215 0.57012319 0.94308784 0.80036485
 0.55254556 0.92944931 0.57876416 0.6410161  0.44734819 0.58700767
 0.74479906] 0.7041586975783124
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7042, Best_mIoU:  0.7130
[0.97631521 0.81437615 0.91475417 0.39209139 0.51417052 0.62662907
 0.66083704 0.76944362 0.91589215 0.57012319 0.94308784 0.80036485
 0.55254556 0.92944931 0.57876416 0.6410161  0.44734819 0.58700767
 0.74479906]
Epoch: [468/484] Iter:[0/24], Time: 0.96, lr: [0.0004648867399628546], Loss: 0.088487
Epoch: [468/484] Iter:[10/24], Time: 0.44, lr: [0.0004539766324267969], Loss: 0.117985
Epoch: [468/484] Iter:[20/24], Time: 0.42, lr: [0.0004430373107857008], Loss: 0.109220
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.9755394  0.80904924 0.91362069 0.39513735 0.49939546 0.62019523
 0.64693462 0.75995529 0.91474097 0.54508282 0.9429897  0.79298919
 0.53440165 0.92275206 0.49936668 0.56221325 0.42387728 0.56230104
 0.73921033] 0.6873553810050052
1 [0.97596423 0.8136058  0.91449443 0.38506254 0.51775139 0.62671478
 0.65968805 0.76786834 0.91576182 0.55794701 0.94438787 0.79876076
 0.55120327 0.92776638 0.5742143  0.6251826  0.42384441 0.58144042
 0.74596769] 0.7004013728192202
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.7004, Best_mIoU:  0.7130
[0.97596423 0.8136058  0.91449443 0.38506254 0.51775139 0.62671478
 0.65968805 0.76786834 0.91576182 0.55794701 0.94438787 0.79876076
 0.55120327 0.92776638 0.5742143  0.6251826  0.42384441 0.58144042
 0.74596769]
Epoch: [469/484] Iter:[0/24], Time: 0.94, lr: [0.00043865320581891964], Loss: 0.155196
Epoch: [469/484] Iter:[10/24], Time: 0.44, lr: [0.0004276714872266172], Loss: 0.122269
Epoch: [469/484] Iter:[20/24], Time: 0.41, lr: [0.0004166583429327311], Loss: 0.115878
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97591291 0.81015167 0.91498232 0.40886843 0.50971193 0.62101995
 0.64902416 0.76042617 0.91446028 0.55707511 0.94336718 0.79340532
 0.53710516 0.92517249 0.51093863 0.5820713  0.45161362 0.56105376
 0.73670468] 0.6927928984049622
1 [0.97626017 0.81448714 0.91650545 0.4155847  0.52813024 0.62761229
 0.66224014 0.7682469  0.91581634 0.57421497 0.94422923 0.79937096
 0.55460322 0.93118831 0.59535296 0.66980656 0.47489042 0.57931169
 0.74337474] 0.7100645494776852
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7101, Best_mIoU:  0.7130
[0.97626017 0.81448714 0.91650545 0.4155847  0.52813024 0.62761229
 0.66224014 0.7682469  0.91581634 0.57421497 0.94422923 0.79937096
 0.55460322 0.93118831 0.59535296 0.66980656 0.47489042 0.57931169
 0.74337474]
Epoch: [470/484] Iter:[0/24], Time: 0.98, lr: [0.00041224405999241927], Loss: 0.118537
Epoch: [470/484] Iter:[10/24], Time: 0.44, lr: [0.00040118519429223906], Loss: 0.104206
Epoch: [470/484] Iter:[20/24], Time: 0.42, lr: [0.0003900923480036151], Loss: 0.109302
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97560025 0.8079653  0.91433739 0.40829563 0.51335371 0.62019429
 0.64973985 0.76158319 0.91443049 0.54388162 0.94272388 0.79409617
 0.5347992  0.92314274 0.50098344 0.57026695 0.42920921 0.56134038
 0.73759918] 0.6896601520847465
1 [0.97578305 0.81105951 0.91601604 0.41696706 0.53241194 0.62734794
 0.66158033 0.76916641 0.91567675 0.5581653  0.94347978 0.79957608
 0.55246471 0.92820306 0.58320794 0.63612701 0.45057031 0.57967845
 0.74495261] 0.7053912767337416
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.244, MeanIU:  0.7054, Best_mIoU:  0.7130
[0.97578305 0.81105951 0.91601604 0.41696706 0.53241194 0.62734794
 0.66158033 0.76916641 0.91567675 0.5581653  0.94347978 0.79957608
 0.55246471 0.92820306 0.58320794 0.63612701 0.45057031 0.57967845
 0.74495261]
Epoch: [471/484] Iter:[0/24], Time: 0.97, lr: [0.0003856454321056124], Loss: 0.123209
Epoch: [471/484] Iter:[10/24], Time: 0.45, lr: [0.00037450300398821307], Loss: 0.114467
Epoch: [471/484] Iter:[20/24], Time: 0.42, lr: [0.0003633236121546624], Loss: 0.112189
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97470503 0.80360829 0.91350424 0.4144455  0.50951529 0.6203194
 0.64615906 0.76099816 0.91442626 0.54631759 0.94225963 0.79386042
 0.53902585 0.92400197 0.5004738  0.57073692 0.41172908 0.55847967
 0.73728315] 0.6885183844203446
1 [0.97483301 0.80636478 0.91517623 0.43549578 0.52971304 0.62714542
 0.65664717 0.76772351 0.91612836 0.56213622 0.94372027 0.7996607
 0.55552058 0.93074317 0.58135657 0.64206311 0.43150959 0.57682716
 0.74359972] 0.7050718095746771
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.246, MeanIU:  0.7051, Best_mIoU:  0.7130
[0.97483301 0.80636478 0.91517623 0.43549578 0.52971304 0.62714542
 0.65664717 0.76772351 0.91612836 0.56213622 0.94372027 0.7996607
 0.55552058 0.93074317 0.58135657 0.64206311 0.43150959 0.57682716
 0.74359972]
Epoch: [472/484] Iter:[0/24], Time: 0.96, lr: [0.00035884119639225694], Loss: 0.155161
Epoch: [472/484] Iter:[10/24], Time: 0.44, lr: [0.0003476076881448455], Loss: 0.093354
Epoch: [472/484] Iter:[20/24], Time: 0.42, lr: [0.00033633368964247693], Loss: 0.092660
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97395029 0.79815077 0.91412371 0.42490929 0.51573026 0.61822059
 0.64569146 0.75935663 0.91417344 0.53905208 0.94189591 0.79307672
 0.53597775 0.92447725 0.49639946 0.5749038  0.42820785 0.55998749
 0.73626679] 0.689186923047042
1 [0.97406837 0.80019317 0.91608754 0.44706749 0.53574127 0.62379074
 0.65712002 0.76648571 0.91582671 0.55077158 0.94257782 0.79969532
 0.55303848 0.93070974 0.57157134 0.65301267 0.44281353 0.57645905
 0.7435087 ] 0.7052915394465477
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.248, MeanIU:  0.7053, Best_mIoU:  0.7130
[0.97406837 0.80019317 0.91608754 0.44706749 0.53574127 0.62379074
 0.65712002 0.76648571 0.91582671 0.55077158 0.94257782 0.79969532
 0.55303848 0.93070974 0.57157134 0.65301267 0.44281353 0.57645905
 0.7435087 ]
Epoch: [473/484] Iter:[0/24], Time: 0.98, lr: [0.00033181238423258223], Loss: 0.138669
Epoch: [473/484] Iter:[10/24], Time: 0.44, lr: [0.0003204788709564733], Loss: 0.124921
Epoch: [473/484] Iter:[20/24], Time: 0.42, lr: [0.0003091006374402392], Loss: 0.132174
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97434076 0.80082484 0.91483529 0.43942898 0.51290659 0.62158623
 0.64903674 0.76166022 0.91512108 0.54871279 0.94277891 0.7941605
 0.54259468 0.92420421 0.50068977 0.57361587 0.44087135 0.5650377
 0.73922333] 0.6927173599612297
1 [0.97456045 0.80402692 0.916757   0.46783878 0.53416857 0.62797346
 0.66130799 0.76925213 0.91679875 0.56896743 0.94369633 0.80076959
 0.55927203 0.9305784  0.5870836  0.657114   0.46500674 0.5839962
 0.74621718] 0.7113360808287521
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7113, Best_mIoU:  0.7130
[0.97456045 0.80402692 0.916757   0.46783878 0.53416857 0.62797346
 0.66130799 0.76925213 0.91679875 0.56896743 0.94369633 0.80076959
 0.55927203 0.9305784  0.5870836  0.657114   0.46500674 0.5839962
 0.74621718]
Epoch: [474/484] Iter:[0/24], Time: 0.97, lr: [0.0003045363754514388], Loss: 0.138159
Epoch: [474/484] Iter:[10/24], Time: 0.44, lr: [0.0002930920978175218], Loss: 0.100088
Epoch: [474/484] Iter:[20/24], Time: 0.42, lr: [0.0002815979375835987], Loss: 0.112940
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97508342 0.80492076 0.91519475 0.43557354 0.51569295 0.62011393
 0.65006885 0.76152818 0.91474221 0.54326242 0.94360512 0.79441803
 0.5401195  0.92469502 0.49929561 0.57849235 0.44036006 0.55604668
 0.7368844 ] 0.6921104093605212
1 [0.97539175 0.80862489 0.9170997  0.46172591 0.53663513 0.62707286
 0.66184833 0.76923268 0.91625033 0.56031006 0.94454443 0.80036915
 0.5570414  0.93048157 0.58679873 0.64771545 0.43457234 0.57219372
 0.74344318] 0.707965873171225
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7080, Best_mIoU:  0.7130
[0.97539175 0.80862489 0.9170997  0.46172591 0.53663513 0.62707286
 0.66184833 0.76923268 0.91625033 0.56031006 0.94454443 0.80036915
 0.5570414  0.93048157 0.58679873 0.64771545 0.43457234 0.57219372
 0.74344318]
Epoch: [475/484] Iter:[0/24], Time: 0.93, lr: [0.0002769857540753147], Loss: 0.031747
Epoch: [475/484] Iter:[10/24], Time: 0.44, lr: [0.0002654175006895353], Loss: 0.103509
Epoch: [475/484] Iter:[20/24], Time: 0.42, lr: [0.0002537929309170121], Loss: 0.109245
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97527252 0.80569866 0.91550889 0.42733622 0.515972   0.62178179
 0.65057858 0.76277077 0.91485902 0.54659933 0.94316484 0.79373473
 0.53799214 0.92441768 0.50938156 0.57524901 0.4452738  0.55896667
 0.73755458] 0.6927427780467171
1 [0.97545022 0.80923569 0.91713913 0.45147813 0.53634193 0.62826773
 0.66247242 0.76995562 0.91610581 0.5640092  0.94329646 0.79992883
 0.55579961 0.93031268 0.59811869 0.65188002 0.45862287 0.57622335
 0.74496221] 0.709978978905139
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.240, MeanIU:  0.7100, Best_mIoU:  0.7130
[0.97545022 0.80923569 0.91713913 0.45147813 0.53634193 0.62826773
 0.66247242 0.76995562 0.91610581 0.5640092  0.94329646 0.79992883
 0.55579961 0.93031268 0.59811869 0.65188002 0.45862287 0.57622335
 0.74496221]
Epoch: [476/484] Iter:[0/24], Time: 0.95, lr: [0.0002491266354885982], Loss: 0.089638
Epoch: [476/484] Iter:[10/24], Time: 0.44, lr: [0.00023741781630677843], Loss: 0.114706
Epoch: [476/484] Iter:[20/24], Time: 0.42, lr: [0.0002256444516680342], Loss: 0.115254
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97550312 0.80634784 0.91497413 0.42904598 0.50769032 0.62134168
 0.65097185 0.76265463 0.91460273 0.53772695 0.94345836 0.79423084
 0.54254929 0.92321133 0.50837647 0.5786249  0.43985594 0.56000236
 0.73790292] 0.6920564019804253
1 [0.97572834 0.80959203 0.91674204 0.45013927 0.52799567 0.62765359
 0.66323459 0.77009422 0.91591138 0.55263904 0.94432398 0.80025258
 0.5578756  0.92901197 0.58591192 0.64951503 0.44532624 0.57940791
 0.74468378] 0.7076862731947938
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7077, Best_mIoU:  0.7130
[0.97572834 0.80959203 0.91674204 0.45013927 0.52799567 0.62765359
 0.66323459 0.77009422 0.91591138 0.55263904 0.94432398 0.80025258
 0.5578756  0.92901197 0.58591192 0.64951503 0.44532624 0.57940791
 0.74468378]
Epoch: [477/484] Iter:[0/24], Time: 0.96, lr: [0.00022091612179404727], Loss: 0.075315
Epoch: [477/484] Iter:[10/24], Time: 0.44, lr: [0.00020904531326437436], Loss: 0.111206
Epoch: [477/484] Iter:[20/24], Time: 0.42, lr: [0.00019709908168310042], Loss: 0.119282
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97542034 0.80583254 0.91503576 0.43129803 0.51096995 0.62180943
 0.65213631 0.76283782 0.9147985  0.54458662 0.94306459 0.79410399
 0.53986745 0.92378957 0.50872505 0.57808597 0.45799044 0.55996918
 0.73781646] 0.6935862098909018
1 [0.97570154 0.80970732 0.91664404 0.45237078 0.53198343 0.62873809
 0.66381856 0.77028079 0.91611293 0.56217801 0.94331643 0.79995675
 0.55473577 0.92935529 0.59624626 0.65617405 0.488929   0.57705899
 0.7443083 ] 0.7114534898463526
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.240, MeanIU:  0.7115, Best_mIoU:  0.7130
[0.97570154 0.80970732 0.91664404 0.45237078 0.53198343 0.62873809
 0.66381856 0.77028079 0.91611293 0.56217801 0.94331643 0.79995675
 0.55473577 0.92935529 0.59624626 0.65617405 0.488929   0.57705899
 0.7443083 ]
Epoch: [478/484] Iter:[0/24], Time: 0.97, lr: [0.00019229823577899684], Loss: 0.223631
Epoch: [478/484] Iter:[10/24], Time: 0.44, lr: [0.00018023676155786228], Loss: 0.127436
Epoch: [478/484] Iter:[20/24], Time: 0.42, lr: [0.00016808485104803625], Loss: 0.124897
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97506736 0.80438076 0.9143464  0.41486314 0.50793745 0.62070404
 0.6502337  0.76231797 0.91472293 0.54521866 0.94376821 0.79352278
 0.5360704  0.92430727 0.50480441 0.56627207 0.43366448 0.55935851
 0.73609445] 0.6898765782047905
1 [0.97541354 0.80859334 0.9158744  0.42394918 0.52861803 0.62609489
 0.66165384 0.77005291 0.91601613 0.56560394 0.94450979 0.7997365
 0.55156123 0.92931968 0.58659757 0.63346993 0.43492436 0.57551714
 0.74268914] 0.7047471331991075
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7047, Best_mIoU:  0.7130
[0.97541354 0.80859334 0.9158744  0.42394918 0.52861803 0.62609489
 0.66165384 0.77005291 0.91601613 0.56560394 0.94450979 0.7997365
 0.55156123 0.92931968 0.58659757 0.63346993 0.43492436 0.57551714
 0.74268914]
Epoch: [479/484] Iter:[0/24], Time: 0.96, lr: [0.0001631970027929206], Loss: 0.095455
Epoch: [479/484] Iter:[10/24], Time: 0.44, lr: [0.00015090459830352705], Loss: 0.107162
Epoch: [479/484] Iter:[20/24], Time: 0.41, lr: [0.00013849977152781127], Loss: 0.112104
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97548677 0.8074385  0.91559529 0.4207746  0.51685437 0.62098291
 0.65140916 0.76241499 0.91432947 0.54796942 0.94269296 0.79491251
 0.54290284 0.92373223 0.50803368 0.57366373 0.44783107 0.55131903
 0.73469975] 0.6922654353390099
1 [0.97579814 0.81140264 0.91712731 0.43854599 0.53500491 0.62715169
 0.66330876 0.76973555 0.91571693 0.5656975  0.94328423 0.80040431
 0.56057116 0.92968206 0.5973726  0.6463663  0.45228588 0.56856171
 0.74102694] 0.7083707681347928
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.241, MeanIU:  0.7084, Best_mIoU:  0.7130
[0.97579814 0.81140264 0.91712731 0.43854599 0.53500491 0.62715169
 0.66330876 0.76973555 0.91571693 0.5656975  0.94328423 0.80040431
 0.56057116 0.92968206 0.5973726  0.6463663  0.45228588 0.56856171
 0.74102694]
Epoch: [480/484] Iter:[0/24], Time: 0.96, lr: [0.00013350365836381595], Loss: 0.185753
Epoch: [480/484] Iter:[10/24], Time: 0.44, lr: [0.00012091986763317612], Loss: 0.110889
Epoch: [480/484] Iter:[20/24], Time: 0.42, lr: [0.00010818854746988272], Loss: 0.108911
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97532665 0.80602807 0.91407153 0.41695452 0.50560381 0.6212641
 0.64844538 0.76241987 0.91444319 0.55176652 0.94228447 0.79432793
 0.54378296 0.92408952 0.50056978 0.58388433 0.4403845  0.5637009
 0.73765802] 0.6919476880838292
1 [0.97557386 0.80942551 0.9156363  0.43914635 0.5229461  0.62659796
 0.66021848 0.76951993 0.91589023 0.57015335 0.94257806 0.80057694
 0.56035182 0.93051981 0.58291764 0.6636307  0.45199636 0.58495453
 0.74516278] 0.7088314050160034
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.243, MeanIU:  0.7088, Best_mIoU:  0.7130
[0.97557386 0.80942551 0.9156363  0.43914635 0.5229461  0.62659796
 0.66021848 0.76951993 0.91589023 0.57015335 0.94257806 0.80057694
 0.56035182 0.93051981 0.58291764 0.6636307  0.45199636 0.58495453
 0.74516278]
Epoch: [481/484] Iter:[0/24], Time: 0.95, lr: [0.00010305007300023879], Loss: 0.099870
Epoch: [481/484] Iter:[10/24], Time: 0.44, lr: [9.00744414038246e-05], Loss: 0.128981
Epoch: [481/484] Iter:[20/24], Time: 0.42, lr: [7.68868495965381e-05], Loss: 0.116549
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97509877 0.80521484 0.9145639  0.41143199 0.51123588 0.62229322
 0.65018183 0.76247906 0.91473405 0.55094264 0.9425421  0.79618844
 0.54379821 0.92364153 0.50351452 0.57807413 0.45580618 0.55802641
 0.73730756] 0.6924776454867142
1 [0.97529578 0.80836474 0.91619012 0.42876795 0.53135313 0.6280644
 0.66252873 0.77002905 0.91610951 0.56790754 0.94241574 0.80191527
 0.56149445 0.92986644 0.58578502 0.65789544 0.49376184 0.57641699
 0.74390912] 0.710424802691223
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.242, MeanIU:  0.7104, Best_mIoU:  0.7130
[0.97529578 0.80836474 0.91619012 0.42876795 0.53135313 0.6280644
 0.66252873 0.77002905 0.91610951 0.56790754 0.94241574 0.80191527
 0.56149445 0.92986644 0.58578502 0.65789544 0.49376184 0.57641699
 0.74390912]
Epoch: [482/484] Iter:[0/24], Time: 0.97, lr: [7.154283909292469e-05], Loss: 0.050230
Epoch: [482/484] Iter:[10/24], Time: 0.44, lr: [5.797680706428415e-05], Loss: 0.114250
Epoch: [482/484] Iter:[20/24], Time: 0.42, lr: [4.4044459536546674e-05], Loss: 0.102130
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97515847 0.80618006 0.91365825 0.40150209 0.5078857  0.62264486
 0.6486063  0.76306536 0.91422383 0.55099609 0.94199862 0.79284466
 0.53533791 0.92312924 0.49969817 0.56265508 0.4329272  0.5602248
 0.73739985] 0.688954554231705
1 [0.97551657 0.80876263 0.91309536 0.36082456 0.5161161  0.62751679
 0.66022702 0.770762   0.91485766 0.5650246  0.94139439 0.79909924
 0.55550379 0.92638313 0.55500286 0.60409571 0.44248065 0.58063103
 0.74393614] 0.6979594852963383
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.245, MeanIU:  0.6980, Best_mIoU:  0.7130
[0.97551657 0.80876263 0.91309536 0.36082456 0.5161161  0.62751679
 0.66022702 0.770762   0.91485766 0.5650246  0.94139439 0.79909924
 0.55550379 0.92638313 0.55500286 0.60409571 0.44248065 0.58063103
 0.74393614]
Epoch: [483/484] Iter:[0/24], Time: 0.98, lr: [3.833885818715131e-05], Loss: 0.095721
Epoch: [483/484] Iter:[10/24], Time: 0.44, lr: [2.3602841451511162e-05], Loss: 0.107451
Epoch: [483/484] Iter:[20/24], Time: 0.42, lr: [7.643689715300727e-06], Loss: 0.114762
0
0
0
0
10
10
10
10
20
20
20
20
30
30
30
30
40
40
40
40
0 [0.97514049 0.80519279 0.91493152 0.41632077 0.5166941  0.62172362
 0.65083336 0.76171095 0.91485191 0.55031464 0.94124041 0.79540511
 0.54178507 0.92416255 0.50897975 0.5784826  0.46095697 0.55961475
 0.73723611] 0.6934514456221058
1 [0.97535252 0.80856366 0.91657702 0.43911488 0.53551976 0.62828967
 0.66250814 0.76979185 0.91626899 0.56746399 0.94125635 0.80166796
 0.55913403 0.92955111 0.59119368 0.65170482 0.48322594 0.57634051
 0.74367364] 0.7103788688238378
=> saving checkpoint to output/subset/0.1/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484checkpoint.pth.tar
Loss: 0.240, MeanIU:  0.7104, Best_mIoU:  0.7130
[0.97535252 0.80856366 0.91657702 0.43911488 0.53551976 0.62828967
 0.66250814 0.76979185 0.91626899 0.56746399 0.94125635 0.80166796
 0.55913403 0.92955111 0.59119368 0.65170482 0.48322594 0.57634051
 0.74367364]
tools/train.py:324: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  logger.info('Hours: %d' % np.int((end-start)/3600))
Hours: 8
Done
Job ID           : 46316367
Cluster          : spartan
User/Project     : njbarry/punim1896
Nodes            : 1
Wall-clock time  : 08:35:38 / 10:00:00

Displaying overall resources usage from 2023-04-12 12:43:01 to 2023-04-12 21:18:39:

NODE            CPU#        TOT%   ( USR   / SYS   / WIO   / IDLE  ) 

spartan-gpgpu127 : 
                CPU# 1    : 38.2   (  28.8 /   9.5 /   0.0 /  61.8 ) 
                CPU# 2    : 37.7   (  28.3 /   9.3 /   0.0 /  62.3 ) 
                CPU# 3    : 40.2   (  30.2 /  10.0 /   0.0 /  59.8 ) 
                CPU# 4    : 39.5   (  29.9 /   9.7 /   0.0 /  60.5 ) 
                CPU# 5    : 40.2   (  30.1 /  10.1 /   0.0 /  59.8 ) 
                CPU# 6    : 38.8   (  29.2 /   9.6 /   0.0 /  61.2 ) 
                CPU# 7    : 42.5   (  31.9 /  10.6 /   0.0 /  57.5 ) 
                CPU# 8    : 39.7   (  29.9 /   9.9 /   0.0 /  60.3 ) 
                CPU# 9    : 39.0   (  29.5 /   9.5 /   0.0 /  61.0 ) 
                CPU# 10   : 39.3   (  29.5 /   9.8 /   0.0 /  60.7 ) 
                CPU# 11   : 38.5   (  29.1 /   9.5 /   0.0 /  61.5 ) 
                CPU# 12   : 39.6   (  29.9 /   9.7 /   0.0 /  60.4 ) 
                CPU# 13   : 39.0   (  29.4 /   9.6 /   0.0 /  61.0 ) 
                CPU# 14   : 39.5   (  29.8 /   9.7 /   0.0 /  60.5 ) 
                CPU# 15   : 39.6   (  29.9 /   9.8 /   0.0 /  60.4 ) 
                CPU# 16   : 36.0   (  27.5 /   8.6 /   0.0 /  64.0 ) 

                GPU# 1    : 35.7  
                GPU# 2    : 36.5  
                GPU# 3    : 35.0  
                GPU# 4    : 34.5  


Allocated CPUs            : 16   
  CPUs with usage <25%    : 0    
  CPUs with usage <50%    : 16   
  CPUs with usage >50%    : 0    


Allocated GPUs            : 4    
  GPUs with usage <25%    : 0    
  GPUs with usage <50%    : 4    
  GPUs with usage >50%    : 0    

Memory used (RAM)         : 10.6%  [26026MB of 245760MB]

--------------------------------------------

