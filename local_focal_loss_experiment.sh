#!/bin/bash

source venv/bin/activate

# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/focal_loss_experiment/fl/ar_focal_loss_gamma2_05_a.yaml
# python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/focal_loss_experiment/fl/ar_focal_loss_gamma2_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/focal_loss_experiment/fl/ar_focal_loss_gamma2_05_c.yaml

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/ohem_loss_experiment/oh/ar_ohem_loss_default_05_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/ohem_loss_experiment/oh/ar_ohem_loss_default_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/ohem_loss_experiment/oh/ar_ohem_loss_default_05_c.yaml

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/focal_loss_experiment/fl/ar_focal_loss_gamma5_05_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/focal_loss_experiment/fl/ar_focal_loss_gamma5_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/focal_loss_experiment/fl/ar_focal_loss_gamma5_05_c.yaml

python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/focal_loss_experiment/fl/ar_focal_loss_gamma05_05_a.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/focal_loss_experiment/fl/ar_focal_loss_gamma05_05_b.yaml
python -m torch.distributed.launch --nproc_per_node 1 tools/train.py --cfg experiments/pascal_ctx/focal_loss_experiment/fl/ar_focal_loss_gamma05_05_c.yaml