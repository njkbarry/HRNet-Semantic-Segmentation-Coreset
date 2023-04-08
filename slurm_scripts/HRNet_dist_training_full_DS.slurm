#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Mar 29 2023 12:41:05 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="HRNet-dist-training-full-DS"

# The project ID which this job should run under:
#SBATCH --account="punim1896"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Number of GPUs requested per node:
#SBATCH --gres=gpu:4
# The amount of memory in megabytes per node:
#SBATCH --mem=245760

# Use this email address:
#SBATCH --mail-user=njbarry@student.unimelb.edu.au

# Send yourself an email when the job:
# begins
#SBATCH --mail-type=BEGIN
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-23:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from this directory:
cd /data/gpfs/projects/punim1896/coresets/repositories/HRNet-Semantic-Segmentation/

# The modules to load:
module purge
module load fosscuda/2020b
module load pytorch/1.7.1-python-3.8.6
module load opencv/4.5.1-python-3.8.6-contrib
module load pyyaml/5.3.1-python-3.8.6
module load cython/0.29.22
module load ninja/1.10.1-python-3.8.6
module load tqdm/4.60.0

# The job command(s):
source venv/bin/activate
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
