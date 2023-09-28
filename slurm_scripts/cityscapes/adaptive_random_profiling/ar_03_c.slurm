#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Mar 29 2023 12:41:05 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="cityscapes_ar_03_c"

# The project ID which this job should run under:
#SBATCH --account="punim1896"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Number of GPUs requested per node:
#SBATCH --gres=gpu:2
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
#SBATCH --time=0-3:30:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from this directory:
cd /data/gpfs/projects/punim1896/coresets/repositories/HRNet-Semantic-Segmentation-Coreset/

# The modules to load:
module purge
module load GCCcore/11.3.0
module load Anaconda3/2022.10
module load Tkinter/3.10.4
 
# The job command(s):
eval "$(conda shell.bash hook)"
conda activate py38
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/adaptive_random_profiling/ar_03_c.yaml

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
