#!/bin/bash
#SBATCH --job-name=electra_hyperparameter_search
#SBATCH --open-mode=append
#SBATCH --output=/scratch_tmp/yg2483/%j_%x.out
#SBATCH --error=/scratch_tmp/yg2483/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --wrap="sleep infinity"
#SBATCH -c 4

## Modify --data_dir, --class_num on demand

singularity exec --nv \
	    --bind /scratch_tmp/yg2483 \
	    --overlay /scratch/yg2483/nlu_pytorch.ext3:ro \
	    /home/yg2483/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
	    /bin/bash -c "
source /ext3/env.sh;
conda activate;
python /scratch_tmp/yg2483/electra/run_hyperparameter_search.py --data_dir /scratch_tmp/yg2483/dataset/R2 --class_num 2
"
