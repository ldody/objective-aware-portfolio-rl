#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=all
#SBATCH --mail-user=leo.dody1@univ-lyon3.fr
#SBATCH --output=DRL_%a.out #/dev/null
#SBATCH --job-name=DRL_%a
#SBATCH --partition=c6420-ib100
#SBATCH --array=0-24%5


module purge
module use /easybuild/AlmaLinux/8/skylake-avx512/mlxln5.5/foss2022b/modules/all
module load Python/3.10.8-GCCcore-12.2.0
source /home_nfs/polytech/leo.dody/PhD/Article_3/PhD_article_3/.venv/bin/activate

export PYTHONUNBUFFERED=TRUE

python3 ../main.py --task-id $SLURM_ARRAY_TASK_ID

deactivate
