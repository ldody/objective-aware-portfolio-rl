#!/bin/sh
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=all
#SBATCH --mail-user=leo.dody@univ-lyon3.fr
#SBATCH --output=venv.out
#SBATCH --job-name=venv


module purge
module use /easybuild/AlmaLinux/8/skylake-avx512/mlxln5.5/foss2022b/modules/all
module load Python/3.10.4-GCCcore-12.2.0
source /home_nfs/polytech/leo.dody/PhD/Article_2/PhD_article_2/.venv/bin/activate

python3 -m pip install -r requirements.txt

deactivate
