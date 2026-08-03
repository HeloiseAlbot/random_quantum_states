#!/bin/bash
#SBATCH --job-name=mps_unitary_transform
#SBATCH --comment "change_under_unitary."
#SBATCH --nodes=1 #8-8   #8-8
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.albot@physik.uni-muenchen.de
#SBATCH --partition=cluster #th-ws,th-cl,large, cluster
##SBATCH --exclude=th-cl-naples27

#SBATCH --time=01:00:00
#SBATCH --mem=1GB #100GB

##SBATCH --constraint=avx512
#SBATCH --chdir=/scratch/h/H.Albot/distribution/
#SBATCH --output=/scratch/h/H.Albot/distribution/slurm/mps_job_%j.out
#SBATCH --error=/scratch/h/H.Albot/distribution/slurm/mps_job_%j.err

# Optional: Uncomment and modify to load your Python environment module or virtualenv
# module load python/3.10
# source path/to/your/venv/bin/activate

echo "Starting heavy calculation job at $(date)"
python3 test/change_under_unitaries.py
echo "Calculation workflow completed at $(date)"