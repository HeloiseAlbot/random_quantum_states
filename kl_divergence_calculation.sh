#!/bin/bash
#SBATCH --job-name=distribution_project
#SBATCH --comment "KL divergence."
#SBATCH --nodes=1-1 #8-8   #8-8
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.albot@physik.uni-muenchen.de
#SBATCH --partition=cluster #th-ws,th-cl,large
##SBATCH --exclude=th-cl-naples27

#SBATCH --time=1:00:00
#SBATCH --mem=1GB #100GB

#SBATCH --constraint=avx512
#SBATCH --chdir=/scratch/h/H.Albot/distribution/
#SBATCH --output=/scratch/h/H.Albot/distribution/slurm/output_%j.sout
#SBATCH --error=/scratch/h/H.Albot/distribution/slurm/output_%j.serr

start_time="$(date -u +%s)"
echo $PATH

# Run Python scripts in parallel
for diameter_disk in 0 1 2 3; do
    python3 test/kl_divergence.py $diameter_disk &
done

echo "Job finished"
sleep 15

echo -n "I'm on Host: "
hostname

echo -n "It's now: "
date
