#!/bin/bash
#SBATCH --job-name=distribution_project
#SBATCH --comment "KL divergence."
#SBATCH --nodes=1 #8-8   #8-8
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.albot@physik.uni-muenchen.de
#SBATCH --partition=cluster #th-ws,th-cl,large, cluster
##SBATCH --exclude=th-cl-naples27

#SBATCH --time=24:00:00
#SBATCH --mem=1GB #100GB

##SBATCH --constraint=avx512
#SBATCH --chdir=/scratch/h/H.Albot/distribution/
#SBATCH --output=/scratch/h/H.Albot/distribution/slurm/output_%j.sout
#SBATCH --error=/scratch/h/H.Albot/distribution/slurm/output_%j.serr

start_time="$(date -u +%s)"
echo $PATH
diameter_disks=(0.01 0.05 0.1)
t=0.1
# Run Python scripts in parallel

# Loop through diameter_disks and run each job in parallel
for diameter_disk in "${diameter_disks[@]}"; do
    python3 test/produce_samples_disk_moving.py $diameter_disk $t &
done

# Wait for all background processes to finish
wait

# python3 test/kl_divergence.py 

echo "Job finished"
sleep 15

echo -n "I'm on Host: "
hostname

echo -n "It's now: "
date
