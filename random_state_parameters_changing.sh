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

#SBATCH --time=30:00:00
#SBATCH --mem=5GB #100GB

##SBATCH --constraint=avx512
#SBATCH --chdir=/scratch/h/H.Albot/distribution/
#SBATCH --output=/scratch/h/H.Albot/distribution/slurm/output_%j.sout
#SBATCH --error=/scratch/h/H.Albot/distribution/slurm/output_%j.serr

start_time="$(date -u +%s)"
echo $PATH
# diameter_disk=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1)
diameter_disk=0.1 #(1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2) #(0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2)# 
chi_max=(16 32 64)
# chi_max=64
number_tests=10000
L=14
# Run Python scripts in parallel

# Loop through diameter_disks and run each job in parallel
# for diameter_disk in "${diameter_disk[@]}"; do
for chi_max in "${chi_max[@]}"; do
    python3 test/test_produce_random_mps.py $diameter_disk $L $chi_max $number_tests &
done

# Wait for all background processes to finish
wait

# python3 test/produce_samples_Haar_measure.py 

echo "Job finished"
sleep 15

echo -n "I'm on Host: "
hostname

echo -n "It's now: "
date
