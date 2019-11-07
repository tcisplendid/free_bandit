#!/bin/bash
#BSUB -G SEAS-Lab-Yeoh
#BSUB -R "select[type==any]"
#BSUB -o cpu_test.%J
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=25]"
#BSUB -n 2
#BSUB -N
#BSUB -J test
python3 experiment.py &> output.txt
