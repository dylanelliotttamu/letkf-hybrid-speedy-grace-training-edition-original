#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
# --export=NONE                #Do not propagate environment
# --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=python_forecast_analysis #Set the job name to "JobExample3"
#SBATCH --time=0-0:30:00            #Set the wall clock limit to 1 Day and 12hr
#SBATCH --ntasks=12                   #Request 8 task
#SBATCH --ntasks-per-node=4          #Request 2 tasks/cores per nod
#SBATCH --mem-per-cpu=80000                  #
#SBATCH --output=python_forecast_analysis.%j      #Send stdout/err to "Example3Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132780093254            #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=dylanelliott@tamu.edu    #Send all emails to email_address 

echo "starting job" 

cd /scratch/user/dylanelliott/plotting_scripts_from_skynet_backup/scripts 
#bash init.sh double
#bash init.sh 
#bash init_generate_climo_ens.sh

python main_forecast_error_script.py 
#time bash run_cycle.sh double
#time bash run_cycle_hybrid.sh hybrid
#time bash run_cycle_hybrid_faster.sh hybrid
#time bash restart_letkf.sh double

echo "Job is done"

