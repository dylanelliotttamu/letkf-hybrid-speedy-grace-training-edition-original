#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
# --export=NONE                #Do not propagate environment
# --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=HybridModel_predict_test_2011_ERA5_weights_RETRYDYLAN      #Set the job name to "JobExample3"
#SBATCH --time=0-30:06:00            #Set the wall clock limit to 1 Day and 12hr
#SBATCH --ntasks=1152                   #Request 8 tasks
#SBATCH --ntasks-per-node=48          #Request 2 tasks/cores per nod
#SBATCH --mem-per-cpu=6500                  #
#SBATCH --output=hybrid_predict_2011_era5_weights_RETRYDYLAN.%j      #Send stdout/err to "Example3Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132780093254 ##132780090616    # 132780093249 #132780099310 # old account=132780093589             #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=dylanelliott@tamu.edu    #Send all emails to email_address 

# Compile command added 
#make imp.exe COMPILER=gcc2021

# now load libraries
ml purge

#ml intel/2020a

#ml arpack-ng

#ml netCDF-Fortran

#ml purge

#ml intel-compilers/2021.2.0
#ml impi/2021.2.0
#ml netCDF-Fortran
#ml imkl
#ml ARPACK-NG

#ml intel/2020a

#ml arpack-ng

#ml netCDF-Fortran

ml gomkl/2021a
ml arpack-ng
ml netCDF-Fortran

ulimit -s unlimited
ulimit -a 
#ulimit -l unlimited

#export H5FDmpio_DEBUG=t

##cd /scratch/user/dylanelliott/letkf-hybrid-speedy/hybrid
cd /scratch/user/dylanelliott/letkf-hybrid-speedy-grace-training-edition-original/hybrid

#export MKL_NUM_THREADS=4
#export I_MPI_HYDRA_TOPOLIB=ipl
#export I_MPI_DEBUG=4
#export FI_LOG_LEVEL=Info
#export UCX_LOG_LEVEL=data
#export UCX_LOG_FILE=/scratch/user/troyarcomano/UCX.log

export OMPI_MCA_sharedfp=^lockedfile,individual

echo "starting job" 

time mpirun -np 1152 ./imp.exe

echo "Job is done"

