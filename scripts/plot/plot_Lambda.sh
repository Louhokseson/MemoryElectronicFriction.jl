#!/bin/bash
# SLURM script for running Julia-based IESH simulation

# SLURM job scheduling options
#SBATCH --time=48:00:00             # Set maximum job run time to 48 hours
#SBATCH --nodes=1                   # Request two nodes
#SBATCH --ntasks=1                  # Total number of tasks across both nodes (40 tasks per node)
#SBATCH --ntasks-per-node=1         # number of task/process in a node
#SBATCH --cpus-per-task=44           # threads per task/process
#SBATCH --mem-per-cpu=3800          # Set memory per CPU to 4000MB



#SBATCH --job-name="Lambda"         # Set the job name to "SH/CVG"
#SBATCH --partition=compchem       # Request the compchem partition



# SLURM email notifications
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=Xuexun.Lu@warwick.ac.uk

# SLURM output files
#SBATCH --output=slurmjob_logs/output_%j.log      # Direct standard output to 'output_<jobID>.log'
#SBATCH --error=slurmjob_logs/error_%j.log        # Direct standard error to 'error_<jobID>.log'


# Pre-job setup
# Ensure the DrWatson package is installed in Julia
# (The actual command to ensure DrWatson is installed is omitted here. 
# It could be something like `julia -e 'using Pkg; Pkg.add("DrWatson")'`)

# Execution
# Run the Julia script with parallel processing enabled, using SLURM's task count
srun julia -t auto plot_Lambda.jl

# End of SLURM script
