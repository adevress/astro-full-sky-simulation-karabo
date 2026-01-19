#!/bin/bash
#SBATCH --output=log_create_ms_%j.out
#SBATCH --error=log_create_ms_%j.err


srun singularity exec --nv --bind $PWD:/workspace/userdir:rw karabo-pipeline_latest.sif bash -c "cd /workspace/userdir && ipython ./create_ms.py"
