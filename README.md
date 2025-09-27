# Physics Informed Machine Learning Part III: Hard-Constraint ODE Method for Structural Dynamics
Paper 2026 IMAC PIML

Tutorial codes demonstrating a variety of PIML methods.
Comparison methods:
1. Pure physics
1. Pure NN

PIML methods:
1. Indirect measurement

## Files too large for GitHub
This repo has some files too large for GitHub. The workaround it to generate these files on your own. 
1. In `physics based modeling`, run `to_numpy.py` to convert .csv files to .npy. This will save `.npz` versions of data in the `PIML examples/data/`.

We may try to get these to work with LFS in the future. For now, you can find the files [here](https://www.dropbox.com/scl/fo/p7u6dwy1t8o83hk3dgnlz/AO3S344MKnTcE55aHe9J2Gs?rlkey=6aor3ivq2wrm9j06h62nnedgn&dl=0).

The files are:
1. PIML examples/data/v5/no_friction.npy
1. PIML examples/data/v5/with_friction.npy
1. PIML examples/model_predictions/pure_physics/k_pred.npy (May or may not be needed. )

## Version control
Nile built the first code in Tensorflow 2.10.0. Austin made small organizational changes and re-trained it in 2.13.0.

### Environment
Use the Anaconda environment in the repo 

## Cluster Computing
Information below is specific to this project. Tutorials on general cluster usage can be found [here](https://uofsc.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=8a7901a7-8611-44c7-af09-b0fc0166d1e1)

### Building an environment
Once in a folder specifically for environments, begin an idev session, then follow the prompts after each of these commands to build your environment.

module load python3/anaconda/2023.9  
conda create --prefix=/work/USERNAME/ENVs/python310_env  
source activate /work/USERNAME/ENVs/python310_env  
conda install python==3.10.18  
conda install keras==2.10.0  
conda install tensorflow==2.10.0  
conda install numpy==1.26.4  
conda install pandas==2.3.1  
conda install matplotlib==3.10.0  

Once this is complete, the idev session can be ended. This environment will be used as long as it is referenced in your .sh file.

### Building a .sh file
In order to run a job, it must be done from an sbatch command so that slurm can control the order of the jobs on the cluster. Below is an example .sh file using the above environment.

\#!/bin/sh  
\#SBATCH --job-name=PIML_training  
\#SBATCH --output train%piml.out  
\#SBATCH --error train%piml.err  
\#SBATCH -N 1  
\#SBATCH -n 32  
\#SBATCH -p defq-64core  
\#SBATCH --exclude=none[174-238]  

\#\# Load modules first:

module load python3/anaconda/2023.9  
source activate /work/USERNAME/ENVS/python310_env  

\#\# Add code here:

hostname  
date  
cd /work/USERNAME/PIML_Code  
python train_indirect_no_friction.py  

 ## Licensing and Citation

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

Cite as:

@Misc{ARTSLab2026Paper2026Test,     
  author = {{ARTS-L}ab},  
  howpublished = {GitHub},    
  title  = {Paper-2026-{PIML}-Part-{III}-Hard-Constraint-{ODE}-Method-for-Structural-Dynamics},    
  groups = {{ARTS-L}ab},    
  year = {2026},   
  url    = {https://github.com/ARTS-Laboratory/Paper-2026-PIML-Part-III-Hard-Constraint-ODE-Method-for-Structural-Dynamics},   
}