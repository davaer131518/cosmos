@echo off

:: Activate Conda environment
call activate cosmos_env

:: Define dataset labels
set datasets=mm mf mfm adult compass 

:: Define method names
set methods=hyper_ln hyper_epo pmtl single_task

:: Iterate over dataset labels
for %%d in (%datasets%) do (
    :: Iterate over method names
    for %%m in (%methods%) do (
        echo Running command: python multi_objective/main.py --dataset %%d --method %%m
        python multi_objective/main.py --dataset %%d --method %%m
    )
)