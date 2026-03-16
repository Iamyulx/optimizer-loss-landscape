# Optimizer Dynamics on Loss Landscape

Visualization of optimization trajectories for several optimizers implemented **from scratch**.

The experiment compares how different algorithms navigate a **non-convex loss landscape**.

## Optimizers Implemented

- SGD
- Adam
- AdamW

## Experiment

A 2D non-convex function is optimized starting from the same initial point.  
Each optimizer follows a different trajectory in parameter space.

## Visualization

The contour plot shows the **loss landscape**, while the colored paths represent the **optimization trajectory** of each algorithm.



## Key Insights

- SGD follows the steepest descent direction.
- Adam adapts the learning rate using moment estimates.
- AdamW introduces decoupled weight decay for improved optimization behavior.

## Project Structure

- optimizers.py → optimizer implementations  
- loss_landscape.py → objective function and gradients  
- run_experiment.py → optimization trajectories  
- plot_landscape.py → visualization  

## Requirements


torch
numpy
matplotlib
