# H-diffusion-model

## Overview

This project models hydrogen (H) diffusion in silicon-based solar cells. It uses PETSc (the Portable, Extensible Toolkit for Scientific Computation) as a finite element solver to simulate the diffusion of hydrogen through each of the layers in a solar cell structure. The codebase is designed for extensibility, reproducibility, and robust parameter sweeps, enabling detailed comparison between simulation results and experimental data.

## Features

- **Layered Diffusion Modeling:** Simulates H diffusion across multiple layers of a silicon solar cell using a finite element approach (via PETSc).
- **Parameter Sweep:** Automates the generation and management of parameter sweeps to efficiently explore the effects of different physical and process parameters on H diffusion.
- **Experimental Comparison:** Provides tools to compare simulation results with experimental data, helping to identify physically realistic model parameters.
- **Batch Processing & HPC Support:** Includes scripts for batch job submission (e.g., SLURM) to facilitate large-scale parameter sweeps on high-performance computing clusters.
- **Modular & Documented:** The codebase is modular, well-documented, and ready for collaborative development and version control.

## Main Components

- `diffusion_calculator.py`: Core simulation engine for H diffusion using PETSc.
- `simulation_data.py`: Utilities and classes for managing simulation parameters, results, and file I/O.
- `create_sweep.py`: Tools for generating and filtering parameter sweeps, avoiding redundant simulations.
- `analyze_diffusion_results.py`: Analysis and plotting utilities for simulation results and experimental comparison.
- `submit_diffusion_jobs.sh`: Example SLURM batch script for submitting parameter sweeps to an HPC cluster.

## Getting Started

1. **Install Dependencies:**
   - Python 3.x
   - PETSc and petsc4py
   - numpy, matplotlib, etc. (see code for details)

2. **Run a Simulation:**
   - Edit parameters in `create_sweep.py` or use the provided sweep utilities.
   - Submit jobs locally or via SLURM using `submit_diffusion_jobs.sh`.

3. **Analyze Results:**
   - Use `analyze_diffusion_results.py` to visualize and compare simulation output with experimental data.

## Project Structure

```
diffusion_calculator.py         # Main simulation code
simulation_data.py              # Parameter/result management utilities
create_sweep.py                 # Parameter sweep generation
analyze_diffusion_results.py    # Analysis and plotting
submit_diffusion_jobs.sh        # SLURM batch script
README.md                       # Project documentation
.gitignore                      # Git ignore rules
```

## License

See LICENSE file for open-source terms.

## Authors

Adam Goga & Zitong(Zeke) Zhao