# phy51171_YH-CO : Particle-In-Cell Simulations of the Two-Stream Instability

## Description
This project simulates a one-dimensional electrostatic system using the Particle-In-Cell (PIC) method. 

## Usage
To run the simulation, execute the following command in the production directory:
`python3 run_simulation.py`

## Directory structure

- production
  * Contains the  PIC simulation. It initializes particle positions, velocities, and fields. It updates the particles' positions and velocities over time, while computing the charge density and field. The results, including particle positions and field data, are collected for analysis and visualization.
- analysis
  * Contains a notebook that computes the kinetic energy and momentum. 
- visualization
  * The computed observables are stored in files here.
- figures
  * Contains the figures for the report.
- report
  * Contains the project report.








