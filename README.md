# phy51171_YH-CO : Particle-In-Cell Simulation of the Two-Stream Instability

## Description
This project simulates a one-dimensional electrostatic two-stream instability using the Particle-In-Cell (PIC) method. The goal is to investigate the conditions under which two counterstreaming electron beams with equal density and opposite velocities lead to the two-stream instability.

## Usage
To run the simulation, execute the following command in the production directory:
`python3 run_simulation.py`

## Directory structure

- production
  * Contains the  PIC simulation. It initializes particle positions, velocities, and fields. It updates the particles' positions and velocities over time, while computing the charge density and field. The results, including particle positions and field data, are collected for analysis and visualization.
- analysis
  * Saves the data.
- visualization
  * The computed observables are stored in files here.
- figures
  * Contains the figures for the report.
- report
  * Contains the project report.








