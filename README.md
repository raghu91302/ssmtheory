# Selection-Stitch Model (SSM) Simulation Repository

This repository contains the source code and simulation data supporting the findings in the paper:

**"Resolving the Hubble Tension via a Topological Phase Transition in a Discrete Vacuum Tensor Network"**

## Overview

The Selection-Stitch Model (SSM) proposes that the Hubble Tension arises from a geometric phase transition in the vacuum lattice structure. This repository includes two distinct Python simulations used to validate the kinematic consistency of the model and numerically verify the predicted expansion boost.

## Repository Contents

### 1. Constructive Geometry Simulation
* **File:** `selection-stitch-simulation.py`
* **Purpose:** Corresponds to **Appendix B** of the paper.
* **Description:** This script simulates the constructive growth of a 3D tensor network using the SSM "Lift and Stitch" operators. It tests whether a stochastic network can naturally saturate into a Face-Centered Cubic (FCC) lattice without "jamming" into an amorphous glass state.
* **Key Output:**
    * Verifies the saturation limit at Coordination Number $K=12$.
    * Generates a 3D visualization of the lattice core vs. surface (Fig. 1 in the paper).
    * Proves the geometric stability of the $K=12$ vacuum ground state.

### 2. Topological Phase Transition (Monte Carlo)
* **File:** `ssm-repair-simulation.py`
* **Purpose:** Corresponds to **Appendix C** of the paper.
* **Description:** This script performs a Monte Carlo analysis on a pre-existing $15^3$ FCC grid. It introduces random "cosmic voids" (vacancies) to model the late-time universe and calculates the effective topological degrees of freedom ($\nu$) available for nucleation.
* **Key Output:**
    * Tracks the shift in active nucleation channels as void fraction increases.
    * Numerically confirms the transition from $\nu_{mean} \approx 12.0$ (Shielded Phase) to $\nu_{mean} \approx 13.0$ (Exposed Phase).
    * Validates the theoretical expansion boost of $\xi = 13/12 \approx 8.3\%$.

## Installation & Usage

### Prerequisites
The simulations require a standard Python 3 environment with the following scientific libraries:

```bash
pip install numpy matplotlib scipy
