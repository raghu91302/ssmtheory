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
 
### 3. Spectral Structure of the Naive Dirac Operator on the FCC Lattice

This repository contains the source code and verification scripts supporting the manuscript:

**"Spectral Structure of the Naive Dirac Operator on the Face-Centered Cubic Lattice"**

* **File:** `fcc_spectral_analysis.py`
* 
## Overview

The script `fcc_spectral_analysis.py` numerically validates the analytical results presented in the paper regarding the spectral landscape of the naive Dirac operator on a Face-Centered Cubic (FCC) lattice.

Unlike hypercubic lattices which generate $2^D$ doublers at zone corners, the FCC lattice ($D=3$) exhibits a unique spectral geometry governed by the Truncated Octahedron Brillouin zone. This code confirms the existence of isolated zero modes and continuous nodal lines that satisfy the Nielsen-Ninomiya theorem.

## Key Features

The simulation performs the following specific verifications:

1.  **Lattice Geometry (Eq. 1):** Generates the 12 nearest-neighbor hopping vectors for the FCC lattice.
2.  **Dispersion Relation (Sec. III):** Computes the magnitude of the kinetic vector field $|f(k)|^2$ along the high-symmetry path $\Gamma \to L \to X \to W \to \Gamma$.
3.  **Zero Mode Classification (Table I):**
    * **$\Gamma$ Point:** Confirms a single zero mode at the zone center.
    * **$L$ Points:** Confirms four isolated zero modes at the hexagonal face centers.
    * **$X-W$ Boundary:** Confirms a continuous nodal line of zeros connecting the square face centers ($X$) to the zone corners ($W$).
4.  **Chirality Calculation (Sec. IV):** Computes the Jacobian determinant $J_{\mu\nu} = \partial f_\mu / \partial k_\nu$ to assign topological charges:
    * $\chi_{\Gamma} = +1$
    * $\chi_{L} = -1$ (Multiplicity 4 $\to$ Net -4)
    * Verifies the topological deficit $\chi_{net} = -3$, necessitating the boundary nodal lines.

## Usage

### Prerequisites
The script requires a standard Python 3 environment with `numpy` and `matplotlib`.

```bash
pip install numpy matplotlib
## Installation & Usage

### Prerequisites
The simulations require a standard Python 3 environment with the following scientific libraries:

```bash
pip install numpy matplotlib scipy
