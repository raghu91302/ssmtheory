# Selection-Stitch Model (SSM): Lattice Saturation Simulation

[SSM Visualization](ssm_saturation_k12.png)
*(Note: This image is generated after running the simulation)*

## Overview

This repository contains a Python computational proof-of-concept for the **Selection-Stitch Model (SSM)**. It procedurally generates a 3D vacuum lattice node-by-node, enforcing the model's core geometric constraints: **Stitch Length Uniformity** and **Geometric Exclusion**.

The simulation empirically demonstrates the **Emergent Saturation Limit**, showing that a discrete network of equal-length constraints naturally converges to a maximum coordination number of **K ≈ 12** (related to the "Kissing Number" problem). This validates the theoretical prediction that the vacuum is a rigid "Mesh Phase" crystal rather than a fluid.

## Features

* **Procedural Lattice Growth:** Simulates the "mining" of new vacuum voxels from a genesis triangle, expanding outward.
* **Geometric Exclusion (The "Selection" Operator):** Implements a spatial stabilizer check using KD-Trees to enforce the exclusion principle (no two nodes can be closer than the Stitch Length).
* **3D Topology Lifting:** Simulates the formation of Tetrahedra (Volume) by selecting valid triangles on the boundary and "lifting" a new node into the third dimension.
* **High-Res Visualization:** Generates a 300 DPI 3D scatter plot where nodes are color-coded by their connectivity (Degree), visually highlighting the saturated core vs. the active frontier.

## Theoretical Background

The code directly implements the two fundamental operators of the SSM:

1.  **The Selection Operator ($\hat{O}$):** A fluctuation is only "selected" if it creates a valid geometry without overlapping existing nodes. In the code, this is the `stabilizer_check`, which rejects candidates within the `HARD_SHELL` radius.
2.  **The Stitch Operator ($\hat{S}$):** Valid nodes are connected to their neighbors to form the lattice graph (`G.add_edge`).

The simulation tracks the local coordination number ($K$) of every node. The emergence of a "Yellow Core" ($K=12$) in the output image represents the **Saturated Vacuum**—regions where the lattice is geometrically locked and "Graph Friction" (Inertia) becomes non-zero.

## Installation

To run this simulation, you need Python 3.x and the following scientific libraries:

```bash
pip install numpy matplotlib networkx scipy
