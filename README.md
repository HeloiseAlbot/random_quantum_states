# Ensembles of Random Quantum States Tunable from Volume Law to Area Law

[![arXiv](https://img.shields.io/badge/arXiv-2604.15300-B31B1B.svg)](https://arxiv.org/abs/2604.15300)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official codebase and numerical routines accompanying the paper:

> **Ensembles of random quantum states tunable from volume law to area law**  
> Héloïse Albot and Sebastian Paeckel  
> *Ludwig-Maximilians-Universität München (LMU), Arnold Sommerfeld Center for Theoretical Physics (ASC), Munich Center for Quantum Science and Technology (MCQST)*  
> **arXiv:** [2604.15300](https://arxiv.org/abs/2604.15300)

---

## Abstract

A standard approach to generate random pure quantum states relies on sampling from the Haar measure. However, due to their volume-law entanglement, Haar-random states pose severe challenges for classical simulations of quantum many-body systems. This repository provides tools to construct and sample from generalized ensembles of random quantum states ($\sigma$-ensembles) whose entanglement properties can be systematically tuned from volume law to area law. By controlling the underlying parameters, this approach circumvents the intractability of Haar-random states and yields states that are more representative of typical physical Hamiltonian ground states, complete with explicit Matrix-Product State (MPS) construction schemes.

---

## Repository Structure

## Repository Structure

The core functionality of the project is organized within the `distribution_functions` package:

```text
random_quantum_states/
├── distribution_functions/
│   ├── __init__.py
│   ├── connected_correlation_functions.py
│   ├── correlation_functions_test.ipynb
│   ├── kl_divergence_tools.py
│   ├── property_calculation_functions.py
│   ├── random_mps_old.py
│   ├── random_mps_v2.py
│   └── standard_random_states_and_matrices.py
├── .gitignore
└── README.md