# Hamiltonian–Liouvillian Learning

> Hamiltonian and Liouvillian learning in weakly-dissipative quantum many-body systems


[![Tests](https://github.com/<username>/<repo>/actions/workflows/tests.yml/badge.svg)](https://github.com/TobiasOlsacher/hamiltonian-liouvillian-learning/actions)

---

## Overview

This package contains tools for Hamiltonian and Liouvillian learning from measurement data based on ``Olsacher et al 2025 Quantum Sci. Technol. 10 015065``.

It also contains a numerical quantum simulator built on top of the **QuTiP** master equation solver.

Example notebooks can be found in the ``../examples`` folder.

The package allows you to:
- generate measurement data from a mock experiment defined by a master equation in Lindblad form
- learn the Hamiltonian and Lindblad operators form the data using one of several learning methods

---

## Installation

### Requirements
- Python ≥ 3.10

### Install from source

```bash
git clone https://github.com/TobiasOlsacher/hamiltonian-liouvillian-learning.git
cd <repo>
pip install -r requirements.txt