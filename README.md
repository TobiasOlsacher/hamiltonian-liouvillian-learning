# Hamiltonian Liouvillian Learning

> Hamiltonian and Liouvillian learning in weakly-dissipative quantum many-body systems.


[![Tests](https://github.com/TobiasOlsacher/hamiltonian-liouvillian-learning/actions/workflows/tests.yml/badge.svg)](https://github.com/TobiasOlsacher/hamiltonian-liouvillian-learning/actions)

---

## Overview

This package contains tools for Hamiltonian and Liouvillian learning from measurement data.
It also contains a numerical quantum simulator built on top of the **QuTiP** master equation solver.

The package allows you to:
- Generate measurement data from a mock experiment defined by a master equation in Lindblad form.
- Learn the Hamiltonian and Lindblad operators from the data using one of several learning methods.
- Calculate error bounds for quantum observables from the uncertainty in the parameters of the learned Hamiltonian and Lindblad operators.

Example notebooks can be found in the ``../examples`` folder.

Relevant references are:
- Olsacher et al. *Hamiltonian and Liouvillian learning in weakly-dissipative quantum many-body systems* (2025)
- Kraft et al. *Bounded-Error Quantum Simulation via Hamiltonian and Lindbladian Learning* (2026)
<!-- - ``França et al. Efficient and robust estimation of many-qubit Hamiltonians (2024)``  -->
<!-- - ``Evans et al. Scalable Bayesian Hamiltonian learning (2019)`` -->

---

## Installation from source

```bash
git clone https://github.com/TobiasOlsacher/hamiltonian-liouvillian-learning.git
cd hamiltonian-liouvillian-learning
pip install -r requirements.txt