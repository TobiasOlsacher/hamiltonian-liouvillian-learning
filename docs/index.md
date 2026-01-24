
# Hamiltonian and Liouvillian learning

Welcome to the **hamiltonian-liouvillian-learning** documentation!  
This project provides easy-to-use tools for quantum Hamiltonian and Liouvillian learning from measurement data.

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](installation.md)
- [Usage](usage.md)
- [API Reference](api.md)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

This package allows you to:
- Generate measurement data from a mock quantum simulation experiment defined by a master equation in Lindblad form (using **QuTiP**).
- Learn the Hamiltonian and Lindblad operators from measurement data using different learning methods from the literature.
- Calculate error bounds for quantum observables from the uncertainty in the parameters of the learned Hamiltonian and Lindblad operators.


The code is based on the references:
- Olsacher et al. Hamiltonian and Liouvillian learning in weakly-dissipative quantum many-body systems. *Quantum Sci. Technol. 10 015065* (2025)
- Kraft et al. Bounded-Error Quantum Simulation via Hamiltonian and Lindbladian Learning. *arXiv:2511.23392*  (2026) 
- Evans et al. Scalable Bayesian Hamiltonian learning. *arXiv:1912.07636* (2019)  	
<!-- - França et al. Efficient and robust estimation of many-qubit Hamiltonians. *Nat Commun 15, 311* (2024)  -->
<!-- - Zubida et al. Optimal short-time measurements for Hamiltonian learning. *arXiv:2108.08824* (2021) -->

---

## Contributing

Contributions are welcome! You can:

- Report issues on GitHub
- Submit pull requests
- Improve the documentation

Please follow the coding style and document any new features.

---

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

---