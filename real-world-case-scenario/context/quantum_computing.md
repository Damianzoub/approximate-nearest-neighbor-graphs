# Quantum Computing

Quantum computing harnesses the principles of quantum mechanics — superposition, entanglement, and interference — to perform computations in ways that classical computers cannot efficiently replicate. While a classical bit is either 0 or 1, a quantum bit (qubit) can exist in a superposition of both states simultaneously, enabling quantum computers to explore many solutions in parallel.

## Fundamental Principles

Superposition allows a qubit to represent 0 and 1 simultaneously until measured. Entanglement links two or more qubits such that the state of one instantaneously determines the state of its partner, regardless of physical distance. Quantum interference allows algorithms to amplify the probability of correct answers and cancel out incorrect ones — the core mechanism that gives quantum algorithms their advantage.

Quantum gates manipulate qubits analogously to classical logic gates but operate on quantum states. Common gates include the Hadamard gate (creating superposition), the Pauli-X gate (quantum NOT), and the CNOT gate (creating entanglement). A sequence of quantum gates forms a quantum circuit, the basic unit of a quantum algorithm.

## Hardware Platforms

Several physical systems are being explored for building qubits. Superconducting qubits — used by IBM, Google, and Rigetti — operate at temperatures near absolute zero (~15 millikelvin) and offer fast gate operations. Trapped ion qubits — used by IonQ and Honeywell — confine individual ions with electromagnetic fields and achieve very high gate fidelities but slower operations.

Photonic qubits encode information in photon properties such as polarisation. Topological qubits, pursued by Microsoft, aim to store quantum information in more robust non-local degrees of freedom, potentially offering intrinsic error protection. Each platform faces the fundamental challenge of decoherence — the loss of quantum information due to environmental noise.

## Quantum Algorithms

Shor's algorithm, published in 1994, demonstrated that a quantum computer could factorise large integers exponentially faster than any known classical algorithm, threatening RSA cryptography. Grover's algorithm provides a quadratic speedup for unstructured search problems. The Quantum Approximate Optimisation Algorithm (QAOA) and Variational Quantum Eigensolver (VQE) target combinatorial optimisation and chemistry simulations on near-term noisy hardware.

Quantum simulation — modelling quantum systems such as molecules and materials — is considered one of the most promising near-term applications, with potential to accelerate drug discovery and materials science.

## Quantum Error Correction

Current quantum devices are noisy — gate errors, decoherence, and measurement errors accumulate quickly. Quantum error correction encodes one logical qubit in many physical qubits and detects errors without directly measuring the fragile quantum state. The surface code is the leading error correction scheme, requiring roughly 1,000 physical qubits per fault-tolerant logical qubit.

Achieving large-scale fault-tolerant quantum computation requires millions of high-quality physical qubits — well beyond today's hundreds to thousands. The path from current noisy intermediate-scale quantum (NISQ) devices to fault-tolerant machines is one of the central challenges in the field.

## Timeline and Applications

Most experts expect commercially relevant quantum advantage in specific domains — optimisation, cryptography, chemistry, and machine learning — within the next decade. Quantum machine learning explores whether quantum computers can accelerate the training or inference of classical ML models, though theoretical advantages remain an active area of research.
