# Quantum Voyager Algorithm
Solving traveling salesman problem with Genetic Algorithms and Quantum Computing

![QuantumVoyagePlatform](https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/QuantumVoyagePlatform.gif)

### Travelling Salesman Problem
Find the best path covering all the cities/points without repeating any. This task aims to minimize the total distance or cost of travel, and is widely used in logistics, planning, and routing.

### How to solve it with Genetic Algorithms?
1. We have population of solutions
<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/qv1.png" width="400">

2. Solutions change with each new generation, mixing and creating potentially better solutions

3. But they often get stuck at some point , and no longer improve in new generations

<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/qv2.png" width="400">

4. In such case, we need some random unexpected change - a **mutation**, that will modify solutions.

<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/qv3.png" width="400">

5. Applying mutations allows us to "unstuck" from local optimal solution.
<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/chartMutation.png" width="400">
 
### How to apply Quantum Computing to enhance it?
We can make use of noise from quantum computing operations.
A gate we use has only two possible outputs: [00, 11]. But on quantum computers it can generate some unexpected outputs.

<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/gate.png" width="400">

```
# create a Quantum Circuit with 2 qubits and 2 classical bits 
qc = QuantumCircuit(2, 2) 
# apply a Hadamard gate on qubit 0 
qc.h(0) 
# apply a CNOT gate on qubit 0 and qubit 1 
qc.cx(0, 1) 
# measure the qubits 
qc.measure([0, 1], [0, 1])
```
<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/chartNoise.png" width="400">

### How to play with Quantum Voyager?

```
todo
```
