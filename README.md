## Quantum Voyager Algorithm

#### Solving travelling salesman problem with Genetic Algorithms and Quantum Computing

Cybersecurity challange on HackYeah2024

![QuantumVoyagePlatform](https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/QuantumVoyagePlatform.gif)

### Travelling Salesman Problem
Find the best path covering all the cities/points without repeating any. This task aims to minimize the total distance or cost of travel, and is widely used in logistics, planning, and routing.

### How to solve it with Genetic Algorithms?
1. We have population of solutions
<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/qv1.png" width="400">

2. Solutions change with each new generation, mixing and creating potentially better solutions

3. But they can get stuck in a local minimum, and no longer improve in new generations

<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/qv2.png" width="400">

4. In such case, we need some random unexpected change - a **mutation**, that will modify solutions.

<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/qv3.png" width="400">

5. Applying mutations allows us to "unstuck" from local optimal solution.
<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/chartMutation.png" width="400">
 
### How to apply Quantum Computing to enhance it?

#### Quantum Computers
- uses quantum bits that represent probability between 0 and 1
- operates on circuits consisting of quantum gates
- generates noise, unexpected outcomes, as a side-effect

#### Noise
Noise is a side-effect, something unexpected, outcome that “shouldn’t be possible”.
Even though we can make use of it! 
For example a gate below has only two possible outputs: [00, 11]. But on quantum computers it can generate some unexpected outputs.

<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/gate.png" width="600">

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

#### Outcome

Outcome per 1024 shots on a  real quantum computer (on non-quantum computer, 0010 or 0001 would never occur)

<img src="https://github.com/HackYeahKabanosy/quantum_genetic_algorithm/blob/main/docs/chartNoise.png" width="400">

Similarly to mutation, it’s an unepxected change that happens randomly.
It’s possible to be received with different frequency, depending on a quantum machine and conditions.

```
# if anomaly occurs, then apply mutation
def should_apply_mutation():
  res = self.qiskit_runtime.run(shots=1)
  return 
    res.get('01', 0) > 0 
    or 
    res.get('10', 0) > 0
```

### How to play with Quantum Voyager?

We also created a platform where you can simulate how this algorithm would work on a quantum computer with a given noise level.

```
git clone git@github.com:HackYeahKabanosy/quantum_genetic_algorithm.git

```
