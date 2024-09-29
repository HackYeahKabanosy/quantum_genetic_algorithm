from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

class QuantumCircuitSimulator:
    def __init__(self, error_prob=0.01):
        # Create a Quantum Circuit with 2 qubits and 2 classical bits
        self.qc = QuantumCircuit(2, 2)

        # Apply a Hadamard gate on qubit 0
        self.qc.h(0)

        # Apply a CNOT gate on qubit 0 and qubit 1
        self.qc.cx(0, 1)

        # Measure the qubits
        self.qc.measure([0, 1], [0, 1])

        # Define noise model
        self.noise_model = NoiseModel()

        # Add depolarizing error to the noise model
        depol_error = depolarizing_error(0.01, 1)  # 1-qubit depolarizing error
        depol_error2 = depolarizing_error(error_prob*2, 2)  # 2-qubit depolarizing error
        self.noise_model.add_all_qubit_quantum_error(depol_error, ['h'])
        self.noise_model.add_all_qubit_quantum_error(depol_error2, ['cx'])

        # Simulator setup
        self.simulator = AerSimulator()

    def mutation_occured(self):
        execution_result = self.run(shots=1)
        return execution_result.get('01', 0) > 0 or execution_result.get('10', 0) > 0

    def run(self, shots=1000, simulator=True, api_token="token"):
        # Execute the circuit on the noisy qasm simulator
        if (simulator):
            result = self.simulator.run(self.qc, noise_model=self.noise_model, shots=shots).result()
        else:
            service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
            backend = service.least_busy(operational=True, simulator=False)
            result = backend.run(self.qc, shots=shots).result()
        # Get the counts (measurement results)
        counts = result.get_counts(self.qc)
        return counts

# If this file is run directly
if __name__ == "__main__":
    # Create an instance of the simulator
    simulator = QuantumCircuitSimulator()

    # Run the simulation
    counts = simulator.run(shots=1000)

    # Display results
    print("\nTotal count for each state are:", counts)
