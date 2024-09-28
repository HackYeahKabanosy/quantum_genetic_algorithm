import json
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from quantum import QuantumCircuitSimulator

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Tour:
    def __init__(self, cities):
        self.cities = cities
        self.distance = self.calculate_distance()

    def calculate_distance(self):
        total_distance = sum(self.cities[i].distance(self.cities[i-1]) for i in range(len(self.cities)))
        return total_distance

    def mutate(self):
        i, j = random.sample(range(len(self.cities)), 2)
        self.cities[i], self.cities[j] = self.cities[j], self.cities[i]
        self.distance = self.calculate_distance()

def create_random_tour(cities):
    return Tour(random.sample(cities, len(cities)))

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1.cities)), 2))
    child_cities = parent1.cities[start:end]
    child_cities += [city for city in parent2.cities if city not in child_cities]
    return Tour(child_cities)

class GeneticAlgorithm:
    def __init__(self, cities, population_size=100, elite_size=20, mutation_rate=0.01, generations=200):
        self.cities = cities
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.simulator = QuantumCircuitSimulator(mutation_rate)
        self.mutation_events = []  # Track mutation events
        self.gen_counter = 0

    def initial_population(self):
        return [create_random_tour(self.cities) for _ in range(self.population_size)]

    def rank_tours(self, tours):
        return sorted(tours, key=lambda x: x.distance)

    def selection(self, ranked_tours):
        selection_results = ranked_tours[:self.elite_size]
        for _ in range(len(ranked_tours) - self.elite_size):
            pick = random.randint(0, len(ranked_tours) - 1)
            selection_results.append(ranked_tours[pick])
        return selection_results

    def breed_population(self, mating_pool):
        children = mating_pool[:self.elite_size]
        for _ in range(len(mating_pool) - self.elite_size):
            parent1, parent2 = random.sample(mating_pool, 2)
            children.append(crossover(parent1, parent2))
        return children

    def mutate_population(self, population, generation):
        for tour in population[self.elite_size:]:
            if self.simulator.mutation_occured():
                tour.mutate()
                best_fitness = min(population, key=lambda x: x.distance).distance
                self.mutation_events.append((generation, best_fitness))  # Track generation and best fitness
        return population

    def next_generation(self, current_gen):
        ranked_tours = self.rank_tours(current_gen)
        selection_results = self.selection(ranked_tours)
        children = self.breed_population(selection_results)
        next_generation = self.mutate_population(children, self.gen_counter)  # Pass current generation index
        self.gen_counter += 1
        return next_generation

    def run(self):
        population = self.initial_population()
        best_distances = []
        for i in range(self.generations):
            population = self.next_generation(population)
            best_tour = min(population, key=lambda x: x.distance)
            best_distances.append(best_tour.distance)
        return best_distances

def load_examples(filename="tsp_examples.json"):
    with open(filename, "r") as f:
        return json.load(f)

def run_genetic_algorithm(cities, population_size, elite_size, mutation_rate, generations):
    ga = GeneticAlgorithm(cities, mutation_rate=mutation_rate, elite_size=elite_size, population_size=population_size, generations=generations)
    best_distances = ga.run()
    return best_distances

def plot_results(best_distances):
    plt.figure(figsize=(10, 5))
    plt.plot(best_distances, label='Best Tour Distance')
    plt.title('Best Tour Distance Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Tour Distance')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Save the plot to a file and return the file path
    plot_file_path = "best_tour_distance_plot.png"
    plt.savefig(plot_file_path)
    plt.close()  # Close the plot to free memory
    return plot_file_path

def run_app(population_size, elite_size, mutation_rate, generations):
    examples = load_examples()
    results = []

    for i, example in enumerate(examples):
        cities = [City(city['x'], city['y']) for city in example]
        best_distances = run_genetic_algorithm(cities, population_size, elite_size, mutation_rate, generations)
        results.append(best_distances)

    # Plot results for the last example and return the plot image path
    return plot_results(results[-1])

iface = gr.Interface(
    fn=run_app,
    inputs=[
        gr.Slider(10, 100, label="Population Size", step=1),
        gr.Slider(5, 50, label="Elite Size" , step=1),
        gr.Slider(0.0, 0.5, step=0.01, label="Mutation Rate"),
        gr.Slider(10, 1000, label="Generations", step=1)
    ],
    outputs=gr.Image(type="filepath"),
    title="Genetic Algorithm TSP Solver",
    description="Adjust the parameters to customize the Genetic Algorithm for solving the Traveling Salesman Problem."
)

if __name__ == "__main__":
    iface.launch()
