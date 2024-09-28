import json
import random
import math
import matplotlib.pyplot as plt

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

counter = 0
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
        global counter
        for tour in population[self.elite_size:]:
            counter += 1
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
        self.gen_counter+=1
        return next_generation

    def run(self):
        population = self.initial_population()
        best_distances = []
        for i in range(self.generations):
            population = self.next_generation(population)
            best_tour = min(population, key=lambda x: x.distance)
            best_distances.append(best_tour.distance)
            if i % 10 == 0:
                print(f"Generation {i+1}: Best distance = {best_tour.distance:.2f}")
        return best_tour, best_distances

def load_examples(filename="tsp_examples.json"):
    with open(filename, "r") as f:
        return json.load(f)

def run_genetic_algorithm(cities, mutation_rate):
    ga = GeneticAlgorithm(cities, mutation_rate=mutation_rate, elite_size=3, population_size=10, generations=1000)
    best_tour, best_distances = ga.run()
    return best_tour, best_distances, ga.mutation_events

def plot_results(mutation_rates, results):
    for mutation_rate, (distances, mutation_events) in results.items():
        plt.plot(distances, label=f'Mutation Rate: {mutation_rate}')
        for generation, fitness in mutation_events:
            plt.plot(generation, fitness, 'ro')  # Red dots for mutation events at correct coordinates
    plt.title('Best Tour Distance Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Tour Distance')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load examples
    examples = load_examples()
    mutation_rates = [0, 0.01, 0.1]
    results = {}

    # Run the genetic algorithm for each example and mutation rate
    for i, example in enumerate(examples):
        print(f"\nRunning example {i+1}")
        cities = [City(city['x'], city['y']) for city in example]

        # Store results for each mutation rate
        for mutation_rate in mutation_rates:
            print(f"  Running with mutation rate: {mutation_rate}")
            best_tour, best_distances, mutation_events = run_genetic_algorithm(cities, mutation_rate)
            results[mutation_rate] = (best_distances, mutation_events)

            print(f"  Example {i+1} results:")
            print(f"  Number of cities: {len(cities)}")
            print(f"  Best tour distance: {best_tour.distance:.2f}")
            print("-" * 40)

    # Plot results
    plot_results(mutation_rates, results)
    print(f"Total mutations: {counter}")
    # Wait
    input("Press Enter to continue...")
