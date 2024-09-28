import json
import random
import math

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
    def __init__(self, cities, population_size=100, elite_size=20, mutation_rate=0.00, generations=1500):
        self.cities = cities
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations

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

    def mutate_population(self, population):
        for tour in population[self.elite_size:]:
            if random.random() < self.mutation_rate:
                tour.mutate()
        return population

    def next_generation(self, current_gen):
        ranked_tours = self.rank_tours(current_gen)
        selection_results = self.selection(ranked_tours)
        children = self.breed_population(selection_results)
        next_generation = self.mutate_population(children)
        return next_generation

    def run(self):
        population = self.initial_population()
        for i in range(self.generations):
            population = self.next_generation(population)
            best_tour = min(population, key=lambda x: x.distance)
            if i % 10 == 0:
                print(f"Generation {i+1}: Best distance = {best_tour.distance:.2f}")
        return best_tour

def load_examples(filename="tsp_examples.json"):
    with open(filename, "r") as f:
        return json.load(f)

def run_genetic_algorithm(cities):
    ga = GeneticAlgorithm(cities, mutation_rate=0.05)
    best_tour = ga.run()
    return best_tour

if __name__ == "__main__":
    # Load examples
    examples = load_examples()
    
    # Run the genetic algorithm for each example
    for i, example in enumerate(examples):
        print(f"\nRunning example {i+1}")
        cities = [City(city['x'], city['y']) for city in example]
        best_tour = run_genetic_algorithm(cities)
        
        print(f"Example {i+1} results:")
        print(f"Number of cities: {len(cities)}")
        print(f"Best tour distance: {best_tour.distance:.2f}")
        # print("Best tour:")
        # for city in best_tour.cities:
        #     print(f"({city.x}, {city.y})")
        print("-" * 40)