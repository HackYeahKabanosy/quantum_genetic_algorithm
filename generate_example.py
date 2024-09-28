import random
import json

def generate_tsp_example(num_cities, min_coord=0, max_coord=100):
    return [
        {"x": random.randint(min_coord, max_coord),
         "y": random.randint(min_coord, max_coord)}
        for _ in range(num_cities)
    ]

def generate_examples(num_examples=10, min_cities=5, max_cities=20, min_coord=0, max_coord=100):
    examples = []
    for _ in range(num_examples):
        num_cities = random.randint(min_cities, max_cities)
        example = generate_tsp_example(num_cities, min_coord, max_coord)
        examples.append(example)
    return examples

def save_examples(examples, filename="tsp_examples.json"):
    with open(filename, "w") as f:
        json.dump(examples, f, indent=2)

if __name__ == "__main__":
    examples = generate_examples(num_examples=1, min_cities=20, max_cities=20)
    save_examples(examples)
    print(f"Generated 10 TSP examples and saved them to tsp_examples.json")
    
    # Print the first example to verify
    print("\nFirst example:")
    print(json.dumps(examples[0], indent=2))