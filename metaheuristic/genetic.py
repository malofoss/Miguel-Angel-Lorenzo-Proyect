import random
import sys
sys.path.append(".")
from src.train import train

# Espacio de búsqueda de hiperparámetros
PARAM_GRID = {
    "learning_rate": [0.01, 0.001, 0.0001],
    "batch_size":    [16, 32, 64],
    "dropout":       [0.2, 0.3, 0.5],
    "epochs":        [10]  # Fijo para que el AG no tarde demasiado
}

def random_individual():
    return {key: random.choice(values) for key, values in PARAM_GRID.items()}

def fitness(individual):
    print(f"\n  Evaluando: {individual}")
    val_loss = train(config=individual)
    # fitness usa pérdida de validación (menor = mejor)
    # La pérdida ahora combina clasificación y regresión
    print(f"  → Val Loss: {val_loss:.4f}")
    return val_loss  # Menor pérdida = mejor

def select_parents(population, scores):
    # Selección por torneo: elige el mejor de 2 aleatorios
    selected = []
    for _ in range(2):
        i, j = random.sample(range(len(population)), 2)
        winner = i if scores[i] < scores[j] else j
        selected.append(population[winner])
    return selected

def crossover(parent1, parent2):
    child = {}
    for key in PARAM_GRID.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual, mutation_rate=0.2):
    for key in PARAM_GRID.keys():
        if random.random() < mutation_rate:
            individual[key] = random.choice(PARAM_GRID[key])
    return individual

def genetic_algorithm(pop_size=6, generations=4):
    print("=== Algoritmo Genético - Búsqueda de Hiperparámetros ===\n")

    # Población inicial aleatoria
    population = [random_individual() for _ in range(pop_size)]

    best_individual = None
    best_score = float("inf")

    for gen in range(generations):
        print(f"\n--- Generación {gen+1}/{generations} ---")

        # Evaluar fitness de cada individuo
        scores = [fitness(ind) for ind in population]

        # Actualizar el mejor global
        for i, score in enumerate(scores):
            if score < best_score:
                best_score = score
                best_individual = population[i].copy()
                print(f"  ★ Nuevo mejor global: {best_individual} → Val Loss: {best_score:.4f}")

        # Crear nueva generación
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, scores)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    print(f"\n=== Resultado Final ===")
    print(f"Mejores hiperparámetros: {best_individual}")
    print(f"Mejor Val Loss: {best_score:.4f}")
    return best_individual, best_score


if __name__ == "__main__":
    best_params, best_mae = genetic_algorithm(pop_size=6, generations=4)