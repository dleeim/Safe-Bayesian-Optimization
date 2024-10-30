import numpy as np

def differential_evolution(func, bounds, pop_size=20, max_gen=1000, mutation_factor=0.8, crossover_prob=0.7,
                           tol=1e-6, stagnation_gen=100):
    dim = len(bounds)
    
    # Initialize population randomly within bounds
    population = np.random.rand(pop_size, dim)
    for i in range(dim):
        population[:, i] = bounds[i][0] + population[:, i] * (bounds[i][1] - bounds[i][0])
    
    # Evaluate initial population
    fitness = np.apply_along_axis(func, 1, population)
    
    # Track best fitness for stagnation check
    best_fitness = np.min(fitness)
    stagnation_counter = 0

    for generation in range(max_gen):
        for i in range(pop_size):
            # Mutation
            indices = np.random.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant = x1 + mutation_factor * (x2 - x3)
            mutant = np.clip(mutant, [b[0] for b in bounds], [b[1] for b in bounds])
            
            # Crossover
            crossover = np.random.rand(dim) < crossover_prob
            if not np.any(crossover):  
                crossover[np.random.randint(0, dim)] = True
            trial = np.where(crossover, mutant, population[i])
            
            # Selection
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
        
        # Track the best and worst fitness values for convergence check
        current_best_fitness = np.min(fitness)
        current_worst_fitness = np.max(fitness)
        
        # Fitness Convergence Check
        if current_worst_fitness - current_best_fitness < tol:
            print(f"Stopping early at generation {generation} due to fitness convergence.")
            break

        # Stagnation Check
        if current_best_fitness >= best_fitness - tol:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_fitness = current_best_fitness
            
        if stagnation_counter >= stagnation_gen:
            print(f"Stopping early at generation {generation} due to stagnation.")
            break

        print(f"Generation {generation}, best fitness: {current_best_fitness}")

    # Return the best solution found
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

# Example usage
def sphere_function(x):
    return sum(x**2)

bounds = [[-5,5]*5] 

best_solution, best_fitness = differential_evolution(sphere_function, bounds)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
