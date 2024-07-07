import numpy as np
import imageio
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0

class GeneticAlgorithm:
    def __init__(self, target_image_path, population_size=8, mating_pool_size=4, mutation_percent=0.01):
        self.target_image = imageio.imread(target_image_path)   
        self.target_genes = self.image_to_genes(self.target_image)
        self.population_size = population_size
        self.mating_pool_size = mating_pool_size
        self.mutation_percent = mutation_percent
        self.population = self.initialze_population()
        self.iterations = 100

    def image_to_genes(self, image_array):
        return np.reshape(a=image_array, newshape=(-1))

    def initialize_population(self):
        initial_population = []
        for _ in range(self.population_size):
            genes = np.random.random(size=len(self.target_genes))
            individual = Individual(genes)
            initial_population.append(individual)
        return initial_population

    def calculate_fitness(self, individual):
        quality = np.mean(np.abs(self.target_genes - individual.genes))
        quality = np.sum(self.target_genes) - quality
        return quality

    def calculate_population_fitness(self):
        for individual in self.population:
            individual.fitness = self.calculate_fitness(individual)

    def select_mating_pool(self):
        parents = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.mating_pool_size]
        return parents

    def crossover(self, parents):
        new_population = []
        new_population.extend(parents)

        for _ in range(self.population_size - self.mating_pool_size):
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            crossover_point = np.random.randint(0, len(self.target_genes))
            child_genes = np.concatenate(
                (parent1.genes[:crossover_point], parent2.genes[crossover_point:])
            )
            new_individual = Individual(child_genes)
            new_population.append(new_individual)

        return new_population

    def mutation(self):
        for individual in self.population[self.mating_pool_size:]:
            mutated_genes_indices = np.random.choice(
                len(self.target_genes),
                size=int(self.mutation_percent * len(self.target_genes)),
                replace=False,
            )
            individual.genes[mutated_genes_indices] = np.random.random(size=len(mutated_genes_indices))

    def save_images(self, iteration):
        if iteration % 100 == 0:
            best_solution = max(self.population, key=lambda x: x.fitness)
            best_solution_img = self.genes_to_image(best_solution.genes)
            plt.imsave(f'solution_{iteration}.png', best_solution_img)

    def genes_to_image(self, genes):
        return np.reshape(a=genes, newshape=self.target_image.shape)


    def plot_fitness_scores(self, fitness_scores):
        generations = range(1, self.iterations + 1)
        plt.plot(generations, fitness_scores, marker='o')
        plt.title('Fitness Scores Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Max Fitness Score')
        plt.show()

    def run(self):
        fitness_scores = []

        for iteration in range(self.iterations):
            self.calculate_population_fitness()
            max_fitness = max(individual.fitness for individual in self.population)
            fitness_scores.append(max_fitness)

            print(f'Quality: {max_fitness}, Iteration: {iteration}')

            parents = self.select_mating_pool()
            self.population = self.crossover(parents)
            self.mutation()
            self.save_images(iteration)

        # Plotting fitness scores over generations
        self.plot_fitness_scores(fitness_scores)

        self.show_individuals()


if __name__ == "__main__":
    target_image_path = "wm.jpg"
    genetic_algorithm = GeneticAlgorithm(target_image_path)
    genetic_algorithm.run()
