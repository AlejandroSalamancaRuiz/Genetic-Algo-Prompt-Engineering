from src.experiment_conditions import Experiment_conditions
import numpy as np
from src.prompt import Prompt, encode_to_binary, decode_from_binary
import random
import pickle
import os
import datetime
import time

class GeneticAlgorithm:

    def __init__(self, experiment_conditions: Experiment_conditions, num_individuals: int, num_generatios: int) -> None:
        """
        Initialize the GeneticAlgorithm with experiment conditions, number of individuals, and number of generations.
        """
        self.experiment_conditions = experiment_conditions

        if num_individuals % 2 != 0:
            raise Exception('Population must have an even number of individuals')

        self.num_individuals = num_individuals
        self.num_generations = num_generatios

        self.generations = {0: []}
        self.fitness_per_gen = {0: []}
        self.run_time = 0

    def save_experiment(self):
        """
        Save the experiment conditions and the run results to disk.
        """
        # Create a unique directory name based on the model, dataset, and current timestamp
        path = self.experiment_conditions.llm.model + '--' + str(self.experiment_conditions.dataset.dataset_name) + '--' + str(datetime.datetime.now())

        # Create the directory to save experiment results
        os.mkdir(path)

        # Initialize a dictionary to store experiment conditions
        exp_cond = {}
        exp_cond['llm'] = {'provider': self.experiment_conditions.llm.provider, 'model': self.experiment_conditions.llm.model}
        exp_cond['mutators'] = self.experiment_conditions.prompt_mutators
        exp_cond['sys_prompts'] = self.experiment_conditions.sys_prompts
        exp_cond['instructions'] = self.experiment_conditions.instructions
        exp_cond['examples'] = self.experiment_conditions.examples
        exp_cond['dataset'] = self.experiment_conditions.dataset.dataset_name

        # Save the experiment conditions to a pickle file
        with open(path + '/exp_conditions.pkl', 'wb') as f:
            pickle.dump(exp_cond, f)

        # Initialize a dictionary to store the run results
        exp_run = {'run_time': self.run_time}

        # Iterate over each generation to save individuals and their fitness
        for i in range(self.num_generations):
            individuals = {}

            # Iterate over each individual in the generation
            for j in range(self.num_individuals):
                individuals[j] = {'sys_prompt_idx': self.generations[i][j].sys_prompt_idx,
                                'instruction_idx': self.generations[i][j].instruction_idx,
                                'num_examples': self.generations[i][j].num_examples,
                                'fitness': self.fitness_per_gen[i][j]}
                
            # Save the individuals of the current generation to the run results
            exp_run[i] = individuals

        # Save the run results to a pickle file
        with open(path + '/ga_run.pkl', 'wb') as f:
            pickle.dump(exp_run, f)
    
    def init_population(self):
        """
        Initialize the population with random individuals.
        """
        # Initialize the first generation with random individuals
        for i in range(self.num_individuals):
            # Randomly select the number of examples, system prompt, and instruction
            num_examples = np.random.randint(self.experiment_conditions.max_num_examples)
            sys_prompt_idx = np.random.randint(len(self.experiment_conditions.sys_prompts))
            instruction_idx = np.random.randint(len(self.experiment_conditions.instructions))

            # Create a new individual with the randomly selected parameters
            individual = Prompt(self.experiment_conditions, sys_prompt_idx, instruction_idx, num_examples)

            # Add the individual to the first generation
            self.generations[0].append(individual)

    def crossover(self, parent_1: Prompt, parent_2: Prompt):
        """
        Perform crossover between two parent individuals to produce two children.
        """
        # Randomly choose a mutation type (0 or 1)
        mutation_type = np.random.randint(2)
        # Randomly choose a cutoff point for crossover
        cutoff = np.random.randint(len(parent_1.encoded_prompt))

        if mutation_type == 0:
            # Perform simple crossover without mutation
            child_1 = np.concatenate((parent_1.encoded_prompt[0:cutoff], parent_2.encoded_prompt[cutoff:]))
            child_2 = np.concatenate((parent_2.encoded_prompt[0:cutoff], parent_1.encoded_prompt[cutoff:]))
        elif mutation_type == 1:
            # Perform crossover with mutation
            child_1 = np.concatenate((parent_1.encoded_prompt[0:cutoff], parent_2.encoded_prompt[cutoff:]))
            child_2 = np.concatenate((parent_2.encoded_prompt[0:cutoff], parent_1.encoded_prompt[cutoff:]))

            # Randomly select mutators for each child
            i = np.random.randint(len(self.experiment_conditions.prompt_mutators))
            j = np.random.randint(len(self.experiment_conditions.prompt_mutators))

            # Decode instruction indices from binary encoding
            inst_child_1_idx = decode_from_binary(child_1[20:]) 
            inst_child_2_idx = decode_from_binary(child_2[20:])

            # Ensure instruction indices are within bounds
            inst_child_1_idx = len(self.experiment_conditions.instructions) - 1 if inst_child_1_idx >= len(self.experiment_conditions.instructions) else inst_child_1_idx
            inst_child_2_idx = len(self.experiment_conditions.instructions) - 1 if inst_child_2_idx >= len(self.experiment_conditions.instructions) else inst_child_2_idx
            
            # Mutate instructions for each child
            new_instruct_child_1 = self.experiment_conditions.llm.mutate(self.experiment_conditions.prompt_mutators[i], 
                                           self.experiment_conditions.instructions[inst_child_1_idx])
            new_instruct_child_2 = self.experiment_conditions.llm.mutate(self.experiment_conditions.prompt_mutators[j],
                                           self.experiment_conditions.instructions[inst_child_2_idx])
            
            # Append new mutated instructions to the list
            self.experiment_conditions.instructions.append(new_instruct_child_1)
            self.experiment_conditions.instructions.append(new_instruct_child_2)

            # Encode new instruction indices to binary
            encoded_new_instruct_1 = encode_to_binary(len(self.experiment_conditions.instructions) - 2, 10)
            encoded_new_instruct_2 = encode_to_binary(len(self.experiment_conditions.instructions) - 1, 10)

            # Replace the instruction part of the child's encoding with the new mutated instruction
            child_1 = np.concatenate((child_1[0:20], encoded_new_instruct_1))
            child_2 = np.concatenate((child_2[0:20], encoded_new_instruct_2))
        else:
            # No crossover or mutation, children are copies of parents
            child_1 = parent_1.encoded_prompt
            child_2 = parent_2.encoded_prompt
        
        return child_1, child_2

    def create_next_generation(self, generation, encoded_children):
        """
        Create the next generation from the encoded children.
        """
        print()
        # Initialize the new generation list
        self.generations[generation] = []
        i = 1

        # Iterate over the encoded children to create new individuals
        for encoded_child in encoded_children:
            # Create a new individual from the encoded child
            individual = Prompt(self.experiment_conditions, None, None, None, binary_encoding=encoded_child)
            # Add the new individual to the current generation
            self.generations[generation].append(individual)
            # Print progress of creating the next generation
            print(f'Creating next generation: {(i * 100) / len(encoded_children):.1f}%', end='\r')
            i += 1
        
        # Initialize the fitness list for the new generation
        self.fitness_per_gen[generation] = []

    def selection_process(self, generation):
        """
        Perform the selection process to choose parents and generate children.
        """
        print()
        encoded_children = []
        # Get the indices of all individuals in the current generation
        individuals_idx = range(self.num_individuals)
        # Select parents based on their fitness using a weighted random choice
        selected_parents_idx = random.choices(individuals_idx, weights=self.fitness_per_gen[generation], k=self.num_individuals)

        i = 0
        # Iterate over the selected parents to perform crossover and generate children
        while i < len(selected_parents_idx):
            # Select two parents
            parent_1 = self.generations[generation][selected_parents_idx[i]]
            parent_2 = self.generations[generation][selected_parents_idx[i + 1]]

            # Perform crossover to generate two children
            child_1, child_2 = self.crossover(parent_1, parent_2)

            # Add the encoded children to the list
            encoded_children.append(child_1)
            encoded_children.append(child_2)

            # Print progress of the selection process
            print(f'Performing selection process: {((i + 1) * 100) / len(selected_parents_idx):.1f}%', end='\r')
            i += 2
        
        # Return the list of encoded children
        return encoded_children

    def calculate_generation_fitness(self, generation):
        """
        Calculate the fitness of each individual in the generation.
        """
        print()
        i = 1
        # Iterate over each prompt in the current generation
        for prompt in self.generations[generation]:
            # Evaluate the fitness of the prompt and append it to the fitness list
            self.fitness_per_gen[generation].append(prompt.evaluate())
            # Print progress of fitness calculation
            print(f'Calculating fitness: {(i * 100) / len(self.generations[generation]):.1f}%', end='\r')
            i += 1

    def run_algorithm(self):
        """
        Run the genetic algorithm for the specified number of generations.
        """
        # Record the start time of the algorithm
        start = time.time()

        # Iterate over the number of generations
        for i in range(self.num_generations):
            print()
            print('Generation:', i)
            # Calculate the fitness of the current generation
            self.calculate_generation_fitness(i)
    
            # If not the last generation, perform selection and create the next generation
            if i != self.num_generations - 1:
                # Perform the selection process to generate encoded children
                encoded_children = self.selection_process(i)
                # Create the next generation from the encoded children
                self.create_next_generation(i + 1, encoded_children)
        
        # Calculate the total run time of the algorithm
        self.run_time = time.time() - start










