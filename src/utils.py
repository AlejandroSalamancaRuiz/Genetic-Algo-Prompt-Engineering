import matplotlib.pyplot as plt
import numpy as np
def load_prompts(file_path):
    """
    Load prompts from a file.

    Args:
        file_path (str): Path to the file containing prompts.

    Returns:
        list: List of prompts.
    """
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]


def plot_gene_frequency(results, cutoff):
    """
    Plot the frequency of genes that meet a fitness cutoff.

    Args:
        results (dict): Dictionary containing the results of the genetic algorithm.
        cutoff (float): Fitness cutoff value.
    """
    generations = len(results.keys()) - 1
    individuals = len(results[0].keys())

    sys_prompt_genes = []
    instruct_genes = []
    num_examples_genes = []

    # Collect genes that meet the fitness cutoff
    for i in range(generations):
        for j in range(individuals):
            if results[i][j]['fitness'] >= cutoff:
                sys_prompt_genes.append(results[i][j]['sys_prompt_idx'])
                instruct_genes.append(results[i][j]['instruction_idx'])
                num_examples_genes.append(results[i][j]['num_examples'])

    # Convert lists to numpy arrays
    sys_prompt_genes = np.array(sys_prompt_genes)
    instruct_genes = np.array(instruct_genes)
    num_examples_genes = np.array(num_examples_genes)

    plt.figure(figsize=(18, 7))

    # Plot System Prompt Index frequency
    counts = np.bincount(sys_prompt_genes)
    numbers = np.arange(len(counts))
    plt.subplot(1, 3, 1)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('System Prompt Index')
    plt.xlabel('Index')
    plt.ylabel('Frequency')

    # Plot Instruction Prompt Index frequency
    counts = np.bincount(instruct_genes)
    numbers = np.arange(len(counts))
    plt.subplot(1, 3, 2)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Instruction Prompt Index')
    plt.xlabel('Index')
    plt.ylabel('Frequency')

    # Plot Number of Examples frequency
    counts = np.bincount(num_examples_genes)
    numbers = np.arange(len(counts))
    plt.subplot(1, 3, 3)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Number of Examples')
    plt.xlabel('Num Examples')
    plt.ylabel('Frequency')

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_fitness_per_generation(results):
    """
    Plot the median, minimum, and maximum fitness per generation.

    Args:
        results (dict): Dictionary containing the results of the genetic algorithm.
    """
    generations = len(results.keys()) - 1
    individuals = len(results[0].keys())

    median = np.zeros(generations)
    min_ = np.zeros(generations)
    max_ = np.zeros(generations)

    # Calculate fitness metrics for each generation
    for i in range(generations):
        fitness_per_gen = np.zeros(individuals)
        for j in range(individuals):
            fitness_per_gen[j] = results[i][j]['fitness']

        median[i] = np.median(fitness_per_gen)
        min_[i] = np.min(fitness_per_gen)
        max_[i] = np.max(fitness_per_gen)

    # Plot fitness metrics
    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), median, label='Median Fitness', color='blue', marker='o')
    plt.plot(range(generations), min_, label='Min Fitness', color='green', marker='v')
    plt.plot(range(generations), max_, label='Max Fitness', color='red', marker='^')
    
    plt.title('Fitness Metrics Per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_gene_frequency_per_gen(results, gen):
    """
    Plot the frequency of genes for a specific generation.

    Args:
        results (dict): Dictionary containing the results of the genetic algorithm.
        gen (int): Generation number to plot.
    """
    individuals = len(results[0].keys())

    sys_prompt_genes = []
    instruct_genes = []
    num_examples_genes = []

    # Collect genes for the specified generation
    for j in range(individuals):
        sys_prompt_genes.append(results[gen][j]['sys_prompt_idx'])
        instruct_genes.append(results[gen][j]['instruction_idx'])
        num_examples_genes.append(results[gen][j]['num_examples'])

    # Convert lists to numpy arrays
    sys_prompt_genes = np.array(sys_prompt_genes)
    instruct_genes = np.array(instruct_genes)
    num_examples_genes = np.array(num_examples_genes)

    plt.figure(figsize=(18, 7))

    # Plot System Prompt Index frequency
    counts = np.bincount(sys_prompt_genes)
    numbers = np.arange(len(counts))
    plt.subplot(1, 3, 1)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('System Prompt Index')
    plt.xlabel('Index')
    plt.ylabel('Frequency')

    # Plot Instruction Prompt Index frequency
    counts = np.bincount(instruct_genes)
    numbers = np.arange(len(counts))
    plt.subplot(1, 3, 2)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Instruction Prompt Index')
    plt.xlabel('Index')
    plt.ylabel('Frequency')

    # Plot Number of Examples frequency
    counts = np.bincount(num_examples_genes)
    numbers = np.arange(len(counts))
    plt.subplot(1, 3, 3)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Number of Instructions')
    plt.xlabel('Num Instructions')
    plt.ylabel('Frequency')

    # Display the plot
    plt.tight_layout()
    plt.show()


def best_prompts(results, exp_conditions, cutoff):
    """
    Print the best prompts that meet a fitness cutoff.

    Args:
        results (dict): Dictionary containing the results of the genetic algorithm.
        exp_conditions (dict): Experimental conditions including prompts and instructions.
        cutoff (float): Fitness cutoff value.
    """
    generations = len(results.keys()) - 1
    individuals = len(results[0].keys())

    # Print prompts that meet the fitness cutoff
    for i in range(generations):
        for j in range(individuals):
            if results[i][j]['fitness'] >= cutoff:
                print('Generation:', i)
                print('Sys Prompt:', exp_conditions['sys_prompts'][results[i][j]['sys_prompt_idx']])
                print('Inst Prompt:', exp_conditions['instructions'][results[i][j]['instruction_idx']])
                print('Num Exp:', results[i][j]['num_examples'])
                print('---------------------------------------------------')
                print()


def experimental_setup(category, exp_conditions):
    """
    Print the experimental setup for a given category.

    Args:
        category (str): Category of the experimental setup.
        exp_conditions (dict): Experimental conditions including prompts and instructions.
    """
    if category not in exp_conditions.keys():
        raise Exception('Invalid category')
    
    for text in exp_conditions[category]:
        print(text)


def best_prompts_per_model(results, exp_conditions, cutoff):
    """
    Save the best prompts per model to a file.

    Args:
        results (dict): Dictionary containing the results of the genetic algorithm.
        exp_conditions (dict): Experimental conditions including prompts and instructions.
        cutoff (float): Fitness cutoff value.
    """
    with open('best_prompts_' + exp_conditions['llm']['model'] + '__.txt', 'w') as f:
        generations = len(results.keys()) - 1
        individuals = len(results[0].keys())

        # Write prompts that meet the fitness cutoff to a file
        for i in range(generations):
            for j in range(individuals):
                if results[i][j]['fitness'] >= cutoff:
                    fitness = results[i][j]['fitness']
                    sys_prompt = exp_conditions['sys_prompts'][results[i][j]['sys_prompt_idx']]
                    inst_prompt = exp_conditions['instructions'][results[i][j]['instruction_idx']]

                    f.write(f'Generation: {i} -- Fitness: {fitness}\n\n')
                    f.write('-----------------------------------------------------------\n')
                    f.write('---------------------- Prompt------------------------------\n')
                    f.write('-----------------------------------------------------------\n\n')
                    f.write('--------------------- System prompt-------------------------\n\n')
                    f.write(sys_prompt + '\n\n')
                    f.write('-------------------- Instruction prompt --------------------\n\n')
                    f.write(inst_prompt + '\n\n')
                    f.write('--------------- Examples given in the prompt ---------------\n\n')

                    for k in range(results[i][j]['num_examples']):
                        example = exp_conditions['examples'][k]
                        f.write(example + '\n')

                    f.write('\n---------------------------------------------------------------\n')
                    f.write('---------------------------------------------------------------\n')
                    f.write('---------------------------------------------------------------\n\n')
