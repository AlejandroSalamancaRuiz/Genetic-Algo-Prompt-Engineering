from datasets import load_dataset
import numpy as np
import re


class Dataset:

    def __init__(self, name='gsm8k') -> None:
        """
        Initializes the Dataset object with a given dataset name.
        
        Args:
            name (str): The name of the dataset to load. Default is 'gsm8k'.
        """
        self.dataset_name = name
        self.dataset = load_dataset(name, 'main')

    def pick_random_examples(self, max_num_examples: int):
        """
        Picks a specified number of random examples from the dataset.
        
        Args:
            max_num_examples (int): The maximum number of examples to pick.
        
        Returns:
            list: A list of randomly picked examples in the format 'Q: <question>\nA: <answer>'.
        """
        examples = []
        indexes = np.random.randint(len(self.dataset['train']), size=max_num_examples)

        for index in indexes:
            if self.dataset_name == 'gsm8k':
                examples.append('Q: ' + self.dataset['train'][int(index)]['question'] + '\nA: ' + self.dataset['train'][int(index)]['answer'])

        return examples

    def pick_random_problems(self, num_problems: int, set='train'):
        """
        Picks a specified number of random problems from the dataset.
        
        Args:
            num_problems (int): The number of problems to pick.
            set (str): The dataset split to pick problems from. Default is 'train'.
        
        Returns:
            tuple: A tuple containing two lists - problems and their corresponding answers.
        """
        problems, answers = []
        indexes = np.random.randint(len(self.dataset[set]), size=num_problems)

        for index in indexes:
            if self.dataset_name == 'gsm8k':
                problems.append('Q: ' + self.dataset[set][int(index)]['question'] + '\nA: ')
                answers.append(self.dataset[set][int(index)]['answer'])

        return problems, answers

    def evaluate(self, real_answer, predicted_answer):
        """
        Evaluates the predicted answer against the real answer.
        
        Args:
            real_answer (str): The actual answer from the dataset.
            predicted_answer (str): The answer predicted by the model.
        
        Returns:
            int: 1 if the predicted answer is correct, 0 otherwise.
        """
        if self.dataset_name == 'gsm8k':
            if last_number(real_answer) == last_number(predicted_answer):
                return 1
            else:
                return 0

def last_number(s):
    """
    Extracts the last number from a given string.
    
    Args:
        s (str): The input string.
    
    Returns:
        int: The last number found in the string, or None if no numbers are found.
    """
    numbers = re.findall(r'\d+', s)
    return int(numbers[-1]) if numbers else None