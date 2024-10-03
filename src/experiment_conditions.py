from src.dataset import Dataset
from src.llm import LLM

class Experiment_conditions:
    """
    A class to represent the conditions for an experiment.
    """

    def __init__(self, dataset: Dataset, prompt_mutators: list[str], sys_prompts: list[str], 
                 instructions: list[str], llm: LLM, max_num_examples: int) -> None:
        """
        Initialize the Experiment_conditions with the given parameters.

        :param dataset: An instance of the Dataset class.
        :param prompt_mutators: A list of strings representing prompt mutators.
        :param sys_prompts: A list of system prompts.
        :param instructions: A list of instructions.
        :param llm: An instance of the LLM class.
        :param max_num_examples: The maximum number of examples to pick from the dataset.
        """
        
        self.dataset = dataset  # Store the dataset instance
        self.prompt_mutators = prompt_mutators  # Store the list of prompt mutators
        self.sys_prompts = sys_prompts  # Store the list of system prompts
        self.instructions = instructions  # Store the list of instructions
        self.llm = llm  # Store the LLM instance
        self.max_num_examples = max_num_examples  # Store the maximum number of examples

        # Pick random examples from the dataset up to the maximum number specified
        self.examples = self.dataset.pick_random_examples(max_num_examples)


