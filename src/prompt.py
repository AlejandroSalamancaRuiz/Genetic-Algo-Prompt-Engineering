from src.experiment_conditions import Experiment_conditions
import numpy as np


class Prompt:

    def __init__(self, experiment_conditions: Experiment_conditions, sys_prompt_idx: int, 
                 instruction_idx: int, num_examples: int, binary_encoding=[]) -> None:
        """
        Initialize the Prompt object. If binary_encoding is provided, decode it to get the prompt parameters.
        Otherwise, use the provided sys_prompt_idx, instruction_idx, and num_examples.
        """
        self.experiment_conditions = experiment_conditions
        
        if len(binary_encoding) > 0:
            self.num_examples, self.sys_prompt_idx, self.instruction_idx = self.decode_prompt(binary_encoding)
        else:
            self.sys_prompt_idx = sys_prompt_idx
            self.instruction_idx = instruction_idx
            self.num_examples = num_examples
            
        self.encoded_prompt = self.encode_prompt()

    def encode_prompt(self):
        """
        Encode the prompt parameters (num_examples, sys_prompt_idx, instruction_idx) into a binary format.
        """
        num_examples_encoded = encode_to_binary(value=self.num_examples, bits_to_use=10)
        sys_instruction_encoded = encode_to_binary(value=self.sys_prompt_idx, bits_to_use=10)
        instruction_encoded = encode_to_binary(value=self.instruction_idx, bits_to_use=10)

        return np.concatenate((num_examples_encoded, sys_instruction_encoded, instruction_encoded))
    
    def decode_prompt(self, encoded_array):
        """
        Decode the binary encoded prompt parameters back into their original values.
        """
        # Split the array back into its original components
        num_examples_encoded = encoded_array[:10]
        sys_instruction_encoded = encoded_array[10:20]
        instruction_encoded = encoded_array[20:]

        # Decode 
        num_examples = decode_from_binary(num_examples_encoded)
        sys_instruction_index = decode_from_binary(sys_instruction_encoded)
        instruction_index = decode_from_binary(instruction_encoded)

        # Ensure the decoded values are within valid ranges
        num_examples = self.experiment_conditions.max_num_examples if num_examples > self.experiment_conditions.max_num_examples else num_examples
        sys_instruction_index = len(self.experiment_conditions.sys_prompts) - 1 if sys_instruction_index >= len(self.experiment_conditions.sys_prompts) else sys_instruction_index
        instruction_index = len(self.experiment_conditions.instructions) - 1 if instruction_index >= len(self.experiment_conditions.instructions) else instruction_index

        return num_examples, sys_instruction_index, instruction_index
    
    def evaluate(self):
        """
        Evaluate the fitness of the prompt by generating answers to a set of problems and comparing them to the correct answers.
        """
        fitness = 0

        problems, answers = self.experiment_conditions.dataset.pick_random_problems(10)

        for i in range(len(problems)):
            examples = ''

            for j in range(self.num_examples):
                examples += self.experiment_conditions.examples[j] + '\n'

            sys_prompt = self.experiment_conditions.sys_prompts[self.sys_prompt_idx]
            instruction = self.experiment_conditions.instructions[self.instruction_idx]
            
            predicted_ans = self.experiment_conditions.llm.generate(sys_prompt, instruction, examples, problems[i])
            fitness += self.experiment_conditions.dataset.evaluate(answers[i], predicted_ans)
        
        fitness /= len(problems)

        if fitness == 0:
            fitness += 0.001

        return fitness 

def calculate_bits(min_value, max_value):
    """
    Calculate the number of bits needed to represent a range of values.
    """
    range_size = max_value - min_value
    bits = int(np.floor(np.log2(range_size)) + 1)
    return bits

def encode_to_binary(value, bits_to_use):
    """
    Encode an integer value into a binary format using a specified number of bits.
    """
    # Format the normalized value as a binary string
    binary_string = format(value, f'0{bits_to_use}b')
    
    # Convert the binary string to a numpy array of integers 
    return np.array(list(binary_string), dtype=int)

def decode_from_binary(binary_array):
    """
    Decode a binary encoded numpy array back into an integer value.
    """
    # Check if the array is a numpy array
    if not isinstance(binary_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Convert the numpy array of 0s and 1s to a string and then to an integer
    binary_string = ''.join(binary_array.astype(str))
    
    # Convert the binary string to an integer
    value = int(binary_string, 2)
     
    return value