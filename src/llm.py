from transformers import pipeline
from openai import OpenAI
import openai
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import fireworks.client as firew_client


class LLM:
    def __init__(self, provider='openai', model='gpt-3.5-turbo', api_key=None):
        '''
        Initialize the LLM object with the specified provider, model, and API key.

        Args: 
            provider (str): The provider of the language model. Default is 'openai'.
            model (str): The model to use for the language model. Default is 'gpt-3.5-turbo'.
            api_key (str): The API key for the provider. Default is None.

        Raises:
            ValueError: If the API key is required for the specified provider.
            NotImplementedError: If the specified provider or model is not supported.
        '''

        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.pipeline = None

        if self.provider == 'openai':
            # Initialize OpenAI API with the provided API key
            if not self.api_key:
                raise ValueError("API key is required for OpenAI provider")

            self.client = OpenAI(api_key=self.api_key)

        elif self.provider == 'fireworks':
            # Initialize OpenAI API with the provided API key
            if not self.api_key:
                raise ValueError("API key is required for Fireworks provider")

            firew_client.api_key = self.api_key

        else:
            raise NotImplementedError("The specified provider is not supported")            

    def generate(self, sys_prompt:str, instruction:str, examples:str, problem:str, **kwargs):
        '''
        Generate a response to a given problem using the specified language model and giving the system prompt, 
        instruction, and examples.

        Args:
            sys_prompt (str): The system prompt to provide context to the model.
            instruction (str): The instruction for the problem.
            examples (str): The examples to help the model understand the problem.
            problem (str): The problem to generate a response for.
            **kwargs: Additional keyword arguments to pass to the model.
        
        Returns:
            str: The generated response to the problem.

        Raises:
            NotImplementedError: If the specified provider or model is not supported.

        '''
        # Check if the provider is OpenAI and the model is supported
        if self.provider == 'openai' and self.model in ['gpt-3.5-turbo', 'gpt-4-1106-preview']:
            done = False
            while not done:
                try:
                    # Create a chat completion request to the OpenAI API
                    response = self.client.chat.completions.create(
                        model=self.model,
                        timeout=5,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": instruction + '\n' + examples + '\n' + problem}
                        ]
                    )
                    done = True
                except:
                    # Wait for 10 seconds before retrying in case of an exception
                    time.sleep(10)
            # Return the content of the response message
            return response.choices[0].message.content
        
        # Check if the provider is Fireworks and the model is supported
        elif self.provider == 'fireworks' and self.model in ['llama-v2-70b-chat', 'falcon-7b', 'mistral-7b-instruct-4k', 'llama-v2-13b-chat']:
            done = False
            while not done:
                try:
                    # Create a chat completion request to the Fireworks API
                    completion = firew_client.ChatCompletion.create(
                        model="accounts/fireworks/models/" + self.model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": instruction + '\n' + examples + '\n' + problem}
                        ],
                        n=1,
                        max_tokens=800,
                        temperature=0.1,
                        top_p=0.9,
                    )
                    done = True
                except:
                    # Wait for 10 seconds before retrying in case of an exception
                    time.sleep(10)
            # Return the content of the response message
            return completion.choices[0].message.content        

        else:
            # Raise an error if the provider or model is not supported
            raise NotImplementedError("The specified provider or model is not supported")
        
    def mutate(self, mutation_prompt:str, instruction:str):
        '''
        Generate a response to a given mutation prompt using the specified language model.

        Args:
            mutation_prompt (str): The prompt to mutate.
            instruction (str): The instruction for the mutation.
        
        Returns:
            str: The generated response to the mutation prompt.
        
        Raises:
            NotImplementedError: If the specified provider is not supported.
        
        '''
        done = False

        if self.provider == 'openai' and self.model in ['gpt-3.5-turbo']:
            # Loop until a successful response is obtained
            while not done:
                try:
                    # Create a chat completion request to the OpenAI API
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": mutation_prompt + '\n' + instruction}
                        ]
                    )
                    done = True
                except:
                    # Wait for 10 seconds before retrying in case of an exception
                    time.sleep(10)
            # Return the content of the response message
            return response.choices[0].message.content

        elif self.provider == 'fireworks':
            # Loop until a successful response is obtained
            while not done:
                try:
                    # Create a chat completion request to the Fireworks API
                    completion = firew_client.ChatCompletion.create(
                        model="accounts/fireworks/models/" + self.model,
                        messages=[
                            {"role": "user", "content": mutation_prompt + '\n' + instruction}
                        ],
                        n=1,
                        max_tokens=1000,
                        temperature=0.1,
                        top_p=0.9,
                    )
                    done = True
                except:
                    # Wait for 10 seconds before retrying in case of an exception
                    time.sleep(10)
            # Return the content of the response message
            return completion.choices[0].message.content

        else:
            # Raise an error if the provider is not supported
            raise NotImplementedError("The specified provider is not supported")
