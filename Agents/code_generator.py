import re
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import get_dataset, get_prompt, get_llm


class CodeGenerator:
    def __init__(self, config):
        self.config = config

        self.task_name = self.config['task_name']
        self.llm = get_llm(self.config)
        
        self.dataset = get_dataset(self.config)
        self.prompts = get_prompt(self.task_name)

        self.output_parser = StrOutputParser()

        code_system_prompt = f'''You are a proficient senior engineer in the field of intelligent transportation, \
capable of writing high-quality, maintainable Python code based on professional ideas or pseudo-code. \
Using your expertise, implement the refined ideas provided. Ensure the code is well-documented, efficient, and handles edge cases appropriately.'''
        
        self.code_prompt = ChatPromptTemplate.from_messages([
            ("system", code_system_prompt),
            ("user", "{input}")
        ])

        self.code_chain = self.code_prompt | self.llm | self.output_parser

        self.test_data = self.dataset.get_test_sampler()
    

    def get_code_refinement_prompt(self, code_snippet, err_message):
        prompt = \
f'''Source code: 
{code_snippet}

Traceback error message for this code: 
{err_message}

Please use the error message to debug and improve source code. 

Important:
    1. The algorithm should be wrapped in ```python ... ``` in the final code snippet.
    2. If you want to import any Python package, please import it inside the function definition.
    3. Ensure the function is self-contained and includes necessary parameter explanations.
    4. Only provide the Python function, and do not include example usage.
'''
        return prompt


    def generate_code(self, prof_advises):
        algorithm_code = self.code_chain.invoke(
            {"input": f'''Professional ideas: {prof_advises}. \n {self.prompts.code_generator_prompt}'''})
        
        code_snippet = re.findall(f"```python\n(.*)\n```", algorithm_code, re.DOTALL)[0]
        func_closure, err_message = self.create_closure_from_code(code_snippet)

        try_time = 0
        while err_message is not None and try_time < self.config['code_gen_try_times']:
            print('########## Iteration:', try_time, '##########')
            
            code_refinement_prompt = self.get_code_refinement_prompt(code_snippet, err_message)
            code_str = self.code_chain.invoke({"input": code_refinement_prompt})
            
            code_snippet = re.findall(f"```python\n(.*)\n```", code_str, re.DOTALL)[0]
            func_closure, err_message = self.create_closure_from_code(code_snippet)
            try_time += 1

        if err_message is not None:
            print("Code generation failed: ", err_message)
            func_closure = None
            
        return code_snippet, func_closure

    def create_closure_from_code(self, func_code, **kwargs):
        func_closure = None 
        message = None
        print("Received function code:\n", func_code)
        function_name = re.findall(r'def (.*)\(', func_code)[0]
        try:
            exec(func_code, globals(), locals())
            func_closure = locals()[function_name]
        except Exception as e:
            message = traceback.format_exc()
            print("\n Function exec error: ", message)

        if func_closure is not None:
            try:
                if self.task_name == 'IDM':
                    spacing, svSpd, lvSpd = self.test_data[0, 0], self.test_data[0, 1], self.test_data[0, 2]
                    params = [17.386, 5.0, 0.742, 0.1, 1.082, 0.1]
                    acc = func_closure(params, spacing, svSpd, lvSpd)
                    print(f"IDM test acceleration: {acc}")
                    
                    # check if acc is valid
                    if acc < -10 or acc > 10:
                        message = f"Invalid acceleration value: {acc}. The accelerate should be with in [-10, 10]. Please check the function implementation."
                    
                elif self.task_name == 'MOBIL':
                    default_params = [30.78, 0.166, 0.143, 0.174, 0.924, 4.55, 0.055, 2.27, 0.108]
                    lane_change_decision = func_closure(self.test_data, default_params)
                    print(f"MOBIL test lane change decision: {lane_change_decision}")
                    
                    # check if lane_change_decision is valid
                    if any(decision not in [0, 1] for decision in lane_change_decision):
                        message = f"Invalid lane change decision: {lane_change_decision}. \
                        The decision should be either 0 or 1. Please check the function implementation."
                    
                elif self.task_name == 'LWR':
                    density_val = self.test_data
                    simulated_speed = func_closure(density_val, Vf=60, rho_max=100)
                    print(f"LWR test speed: {simulated_speed[:5]}")
                    
                    # check if simulated_speed is valid, speed should be in range [0, 500]
                    if any(speed < 0 or speed > 500 for speed in simulated_speed):
                        message = f"Invalid speed values: {simulated_speed}. The speed should be in range [0, 500]. \
                        Please check the function implementation."
    
                else:
                    raise ValueError(f"Invalid task name: {self.task_name}")
            except Exception as e:
                message = traceback.format_exc()
                print("\n Function test error:", message)

        return func_closure, message


if __name__ == "__main__":
    import os
    from config import configs
    os.environ["OPENAI_API_KEY"] = configs['openai_api_key']


    code_generator = CodeGenerator(config=configs)
    prof_adverses = f'''1. The IDM model is a car-following model used to simulate traffic flow and congestion dynamics, particularly in congested traffic states.\
2. The limitations of the IDM model include the potential for specific vehicle velocities to become negative or diverge to negative infinity in finite time. \
3. To improve the IDM model, modifications are suggested to prevent ill-posedness, such as adjusting acceleration to avoid negative velocities and ensuring well-posedness by implementing straightforward improvements.'''
    
    code_snippet, func_closure = code_generator.generate_code(prof_adverses)
    print(f"Code snippet: \n {code_snippet}")