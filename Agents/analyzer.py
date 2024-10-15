import re
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import get_llm


class Analyzer:
    def __init__(self, config, logger_filename='', **kwargs):
        self.config = config
        self.logger_filename = logger_filename

        self.llm = get_llm(self.config)
        
        self.output_parser = StrOutputParser()

        system_prompt = f'''You are an experiment analyst. You have been given a log file of a series of trials. \
Please carefully analyze the trial log. If the results of the last trial did not meet the performance requirements, \
analyze why and provide suggestions for further improvement. If the performance requirements were met, \
analyze the reasons for the improvement and document the successful factors.'''
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm | self.output_parser
        

    def analyse_success(self):
        ### load log file
        base_model_log, iterations, _ = self.split_log()
        last_iteration = iterations[-1]
        
        ### analyse log file
        user_prompt = \
f'''The last trial has successfully met the performance requirements. Below are the details for analysis:

Baseline Model Information: 
{base_model_log}

Last Iteration Experiment Log: 
{last_iteration}

Analysis Requirements:
Provide a specific and concrete analysis, based on the provided information, \
outlining the reasons for the observed improvement. Limit your analysis to 200 words.
'''

        outputs = self.chain.invoke({"input": user_prompt})
        return outputs
    

    def analyse_failure(self):
        ### load log file
        base_model_log, iterations, improved_idx = self.split_log()
        # improved_iterations = [iterations[i] for i in improved_idx]
        last_three_iterations = iterations[-3:] if len(iterations) >= 3 else iterations
        ### analyse log file
        user_prompt = f'''The last trial did not meet the performance requirement. Here are the details for analysis:

Baseline Model Information: 
{base_model_log}

Experiment Log of the last {len(last_three_iterations)} Iterations: 
{''.join(last_three_iterations)}

Please analyze the reasons for the low performance and provide suggestions for further improvement. \
Remember the following guidelines:
     1. Provide suggestions without rejecting any possibilities.
     2. Avoid suggestions involving deep learning or machine learning technologies; focus on concrete mathematical formulations.
     3. Do not provide suggestions involving parameter optimization or similar techniques.
     4. Ensure suggestions are specific and grounded in the analysis of the log file.

Limit your analysis tn 300 words.
'''
        outputs = self.chain.invoke({"input": user_prompt})

        new_question_prompt = f'''Based on the low performance analysis, please identify additional areas we need to explore to improve the model. \
Formulate your response as a list of questions (no more than 5), each targeting specific aspects for the next iteration of improvements.

Focus on:
    - Further understanding the internal mechanisms of the mathematical formulations.
    - Investigating current directions of improvements pursued by others.

Avoid:
    - Suggestions involving deep learning or machine learning technologies.
    - Queations related to datasets.
    - Hyperparameter tuning.
    - Training and evaluation processes.
'''
        new_questions = self.chain.invoke({"input": new_question_prompt})

        return outputs, new_questions


    def split_log(self):
        with open(self.logger_filename, 'r') as f:
            log = f.read()
        
        split_str = f'research_agent_logger - INFO - ============ Iteration'
        parts = log.split(split_str)
        
        base_model_log = parts[0]
        # Remove the first two lines
        base_model_log_lines = base_model_log.splitlines(True)
        base_model_log = ''.join(base_model_log_lines[2:-1])
        
        iterations = []
        improved_idx = []
        for idx in range(1, len(parts)):
            part = parts[idx]
            if '- INFO - Code generation failed, skip this iteration' in part:
                continue
            part = split_str + part
            
            if idx < len(parts) - 1:
                # remove last line
                part_lines = part.splitlines(True)
                part = ''.join(part_lines[:-1])

            iterations.append(part)

            improve_rate = re.findall(r'research_agent_logger - INFO - .*, improved rate: (.*)%', part)[0]
            if float(improve_rate) > 0:
                improved_idx.append(1)
            else:
                improved_idx.append(0)
        
        improved_idx = np.where(np.array(improved_idx) == 1)[0]  

        return base_model_log, iterations, improved_idx
    