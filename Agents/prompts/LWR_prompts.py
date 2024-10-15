class LWR_PROMPT:
     
     idea_generator_prompt = \
f'''**Objective**:
You are tasked with improving the original Lighthill-Whitham-Richards (LWR) traffic flow model using mathematical approaches. Your goal is to address the deficiencies of the LWR model through mathematical modifications, without resorting to parameter optimization, machine learning, or deep learning methods.

**Resources**:
     **1. Offline Literature**: A collection of pre-filtered LWR-related research papers and documents.
     **2. Online Search**: An online search tool for supplementary information.

**Instructions**:
Follow these steps to develop a mathematically enhanced LWR traffic flow model:

**1. Understand the LWR Traffic Flow Model**:
     - Task: Describe the fundamental principles and mathematical equations of the LWR traffic flow model.
     - Action: Begin by thoroughly reviewing the offline LWR literature or online research to gather foundational knowledge.

**2. Identify Deficiencies**:
     - Task: Identify and elaborate on the limitations and shortcomings of the original LWR model from a mathematical perspective.
     - Action: Focus on the offline literature to find any documented deficiencies. If necessary, use the Online Search Tool to find additional information.

**3. Propose Mathematical Improvements**:
     - Task: Develop a mathematically rigorous proposal to improve the LWR model, addressing the identified deficiencies.
     - Action: Synthesize information from both offline literature and online searches to formulate your proposal. Ensure your solutions are based on mathematical modeling principles.

**Format of Deliverables**:
- Explanation of the LWR Model: Provide a comprehensive description of the LWR traffic flow model, including its fundamental principles and equations.
- Analysis of Deficiencies: Detail the mathematical limitations and shortcomings of the original LWR model.
- Proposal for Improvements: Present a well-founded, mathematically rigorous proposal for improving the LWR model. Clearly explain the rationale behind each suggested modification.

**Note**:
Prioritize using the offline literature for initial research and insights. Resort to the Online Search Tool when additional information is required or if certain aspects are not sufficiently covered in the offline materials.

**Formatting**:
- Ensure clarity and conciseness in your explanations.
- Use mathematical equations and diagrams where necessary to support your proposal.
- Reference all sources used, distinguishing between offline literature and online searches.
- Outputs shold contain a detailed explanation of the LWR model, an analysis of its deficiencies, and a proposal for mathematical improvements.
'''
     
     code_generator_prompt = f'''Based on the provided improved ideas of the LWR model, you need to write a Python function to implement the improved LWR model. 
Below is the template of the "Improved_LWR" function in Python. Complete the TODO part:

def Improved_LWR(density, Vf, rho_max):
     """
     Simulate the LWR model to compute speed and density.

     :param density: numpy ndarray, initial density of vehicles.
     :param Vf: scaler, free flow speed.
     :param rho_max: scaler, maximum density.
     :return: numpy ndarray, corresponding flow speed.
     """   

     # Import necessary libraries
     import math            
     import numpy as np 
     
     ### Your code here (TODO) ###
     
     return simulated_speed

                    
Important:
     1. The algorithm should be wrapped in ```python ... ``` in the final code snippet.
     2. Import any additional packages after the line `import numpy as np`.
     3. Provide only the Python function without example usage.
     4. Please handle the overflow in scalar operations.
     5. The input parameters are fixed and should not be modified, adding any other parameter is not allowed.
'''