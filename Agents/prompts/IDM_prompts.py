class IDM_PROMPT:
     
     idea_generator_prompt = \
f'''**Objective**:
You are tasked with improving the original Intelligent Driver Model (IDM) car-following model using mathematical approaches. Your goal is to address the deficiencies of the IDM model through mathematical modifications, without resorting to parameter optimization, machine learning, or deep learning methods.

**Resources**:
     **1. Offline Literature**: A collection of pre-filtered IDM-related research papers and documents.
     **2. Online Search**: An online search tool for supplementary information.

**Instructions**:
Follow these steps to develop a mathematically enhanced IDM car-following model:

**1. Understand the IDM Car-Following Model**:
     - Task: Describe the fundamental principles and mathematical equations of the IDM car-following model.
     - Action: Begin by thoroughly reviewing the offline IDM literature or online research to gather foundational knowledge.

**2. Identify Deficiencies**:
     - Task: Identify and elaborate on the limitations and shortcomings of the original IDM model from a mathematical perspective.
     - Action: Focus on the offline literature to find any documented deficiencies. If necessary, use the Online Search Tool to find additional information.

**3. Propose Mathematical Improvements**:
     - Task: Develop a mathematically rigorous proposal to improve the IDM model, addressing the identified deficiencies.
     - Action: Synthesize information from both offline literature and online searches to formulate your proposal. Ensure your solutions are based on mathematical modeling principles.

**Format of Deliverables**:
- Explanation of the IDM Model: Provide a comprehensive description of the IDM car-following model, including its fundamental principles and equations.
- Analysis of Deficiencies: Detail the mathematical limitations and shortcomings of the original IDM model.
- Proposal for Improvements: Present a well-founded, mathematically rigorous proposal for improving the IDM model. Clearly explain the rationale behind each suggested modification.

**Note**:
Prioritize using the offline literature for initial research and insights. Resort to the Online Search Tool when additional information is required or if certain aspects are not sufficiently covered in the offline materials.

**Formatting**:
- Ensure clarity and conciseness in your explanations.
- Use mathematical equations and diagrams where necessary to support your proposal.
- Reference all sources used, distinguishing between offline literature and online searches.
- Outputs shold contain a detailed explanation of the IDM model, an analysis of its deficiencies, and a proposal for mathematical improvements.
'''
     
     code_generator_prompt = f'''Based on the provided improved ideas of the IDM model, you need to write a Python function to implement the improved IDM model. 
Below is the template of the "Improved_IDM" function in Python. Complete the TODO part:

def Improved_IDM(params, spacing, svSpd, lvSpd):
     """
    Implement the improved IDM model.
    
    :param params: list of parameters [desired_speed, desired_time_window, max_acc, comfort_acc, beta, jam_space]
        - desired_speed (float): Desired speed of the following vehicle [m/s]
        - desired_time_window (float): Desired time headway [s]
        - max_acc (float): Maximum acceleration [m/s^2]
        - comfort_acc (float): Comfortable acceleration [m/s^2]
        - beta (float): Exponent parameter for acceleration
        - jam_space (float): Minimum gap between vehicles in traffic jam [m]

    :param spacing: scaler, gap between two vehicles [m]
    :param svSpd: scaler, speed of the following vehicle [m/s]
    :param lvSpd: scaler, speed of the lead vehicle [m/s]
    :return: scaler, acceleration of the following vehicle in the next step [m/s^2]
    """

    # Import necessary libraries here
    import numpy as np
    

    # Unpack parameters
    desiredSpd, desiredTimeHdw, maxAcc, comfortAcc, beta, jamSpace = params

    ### Your code here (TODO) ###
    # Implement the improved IDM model based on the improved ideas
    
    #############################
    return acceleration

Important:
     1. The algorithm should be wrapped in ```python ... ``` in the final code snippet.
     2. Import any additional packages inside the function definition.
     3. Try to use numpy for mathematical operations.
     4. Provide only the Python function without example usage.
     5. The input parameters are fixed and should not be modified, adding any other parameter is not allowed.
'''