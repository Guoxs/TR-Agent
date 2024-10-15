class MOBIL_PROMPT:

     idea_generator_prompt = \
f'''**Objective**:
You are tasked with improving the original MOBIL lane-changing model using mathematical approaches. Your goal is to address the deficiencies of the MOBIL model through mathematical modifications, without resorting to parameter optimization, machine learning, or deep learning methods.

**Resources**:
     **1. Offline Literature**: A collection of pre-filtered MOBIL-related research papers and documents.
     **2. Online Search**: An online search tool for supplementary information.

**Instructions**:
Follow these steps to develop a mathematically enhanced MOBIL lane-changing model:

**1. Understand the MOBIL Lane-Changing Model**:
     - Task: Describe the fundamental principles and mathematical equations of the MOBIL lane-changing model.
     - Action: Begin by thoroughly reviewing the offline MOBIL literature or online research to gather foundational knowledge.

**2. Identify Deficiencies**:
     - Task: Identify and elaborate on the limitations and shortcomings of the original MOBIL model from a mathematical perspective.
     - Action: Focus on the offline literature to find any documented deficiencies. If necessary, use the Online Search Tool to find additional information.

**3. Propose Mathematical Improvements**:
     - Task: Develop a mathematically rigorous proposal to improve the MOBIL model, addressing the identified deficiencies.
     - Action: Synthesize information from both offline literature and online searches to formulate your proposal. Ensure your solutions are based on mathematical modeling principles.

**Format of Outputs**:
- Explanation of the MOBIL Model: Provide a comprehensive description of the MOBIL lane-changing model, including its fundamental principles and equations.
- Analysis of Deficiencies: Detail the mathematical limitations and shortcomings of the original MOBIL model.
- Proposal for Improvements: Present a well-founded, mathematically rigorous proposal for improving the MOBIL model. Clearly explain the rationale behind each suggested modification.

**Note**:
Prioritize using the offline literature for initial research and insights. Resort to the Online Search Tool when additional information is required or if certain aspects are not sufficiently covered in the offline materials.

**Formatting**:
- Ensure clarity and conciseness in your explanations.
- Use mathematical equations and diagrams where necessary to support your proposal.
- Reference all sources used, distinguishing between offline literature and online searches.
- Outputs shold contain a detailed explanation of the MOBIL model, an analysis of its deficiencies, and a proposal for mathematical improvements.
'''
     
     code_generator_prompt = f'''Based on the provided improvements to the MOBIL model, you need to write a Python function to implement the improved MOBIL model. Below is the template of the "Improved_MOBIL" function in Python. Complete the TODO section:

def Improved_MOBIL(event_data, params):
     """Improved MOBIL model for lane change decision
          event_data:  [N, 11] ndarray, event datas for test vehicles, each row contains the following data
          [
               v: speed of the following vehicle
               s: headway distance between the leading and following vehicle
               of_v: speed of the original lane front vehicle
               or_v: speed of the original lane rear vehicle
               tf_v: speed of the target lane front vehicle
               tr_v: speed of the target lane rear vehicle
               rtf_x: relative distance to the target lane front vehicle
               rtr_x: relative distance to the target lane rear vehicle
               rr_x: relative distance to the rear vehicle
               or_acc: acceleration of the original lane rear vehicle
               tr_acc: acceleration of the target lane rear vehicle
          ]
          params: Scaler list, parameters for IDM model and improved MOBIL model 
          [
               desired_speed, jam_space, desired_time_window, max_acc, comfort_acc, beta, # IDM parameters
               politeness, b_safe, acc_thres # MOBIL parameters # MOBIL parameters
          ]
        """

     import math
     import numpy as np

     def calculate_idm_acceleration(leading_v, v, s, params):
            """Calculate acceleration of the following vehicle using IDM model
                leading_v: (N,), ndarray, speed of the leading vehicles
                v: (N,), ndarray, speed of the following vehicles
                s: (N,), ndarray, headway distances between the leading and following vehicles
                params: [desired_speed, jam_space, desired_time_window, max_acc, comfort_acc, beta]
            """
            desired_speed, jam_space, desired_time_window, max_acc, comfort_acc, beta = params
                        
            delta_v = leading_v - v
            s_star = jam_space + np.maximum(0, v * desired_time_window - v * delta_v / (2 * np.sqrt(max_acc * comfort_acc)))
            acceleration = max_acc * (1 - np.power(v / (desired_speed + 1e-6), beta) - np.power(s_star / (s + 1e-6), 2))

            # handle the negative spacing
            acceleration[s <= 0] = -max_acc

            return acceleration
     
     # Extract event data and parameters
     v, s, of_v, or_v, tf_v, tr_v, rtf_x, rtr_x, rr_x, or_acc, tr_acc = event_data
     desired_speed, jam_space, desired_time_window, max_acc, comfort_acc, beta, politeness, b_safe, acc_thres = params
     idm_params = [desired_speed, jam_space, desired_time_window, max_acc, comfort_acc, beta]
     
     change_decision = np.zeros_like(v)

     ### Your code here (TODO) ###
     # Implement the improved MOBIL model based on the provided ideas and guidelines.
     # Ensure that the decision to change lanes is based on calculated safety and efficiency criteria.
     # Remember to handle any potential overflow in vectorized operations.

     #############################
     return change_decision

Remember:
1. The algorithm should be wrapped in ```python ... ``` in the final code snippet.
2. Import any additional packages after the line `import numpy as np`.
3. Use the vectorized operations for efficient computation.
4. Provide only the Python function without example usage.
5. Handle any potential overflow in vectorized operations.
6. The input parameters are fixed and should not be modified; adding any other parameter is not allowed.
'''