import os
import inspect

from idea_generator import IdeaGenerator
from code_generator import CodeGenerator
from analyzer import Analyzer
from config import configs
from utils import setup_logger, get_prompt
from evaluators import IDM_evaluator, MOBIL_evaluator, LWR_evaluator

os.environ["OPENAI_API_KEY"] = configs['openai_api_key']
os.environ['TAVILY_API_KEY'] = configs['tavily_api_key']

from utils import setup_logger


def get_evaluator(config):
    task_name = config['task_name']
    if task_name == 'IDM':
        return IDM_evaluator.IDMEvaluator(config)
    elif task_name == 'MOBIL':
        return MOBIL_evaluator.MOBILEvaluator(config)
    elif task_name == 'LWR':
        return LWR_evaluator.LWREvaluator(config)
    else:
        raise ValueError('Invalid task name')


def get_refined_idea_generator_prompt(previous_ideas, improve_advice, new_questions):
    prompts = \
f'''Your proposed ideas to improve the model in the last iteration are as follows: 
{previous_ideas} 

These ideas were experimentally confirmed not to achieve the expected performance improvements. \
Here are some further improvement suggestions from the Analyzer based on the last iteration:
{improve_advice} 

The Analyzer also raised some new questions for you to consider in the next iteration: 
{new_questions} 

Please refine your proposed ideas to further improve the model's performance. \
Use the tools provided and your own knowledge to develop more effective solutions.'''
    
    return prompts


class ResearchAgent:
    def __init__(self, config, logger=None, logger_filename='', **kwargs):
        self.config = config
        self.logger = logger
        self.logger_filename = logger_filename

        self.task_name = self.config['task_name']
        self.prompts = get_prompt(self.task_name)

        self.idea_generator = IdeaGenerator(config=self.config)
        self.code_generator = CodeGenerator(config=self.config)
        self.evaluator = get_evaluator(config=self.config)
        self.analyzer = Analyzer(config=self.config, logger_filename=self.logger_filename)

    def run(self):

        self.logger.info(f'''Configurations: \n {self.config}''')

        self.logger.info(f'''Baseline model defination: \n {inspect.getsource(self.evaluator.baseline_model)}''')

        base_model_metric = self.evaluator.evaluate_baseline()
        if self.config['task_name'] == 'IDM':
            if type(base_model_metric) == tuple:
                base_model_metric, state_loss = base_model_metric
                self.logger.info(f'''Base model total loss: {base_model_metric:.3f}''')
                self.logger.info(f'''Base model loss for each driving scenarios: [free driving: {state_loss[0]:.4f}, following: {state_loss[1]:.4f}, closing in: {state_loss[2]:.4f},  emergency braking: {state_loss[3]:.4f}].''')
            else:
             self.logger.info(f'''Base model loss: {base_model_metric:.3f}''')
        elif self.config['task_name'] == 'MOBIL':
            base_model_metric, eval_results, conf_results = base_model_metric
            self.logger.info(f'''Base model loss: {base_model_metric:.3f}''')
            self.logger.info(f'''Base model evaluation results: [{', '.join([f"{k}: {v:.3f}" for k, v in eval_results.items()])}], ''')
            self.logger.info(f'''Base model confusion matrix: [{', '.join([f"{k}: {v}" for k, v in conf_results.items()])}]''')
        elif self.config['task_name'] == 'LWR':
            base_model_metric, density_loss = base_model_metric
            self.logger.info(f'''Base model loss: {base_model_metric:.4f}''')
            self.logger.info(f'''Base model loss for different density levels: [low (0~0.3): {density_loss[0]:.4f}, medium (0.3~0.6): {density_loss[1]:.4f}, high (0.6~1.0): {density_loss[2]:.4f}]''')
        else:
            raise ValueError('Invalid task name')

        best_model = None
        improve_rate_list = []
        improve_advice = None
        new_questions = None
        idea = None
        
        for i in range(self.config['max_iter']):
            self.logger.info(f"============ Iteration {i} ============")
            
            if i == 0 or improve_advice is None:
                idea_prompts = self.prompts.idea_generator_prompt
            else:
                idea_prompts = get_refined_idea_generator_prompt(idea, improve_advice, new_questions)
            
            idea = self.idea_generator.run(idea_prompts)
            self.logger.info(f'''Idea: \n {idea}''')

            code_prompts = f'''Advice from last iteration: \n {improve_advice}. \n New idea: {idea}.'''
            improved_alg, model_func = self.code_generator.generate_code(code_prompts)
            
            if model_func is None:
                self.logger.info("Code generation failed, skip this iteration.")
                continue
            
            self.logger.info(f'''Code: \n {improved_alg}''')
            
            model_metric = self.evaluator.evaluate(model_func)

            if self.config['task_name'] == 'IDM':
                if type(model_metric) == tuple:
                    totol_loss, state_loss = model_metric
                    self.logger.info(f'''Model total loss: {totol_loss:.3f}''')
                    self.logger.info(f'''Model Loss for each driving scenarios: [free driving: {state_loss[0]:.4f},  following: {state_loss[1]:.4f}, closing in: {state_loss[2]:.4f},  emergency braking: {state_loss[3]:.4f}].''')
                    improve_rate = (base_model_metric - totol_loss) / (base_model_metric + 1e-6)
                    improve_rate_list.append(round(100 * improve_rate, 2))
                    result_message = f'''Baseline model loss: {base_model_metric:.3f}, improved model loss: {totol_loss:.3f}, improved rate: {100 * improve_rate:.2f}%. '''
                    
                    state_map = {0: 'free driving', 1: 'following', 2: 'closing in', 3: 'emergency braking'}
                    state_loss_list = state_loss.tolist()
                    max_state_loss_idx = state_loss_list.index(max(state_loss_list))
                    result_message += f'''Throughout the driving event, the model performs worst in the {state_map[max_state_loss_idx]} phase.'''
                    self.logger.info(result_message)
                else:
                    self.logger.info(f'''Model loss: {model_metric:.3f}''')
                    improve_rate = (base_model_metric - model_metric) / (base_model_metric + 1e-6)
                    improve_rate_list.append(round(100 * improve_rate, 2))
                    result_message = f'''Baseline model loss: {base_model_metric:.3f}, improved model loss: {model_metric:.3f}, improved rate: {100 * improve_rate: .2f}%.'''
                    self.logger.info(result_message)
            
            elif self.config['task_name'] == 'MOBIL':
                model_metric, eval_results, conf_results = model_metric
                self.logger.info(f'''Model loss: {model_metric:.3f}''')
                improve_rate = (base_model_metric - model_metric) / (base_model_metric + 1e-6)
                improve_rate_list.append(round(100 * improve_rate, 2))
                result_message = f'''Model evaluation results: [{', '.join([f"{k}: {v:.3f}" for k, v in eval_results.items()])}], '''
                result_message += f'''Model confusion matrix: [{', '.join([f"{k}: {v}" for k, v in conf_results.items()])}]'''
                self.logger.info(result_message)
                self.logger.info(f'''Baseline model loss: {base_model_metric:.3f}, improved model loss: {model_metric:.3f}, improved rate: {100 * improve_rate: .2f}%.''')
            
            elif self.config['task_name'] == 'LWR':
                model_metric, density_loss = model_metric
                self.logger.info(f'''Model loss: {model_metric:.3f}''')
                improve_rate = (base_model_metric - model_metric) / (base_model_metric + 1e-6)
                improve_rate_list.append(round(100 * improve_rate, 2))
                result_message = f'''Loss for different density levels: [low (0~0.3): {density_loss[0]:.4f}, medium (0.3~0.6): {density_loss[1]:.4f}, high (0.6~1.0): {density_loss[2]:.4f}]. '''
                result_message += f'''Based on the results, the model performs worst in the {["low", "medium", "high"][density_loss.index(max(density_loss))]} density scenario.'''
                self.logger.info(result_message)
                self.logger.info(f'''Baseline model loss: {base_model_metric:.3f}, improved model loss: {model_metric:.3f}, improved rate: {100 * improve_rate:.2f}%.''')
            else:
                raise ValueError('Invalid task name')

            if improve_rate > self.config['improve_rate']:
                self.logger.info("Improved model found!")
                best_model = improved_alg
                success_factors = self.analyzer.analyse_success()
                self.logger.info(f"Success factors: \n {success_factors}") 
                break

            improve_advice, new_questions = self.analyzer.analyse_failure()
            self.logger.info(f"Improve advice: \n {improve_advice}")
            self.logger.info(f"New questions: \n {new_questions}")
        
        self.logger.info(f'best_algorithm: \n {best_model}')
        self.logger.info(f'improve rate list (%): {improve_rate_list}')
        return best_model
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Research Agent')
    parser.add_argument('--task_name', type=str, default='MOBIL', help='Task name')

    args = parser.parse_args()
    configs['task_name'] = args.task_name

    print(f"Task name: {configs['task_name']}")

    logger, logger_filename = setup_logger('research_agent_logger', configs['task_name'], configs['llm_model'])
    agent = ResearchAgent(config=configs, logger=logger, logger_filename=logger_filename)
    agent.run()
