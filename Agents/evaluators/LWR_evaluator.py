import sys
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from functools import partial

sys.path.append("..") 
from utils import get_dataset

class LWREvaluator:
    def __init__(self, config):
        self.config = config

        lwr_dataset = get_dataset(config=self.config)

        self.trainset = {'input': lwr_dataset.density_train, 'label': lwr_dataset.speed_train}
        self.valset = {'input': lwr_dataset.density_val, 'label': lwr_dataset.speed_val}

        self.fixed_speed = self.config.get('fixed_speed', 60) 

    def evaluate(self, model_func):
        best_param = self.model_calibration(model_func=model_func)
        loss = self.compute_loss(model_func=model_func, dataset=self.valset, params=best_param)
        return loss

    def evaluate_baseline(self):
        return self.evaluate(self.baseline_model)

    def baseline_model(self, density, Vf, rho_max):
        """
        Simulate the LWR model to compute speed given density.
        
        :param density: initial density of vehicles.
        :param Vf: free flow speed.
        :param rho_max: maximum density.
        :return: simulated flow speed.
        """
        simulated_speed = Vf * (1 - (density / (rho_max + 1e-6)))
        return simulated_speed

    def compute_loss(self, model_func, dataset, params, if_calib=False):
        Vf, rho_max = params
        Vf *= self.fixed_speed
        rho_max *= dataset['input'].max()
        
        speed_pred = model_func(dataset['input'], Vf, rho_max)
        speed_gt = dataset['label']

        loss_list = np.abs(speed_gt - speed_pred)
        total_loss = np.mean(loss_list)
        if if_calib:
            return total_loss
        else:
            # 0-0.3 low density, 0.3-0.6 medium density, 0.6-1 high density
            low_density_loss = np.mean(loss_list[dataset['input'] < 0.3])
            medium_density_loss = np.mean(loss_list[(dataset['input'] >= 0.3) & (dataset['input'] < 0.6)])
            high_density_loss = np.mean(loss_list[dataset['input'] >= 0.6])
            
            return total_loss, [low_density_loss, medium_density_loss, high_density_loss]

    
    def model_calibration(self, model_func):
        fitness_func = partial(self.compute_loss, model_func, self.trainset, if_calib=True)
        var_bound = np.array([[0, 150], [0, 1]])
        np.random.seed(self.config['seed'])
        
        algorithm_param = {
            'max_num_iteration': 200,
            'population_size': 150,
            'mutation_probability': 0.2,
            'elit_ratio': 0.01,
            'crossover_probability': 0.3,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': 100
        }

        model = ga(function=fitness_func, dimension=len(var_bound), variable_type='real',
                   variable_boundaries=var_bound, algorithm_parameters=algorithm_param, convergence_curve=False)
        model.run()

        return model.output_dict['variable']


if __name__ == "__main__":
    from config import configs
    evaluator = LWREvaluator(config=configs)
    total_loss_base, density_loss_base = evaluator.evaluate_baseline()
    
    print(f'Baseline model total loss: {total_loss_base:.4f}. Loss at different density levels: [low (0~0.3): {density_loss_base[0]:.4f}, medium (0.3~0.6): {density_loss_base[1]:.4f}, high (0.6~1.0): {density_loss_base[2]:.4f}]')