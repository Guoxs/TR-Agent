import sys
import numpy as np
from functools import partial
from geneticalgorithm import geneticalgorithm as ga


sys.path.append("..") 
from utils import get_dataset

def callback(xk, convergence):
    print(f"Current solution: {xk}, Convergence: {convergence}")

def car_following_simulation(model_func, time_interval, params, lvSpd_obs, svSpd_init, spacing_init):
    n = len(lvSpd_obs)
    svSpd_sim = np.zeros(n)
    spacing_sim = np.zeros(n)

    svSpd_sim[0] = svSpd_init
    spacing_sim[0] = spacing_init

    svSpd, spacing = svSpd_init, spacing_init
    relSpd = lvSpd_obs[0] - svSpd_init

    for i in range(1, n):
        # calculate next_step acceleration using IDM model
        acc = model_func(params, spacing, svSpd, lvSpd_obs[i-1])
        # state update based on Newton's motion law
        svSpd_ = max(0.001, svSpd + acc * time_interval)  # next step svSpd
        relSpd_ = lvSpd_obs[i] - svSpd_
        spacing_ = spacing + time_interval * (relSpd_ + relSpd) / 2

        # update state variables
        svSpd = svSpd_
        relSpd = relSpd_
        spacing = spacing_

        # store simulation results
        svSpd_sim[i] = svSpd
        spacing_sim[i] = spacing

    return svSpd_sim, spacing_sim


class IDMEvaluator:
    def __init__(self, config):
        self.config = config
        self.time_interval = config['IDM']['time_interval']

        idm_dataset = get_dataset(config)
        print(idm_dataset)
        self.trainset = idm_dataset.trainset
        self.testset = idm_dataset.testset

    def evaluate(self, model_func):
        np.random.seed(self.config['seed'])
        train_len = len(self.trainset)
        sample_idx = np.random.choice(range(train_len), self.config['IDM']['calib_data_len'], replace=False)
        
        if isinstance(self.trainset, np.ndarray):
            calib_data = self.trainset[sample_idx]
            print(f'Calibration data shape: {calib_data.shape}')
        else:
            calib_data = [self.trainset[i] for i in sample_idx]
            print(f'Calibration data shape: {len(calib_data)}')

        # model calibration
        best_param = self.model_calibration(model_func=model_func, calib_data=calib_data)
        # model evaluation
        loss = self.compute_loss(model_func=model_func, data=self.testset, params=best_param)

        return loss

    def evaluate_baseline(self):
        return self.evaluate(self.baseline_model)
    

    def baseline_model(self, params, spacing, svSpd, lvSpd):
        '''Calculate acceleration of the following vehicle using IDM model
            spacing: scaler, headway distance between the leading and following vehicle
            svSpd: scaler, speed of the following vehicle
            lvSpd: scaler, speed of the leading vehicle  
            params: [desired_speed, desired_time_window, max_acc, comfort_acc, beta, jam_space]
            
            return acc: scaler, acceleration of the following vehicle
        '''
        desiredSpd, desiredTimeHdw, maxAcc, comfortAcc, beta, jamSpace = params
        relSpd = lvSpd - svSpd
        desiredSpacing = jamSpace + np.maximum(0, svSpd * desiredTimeHdw - (svSpd * relSpd) / (2 * np.sqrt(maxAcc * comfortAcc)))
        acc = maxAcc * (1 - svSpd / (desiredSpd + 1e-6) ** beta - desiredSpacing / (spacing + 1e-6) ** 2)
        # handle the negative spacing
        if spacing < 0:
            acc = -maxAcc
        return acc

    def compute_loss(self, model_func, data, params, if_calib=False):
        total_loss = 0
        event_count = np.zeros(4)  # for spacing_with_state, count the number of events for each state

        eval_metric = self.config['IDM']['eval_metric']
        if if_calib and eval_metric == 'spacing_with_state':
            eval_metric = 'spacing'

        for event in data:
            spacing, svSpd, lvSpd = event[:, 0], event[:, 1], event[:, 2]
            spacing_init, svSpd_init = event[0, 0], event[0, 1]

            svSpd_sim, spacing_sim = car_following_simulation(
                model_func=model_func,
                time_interval=self.time_interval,
                params=params,
                lvSpd_obs=lvSpd,
                svSpd_init=svSpd_init,
                spacing_init=spacing_init
            )

            if eval_metric == 'all':
                loss = np.mean(np.abs(svSpd_sim - svSpd[:len(svSpd_sim)])) + \
                    np.mean(np.abs(spacing_sim - spacing[:len(spacing_sim)]))
            elif eval_metric == 'speed':
                loss = np.mean(np.abs(svSpd_sim - svSpd[:len(svSpd_sim)]))
            elif eval_metric == 'spacing':
                loss = np.mean(np.abs(spacing_sim - spacing[:len(spacing_sim)]))
            elif eval_metric == 'spacing_with_state':
                loss_array = np.abs(spacing_sim - spacing[:len(spacing_sim)])
                state = event[:, -1].astype(int)
                loss = np.array([np.sum(loss_array[state == i]) for i in range(4)])
                event_count += np.array([np.sum(state == i) for i in range(4)])
            else:
                print(f'Error!! Metric should be one of the following: [speed, spacing, all, spacing_with_state]')
                break

            total_loss += loss

        if eval_metric == 'spacing_with_state':
            total_loss /= np.maximum(event_count, 1)  # Avoid division by zero
            return np.mean(total_loss), total_loss
        else:
            return total_loss / len(data)

    def model_calibration(self, model_func, calib_data):
        np.random.seed(self.config['seed'])

        fitness_func = partial(self.compute_loss, model_func, calib_data, if_calib=True)

        algorithm_param = {
            'max_num_iteration': 50, 
            'population_size': 50,
            'mutation_probability': 0.2, 
            'elit_ratio': 0.01,
            'crossover_probability': 0.3, 
            'parents_portion': 0.3,
            'crossover_type': 'uniform', 
            'max_iteration_without_improv': 20
        }
        
        var_bound = np.array([[1., 42], [0.1, 5.], [0.1, 5.], [0.1, 5.], [1., 10.], [0.1, 10.]])

        model = ga( 
            function=fitness_func, 
            dimension=len(var_bound), 
            variable_type='real', 
            function_timeout=10,
            variable_boundaries=var_bound, 
            algorithm_parameters=algorithm_param, 
            convergence_curve=False
            )

        try:
            model.run()
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
            model.best_variable = None
        return model.best_variable
        

if __name__ == "__main__":
    from config import configs
    import time

    evaluator = IDMEvaluator(config=configs)

    start_time = time.time()
    loss_base = evaluator.evaluate_baseline() 
    print(f"Time elapsed: {time.time() - start_time:.6f} seconds.")

    print(loss_base)
    if type(loss_base) == tuple:
        print(f'''Evaluation loss_base [ total: {loss_base[0]:.4f}, free driving: {loss_base[1][0]:.4f},  \
following: {loss_base[1][1]:.4f},  closing in: {loss_base[1][2]:.4f},  emergency braking: {loss_base[1][3]:.4f}].''')
    else:
        print(f'Evaluation loss_base: {loss_base:.4f}.')