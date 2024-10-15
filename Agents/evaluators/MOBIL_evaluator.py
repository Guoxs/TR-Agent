import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from functools import partial
from geneticalgorithm import geneticalgorithm as ga

sys.path.append("..")
from utils import get_dataset

class MOBILEvaluator:
    def __init__(self, config):
        self.config = config

        mobil_dataset = get_dataset(config=self.config)
        self.test_dataset = mobil_dataset.all_dataset
        self.calib_dataset = mobil_dataset.calib_dataset

    def evaluate(self, model_func):
        # model calibration
        best_param = self.model_calibration(model_func=model_func)
        # model evaluation
        loss = self.compute_loss(model_func=model_func, eval_dataset=self.test_dataset, params=best_param)
        return loss

    def evaluate_baseline(self):
        return self.evaluate(self.baseline_model)
    

    def model_calibration(self, model_func):
        fitness_func = partial(self.compute_loss, model_func, self.calib_dataset, if_calib=True)
        
        algorithm_param = {'max_num_iteration': 100, 'population_size': 100,
                        'mutation_probability': 0.1, 'elit_ratio': 0.01,
                        'crossover_probability': 0.5, 'parents_portion': 0.3,
                        'crossover_type': 'uniform', 'max_iteration_without_improv': 100}
        
        var_bound = np.array([[1., 42.], [0.1, 10.], [0.1, 5.], [0.1, 5.], [0.1, 5.], [1., 10.], [0.05, 1.0], [0.5, 5.0], [0.1, 1.0]])

        print("Model calibration started...")
        model = ga(function=fitness_func, dimension=len(var_bound), variable_type='real',
                variable_boundaries=var_bound, algorithm_parameters=algorithm_param, convergence_curve=False)
        model.run()
        return model.best_variable
    
    def compute_loss(self, model_func, eval_dataset, params, if_calib=False):
        labels = eval_dataset['lane_change_event']
        preds = np.zeros_like(labels)

        valid_column_names = ['vehicle_speed', 'headway_distance', 
                          'original_lane_front_vehicle_speed', 'original_lane_rear_vehicle_speed', 
                          'target_lane_front_vehicle_speed', 'target_lane_rear_vehicle_speed', 
                          'relative_distance_target_lane_front_vehicle_x', 'relative_distance_target_lane_rear_vehicle_x', 
                          'relative_distance_rear_vehicle_x', 
                          'original_lane_rear_vehicle_acceleration', 'target_lane_rear_vehicle_acceleration']
        
        event_data = eval_dataset[valid_column_names].values
        preds = model_func(event_data, params)

        if if_calib:
            return 1 - f1_score(labels, preds, zero_division=1)
        else:
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            # accuracy = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, zero_division=1)
            recall = recall_score(labels, preds, zero_division=1)
            f1 = f1_score(labels, preds, zero_division=1)
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

            loss = 1 - f1
            eval_results = {"Precision": precision, "Recall": recall, "F1": f1, "Specificity": specificity}
            cm_results = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
            return loss, eval_results, cm_results


    def baseline_model(self, event_data, params):
        ''' MOBIL baseline model for lane change decision
            event_data:  [N, 12] ndarray, event datas for test vehicles, each row contains the following data
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
            params: Scaler list, parameters for IDM model and MOBIL model 
                [
                    jam_space, desired_time_window, max_acc, comfort_acc, beta, # IDM parameters
                    politeness, b_safe, acc_thres # MOBIL parameters
                ]
        '''
        def calculate_idm_acceleration(leading_v, v, s, params):
            '''Calculate acceleration of the following vehicle using IDM model
                leading_v: (N,), ndarray, speed of the leading vehicles
                v: (N,), ndarray, speed of the following vehicles
                s: (N,), ndarray, headway distances between the leading and following vehicles
                params: [desired_speed, jam_space, desired_time_window, max_acc, comfort_acc, beta]
            '''
            desired_speed, jam_space, desired_time_window, max_acc, comfort_acc, beta = params
                        
            delta_v = leading_v - v
            s_star = jam_space + np.maximum(0, v * desired_time_window - v * delta_v / (2 * np.sqrt(max_acc * comfort_acc)))
            acceleration = max_acc * (1 - np.power(v / (desired_speed + 1e-6), beta) - np.power(s_star / (s + 1e-6), 2))

            # handle the negative spacing
            acceleration[s <= 0] = -max_acc

            return acceleration
        
        # Extract event data and parameters
        v, s, of_v, or_v, tf_v, tr_v, rtf_x, rtr_x, rr_x, or_acc, tr_acc = event_data.T # [12, N]
        desired_speed, jam_space, desired_time_window, max_acc, comfort_acc, beta, politeness, b_safe, acc_thres = params
        idm_params = [desired_speed, jam_space, desired_time_window, max_acc, comfort_acc, beta]
        
        # Calculate acceleration of the following vehicle
        acc = calculate_idm_acceleration(of_v, v, s, idm_params)

        # Calculate acceleration of the following vehicle in the new lane
        acc_new = calculate_idm_acceleration(tf_v, v, rtf_x, idm_params)

        # Calculate acceleration of the target lane rear vehicle
        tr_acc_new = calculate_idm_acceleration(v, tr_v, rtr_x, idm_params)

        # Calculate acceleration of the original lane rear vehicle
        or_acc_new = calculate_idm_acceleration(v, or_v, rr_x, idm_params)

        # Calculate acceleration differences
        acc_diff = acc_new - acc
        tr_acc_diff = tr_acc_new - tr_acc
        or_acc_diff = or_acc_new - or_acc

        # Check if the lane change is beneficial
        benefit = acc_diff + politeness * (tr_acc_diff + or_acc_diff)
        benefit_idx = benefit > acc_thres
        
        # Check if the target lane rear vehicle is safe
        safe_idx = tr_acc_new <= b_safe

        # Make lane change decision
        lane_change_decision = np.zeros_like(v)
        lane_change_decision[benefit_idx & safe_idx] = 1

        return lane_change_decision
        


if __name__ == "__main__":
    from config import configs

    evaluator = MOBILEvaluator(config=configs)
    loss, eval_results, cm_results = evaluator.evaluate_baseline()
    print(f'''Baseline Evaluation Loss: {loss:.3f}.
Base model evaluation results: [{', '.join([f"{k}: {v:.3f}" for k, v in eval_results.items()])}]. 
Confusion Matrix: [{', '.join([f"{k}: {v}" for k, v in cm_results.items()])}].''')