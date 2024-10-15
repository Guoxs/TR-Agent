import os
import pickle
import numpy as np

class IDMDataLoader:
    def __init__(self, config, **kwargs):
        if config['task_name'] != 'IDM':
            raise ValueError('IDMDataLoader only supports IDM task')
        
        data_path = os.path.join(config['dataset_path'], config['task_name'])

        sample_len = config[config['task_name']]['data_sample_len']
        dataset_name = config[config['task_name']]['dataset_name']

        if dataset_name == 'SH_Fol':
            data_path = os.path.join(data_path, 'SH_Fol_with_state.pkl')
            all_dataset = []
            with open(data_path, 'rb') as f:
                all_dataset = pickle.load(f)
            
            # filter out the event less than 100
            all_dataset = [event for event in all_dataset if event.shape[0] > 100]

            event_length = [event.shape[0] for event in all_dataset]
            print(f'Number of events: {len(all_dataset)}, Min length: {min(event_length)}, Max length: {max(event_length)}')

            # [space, svSpd, lvSpd, xSv, xLv, state]
            self.trainset = all_dataset[:int(len(all_dataset) * 0.2)]
            self.testset = all_dataset[int(len(all_dataset) * 0.2):]

        else:
            train_path = os.path.join(data_path, f'{dataset_name}_train_data_with_state_{sample_len}s.pkl')
            test_path = os.path.join(data_path, f'{dataset_name}_test_data_with_state_{sample_len}s.pkl')

            train_dataset = []
            with open(train_path, 'rb') as f:
                train_dataset = pickle.load(f)

            test_dataset = []
            with open(test_path, 'rb') as f:
                test_dataset = pickle.load(f)

            all_dataset = train_dataset + test_dataset
            # filter out the event less than 100
            all_dataset = [event for event in all_dataset if event.shape[0] > 100]

            # [space, svSpd, lvSpd, xSv, xLv, state]
            self.trainset = all_dataset[:int(len(all_dataset) * 0.2)]
            self.testset = all_dataset[int(len(all_dataset) * 0.2):]

    def __str__(self):
        return f'Trainset size: {len(self.trainset)}, Testset size: {len(self.testset)}'
    
    def get_test_sampler(self, idx=2):
        return self.testset[idx]


if __name__ == "__main__":
    config = {
        'task_name': 'IDM',
        'dataset_path': '../../datasets',
        'seed': 2024,
        'IDM': {
            'data_sample_len': 15,
            'dataset_name': 'SH_Fol', 
            'calib_data_len': 256,
        }
    }

    idm_dataloader = IDMDataLoader(config)
    print(idm_dataloader)
    test_sampler = idm_dataloader.get_test_sampler()
    print(test_sampler.shape)