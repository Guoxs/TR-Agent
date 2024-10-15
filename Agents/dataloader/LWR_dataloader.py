import os
import numpy as np


class LWRDataLoader:
    def __init__(self, config, **kwargs):
        if config['task_name'] != 'LWR':
            raise ValueError('LWRDataLoader only supports LWR task')
        
        self.config = config
        self.data_path = os.path.join(config['dataset_path'], config['task_name'])

        self.density_train = np.load(os.path.join(self.data_path, 'density_train.npy'))
        self.density_val = np.load(os.path.join(self.data_path, 'density_val.npy'))
        self.speed_train = np.load(os.path.join(self.data_path, 'speed_train.npy'))
        self.speed_val = np.load(os.path.join(self.data_path, 'speed_val.npy'))

    def __str__(self):
        return f'Trainset size: {self.density_train.shape}, Validation set size: {self.density_val.shape}'

    def get_test_sampler(self):
        return self.density_val[:100]


if __name__ == '__main__':
    config = {
        'task_name': 'LWR',
        'dataset_path': os.path.join('../../', 'datasets'),
    }

    lwr_data_loader = LWRDataLoader(config=config)
    print(lwr_data_loader)

    # 0-0.3 low density, 0.3-0.6 medium density, 0.6-1 high density
    density_val_low = lwr_data_loader.density_val[lwr_data_loader.density_val < 0.3]
    density_val_medium = lwr_data_loader.density_val[(lwr_data_loader.density_val >= 0.3) & (lwr_data_loader.density_val < 0.6)]
    density_val_high = lwr_data_loader.density_val[lwr_data_loader.density_val >= 0.6]

    print(f'Low density: {density_val_low.shape[0]}, Medium density: {density_val_medium.shape[0]}, High density: {density_val_high.shape[0]}')