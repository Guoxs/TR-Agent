import os
import numpy as np
import pandas as pd

class MOBILDataLoader:
    def __init__(self, config, **kwargs):
        self.config = config

        if config['task_name'] != 'MOBIL':
            raise ValueError('MOBILDataLoader only supports MOBIL task')
        
        self.config = config
        
        self.data_path = os.path.join(config['dataset_path'], config['task_name'], 
                                      config[config['task_name']]['dataset_name'])
        self.all_dataset = self.load_data()
        self.calib_dataset = self.all_dataset.sample(frac=0.2, random_state=config['seed'])
        self.calib_dataset = self.calib_dataset.reset_index(drop=True)

    def __str__(self):
        return 'MOBIL dataset'
    
    def load_data(self):
        data = np.load(self.data_path, allow_pickle=True)
        columns = ['vehicle_id', 'time', 'lane', 'vehicle_speed', 'vehicle_acceleration', 
                   'front_vehicle_id', 'rear_vehicle_id', 'headway_distance', 'headway_time', 
                   'original_lane_front_vehicle_id', 'original_lane_front_vehicle_speed', 'original_lane_front_vehicle_acceleration', 
                   'original_lane_rear_vehicle_id', 'original_lane_rear_vehicle_speed', 'original_lane_rear_vehicle_acceleration', 
                   'target_lane_front_vehicle_id', 'target_lane_front_vehicle_speed', 'target_lane_front_vehicle_acceleration', 
                   'target_lane_rear_vehicle_id', 'target_lane_rear_vehicle_speed', 'target_lane_rear_vehicle_acceleration', 
                   'relative_distance_front_vehicle_x', 'relative_distance_front_vehicle_y', 'relative_speed_front_vehicle', 
                   'relative_distance_rear_vehicle_x', 'relative_distance_rear_vehicle_y', 'relative_speed_rear_vehicle', 
                   'relative_distance_target_lane_front_vehicle_x', 'relative_distance_target_lane_front_vehicle_y', 'relative_speed_target_lane_front_vehicle', 
                   'relative_distance_target_lane_rear_vehicle_x', 'relative_distance_target_lane_rear_vehicle_y', 'relative_speed_target_lane_rear_vehicle']
        
        data = pd.DataFrame(data, columns=columns)
        data['desired_speed'] = data.groupby('vehicle_id')['vehicle_speed'].transform('max')
        data['lane_change_event'] = data.sort_values(['vehicle_id', 'time']).groupby('vehicle_id')['lane'].diff().fillna(0) != 0
        data['lane_change_event'] = data['lane_change_event'].astype(int)  
        data['speed_change_rate'] = data.groupby('vehicle_id')['vehicle_speed'].diff().fillna(0)
        data['acceleration_change_rate'] = data.groupby('vehicle_id')['vehicle_acceleration'].diff().fillna(0)
        
        lane_change_indices = data[data['lane_change_event'] == 1].index
        lane_change_data = pd.DataFrame()
        
        for idx in lane_change_indices:
            vehicle_id = data.at[idx, 'vehicle_id']
            lane_change_time = data.at[idx, 'time']
            start_time = lane_change_time - 30 # 3 seconds before lane change event
            
            vehicle_data = data[(data['vehicle_id'] == vehicle_id) & 
                                (data['time'] >= start_time) & 
                                (data['time'] <= lane_change_time)]
            
            data.loc[vehicle_data.index, 'lane_change_event'] = 1
            lane_change_data = pd.concat([lane_change_data, vehicle_data], ignore_index=True)
        
        # select data that lane_change_event == 1
        lane_change_data = data[data['lane_change_event'] == 1]
        lane_change_data = lane_change_data.drop_duplicates().reset_index(drop=True)
        print(f'Number of lane change event data groups: {lane_change_data.shape[0]}')
        
        # Select equal amount of non-lane change events
        no_lane_change_data = data[data['lane_change_event'] == 0].sample(
                                n=lane_change_data.shape[0], random_state=self.config['seed'])
        print(f'Number of non-lane change event data groups: {no_lane_change_data.shape[0]}')
      
        # Combine data and shuffle order
        combined_data = pd.concat([lane_change_data, no_lane_change_data]).drop_duplicates().sample(
                        frac=1, random_state=self.config['seed']).reset_index(drop=True)
        print(f'Final number of data groups: {combined_data.shape[0]}')

        output_columns = ['vehicle_id', 'time', 'vehicle_speed', 'desired_speed', 'headway_distance', 
                          'original_lane_front_vehicle_speed', 'original_lane_rear_vehicle_speed', 
                          'target_lane_front_vehicle_speed', 'target_lane_rear_vehicle_speed', 
                          'relative_distance_target_lane_front_vehicle_x', 'relative_distance_target_lane_rear_vehicle_x', 
                          'relative_distance_rear_vehicle_x', 
                          'original_lane_rear_vehicle_acceleration', 'target_lane_rear_vehicle_acceleration', 
                          'lane_change_event']

        output_dataset = combined_data[output_columns]
        return output_dataset

    def get_test_sampler(self):
        valid_column_names = ['vehicle_speed', 'headway_distance', 
                          'original_lane_front_vehicle_speed', 'original_lane_rear_vehicle_speed', 
                          'target_lane_front_vehicle_speed', 'target_lane_rear_vehicle_speed', 
                          'relative_distance_target_lane_front_vehicle_x', 'relative_distance_target_lane_rear_vehicle_x', 
                          'relative_distance_rear_vehicle_x', 
                          'original_lane_rear_vehicle_acceleration', 'target_lane_rear_vehicle_acceleration']
    
        event_data = self.calib_dataset[valid_column_names].values
        return event_data


if __name__ == '__main__':
    config = {
        'task_name': 'MOBIL',
        'dataset_path': '../../datasets',
        'seed': 2024,
        'MOBIL': {
            'dataset_name': 'ngsim_data.npy',
        }
    }
    
    mobil_data_loader = MOBILDataLoader(config)
    print(mobil_data_loader)

    print(mobil_data_loader.get_test_sampler())