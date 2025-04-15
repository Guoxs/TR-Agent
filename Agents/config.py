import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

configs = {
    'task_name': 'IDM', # ['IDM', 'MOBIL', 'LWR']
    'llm_model': 'gpt-4-turbo', # ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o', 'claude-3-opus-20240229']
    'openai_api_key': '<your_openai_api_key>',
    'tavily_api_key': '<your_tavily_api_key>',
    'llm_temperature': 0.5,
    'max_iter': 10,
    'improve_rate': 0.5,
    'seed': 2024,
    'dataset_path': os.path.join(project_root, 'datasets'),
    'offline_paper_path': os.path.join(project_root, 'papers'),
    'code_gen_try_times': 5,
    
    'use_RAG': False,

    'IDM': {
        'data_sample_len': 15,
        'dataset_name': 'SH_Fol', 
        'calib_data_len': 256,
        'time_interval': 0.1,
        'eval_metric': 'spacing_with_state', # ['spacing', 'speed', 'all', 'spacing_with_state']
    },

    'MOBIL': {
        'dataset_name': 'ngsim_data.npy',
    },

    'LWR': {
        'dataset_name': 'PeMS',
    }
}