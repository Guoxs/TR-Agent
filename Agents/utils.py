import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime

from prompts import IDM_prompts, MOBIL_prompts, LWR_prompts
from dataloader import IDM_dataloader, MOBIL_dataloader, LWR_dataloader


def get_llm(config):
    model_name = config['llm_model']
    if 'gpt' in model_name:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(temperature=config['llm_temperature'], model=model_name)
    elif 'claude' in model_name:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(temperature=config['llm_temperature'], model=model_name)
    else:
        raise ValueError('Invalid model name')


def get_prompt(task_name):
    if task_name == 'IDM':
        return IDM_prompts.IDM_PROMPT
    elif task_name == 'MOBIL':
        return MOBIL_prompts.MOBIL_PROMPT
    elif task_name == 'LWR':
        return LWR_prompts.LWR_PROMPT
    else:
        raise ValueError('Invalid task name')


def get_dataset(config):
    task_name = config['task_name']
    if task_name == 'IDM':
        return IDM_dataloader.IDMDataLoader(config=config)
    elif task_name == 'MOBIL':
        return MOBIL_dataloader.MOBILDataLoader(config=config)
    elif task_name == 'LWR':
        return LWR_dataloader.LWRDataLoader(config=config)
    else:
        raise ValueError('Invalid task name')


def setup_logger(name, task_name, model_name, level=logging.INFO):
    """Function to setup as many loggers as you want"""

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Ensure the logger has no handlers before adding new ones
    if not logger.hasHandlers():
        # Create a console handler
        c_handler = logging.StreamHandler()
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        # Ensure the logs directory exists
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(dir_path, 'logs', task_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create a file handler that logs to a file named with the current date
        log_filename = os.path.join(log_dir, f"{model_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
        f_handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1, backupCount=7)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        f_handler.suffix = "%Y-%m-%d"
        logger.addHandler(f_handler)

    return logger, log_filename