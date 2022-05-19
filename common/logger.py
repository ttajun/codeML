
import os
import logging
from logging.handlers import RotatingFileHandler

def make_logger(name=None):

    program_path = os.path.dirname(os.path.dirname(__file__))
    _, program_name = os.path.split(program_path)

    str_log_file = f'{program_name}.log'
    log_file = os.path.join(program_path, str_log_file)

    # logger instance를 만든다.
    logger = logging.getLogger(name)

    # logger의 level을 가장 낮은 수준인 DEBUG로 설정해둔다.
    logger.setLevel(logging.DEBUG)

    # formatter 지정
    formatter = logging.Formatter("%(asctime)s %(name)s (%(lineno)d) [%(levelname)s] %(message)s", 
                datefmt='%y-%m-%d %H:%M:%S')
    
    console = logging.StreamHandler()
    file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=1024*1024, encoding='utf-8')
    
    # handler 별로 다른 level 설정
    console.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    # handler 출력 format 지정
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # logger에 handler 추가
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger

