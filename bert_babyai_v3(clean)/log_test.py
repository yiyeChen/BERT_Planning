#python 3.x
import logging
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO)
horizn = 2
handler = logging.FileHandler(f'log_file {horizn}.log')
formatter = logging.Formatter('%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
num_list = [1, 2, 3, 4, 5]
logger.info(str(("Numbers in num_list are: ",num_list)))


