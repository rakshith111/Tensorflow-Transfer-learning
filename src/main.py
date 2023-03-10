import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('CRITICAL')
import sys

import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from libs._base_logger import logger
from libs.common import check_gpu
if __name__ == "__main__":
    if not os.path.exists("src/dataset/train") and not os.path.exists("src/dataset/test"):
        os.makedirs("src/dataset/train")
        os.makedirs("src/dataset/test")
        logger.info("Created dataset directory")
        logger.info("Add the dataset to the dataset directory Class name should be the same as the folder name")
        exit()
       
    logger.info(f"Tensorflow version: {tf.__version__}")
    logger.info(f"Gpu available: {check_gpu()}")
