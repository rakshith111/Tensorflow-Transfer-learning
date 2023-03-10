import tensorflow as tf
from numba import cuda 

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from libs._base_logger import logger

def check_gpu() -> bool:
    '''
    This function checks if the GPU is available and if CUDA is installed
    :return: `True` if the GPU is available and CUDA is installed else `False`
    :rtype: bool
    '''

    devices = tf.config.list_physical_devices()
    if devices:
        try:
            for device in devices:
                logger.info(
                    f"Device name: {device.name} Device type: {device.device_type}")
            physical_gpus = tf.config.list_logical_devices('GPU')
            logger.info("Num GPUs Available: {}".format(len(physical_gpus)))
            if len(physical_gpus) > 0:
                device = cuda.get_current_device()
                if device is not None:
                    device.reset()
                    logger.info(f"Device name: {device.name} ")
                    if tf.test.is_built_with_cuda():
                        logger.info("CUDA is available !")
                        return True
                    else:
                        logger.error("CUDA is not available !")
                        logger.info("Have you installed CUDA ?\n or maybe old GPU drivers ?")
                        return False
                else:
                    logger.error("No device found")
                    return False
            else:
                logger.error("No GPU found")
                logger.info("Have you installed CUDA ?")
                return False

        except RuntimeError as e:

            logger.error(e)
            print(e)
    else:
        logger.error("No GPU found")
        return False
