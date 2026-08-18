import logging
import numpy as np
from torchvision import transforms as tforms

from credit.transforms.transforms_wrf import Normalize_WRF, ToTensor_WRF
from credit.transforms.transforms_dscale import Normalize_Dscale, ToTensor_Dscale

logger = logging.getLogger(__name__)


def load_transforms(conf, scaler_only=False):
    """Load transforms.

    Args:
        conf (str): path to config
        scaler_only (bool): True --> retrun scaler; False --> return scaler and ToTensor

    Returns:
        tf.tensor: transform

    """
    # ------------------------------------------------------------------- #
    # transform class
    if conf["data"]["scaler_type"] == "std-wrf":
        transform_scaler = Normalize_WRF(conf)

    elif conf["data"]["scaler_type"] == "std-dscale":
        transform_scaler = Normalize_Dscale(conf)
        
    else:
        logger.log("scaler type not supported check data: scaler_type in config file")
        raise

    if scaler_only:
        return transform_scaler

    # ------------------------------------------------------------------- #
    # ToTensor class

    if conf["data"]["scaler_type"] == "std-wrf":
        to_tensor_scaler = ToTensor_WRF(conf)

    elif conf["data"]["scaler_type"] == "std-dscale":
        to_tensor_scaler = ToTensor_Dscale(conf)
        
    else:
        # the old ToTensor
        to_tensor_scaler = ToTensor(conf=conf)

    # ------------------------------------------------------------------- #
    # combine transform and ToTensor

    if transform_scaler is not None:
        transforms = [transform_scaler, to_tensor_scaler]

    else:
        transforms = [to_tensor_scaler]

    return tforms.Compose(transforms)
