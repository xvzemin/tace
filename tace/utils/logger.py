################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
import sys
from datetime import datetime
from typing import Union

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from ..__init__ import __version__

LOG_LEVELS = {
    10: logging.DEBUG,
    20: logging.INFO,
    30: logging.WARNING,
    40: logging.ERROR,
    50: logging.CRITICAL,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def num_params(model) -> int:
    """Return the number of trainable scalar parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_parameters(model) -> None:
    """Log trainable parameter counts for the model and each parameter tensor."""
    logging.debug("Total number of parameters in the model: %s", num_params(model))
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.debug("Layer: %s, Number of parameters: %s", name, param.numel())


def set_logger(_level: Union[int, str] = "info", _rank_zero_only: bool = True) -> None:
    log_level = LOG_LEVELS[_level]
    root_logger = logging.getLogger()
    if root_logger.handlers:
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
    else:
        logging.basicConfig(
            level=log_level,
            format="[%(levelname)s] %(message)s",
        )
    root_logger.setLevel(log_level)
    if _rank_zero_only:
        logging.debug = rank_zero_only(logging.debug)
        logging.info = rank_zero_only(logging.info)
        logging.warning = rank_zero_only(logging.warning)
        logging.error = rank_zero_only(logging.error)
        logging.critical = rank_zero_only(logging.critical)
    logging.info("Training started at %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logging.info(
        f"\nTACE version: {__version__}\n"
        f"Torch version: {torch.__version__}\n"
        f"Lightning version: {L.__version__}\n"
        f"Python version: {sys.version}\n"
        r"""
        ,----,                                                                                             
      ,/   .`|                                                                                             
    ,`   .'  :                                                           ,---,         ,----..      ,---,. 
  ;    ;     /                                                          '  .' \       /   /   \   ,'  .' | 
.'___,/    ,'              ,---,              ,---.    __  ,-.         /  ;    '.    |   :     :,---.'   | 
|    :     |           ,-+-. /  | .--.--.    '   ,'\ ,' ,'/ /|        :  :       \   .   |  ;. /|   |   .' 
;    |.';  ;   ,---.  ,--.'|'   |/  /    '  /   /   |'  | |' |        :  |   /\   \  .   ; /--` :   :  |-, 
`----'  |  |  /     \|   |  ,"' |  :  /`./ .   ; ,. :|  |   ,'        |  :  ' ;.   : ;   | ;    :   |  ;/| 
    '   :  ; /    /  |   | /  | |  :  ;_   '   | |: :'  :  /          |  |  ;/  \   \|   : |    |   :   .' 
    |   |  '.    ' / |   | |  | |\  \    `.'   | .; :|  | '           '  :  | \  \ ,'.   | '___ |   |  |-, 
    '   :  |'   ;   /|   | |  |/  `----.   \   :    |;  : |           |  |  '  '--'  '   ; : .'|'   :  ;/| 
    ;   |.' '   |  / |   | |--'  /  /`--'  /\   \  / |  , ;           |  :  :        '   | '/  :|   |    \ 
    '---'   |   :    |   |/     '--'.     /  `----'   ---'            |  | ,'        |   :    / |   :   .' 
             \   \  /'---'        `--'---'                            `--''           \   \ .'  |   | ,'   
              `----'                                                                   `---`    `----'     
                                                                                                                       
        """
    )
