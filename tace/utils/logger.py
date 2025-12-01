################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import sys
import logging
from datetime import datetime

import torch
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only


from ..__init__ import __version__


LOG_LEVELS = {
    10: logging.DEBUG,
    20: logging.INFO,
    30: logging.WARNING,
    40: logging.ERROR,
    50: logging.CRITICAL,
}


def set_logger(_level: int = 20, _rank_zero_only: bool = True) -> None:
    assert _level in [10, 20, 30, 40, 50]
    root_logger = logging.getLogger()
    if root_logger.handlers:
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
    else:
        logging.basicConfig(
            level=LOG_LEVELS[_level],
            format="[%(levelname)s] %(message)s",
        )
    root_logger.setLevel(LOG_LEVELS[_level])
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

