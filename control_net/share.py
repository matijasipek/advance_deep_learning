import sys
import os
project_root = '/zhome/48/2/181238/cv_project/advance_deep_learning/control_net'
sys.path.append(project_root)

import config
from cldm.hack import disable_verbosity, enable_sliced_attention # solved


disable_verbosity()

if config.save_memory:
    enable_sliced_attention()
