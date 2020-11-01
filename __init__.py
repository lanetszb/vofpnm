import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path))

from vofpnm.pnm_bind import *
from vofpnm.vof_bind import *
