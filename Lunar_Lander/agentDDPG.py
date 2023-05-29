import os
import sys
sys.path.append(os.path.abspath("../agent/"))
from agent import *


a = AgentDQN_TargetNetwork(3,3)
print(a)