import os
import sys
sys.path.append(os.path.abspath("../Cart_Pole"))
from agentDQN_targetNetwork import *


a = AgentDQN_TargetNetwork(3,3)
print(a)