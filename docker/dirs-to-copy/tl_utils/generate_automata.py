import os
import sys
import numpy as np
import time

default_config = {}

class GenerateAutomata(object):

    def __init__(self, config={}):
        self.GenerateAutomata_config = default_config
        self.GenerateAutomata_config.update(config)