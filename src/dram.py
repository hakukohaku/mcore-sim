import simpy
import logging
import contextlib
from enum import IntEnum
from src.arch_config import LinkConfig, RouterConfig, NoCConfig
from src.sim_type import Data, Message, ceil, Slice, Direction, DataType
from src.common import MonitoredResource

class Dram:
    def __init__(self, env, config, index):
        self.env = env
        self.index = index
        self.bandwidth = config.dram_bw
        self.capacity = config.dram_capacity
        self.space_allocated = config.dram_capacity
        self.mem_resource = MonitoredResource(env, capacity=1)
    def read(self, size, ins):
        if size > self.space_allocated:
            raise ValueError(f"Dram {self.index} allocated space is {self.space_allocated}, but read size is {size}")
        else:
            yield self.mem_resource.execute("READ"+str(self.index), ceil(size, self.bandwidth), ins, self.index)
            
    def write(self, size, ins):
        if size > self.capacity - self.space_allocated:
            raise ValueError(f"Dram {self.index} remaining space is {self.capacity - self.space_allocated}, but write size is {size}")
        else:
            yield self.mem_resource.execute("WRITE"+str(self.index), ceil(size, self.bandwidth), ins, self.index)
            self.space_allocated += size
            
    def free_space(self, size):
        self.space_allocated -= size  
        
        
    def get_index(self):
        return self.index