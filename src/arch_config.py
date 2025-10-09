from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SPMConfig(BaseModel):
    size: int
    delay: int

class TPUConfig(BaseModel):
    flops: int
    vect_flops: int

class LSUConfig(BaseModel):
    width: int

class CoreConfig(BaseModel):
    type: str
    x: int
    y: int
    width: int
    blk_size: int
    spm: SPMConfig
    compute: TPUConfig
    lsu: LSUConfig

class RouterConfig(BaseModel):
    type: str
    vc: int

class LinkConfig(BaseModel):
    width: int
    delay: int

class NoCConfig(BaseModel):
    type: str
    x: int
    y: int
    router: RouterConfig
    link: LinkConfig

class MemConfig(BaseModel):
    type: str
    num: int
    dram_bw: int
    dram_capacity: int
    mem_core: Optional[Dict[str, dict]] = None

class ArchConfig(BaseModel):
    core: CoreConfig
    noc: NoCConfig
    mem: MemConfig

class ScratchpadConfig(BaseModel):
    size: int
    delay: int

