from pydantic import BaseModel


class AcceleratorConfiguration(BaseModel):
    pe_array_x: int
    pe_array_y: int
    precision: int
    sram_size: int
    ifmap_spad_size: int
    weights_spad_size: int
    psum_spad_size: int
