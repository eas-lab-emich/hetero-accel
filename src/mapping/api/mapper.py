from abc import ABC
from concurrent.futures import Future
from uuid import UUID

from pydantic import BaseModel


class AcceleratorConfiguration(BaseModel):
    pe_array_x: int
    pe_array_y: int
    precision: int
    sram_size: int
    ifmap_spad_size: int
    weights_spad_size: int
    psum_spad_size: int


class ConvolutionProblem(BaseModel):
    input_channels: int  # C
    output_channels: int  # K/M
    input_width: int
    input_height: int
    padding_width: int
    padding_height: int
    stride_width: int
    stride_height: int
    batch_size: int  # N
    kernel_width: int  # S
    kernel_height: int  # R


class MappingRequest(BaseModel):
    id: UUID
    name: str | None = None
    accelerator_config: AcceleratorConfiguration
    problem: ConvolutionProblem


class MappingStats(BaseModel):
    id: UUID
    gflops: float
    utilization: float
    cycles: int
    energy: float
    edp: float
    area: float


class AcceleratorMapper(ABC):
    def map(self, request: MappingRequest) -> MappingStats:
        pass


class AsyncAcceleratorMapper(ABC):
    def map(self, request: MappingRequest) -> Future[MappingStats]:
        pass

    def start(self):
        pass

    def stop(self):
        pass
