from uuid import UUID

from pydantic import BaseModel

from src.worker.api.accelerator import AcceleratorConfiguration
from src.worker.api.problem import ConvolutionProblem


class MappingRequest(BaseModel):
    id: UUID
    accelerator_config: AcceleratorConfiguration
    problem: ConvolutionProblem
