from uuid import UUID

from pydantic import BaseModel

from src.mapping.api.accelerator import AcceleratorConfiguration
from src.mapping.api.problem import ConvolutionProblem


class MappingRequest(BaseModel):
    id: UUID
    accelerator_config: AcceleratorConfiguration
    problem: ConvolutionProblem
