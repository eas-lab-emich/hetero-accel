from pydantic import BaseModel


class ConvolutionProblem(BaseModel):
    input_channels: int  # C
    output_channels: int  # K/M
    # int((self.dims['Xi'] - self.dims['S'] + 2 * self.dims['Wpad']) / self.dims['Wstr']) + 1
    # int((self.dims['Yi'] - self.dims['R'] + 2 * self.dims['Hpad']) / self.dims['Hstr']) + 1
    input_width: int
    input_height: int
    padding_width: int
    padding_height: int
    stride_width: int
    stride_height: int
    batch_size: int  # N
    kernel_width: int  # S
    kernel_height: int  # R
