import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    A custom activation layer that applies the SiLU (Sigmoid Linear Unit) activation
    function followed by element-wise multiplication with the input tensor.
    """

    def __init__(self):
        super().__init__()

    # 算子融合，前面的gate和up矩阵沿最后一维拼接起来了
    # 所以算出来的x包括了gate值和up值，因此将其沿最后一维再切开
    # 然后使用一个算子进行门控操作
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


if __name__ == "__main__":
    # Example usage
    layer = SiluAndMul().cuda()
    input_tensor = torch.randn(8, 4000, 8000).cuda()  # Example input tensor with shape (400, 800)
    
    for _ in range(10):  # Warm-up iterations
        _ = layer(input_tensor)

    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
