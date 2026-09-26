import torch
from torch.profiler import ProfilerActivity, profile

from equations import flops
from models import SmallCNN


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

s, b = 112, 33
model = SmallCNN().cuda().eval()
x = torch.randn(b, 3, s, s, device="cuda")

# Прогрев не входит в профилируемый проход.
with torch.inference_mode():
    model(x)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof:
    with torch.inference_mode():
        model(x)
    torch.cuda.synchronize()

profiled_flops = 0
for event in prof.key_averages():
    if event.flops:
        print(f"{event.key}: {int(event.flops):,} FLOPs")
        profiled_flops += event.flops

print("Оценка профилировщика:", int(profiled_flops))
print("Наша формула:", int(flops(s, b)))
print("Разница:", int(flops(s, b) - profiled_flops))
