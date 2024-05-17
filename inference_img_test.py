import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


from model.model import Model
model = Model()
model.device()
model.load_model('train_log')
model.eval()

img0 = torch.randn(1, 3, 256, 256)
img1 = torch.randn(1, 3, 256, 256)
img0, img1 = img0.cuda(), img1.cuda()

for i in range(20):
    with torch.no_grad():
        img_list = model.inference(img0, img1, timestep=[0.5])
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        for _ in range(50):
            img_list = model.inference(img0, img1, timestep=[0.5])

# print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))