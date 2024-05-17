import os

import torch
from dataset import AdobeDataset
from bgr2yuv import bgr2ycbcr_torch
from tqdm import tqdm
from model.model import Model
from utils.metrics import ssim_tensor_function, psnr_tensor_function
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import cv2
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_device(model):
    return next(model.flownet.parameters()).device


class Vid4Dataset(Dataset):
    def __init__(self, root_dir, interval=8, get_lowres=True):
        self.root_dir = root_dir
        self.interval = interval
        self.get_lowres = get_lowres

        self.gt_image_path_tuple_list, self.intermediate_image_path_tuple_list = self.get_img_list(self.root_dir, self.interval)

    def get_img_list(self, path, interval=8):
        file_list = os.listdir(path)
        # extract only txt file
        txt_files = [file for file in file_list if file.endswith(".txt")]

        gt_image_path_tuple_list = []
        intermediate_image_path_tuple_list = []

        for txt_file in txt_files:
            with open(os.path.join(path, txt_file), "r") as f:
                data = f.readlines()
                temp_list = []
                for line in data:
                    line = line.strip("\n")
                    if line == "":
                        continue
                    image_path = os.path.join(path, f"{line}")
                    temp_list.append(image_path)

                for i in range(0, len(temp_list) - (interval + 1), interval):
                    gt_image_path_tuple_list.append((temp_list[i], temp_list[i + interval]))
                    intermediate_image_path_tuple_list.append([temp_list[i + j] for j in range(1, interval)])

        return gt_image_path_tuple_list, intermediate_image_path_tuple_list

    def __len__(self):
        return len(self.gt_image_path_tuple_list)

    def __getitem__(self, item):
        gt_image_path_tuple = self.gt_image_path_tuple_list[item]
        intermediate_image_path_tuple = self.intermediate_image_path_tuple_list[item]

        gt_images = []
        intermediate_images = []

        for gt_image_path in gt_image_path_tuple:
            img = cv2.imread(gt_image_path) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            gt_images.append(img)

        for intermediate_image_path in intermediate_image_path_tuple:
            img = cv2.imread(intermediate_image_path) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            intermediate_images.append(img)

        ground_truth = torch.cat((gt_images[0], *intermediate_images, gt_images[-1]), dim=0)

        # lowres = F.interpolate(ground_truth.unsqueeze(0), scale_factor=0.25, mode="bicubic", align_corners=False).squeeze(0)
        # lowres = F.interpolate(lowres.unsqueeze(0), scale_factor=4, mode="bilinear", align_corners=False).squeeze(0)

        highres_path_list = [gt_image_path_tuple[0], *intermediate_image_path_tuple, gt_image_path_tuple[-1]]
        lowres_list = []
        for highres_path in highres_path_list:
            img = cv2.imread(highres_path)
            if self.get_lowres:
                img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
                img = cv2.resize(img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
            img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
            lowres_list.append(img)
        lowres = torch.cat(lowres_list, dim=0)

        return ground_truth, lowres, 0.5

# Vid4 PSNR: 16.9774, SSIM: 0.3393

@torch.no_grad()
def validate_vid4(model, root_dir, visualization=False, early_stop=-1, test_lowres=False, interval=2): # timestep 0.5 only
    validate_dataset = Vid4Dataset(root_dir, interval=interval, get_lowres=test_lowres)
    validate_dataloader = DataLoader(validate_dataset, batch_size=1, shuffle=False)
    model.eval()

    total_psnr, total_ssim = 0., 0.
    total_psnr_counter, total_ssim_counter = 0, 0

    for data in tqdm(validate_dataloader):
        if early_stop != -1 and total_psnr_counter >= early_stop:
            break

        gt_frames, lowres_frames, timestep = data
        gt_frames, lowres_frames, timestep = gt_frames.to(get_model_device(model), non_blocking=True), lowres_frames.to(get_model_device(model), non_blocking=True), timestep.to(get_model_device(model), non_blocking=True).float()
        timestep = timestep.expand(gt_frames.size(0), 1, gt_frames.size(2) // 2, gt_frames.size(3) // 2)

        gt_frame_list, lowres_frame_list = gt_frames.chunk(interval + 1, dim=1), lowres_frames.chunk(interval + 1, dim=1)
        result = model.inference(lowres_frame_list[0], lowres_frame_list[-1], timestep=[0.5])
        result = result[0]

        gt_y = bgr2ycbcr_torch(gt_frame_list[len(gt_frame_list) // 2], only_y=True) / 255.
        result_y = bgr2ycbcr_torch(result, only_y=True) / 255.

        # gt_y = gt_frame_list[len(gt_frame_list) // 2]
        # result_y = result

        total_psnr += psnr_tensor_function(result_y, gt_y)
        total_ssim += ssim_tensor_function(result_y, gt_y)
        total_psnr_counter += 1
        total_ssim_counter += 1

        # visualization
        if visualization:
            concat_image = torch.cat((gt_y, result_y), dim=-1).squeeze().detach().cpu().numpy()
            if len(concat_image.shape) == 3:
                concat_image = concat_image.transpose(1, 2, 0)
            concat_image = concat_image * 255
            concat_image = concat_image.astype('uint8')
            cv2.imshow("result", concat_image)
            cv2.waitKey(3)

        # memory release
        del gt_frames, lowres_frames, timestep, gt_frame_list, lowres_frame_list, result, gt_y, result_y

    print("Vid4 PSNR: {:.4f}, SSIM: {:.4f}".format(total_psnr / total_psnr_counter, total_ssim / total_ssim_counter))
    return total_psnr / total_psnr_counter, total_ssim / total_ssim_counter

if __name__ == '__main__':
    model = Model()
    model.device()
    model.load_model('train_log_240425')
    model.eval()

    psnr, ssim = validate_vid4(model, "/media/ms-neo2/ms-ssd11/1.dataset/VFI/Vid4", visualization=True, early_stop=-1, test_lowres=True, interval=8)

    # Vid4 PSNR: 17.3944, SSIM: 0.3561