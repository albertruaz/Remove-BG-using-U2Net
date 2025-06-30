import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from u2net import U2NET # full size version 173.6 MB
from u2net import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name, pred, d_dir):
    # 마스크
    predict = pred.squeeze().cpu().data.numpy()
    mask = (predict * 255).astype(np.uint8)

    # 원본 이미지 불러오기 (RGB)
    image = Image.open(image_name).convert("RGB")
    image = np.array(image)

    # 마스크 리사이즈 (원본 크기 맞춤)
    mask = Image.fromarray(mask).resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    mask = np.array(mask) / 255.0  # 0~1로 정규화

    # 흰 배경
    white_bg = np.ones_like(image, dtype=np.uint8) * 255

    # 마스크 적용 (배경은 흰색, 전경은 원본)
    result = image * mask[..., None] + white_bg * (1 - mask[..., None])
    result = result.astype(np.uint8)

    # 저장
    img_name = os.path.basename(image_name)
    name_no_ext = os.path.splitext(img_name)[0]
    result_pil = Image.fromarray(result)
    result_pil.save(os.path.join(d_dir, f"{name_no_ext}_whitebg.png"))


def main():


    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', 'u2net' + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', 'u2net.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)


    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    net = U2NET(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir, weights_only=False))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu', weights_only=False))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
