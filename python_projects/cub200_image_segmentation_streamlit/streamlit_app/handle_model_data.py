import os
import requests
import numpy as np

from typing import Tuple, Optional
from numpy.typing import NDArray

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model_class import Unet


# Number of pixels (in one direction) and number of channels for classifier
N_PIXEL = 128
N_CHANNEL = 3

def download_file(url: str,
                  output_path: str,
                  chunk_size: int = 6000,
                  overwrite: bool = False) -> None:
    # Correct Dropbox url if necessary
    if ('.dropbox.' in url) and url.endswith('&dl=0'):
        url.replace('&dl=0', '&dl=1')

    # Load file in chunks
    if (not os.path.isfile(output_path)) or overwrite:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f_out:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f_out.write(chunk)

def download_all_files(model_path: str,
                       image_path: str,
                       mask_path: str):
    model_url = 'https://www.dropbox.com/scl/fi/wptqttmjfo6hvx48qzr9h/Unet.pt?rlkey=z2ftw59b0hexdcp6mj7ltm9c0&st=1zi87neu&dl=1'
    image_url = 'https://www.dropbox.com/scl/fi/3qavnovyp7dluqvylr3ir/test_images.pt?rlkey=wnlp2s8cwlekivsyv0unz08mz&st=xrmemaok&dl=1'
    mask_url = 'https://www.dropbox.com/scl/fi/mq40h9cacd8cig8ckqhsp/test_masks.pt?rlkey=7m9jfre1wivi32cei2kc1ukeo&st=80eznfbd&dl=1'

    download_file(url=model_url,
                  output_path=model_path)
    download_file(url=image_url,
                  output_path=image_path)
    download_file(url=mask_url,
                  output_path=mask_path)


def load_model(model_path: str) -> Unet:
    # Model architecture
    model = Unet(channels_in=N_CHANNEL, channels_out=2)

    # Load checkpoint state dictionary
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device('cpu'))

    return model


def load_images_masks(image_path: str,
                      mask_path: str) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    images = torch.load(image_path)
    masks = torch.load(mask_path)

    return images, masks


def get_relevant_images_mask(
        random_image: bool,
        uploaded_file: Optional[str],
        images: list[torch.Tensor],
        masks: list[torch.Tensor]
    ) -> Tuple[NDArray, torch.Tensor, Optional[torch.Tensor]]:
    
    # Transform for classification
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    transform_display = A.Compose([A.SmallestMaxSize(max_size=N_PIXEL),
                                   A.CenterCrop(height=N_PIXEL, width=N_PIXEL)])
    transform = A.Compose([A.SmallestMaxSize(max_size=N_PIXEL),
                           A.CenterCrop(height=N_PIXEL, width=N_PIXEL),
                           A.Normalize(mean=data_mean, std=data_std),
                           ToTensorV2()])

    # Initialize mask to None
    mask = None

    # Image to display
    if random_image:
        # Select random image and associated label
        idx = np.random.randint(len(images))
        image = images[idx].numpy()
        mask = masks[idx].unsqueeze(2).float().numpy()

        image_display = np.transpose(image, (1, 2, 0))
        image_display = cv2.normalize(image_display, None, 0., 1., cv2.NORM_MINMAX)
        image_display[image_display < 0] = 0.
        image_display[image_display > 1] = 1.

        image_for_model = torch.unsqueeze(images[idx], 0)

    elif uploaded_file:
        # Load file and process for display
        with Image.open(uploaded_file).convert('RGB') as image:
            image_display = transform_display(image=np.asarray(image))['image']
            image_display = np.reshape(image_display,
                                       (N_PIXEL, N_PIXEL, N_CHANNEL))
            image_display = image_display/255

        # Process image for model
        # image_for_model = Image.fromarray(image_display)
        image_for_model = transform(image=np.asarray(image))['image']
        image_for_model = torch.unsqueeze(image_for_model, 0)

    return image_display, image_for_model, mask


def mask_intersection_over_union(mask_pred, mask):

        # compute the area of intersection rectangle
        interArea = np.sum(mask_pred*mask)
        area1 = np.sum(mask_pred)
        area2 = np.sum(mask)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / (area1 + area2 - interArea + 1e-5)

        # return the intersection over union value
        return iou