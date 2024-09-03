import os
import requests
import numpy as np

from typing import Tuple, Optional
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

# Number of pixels (in one direction) and number of channels for classifier
N_PIXEL = 96
N_CHANNELS = 3

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
                       label_path: str) -> None:
    model_url = 'https://www.dropbox.com/scl/fi/q1h5szzia5gn6fv58qdg4/ResNet18_STL10_rotated_pretrained.pt?rlkey=0qtg8q7ybx4l9kbs08739647p&st=twhfejhg&dl=1'
    image_url = 'https://www.dropbox.com/scl/fi/obyczlakb3hsuns6jt4xu/test_X.bin?rlkey=6ivb829jjqgec3y97amwe8qw9&st=68hwdrs7&dl=1'
    label_url = 'https://www.dropbox.com/scl/fi/cvss6vxfn72ogjxmlx6bt/test_y.bin?rlkey=h7huquh724egiug95nsbsz83y&st=0ltx8y4i&dl=1'

    download_file(url=model_url,
                  output_path=model_path)
    download_file(url=image_url,
                  output_path=image_path)
    download_file(url=label_url,
                  output_path=label_path)


def load_model(model_path: str) -> models.resnet.ResNet:
    # Number of classes
    n_classes = len(get_classes())

    # Define model (resnet18 with modified final layer)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    # Load checkpoint state dictionary
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device('cpu'))

    return model


def load_images_labels(image_path: str,
                       label_path: str) -> Tuple[NDArray, NDArray]:
    # Read images
    with open(image_path, 'rb') as f:
        images = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(images, (-1, N_CHANNELS, N_PIXEL, N_PIXEL))

    # Read labels
    with open(label_path, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
    
    return images, labels


def get_relevant_images_label(
        random_image: bool,
        uploaded_file: Optional[str],
        images: NDArray,
        labels: NDArray
    ) -> Tuple[NDArray, torch.Tensor, Optional[int]]:
    
    # Transform for classification
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    transform_for_classifier = transforms.Compose(
        [transforms.Resize(N_PIXEL),
         transforms.ToTensor(),
         transforms.Normalize(data_mean, data_std)]
    )

    # Initialize label to None
    label = None

    # Image to display
    if random_image:
        # Select random image and associated label
        idx = np.random.randint(images.shape[0])
        image = images[idx, :, :, :]
        label = labels[idx] - 1

        # Transpose for display
        image_display = np.transpose(image, (2, 1, 0))

    elif uploaded_file:
        # Load file and process for display
        with Image.open(uploaded_file).convert('RGB') as image:
            image_display = image.resize((N_PIXEL, N_PIXEL))
            image_display = np.reshape(image_display,
                                       (N_PIXEL, N_PIXEL, N_CHANNELS))

    # Process image for classifier
    image_for_classifier = Image.fromarray(image_display)
    image_for_classifier = transform_for_classifier(image_for_classifier)
    image_for_classifier = torch.unsqueeze(image_for_classifier, 0)

    return image_display, image_for_classifier, label


def get_classes() -> list[str]:
    return ['plane', 'bird', 'car', 'cat', 'deer',
            'dog', 'horse', 'monkey', 'ship', 'truck']
