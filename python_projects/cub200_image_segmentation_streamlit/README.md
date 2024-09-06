# Python project - CUB-200 image segmentation using U-Net - Streamlit web application

## Introduction
This project's aims are twofold:
- segmenting 128x128 color images of birds using a custom U-Net.
- deploying the resulting model online using Streamlit.

Link to the Streamlit application: [https://stl10-image-classification-ar.streamlit.app](https://cub200-image-segmentation-ar.streamlit.app).

## Tools I used
This project was carried out using the following tools:
- **Python (Numpy, Matplotlib, Seaborn)** - backbone of the data analysis and visualization.
- **Python (Pytorch)** - machine learning toolbox.
- **Python (Streamlit)** - model deployment within online application. 
- **Jupyter Notebooks** - facilitating table and plot visualizations during the analysis.
- **Visual Studio Code** - my go-to code editor.
- **Git/Github** - essential for version control and code sharing.

## Model architecture

The model design, training and evaluation are described in the Jupyter Notebook [model_training/segmentation.ipynb](./model_training/segmentation.ipynb). The model architecture is structured as follows.

```python
class UnetDown(nn.Module):
    def __init__(self, input_size, output_size):
        super(UnetDown, self).__init__()
        
        model = [nn.BatchNorm2d(input_size),
                 nn.ELU(),
                 nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(output_size),
                 nn.ELU(),
                 nn.MaxPool2d(2),
                 nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):        
        return self.model(x)
      

class UnetUp(nn.Module):
    def __init__(self, input_size, output_size):
        super(UnetUp, self).__init__()

        model = [nn.BatchNorm2d(input_size),
                 nn.ELU(),
                 nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(output_size),
                 nn.ELU(),
                 nn.Upsample(scale_factor=2, mode="nearest"),  # Counterpart of the MaxPool2d
                 nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1)]
          
        self.model = nn.Sequential(*model)
            
    def forward(self, x):
        return self.model(x)
            
         
class Unet(nn.Module):
    def __init__(self, channels_in, channels_out=2):
        super(Unet, self).__init__()
        
        self.conv_in = nn.Conv2d(channels_in, 64, 
                                 kernel_size=3, stride=1, padding=1)  # H x W --> H x W
        
        self.down1 = UnetDown(64, 64)  # H x W --> H/2 x W/2
        self.down2 = UnetDown(64, 128)  # H/2 x W/2 --> H/4 x W/4
        self.down3 = UnetDown(128, 128)  # H/4 x W/4 --> H/8 x W/8
        self.down4 = UnetDown(128, 256)  # H/8 x W/8 --> H/16 x W/16

        # The "*2"s below come from the skip-connection concatenations
        self.up4 = UnetUp(256, 128)  # H/16 x W/16 --> H/8 x W/8
        self.up5 = UnetUp(128*2, 128)  # H/8 x W/8 --> H/4 x W/4
        self.up6 = UnetUp(128*2, 64)  # H/4 x W/4 --> H/2 x W/2
        self.up7 = UnetUp(64*2, 64)  # H/2 x W/2 --> H x W
        
        self.conv_out = nn.Conv2d(64*2, channels_out, 
                                  kernel_size=3, stride=1, padding=1)  # H x W --> H x W

    def forward(self, x):
        x0 = self.conv_in(x)  # 64 x H x W
        
        x1 = self.down1(x0)  # 64 x H/2 x W/2
        x2 = self.down2(x1)  # 128 x H/4 x W/4
        x3 = self.down3(x2)  # 128 x H/8 x W/8
        x4 = self.down4(x3)  # 256 x H/16 x W/16

        # Bottleneck --> 256 x H/16 x W/16

        x5 = self.up4(x4)  # 128 x H/8 x W/8
        
        x5_ = torch.cat((x5, x3), 1)  # 256 x H/8 x W/8 (skip connection)
        x6 = self.up5(x5_)  # 128 x H/4 x W/4
        
        x6_ = torch.cat((x6, x2), 1)  # 256 x H/4 x W/4 (skip connection)
        x7 = self.up6(x6_)  # 64 x H/2 x W/2
        
        x7_ = torch.cat((x7, x1), 1)  # 128 x H/2 x W/2 (skip connection)
        x8 = self.up7(x7_)  # 64 x H x W
        
        x8_ = F.elu(torch.cat((x8, x0), 1))  # 128 x H x W        
        return self.conv_out(x8_)  # channels_out x H x W
```

Details regarding the model training and evaluation can be found in the aforementioned Jupyter notebook. Note that the `ModelTrainer` and `CUB200` classes used in this project were taken and adapted from [Luke Ditria's GitHub repository](https://github.com/LukeDitria/pytorch_tutorials/tree/main/section08_detection/solutions).

## Streamlit application

The resulting U-Net classifier was deployed online using Streamlit: see [https://stl10-image-classification-ar.streamlit.app](https://cub200-image-segmentation-ar.streamlit.app). The code underlying this application is located in the [streamlit_app folder](./streamlit_app/). Note the model and CUB-200 test images/labels were stored in Dropbox to be easily loaded while spinning the application without taking space on GitHub.

Whereas the main part of the Streamlit application is coded in [streamlit_app/app.py](./streamlit_app/app.py), various helper functions are defined in [streamlit_app/handle_model_data.py](./streamlit_app/handle_model_data.py), namely those appearing in this package importation.

```python
from handle_model_data import (download_all_files,
                               load_model,
                               load_images_masks,
                               get_relevant_images_mask,
                               mask_intersection_over_union)
```

The Streamlit application is organized into three parts.

### Model and data (down)loading

The page starts with a description of the application, which gives time for the model and data to be downloaded from Dropbox while the user is reading this description. By "data", we refer to 500 randomly selected test images and their associated masks of the CUB-200 dataset (500 were selected for faster loading).

```python
# App title and description
st.set_page_config(layout='wide')
st.title('CUB-200-2011 Segmentation')

st.header('Description', divider='blue')
st.markdown('''
This web application coded using Streamlit enables you to...
...
...
''')

# Spin app, (down)loading relevant files
st.header('Loading', divider='blue')

model_path = './model.pt'
image_path = './test_images.pt'
mask_path = './test_masks.pt'
with st.spinner('Loading model, test images and masks...'):
    download_all_files(model_path, image_path, mask_path)
    model = load_model(model_path)
    images, masks = load_images_masks(image_path, mask_path)
st.success("Model, test images and masks loaded!")
```


### Image selection

The user can choose between:
1. uploading their own `.png` or `.jpg` image, which will be resized to 96x96 pixel resolution for classification.
2. select a random image from the 500 randomly selected CUB-200 test images and masks loaded in the background of this application.

Selecting the latter option allows for a direct evaluation of the model prediction, as the ground-truth label is known. Instead, the former option requires a visual assessment of the model prediction.

```python
# Choose the method for image input
st.header('Choose an image to classify', divider='blue')
col1, col2, col3 = st.columns([0.4, 0.5, 0.1], gap='large')
with col1:
    st.markdown('- Upload your own .jpg or .png image')
    uploaded_file = st.file_uploader('Classify your own .jpg or .png image',
                                      type=['jpg', 'png'],
                                      label_visibility='collapsed')
with col2:
    st.markdown('- Randomly select an image from the CUB-200 test dataset')
    random_image = st.button('Select random test image')
```


### Segmentation and visualization

Regardless of the option chosen above, the image of interest is displayed and segmented using the U-Net model. The user can see:
- the displayed image.
- the predicted bird segmentation.
- if the ground-truth mask is known (_i.e._, the user chose to use a random CUB-200 test image/mask):
    - the ground-truth bird segmentation.
    - a comparison of the ground-truth and predicted segmentation masks.
    - the value of the intersection over union of the aforementioned masks.

```python
# Segmentation
if random_image or uploaded_file:
    # Get relevant images and mask
    image_display, image_for_model, mask \
        = get_relevant_images_mask(random_image=random_image,
                                   uploaded_file=uploaded_file,
                                   images=images,
                                   masks=masks)

    # Run segmentation
    st.header('Segmentation', divider='blue')

    model.eval()
    with torch.no_grad():
        mask_pred = model(image_for_model).argmax(1).cpu().float().numpy()
        mask_pred = np.transpose(mask_pred, (1, 2, 0))

    if mask is not None:
        # Show intersection over union
        iou = mask_intersection_over_union(mask_pred, mask)
        st.subheader(f'Intersection over union: {100*iou:.2f}%')
        
    # Create display columns
    col1, col2 = st.columns(2)

    with col1:
        # Show image
        if mask is not None:
            st.markdown('**Randomly selected image**')
        else:
            st.markdown('**Uploaded image (resized and center-cropped to 128 x 128)**')
        st.image(image=image_display, use_column_width=True)

        # Show ground-truth segmentation
        if mask is not None:
            st.markdown('**Segmented bird (ground-truth mask)**')
            st.image(image=image_display*mask, use_column_width=True)

    with col2:
        # Show predicted segmentation
        st.markdown('**Segmented bird (predicted mask)**')
        st.image(image=image_display*mask_pred, use_column_width=True)

        if mask is not None:
            # Compare the ground-truth and predicted segmentations
            comparison = np.zeros((np.prod(mask.shape), 3))
            overlap = (mask.ravel()>0)*(mask_pred.ravel()>0)
            gt_only = (mask.ravel()>0)*(np.logical_not(overlap))
            pred_only = (mask_pred.ravel()>0)*(np.logical_not(overlap))

            comparison[gt_only, 2] = 1.  # Blue
            comparison[overlap, 1] = 1.  # Green
            comparison[pred_only, 0] = 1.  # Red

            comparison = np.reshape(comparison, mask.shape[:2] + (3,))

            st.markdown('**Mask comparison: :green[overlap], :blue[ground truth only], :red[prediction only]**')
            st.image(image=comparison, use_column_width=True)
```
