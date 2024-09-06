# Python project - STL-10 image-classification application using Streamlit

## Introduction
This project's aims are twofold:
- classifying 96x96 color images into 10 classes using a convolutional neural network leveraging unsupervised learning for pretraining.
- deploying the resulting model online using Streamlit.

Link to the Streamlit application: [https://stl10-image-classification-ar.streamlit.app](https://stl10-image-classification-ar.streamlit.app).

## Tools I used
This project was carried out using the following tools:
- **Python (Numpy, Matplotlib, Seaborn)** - backbone of the data analysis and visualization.
- **Python (Pytorch)** - machine learning toolbox.
- **Python (Streamlit)** - model deployment within online application. 
- **Jupyter Notebooks** - facilitating table and plot visualizations during the analysis.
- **Visual Studio Code** - my go-to code editor.
- **Git/Github** - essential for version control and code sharing.

## Model design, training and evaluation

In this README, let us not go into details regarding the model training and evaluation (to better focus on Streamlit). That being said, the model design, training and evaluation are described in the Jupyter Notebook [model_training/unsupervised_learning.ipynb](./model_training/unsupervised_learning.ipynb).

In summary:
- The CNN classifier is based on the residual network with 18 layers (ResNet18) listed in the Pytorch model zoo (with randomly initialized weights).
- The model was pretrained using an unsupervised learning strategy leveraging randomly rotated images from the training and unlabeled STL-10 datasets.
- After training on the actual STL-10 dataset (without further data augmentation), the final classifier reached an accuracy of 69.2% on the test dataset.

## Streamlit application

The resulting CNN classifier was deployed online using Streamlit: see [https://stl10-image-classification-ar.streamlit.app](https://stl10-image-classification-ar.streamlit.app). The code underlying this application is located in the [streamlit_app folder](./streamlit_app/). Note the model and STL-10 test images/labels were stored in Dropbox to be easily loaded while spinning the application without taking space on GitHub.

Whereas the main part of the Streamlit application is coded in [streamlit_app/app.py](./streamlit_app/app.py), various helper functions are defined in [streamlit_app/handle_model_data.py](./streamlit_app/handle_model_data.py), namely those appearing in this package importation.

```python
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from handle_model_data import (download_all_files,
                               load_model,
                               load_images_labels,
                               get_relevant_images_label,
                               get_classes)
```

The Streamlit application is organized into three parts.

### Model and data (down)loading

The page starts with a description of the application, which gives time for the model and data to be downloaded from Dropbox while the user is reading this description. By "data", we refer to the test images and labels of the STL-10 dataset, stored in binary files.

```python
# App title and description
st.set_page_config(layout='wide')
st.title('STL-10 Web Classifier')

st.header('Description', divider='blue')
st.markdown('''
This web application coded using Streamlit enables you to...
...
...
''')

# Spin app, (down)loading relevant files
st.header('Loading', divider='blue')

model_path = './model.pt'
image_path = './test_X.bin'
label_path = './test_y.bin'
with st.spinner('Loading model, test images and labels...'):
    download_all_files(model_path=model_path,
                       image_path=image_path,
                       label_path=label_path)
st.success("Model, test images and labels loaded!")

model = load_model(model_path=model_path)
images, labels = load_images_labels(image_path=image_path,
                                    label_path=label_path)
```


### Image selection

The user can choose between:
1. uploading their own `.png` or `.jpg` image, which will be resized to 96x96 pixel resolution for classification.
2. select a random image from the STL-10 test dataset loaded in the background of this application.

Selecting the latter option allows for a direct evaluation of the model prediction, as the ground-truth label is known. Instead, the former option requires a visual assessment of the model prediction.

```python
# Choose the method for image input
st.header('Choose an image to classify', divider='blue')
col1, col2, col3 = st.columns([0.4, 0.4, 0.2], gap='large')
with col1:
    st.markdown('- Upload your own .jpg or .png image')
    uploaded_file = st.file_uploader('Classify your own .jpg or .png image',
                                      type=['jpg', 'png'],
                                      label_visibility='collapsed')
with col2:
    st.markdown('- Randomly select an image from the STL-10 test dataset')
    random_image = st.button('Select random test image')
```


### Classification and visualization

Regardless of the option chosen above, the image of interest is displayed and classified using the CNN model. The user can see:
- the displayed image.
- the distribution of class-specific scores (to better understand the model decision).
- if the ground-truth label is known (_i.e._, the user chose to use a random STL-10 test image/label):
    - some colored text indicating the classification success/failure.

```python
# Classification
classes = get_classes()
if random_image or uploaded_file:
    # Get relevant images and label
    image_display, image_for_classifier, label \
        = get_relevant_images_label(random_image=random_image,
                                    uploaded_file=uploaded_file,
                                    images=images,
                                    labels=labels)

    # Run classification
    st.header('Classification', divider='blue')

    model.eval()
    with torch.no_grad():
        predictions: torch.Tensor = model(image_for_classifier)
    scores = predictions.softmax(1)
    predicted_label: int = scores.argmax(1).item()

    if label is not None:
        # Show general classification result
        good_result = (predicted_label == label)
        if good_result:
            st.subheader(':green[Correct classification!]')
        else:
            st.subheader(':red[Incorrect classification...]')

    # Create display columns
    col1, col2 = st.columns(2)

    # Show image
    with col1:
        if label is not None:
            st.markdown('**Randomly selected image**')
            st.markdown(f'- True label - **{classes[label].capitalize()}**')
        else:
            st.markdown('**Uploaded image (resized to 96 x 96)**')
        st.image(image=image_display, use_column_width=True)
    
    # Show score distribution
    scores = 100*scores.cpu().numpy().ravel()
    idx_sort = np.argsort(scores)[::-1]

    fig, ax = plt.subplots()
    sns.barplot(x=scores[idx_sort], y=np.arange(len(classes)),
                orient='h', ax=ax, legend=False,
                hue=np.arange(len(classes)), palette='dark:b_r')
    ax.set_yticks(ax.get_yticks())  # To suppress a warning
    ax.set_yticklabels([classes[i] for i in idx_sort])
    ax.set_xlim([0, 100])
    ax.set_xlabel('Score (%)')
    
    with col2:
        st.markdown('**Score distribution**')
        st.markdown(f'- Predicted label - **{classes[predicted_label].capitalize()}**')
        st.pyplot(fig)
```