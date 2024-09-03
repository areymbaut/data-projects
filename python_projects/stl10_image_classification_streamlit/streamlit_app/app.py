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


# App title and description
st.set_page_config(layout='wide')
st.title('STL-10 Web Classifier')

st.header('Description', divider='blue')
st.markdown('''
This web application coded using Streamlit enables you to classify images using
a **:blue[convolutional neural network]** (CNN). Given this network was trained
on the so-called STL-10 dataset, images can only be classified as belonging to
one of the following classes: plane, bird, car, cat, deer, dog, horse, monkey,
ship, truck. Moreover, images will be automatically resized to a resolution of
96 x 96 before classification.
            
The CNN classifier is based on the **:blue[residual network with 18 layers]**
(ResNet18) listed in the Pytorch model zoo (with randomly initialized weights).
The model was **:blue[pretrained using an unsupervised learning strategy]**
leveraging randomly rotated images from the training and unlabeled STL-10 datasets.
After training on the actual STL-10 dataset (without further data augmentation),
the final classifier reached an **:blue[accuracy of 69.2%]** on the test dataset.
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