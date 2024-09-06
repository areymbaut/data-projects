import streamlit as st
import numpy as np
import torch

from handle_model_data import (download_all_files,
                               load_model,
                               load_images_masks,
                               get_relevant_images_mask,
                               mask_intersection_over_union)


# App title and description
st.set_page_config(layout='wide')
st.title('CUB-200-2011 Segmentation')

st.header('Description', divider='blue')
st.markdown('''
This web application coded using Streamlit enables you to segment bird images
using a convolutional neural network, namely a **:blue[U-Net]**. Note
that uploaded images will be automatically resized and center-cropped to a
resolution of 128 x 128 before segmentation. Trained on the CUB-200-2011 dataset,
the custom U-Net employed here reached an **:blue[accuracy of 83.4%]**, this accuracy
being evaluated as the average intersection over union (IoU) of the segmentation masks.

While the neural network was trained on bird images, nothing prevents you from
uploading images of other animals/objects and see what comes out of it.
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