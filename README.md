# data-projects

This repository collects various personal projects aiming at enhancing my data-analysis and data-science skills. While each project is further detailed within their respective README file, please find below summaries of these projects and the main skills they involved.

## Python projects

### Data analysis

- [Data job market analysis](/python_projects/data_job_market_analysis/)
  - Extract insights regarding the 2023 data job market.
  - Focus on most demanded and high-paying roles/skills, monthly trends and the existence (or not) of most optimal skills to learn.
  - ![Static Badge](https://img.shields.io/badge/Pandas-blue) ![Static Badge](https://img.shields.io/badge/Matplotlib-darkgreen) ![Static Badge](https://img.shields.io/badge/Seaborn-darkgreen)
- [World population analysis](/python_projects/world_population_analysis/)
  - Take a quick look at the world population between 1970 and 2022.
  - Identify if certain continental populations have grown faster than others.
  - ![Static Badge](https://img.shields.io/badge/Pandas-blue) ![Static Badge](https://img.shields.io/badge/Matplotlib-darkgreen) ![Static Badge](https://img.shields.io/badge/Seaborn-darkgreen)
- [Customer call list cleanup](/python_projects/customer_call_list_cleanup/)
  - Clean up a small dataset to generate a list of customers that can be called by a fictitious commercial team.
  - ![Static Badge](https://img.shields.io/badge/Pandas-blue)

### Predictive machine learning

- [Bank customer churn prediction](/python_projects/bank_customer_churn_prediction)
  - Predict customer churn for a fictitious bank using various classical machine-learning classifiers.
  - Perform feature engineering and model hyperparameter tuning.
  - ![Static Badge](https://img.shields.io/badge/Scikit_Learn-orangered) ![Static Badge](https://img.shields.io/badge/Pandas-blue) ![Static Badge](https://img.shields.io/badge/Numpy-blue) ![Static Badge](https://img.shields.io/badge/Matplotlib-darkgreen) ![Static Badge](https://img.shields.io/badge/Seaborn-darkgreen)
- [NVIDIA stock price prediction](/python_projects/nvidia_stock_price_prediction/)
  - Predict the NVIDIA stock price using a Long Short-Term Memory (LSTM) neural network.
  - ![Static Badge](https://img.shields.io/badge/Pytorch-orangered) ![Static Badge](https://img.shields.io/badge/Pandas-blue) ![Static Badge](https://img.shields.io/badge/Numpy-blue) ![Static Badge](https://img.shields.io/badge/Matplotlib-darkgreen) ![Static Badge](https://img.shields.io/badge/Seaborn-darkgreen)

### Computer vision

- [Handwritten digit classification (classical)](/python_projects/handwritten_digit_classification)
  - Classify handwritten digits using dimensionality reduction (UMAP) and a classical machine-learning classifier.
  - ![Static Badge](https://img.shields.io/badge/Scikit_Learn-orangered) ![Static Badge](https://img.shields.io/badge/Numpy-blue) ![Static Badge](https://img.shields.io/badge/Matplotlib-darkgreen)
- [Handwritten digit classification (convolutional neural network)](/python_projects/handwritten_digit_classification_cnn)
  - Classify a bigger dataset of handwritten digits using a LeNet-5 convolutional neural network.
  - Investigate some feature maps.
  - ![Static Badge](https://img.shields.io/badge/Pytorch-orangered) ![Static Badge](https://img.shields.io/badge/Numpy-blue) ![Static Badge](https://img.shields.io/badge/Matplotlib-darkgreen)
- [CIFAR-10 image classification](/python_projects/cifar10_image_classification)
  - Classify color images into 10 classes using a rather deep convolutional neural network.
  - Compare the performance of:
    - a traditional convolutional network.
    - a convolutional network comprising residual connections (ResNet).
    - a convolutional network comprising skip connections.
  - ![Static Badge](https://img.shields.io/badge/Pytorch-orangered) ![Static Badge](https://img.shields.io/badge/Numpy-blue) ![Static Badge](https://img.shields.io/badge/Matplotlib-darkgreen) ![Static Badge](https://img.shields.io/badge/Seaborn-darkgreen)
- [STL-10 image-classification application using Streamlit](/python_projects/stl10_image_classification_streamlit)
  - Leverage unsupervised learning to train a rather deep residual-network (ResNet) classifier.
  - Deploy the resulting model online as a [Streamlit application](https://stl10-image-classification-ar.streamlit.app).
  - ![Static Badge](https://img.shields.io/badge/Streamlit-purple) ![Static Badge](https://img.shields.io/badge/Pytorch-orangered) ![Static Badge](https://img.shields.io/badge/Numpy-blue) ![Static Badge](https://img.shields.io/badge/Matplotlib-darkgreen) ![Static Badge](https://img.shields.io/badge/Seaborn-darkgreen)
- [Object detection](/python_projects/object_detection)
  - Detect objects and classify them into 20 categories using a region-based convolutional neural network (R-CNN).
  - Take advantage of transfer learning to do so.
  - ![Static Badge](https://img.shields.io/badge/Pytorch-orangered) ![Static Badge](https://img.shields.io/badge/Numpy-blue) ![Static Badge](https://img.shields.io/badge/Matplotlib-darkgreen)

## SQL projects

- [Data job market analysis](/sql_projects/data_job_market_analysis/)
  - Extract insights regarding the 2023 data-analyst job market.
  - Focus on high-paying jobs and most relevant skills.
  - ![Static Badge](https://img.shields.io/badge/PostgreSQL-gold)
- [Worldwide layoff analysis](/sql_projects/world_layoffs_analysis/)
  - Cleanup data and investigate layoffs that occurred around the world between March 2020 and March 2023.
  - ![Static Badge](https://img.shields.io/badge/PostgreSQL-gold)

