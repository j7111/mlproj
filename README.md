# Machine Learning Projects

## Description

This repository contains a collection of projects that I have developed to deepen my understanding of various machine learning concepts and their mathematical foundations. While many of these projects replicate the functionality of established libraries like scikit-learn or PyTorch, they have been instrumental in enhancing my comprehension of the inner workings of these models.

## Purpose

The primary objective of these projects is educational. The focus is on constructing and exploring models to gain insights, rather than on minimizing errors or fine-tuning for optimal performance. Consequently, these tasks are often not prioritized.

## Project Structure

- From Scratch Implementations: Some projects involve building models entirely from scratch to understand their mechanics thoroughly.
- Hybrid Approaches: In other projects, while certain elements are developed from the ground up (such as the U-Net architecture), others rely on pre-existing components (like convolutional neural networks).

## Model Comparison

For projects developed from scratch, comparisons are typically made with standard implementations from popular machine learning libraries (mainly scikit-learn). These comparisons demonstrate that while custom models may match the performance of standard models, they may lack additional features and optimizations.

# Projects

- `unet.ipynb`: Given a convolutional neural network, we develop a u-net architecture from scratch, following the original research paper, using TensorFlow and Keras API. We also preprocess a batch of labeled medical images with techniques such as normalizatoin, transformation, and augmentation to increase the size of the dataset. We train the model for image segmentation, and evaluate its performance via loss and accuracy functions.

- `time-series.ipynb`: We create a long short-term memory (LTSM) recurrent neural network using PyTorch and train it on seasonal production data to forecast future data. This is then compared to a traditional ARIMA model, using root mean squared, mean absolute, and mean absolute percentage errors.

- `knn.ipynb`: Using NumPy to create random labeled clusters, we implement a KNN algorithm from scratch, using the underlying mathematical principles. This is later compared to scikit-learn's implementation of KNN.

- `kmeans.ipynb`: Using unlabeled clusters, we implement a k-Means clustering algorithm from scratch, again from the underlying mathematical principles. This is an unsupervised learning technique to cluster data points based on their distance from centroids. Our implementation is also compared to scikit-learn's.

- `naive_bayes.ipynb`: We build a naive Bayes classifier, utilizing the mathematics of Bayes' Theorem. We compare our results and accuracy scores to scikit-learn's standard model.

- `xgboost.ipynb`: Following along the mathematical foundations of standard gradient boosting + XGBoost, we developed a basic XGBoost model from scratch. This project emphasizes understanding the underlying mathematics rather than achieving optimal model performance, as demonstrated by comparisons with the standard XGBoost implementation."# decentralized-app" 
"# decentralized-app" 
"# mlproj" 
