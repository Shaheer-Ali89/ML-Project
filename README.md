# FashionMNIST Classification

## Team Members
- M. Shaheer Ali Khan (509801)  
- Kinza Ejaz (503623)  

## Abstract
This project compares classical machine learning models and deep learning techniques for image classification using the FashionMNIST dataset. An MLP-based encoder was used to extract features for classical models such as SVM and Random Forest, while a CNN was trained end-to-end on image data. Experimental results show that the CNN significantly outperformed classical models, demonstrating the effectiveness of deep learning for image-based tasks.

## Introduction
Image classification is a key problem in computer vision with applications in fashion, healthcare, and automation. Classical machine learning models depend on extracted features, which limits performance. Deep learning models, especially CNNs, automatically learn spatial features from images. This project aims to compare both approaches and analyze their performance on the same dataset.

## Dataset Description
The FashionMNIST dataset consists of 70,000 grayscale images of fashion items across 10 classes.  
- Images resized from 28×28 to 32×32 and converted to tensors  
- Split into 60,000 training and 10,000 testing samples  

## Methodology
1. Images were preprocessed and resized to 32×32  
2. An MLP encoder extracted 32-dimensional features  
3. Classical models (SVM and Random Forest) trained on extracted features  
4. Deep learning model (CNN) trained end-to-end  
5. Models evaluated using accuracy, precision, recall, and F1-score  

## Results & Analysis
- **SVM Accuracy:** ~21%  
- **CNN Accuracy:** ~91.7%
<img width="771" height="501" alt="image" src="https://github.com/user-attachments/assets/a0328c9d-c9c1-43f3-b855-0c16214a3052" />


Key observations:  
- CNN significantly outperformed classical models  
- CNN learned spatial features directly from images  
- Classical models were limited by weak feature representation  

## Conclusion & Future Work
**Conclusion:**  
CNN achieved superior performance compared to classical models, proving deep learning’s effectiveness for image classification tasks.

**Future Work:**  
- Use deeper CNN architectures (ResNet, VGG)  
- Apply data augmentation  
- Perform hyperparameter tuning  
- Extend to color images and larger datasets  

## References
- Xiao et al., *Fashion-MNIST Dataset*, 2017  
- LeCun et al., *Deep Learning*, Nature, 2015  
- Cortes & Vapnik, *Support Vector Machines*, 1995  
- Breiman, *Random Forests*, 2001  
- Paszke et al., *PyTorch*, 2019  
