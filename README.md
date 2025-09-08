# Multimodal Image Fusion for Facial Analysis

This repository provides tools and algorithms for **image registration** and **multimodal image fusion**, using RGB and thermal facial images. It is designed for medical applications such as **remote photoplethysmography (rPPG)** and **facial skin anomaly detection**.

![Multimodal Image Fusion](images/reg1.png)

## Overview
This project implements image registration techniques to enable multimodal image fusion of **facial RGB and thermal image** data. It supports medical analysis by combining RGB and thermal imaging modalities and aligning them temporally. This repository implements three methods to register **facial RGB and thermal images** and align them temporally using timestamp differences:
1. **Enhanced Correlation Coefficient (ECC)**:
    - Robust alignment for global motion  
    - Fusion step is non-adaptive
2. **SimpleITK Registration Framework**
    - High-precision multimodal image registration
3. **Deep Learning-based Multimodal Image Fusion**
    - (In progress)
   
## Highlights
- **Multimodal Image Fusion**: Combines RGB and thermal images using ECC (OpenCV) and SimpleITK for enhanced feature extraction.
- **Temporal Alignment**: Matches RGB and thermal images across time frames using timestamp-based temporal difference calculations.
- **Evaluation Metrics**: Includes the **Structural Similarity Index (SSIM)** metric to evaluate the quality of image alignment and fusion.
- **Medical Applications**: Supports rPPG (e.g., heart rate extraction) and skin anomaly detection in facial image data.


## Repository Structure
```
├── RGB_thermal_Fusion.ipynb          # RGB-thermal image fusion using ECC (OpenCV) and SimpleITK
├── time_stamp_matching.ipynb         # Temporal alignment of RGB and thermal images using timestamps
├── images 
└── README.md   
```

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/usmanraza121/Multimodal-Image-Fusion.git
    cd Multimodal-Image-Fusion
    
    ```

2. **Create a virtual environment**:
   ```bash
   conda create -n mmfusion
   conda activate mmfusion
   ```

3. **Install dependencies**:
   Install required packages:
   ```bash
   pip install numpy opencv-python scikit-image SimpleITK matplotlib
   ```



## Image Registration
![Image Registration Example](images/fuse1.png)

## Temporal Matching
![Time-stamp mismatching example](images/diff.png)

![Time-stamp matching example](images/temp1.png)

![Time-stamp matching example](images/temp2.png)
