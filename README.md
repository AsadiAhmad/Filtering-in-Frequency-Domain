# Filtering-in-Frequency-Domain
This project demonstrates how to simulate, visualize, and remove periodic noise from a grayscale image using Fourier Transform and frequency-domain filtering in Python. The method applies a custom-designed cross mask to suppress high-frequency noise components.

## Tech :hammer_and_wrench: Languages and Tools :

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/jupyter/jupyter-original.svg" title="Jupyter Notebook" alt="Jupyter Notebook" width="40" height="40"/>&nbsp;
  <img src="https://assets.st-note.com/img/1670632589167-x9aAV8lmnH.png" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp;
  <img src="https://github.com/AsadiAhmad/AsadiAhmad/blob/main/Logo/Python/math.png" title="Math" alt="Math" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="OpenCV" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Numpy" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/matplotlib/matplotlib-original.svg"  title="MatPlotLib" alt="MatPlotLib" width="40" height="40"/>&nbsp;
</div>

- Python : Popular language for implementing Neural Network
- Jupyter Notebook : Best tool for running python cell by cell
- Google Colab : Best Space for running Jupyter Notebook with hosted server
- Math : Essential Python library for basic mathematical operations and functions
- OpenCV : Best Library for working with images
- Numpy : Best Library for working with arrays in python
- MatPlotLib : Library for showing the charts in python

## 💻 Run the Notebook on Google Colab

You can easily run this code on google colab by just clicking this badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsadiAhmad/Filtering-in-Frequency-Domain/blob/main/Code/Filtering_in_Frequency_Domain.ipynb)

## 📝 Tutorial

### Step 1: Import Libraries

we need to import these libraries :

`cv2`, `numpy`, `math`, `matplotlib`, `itertools`

```python
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from itertools import chain
```

### Step 2: Download Resources

We need to download the Image for our work.

```python
!wget https://raw.githubusercontent.com/AsadiAhmad/Filtering-in-Frequency-Domain/main/Pictures/original_image.png -O original_image.png
```

### Step 3: Load Image

We need to load images into `python` variables we ues `OpenCV` library to read the images also the format of the images are `nd.array`.

```python
original_image = cv2.imread("original_image.png", cv2.IMREAD_GRAYSCALE)
```

<div display=flex align=center>
  <img src="/Pictures/original_image.png" width="400px"/>
</div>

### Step 4: Add Periodic Noise

in here we add periodic noise to our image.

```python
def f(x,y):
  return np.sin((1/2)*np.pi*x)+np.cos((1/3)*np.pi*y)
```

```python
X, Y = original_image.shape
noise = np.zeros((X, Y))
for i in range(X):
    for j in range(Y):
        noise[i,j] = f(i,j)*100

noisy_image = original_image + noise
```

<div display=flex align=center>
  <img src="/Pictures/noisy_image.png" width="400px"/>
</div>

### Step 5: Convert Image into the Frequency Domain

for removing periodic noise we need to transform our image into frequency domain.

```python
def convert_image_frequency(noisy_image):
    f_transform = np.fft.fft2(noisy_image)
    f_shifted = np.fft.fftshift(f_transform)
    return f_shifted
```

```python
f_shifted = convert_image_frequency(noisy_image)
```

<div display=flex align=center>
  <img src="/Pictures/frequency_domain.png" width="800px"/>
</div>

### Step 6: Define a cross filter mask

We define a filter mask for removing some sections that seems it's the periodic noise this sections can be identified as bright sections (not the center section it's for brightness of whole image).

```python
def define_cross_filter_mask(shape, circle_reduce):
    height, width = shape
    filter_mask = np.ones((height, width), dtype=np.uint8)
    rectangle_width = (width // 2) - circle_reduce
    rectangle_height = (height // 2) - circle_reduce

    for i in chain(range(0, rectangle_height), range(height - rectangle_height, height)):
        for j in range(rectangle_width, width - rectangle_width):
            filter_mask[i, j] = 0

    for i in range(rectangle_height, height - rectangle_height):
        for j in chain(range(0, rectangle_width), range(width - rectangle_width, width)):
            filter_mask[i, j] = 0

    return filter_mask
```

```python
mask = define_cross_filter_mask(f_shifted.shape, 2)
frequency_filtered = f_shifted * mask
```

```python
mask_visualization = mask * 255
magnitude_spectrum_filtered = magnitude_spectrum  * mask

plt.figure(figsize=[13, 6])
plt.subplot(121), plt.imshow(mask_visualization, cmap='gray'), plt.title('Mask'), plt.axis('off')
plt.subplot(122), plt.imshow(magnitude_spectrum_filtered, cmap='gray'), plt.title('Filtered Frequency'), plt.axis('off')
plt.show()
```

<div display=flex align=center>
  <img src="/Pictures/filter_mask.png" width="800px"/>
</div>

### Step 7: Inverse FFT

After removing the noise we need to convert the images into it's primary format (RGB).

```python
frequency_ishift = np.fft.ifftshift(frequency_filtered)
image_reconstructed = np.fft.ifft2(frequency_ishift)
image_reconstructed = np.abs(image_reconstructed)
```

<div display=flex align=center>
  <img src="/Pictures/reconstructed_comparison.png" width="800px"/>
</div>

### Step 8: Compare Original Image with Reconstructed Image

<div display=flex align=center>
  <img src="/Pictures/final_comparison.png" width="800px"/>
</div>

### Step 9: Quality Assurance with PSNR

```python
def psnr(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr_value
```

```python
original_float = original_image.astype(np.float32)
noisy_float = noisy_image.astype(np.float32)
reconstructed_float = image_reconstructed.astype(np.float32)
```

```python
psnr_noisy = psnr(original_float, noisy_float)
psnr_reconstructed = psnr(original_float, reconstructed_float)

print(f"PSNR (Original vs Noisy): {psnr_noisy:.2f} dB")
print(f"PSNR (Original vs Reconstructed): {psnr_reconstructed:.2f} dB")
```

PSNR (Original vs Noisy): 8.12 dB
PSNR (Original vs Reconstructed): 19.47 dB

## 🪪 License

This project is licensed under the MIT License.
