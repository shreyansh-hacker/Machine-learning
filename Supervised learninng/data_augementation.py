import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Dataset load karte hain
df = pd.read_csv('fashion-mnist_train.csv')

# 2. Pehli image ka data nikalte hain (Label ko chhodkar baki 784 pixels)
# Fashion MNIST ki images 28x28 pixels ki hoti hain
image_data = df.iloc[0, 1:].values.astype('uint8').reshape(28, 28)
label = df.iloc[0, 0]

# 3. NumPy array ko Image mein badalte hain taaki hum changes kar sakein
img = Image.fromarray(image_data)

# 4. Data Augmentation Apply karte hain
# - Flip karna (Palatna)
img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)

# - Rotate karna (15 degree ghumana)
img_rotate = img.rotate(15)

# - Thoda shift (translate) karna
# (Isme hum left-right shift kar rahe hain)
from PIL import ImageChops
img_shift = ImageChops.offset(img, 3, 0)

# 5. Sabhi images ko ek sath dikhane ke liye plot banate hain
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("1. Original Image")
axes[0].axis('off')

axes[1].imshow(img_flip, cmap='gray')
axes[1].set_title("2. Flipped")
axes[1].axis('off')

axes[2].imshow(img_rotate, cmap='gray')
axes[2].set_title("3. Rotated (15°)")
axes[2].axis('off')

axes[3].imshow(img_shift, cmap='gray')
axes[3].set_title("4. Shifted")
axes[3].axis('off')

plt.tight_layout()
plt.savefig('augmented_fashion.png')
print("Augmentation complete and image saved!")