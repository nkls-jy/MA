import os
import numpy as np
#import matplotlib.pyplot as plt
import rasterio
import torch
import sys
sys.path.append('.')

#import dataloader as dl

train_path = ".\\sets\\train\\"

file = rasterio.open(train_path + "000000000023.tif")
arr = file.read()
print(arr.shape)
print(type(arr))

#train_path = ".\\sets\\train_bands\\tile0.tiff"
#train_path = r'.\sets\train_bands\tile0.tiff'
'''
train_path = r"T:\Jaggy\Masterarbeit\MA\sets\train_bands\tile1.tif"

#load image
image = gdal.Open(train_path)

arr = np.array(image.ReadAsArray())

size = 25

y1 = np.random.randint(0, arr.shape[1] - size)
x1 = np.random.randint(0, arr.shape[2] - size)

y2 = y1 + size
x2 = x1 + size

arr_c = arr[:, y1:y2, x1:x2]
print(arr_c.shape)

t = torch.from_numpy(arr_c.copy(order='C'))

preprocess = dl.PreprocessingIntNoisyFromAmp()

noisy, target = preprocess(t)

print(noisy.shape)
print(target.shape)
#print(f'y1: {y1}, x1: {x1}')


#noisy = arr[0]
#target = arr[1]
'''
'''
path = r"T:\Jaggy\Masterarbeit\MA\sets\train_bands\\"
fig = plt.figure(figsize=(8,8))

i=1
for filename in os.listdir(path):
    print(filename)
    if filename.endswith('.tif'):
        img = gdal.Open(path + filename)
        arr = np.array(img.ReadAsArray())
        print(arr.shape)
        #fig.add_subplot(1, i, 1)
        #plt.imshow(arr[0])
        #plt.show()
        i += 1

#plt.show()

#print(image.GetMetadata())
#print(image.RasterCount)
# width
#print(image.RasterXSize)

# height
#print(image.RasterYSize)



dataset = gdal.Open(train_path + "tile0.tif")
image = dataset.ReadAsArray()

# extract bands
#noisy = image[0]
#target = image[1]

#print(type(image))
#print(image.shape)

#plt.plot(image)
#plt.shop()
'''

'''
train_path = ".\\sets\\train\\"
valid_path = ".\\sets\\valid\\"

def image_filter(filename):
    from torchvision.datasets.folder import IMG_EXTENSIONS
    filename_lower = filename.lower()
    print(filename_lower)
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def find_files(dir, filter=None):
    images = list()
    if filter is None:
        filter = lambda x: True

    #print(dir)
    for fname in sorted(os.listdir(dir)):
        if filter(fname):
            images.append(os.path.join(dir, fname))
            print(images)
    return images
find_files(train_path, image_filter)
'''

# Tensor Slicing Testing

'''
 # create an 3 D tensor with 8 elements each
a = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8],
                   [10, 11, 12, 13, 14, 15, 16, 17]],
                    
                  [[71, 72, 73, 74, 75, 76, 77, 78],
                   [81, 82, 83, 84, 85, 86, 87, 88]]])
  
# display actual  tensor
#print(a)
print(a.shape)
#print(torch.split(a, 1, dim=0))
tensorList = torch.split(a, 1, dim=0)
t1 = tensorList[0]
t2 = tensorList[1]

print(f't1: {t1}')
print(f't1 shape: {t1.shape}')
print(f't2: {t2}')
'''
