import os


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


