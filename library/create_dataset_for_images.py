import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import glob


def load_and_resize_image(image_path, size=(32, 32)):
    img = Image.open(image_path).convert('RGB') # L for grayscale
    img = img.resize(size)
    return img

'''
Library for dog classification.
'''

def resize_dataset_dog(dataset_path, size = (32, 32)):

    for im_path in glob.glob(os.path.join(dataset_path, "*/*")):
        im = load_and_resize_image(im_path, size)
        label = im_path.split("/")[1]
        os.makedirs(os.path.join(dataset_path + "_resized", label), exist_ok=True)
        path_dir = im_path.replace(dataset_path, dataset_path + "_resized")
        im.save(path_dir)


def create_dataset_dog(dataset_path):
    dataset, labels = [], []
    for im_path in glob.glob(os.path.join(dataset_path, "*/*")):
        img = Image.open(im_path).convert('RGB')

        dataset.append(np.array(img).flatten())
        labels.append(im_path.split("/")[1])
    
    return dataset, labels


# resize_datase_dog("dog_emotion")
dataset, labels = create_dataset_dog("dog_emotion_resized")
print(type(dataset[0]))
print(dataset[0].shape)
pca = PCA(n_components=100)
new = pca.fit_transform(dataset)
print(new.shape)

'''
Library for fabric classification.
'''

def get_labels(im_path):
    tag_dir = os.path.join(*im_path.split("/")[:-1])

    with open(tag_dir + "/tag.txt", "r") as file:
        tag_label = file.read().lower()

    label = "cotton" if "cotton" in tag_label else "other"

    return label


def resize_dataset_fabric(dataset_path, size = (32, 32)):

    for im_path in glob.glob(os.path.join(dataset_path, "*/*/*")):

        if im_path.split(".")[-1] == "png":
            label = get_labels(im_path)
            im = load_and_resize_image(im_path, size)
            type_fabric = im_path.split("/")[1]
            sample_fabric = im_path.split("/")[2]
            img_name = im_path.split("/")[-1]
            os.makedirs(os.path.join(dataset_path + "_resized", type_fabric, sample_fabric), exist_ok=True)
            path_dir = im_path.replace(dataset_path, dataset_path + "_resized")
            path_dir = path_dir.replace(img_name, label + "_" + img_name)
            im.save(path_dir)


def create_dataset_fabric(dataset_path):
    dataset, labels = [], []
    for im_path in glob.glob(os.path.join(dataset_path, "*/*/*")):
        img = Image.open(im_path).convert('RGB')

        dataset.append(np.array(img).flatten())
        
        l = im_path.split("/")[-1]

        if "cotton" in l:
            labels.append("cotton")
        else:
            labels.append("other")
    
    return dataset, labels

# resize_dataset_fabric("fabric_dataset")
# dataset, labels = create_dataset_fabric("fabric_dataset_resized")
# print(dataset[0])
# print(labels[1])