import struct
import numpy as np
import torch.utils.data as data

# custom dataset class for loading the dataset
class mnsit_dataset(data.Dataset):
    def __init__(self, images, labels) -> None:
        super().__init__()
        self.images = np.expand_dims(images,1)
        self.labels = labels
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

# function to load and reshape the dataset
def load_mnist(imgs_path, labels_path):
    # reading the ubyte file using python packages
    with open(labels_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(imgs_path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(n, rows*cols)
    
    # reshape the 784 flattened image to 28x28
    images = images.reshape(-1,rows, cols)

    return images, labels, rows, cols

def load(train_img_filepath, train_label_filepath, test_img_filepath, test_label_filepath, BATCH_SIZE=64):
    train_images, train_labels, rows, cols = load_mnist(train_img_filepath, train_label_filepath)
    test_images, test_labels, rows, cols = load_mnist(test_img_filepath, test_label_filepath)

    train_images = train_images/255.0
    test_images = test_images/255.0

    train_dataset = mnsit_dataset(train_images, train_labels)
    test_dataset = mnsit_dataset(test_images, test_labels)

    # creating the dataloader for training and testing
    # 'Dataloader' handles randomization and batching
    train_loader = data.DataLoader(train_dataset,
            batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test_dataset,
            batch_size=BATCH_SIZE, shuffle=False)


    return train_loader, test_loader