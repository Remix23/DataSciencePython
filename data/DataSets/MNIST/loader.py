import os.path
import numpy as np
from PIL import Image

def loadIDX (pathToIDX):
    if not os.path.exists(pathToIDX): raise (FileNotFoundError)

    with open(pathToIDX, mode = "rb") as f:
        magic = f.read(4)
        magic = np.frombuffer(magic, np.uint8)
        match magic[3]:
            case 8:
                data_type = np.uint8
            case 9:
                data_type = np.int8
            case 11:
                data_type = np.short
            case 12:
                data_type = np.int32
            case 13:
                data_type = np.float32
            case _:
                data_type = np.double


        dim = magic[3]
        dims = f.read(4 * dim)
        dims = np.frombuffer(dims, np.uint32)

        data = f.read(np.prod(dims).is_integer())

    data = np.frombuffer(data, data_type)
    data = data.reshape(dims)

    return dims

def showGreyIMG (data):
    im = Image.fromarray(data)
    im.show()

toInt = lambda x: int.from_bytes(x)

def convertLabel (x : int):
    return [int(x == i) for i in range(10)]

def convertLabels (x : np.ndarray) -> np.ndarray:
    return np.array([convertLabel(xi) for xi in x])

def loadImgs (path, normalise = True) -> tuple[np.ndarray, int]:

    if not os.path.exists(path): raise(FileNotFoundError)
    with open(path, mode = "rb") as f:
        magic_number = toInt(f.read(4))
        num_of_images = toInt(f.read(4))
        num_rows = toInt(f.read(4))
        num_cols = toInt(f.read(4))

        imgs = f.read(num_of_images * num_cols * num_rows)

    imgs = np.frombuffer(imgs, dtype = np.uint8)
    if normalise: imgs = imgs / 255
    imgs = imgs.reshape((num_of_images, num_rows * num_cols))

    return (imgs, num_of_images)

def loadLabels (path) :

    if not os.path.exists(path): raise(FileNotFoundError)

    with open(path, mode="rb") as f:
        magic_num = toInt(f.read(4))
        num_items = toInt(f.read(4))

        labels = f.read(num_items)

    labels = np.frombuffer(labels, dtype = np.uint8)

    return (labels, num_items)

def loadCombined (training = True, normalise = True) -> tuple[np.ndarray, np.ndarray]: ### imgs, labels:

    labels_path = "train-labels.idx1-ubyte"
    images_path = "train-images.idx3-ubyte"

    if not training:
        labels_path = "t10k-labels.idx1-ubyte"
        images_path = "t10k-images.idx3-ubyte"

    file_dir = os.path.dirname(os.path.abspath(__file__))
    labels_path = os.path.join(file_dir, labels_path)
    images_path = os.path.join(file_dir, images_path)

    labels = loadLabels (labels_path)[0]
    imgs = loadImgs (images_path, normalise = normalise)[0]

    return imgs, labels


if __name__ == "__main__":
    imgs, labels = loadCombined(False, normalise=True)


