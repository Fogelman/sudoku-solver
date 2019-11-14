import cv2
import os
import numpy as np

from skimage import morphology, filters, color, measure
from tensorflow import keras

try:
    import pprint
except ImportError:
    pass  # module doesn't exist, deal with it.
    pprint = None


size = (2, 2)


def contrast(img):
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(15, 15))
    return clahe.apply(img)


def threshold(img):
    thresh = filters.threshold_local(img, 91, offset=18)
    return img > thresh


def load(filename, folder="sudoku"):
    basedir = os.path.join(os.getcwd(), folder, filename)
    return cv2.imread(basedir, cv2.IMREAD_GRAYSCALE)


def show(img, figsize=(8, 8), title=None):
    plt.figure(figsize=figsize)

    if(title):
        plt.title(title)
    plt.imshow(img, interpolation="none", cmap="gray")
    plt.show()


def outliers(data):
    arr = []
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)

    for y in data:
        z_score = (y - mean)/std
        if np.abs(z_score) <= threshold:
            arr.append(y)
    return arr


def select(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    min_area = q1 - (1.5 * iqr)
    max_area = q3 + (1.5 * iqr)

    return[min_area, max_area]


def empty(matrix, boxes):
    x = []
    for i in range(len(matrix)):
        line = ""
        for j in range(len(matrix[i])):
            index = matrix[i][j][0]
            if boxes[index]['filled'] != 1:
                line += "x"
            else:
                line += "-"
        x.append(line)
    return x


def predict(matrix, boxes, modelname):
    model_cnn = keras.models.load_model(modelname, compile=True)
    x = []
    for i in range(len(matrix)):
        line = ""
        for j in range(len(matrix[i])):
            index = matrix[i][j][0]
            if boxes[index]['filled'] < 1:
                image = ~boxes[index]['clean']
                h, w = image.shape
                w4 = int(w/3.5)
                h4 = int(h/3.5)
                cropped = image[h4:h-h4, w4:w-(w4)]
                resized = cv2.resize((cropped).astype(np.uint8), (28, 28))
                pred = model_cnn.predict_proba(
                    resized.reshape(1, 28, 28, 1).astype(np.float16))
                line += str(np.argmax(pred))
            else:
                line += "-"
        x.append(line)
    return x


def compare(x, y, blank=False, verbose=False):
    if len(x) != len(y):
        print('Size mismatch X & Y')
        return None
    errors = 0
    for i in range(len(x)):
        for j in range(len(x[0])):
            if len(x[i]) != len(y[i]):
                print('Size mismatch X & Y')
                return None

            if(blank):
                if(y[i][j] == "-" and x[i][j] != "-" or x[i][j] == "-" and y[i][j] != "-"):
                    errors += 1
            else:
                if(y[i][j] != x[i][j]):
                    errors += 1
                    if verbose:
                        print(
                            f"Esperado {y[i][j]} (y) e recebeu {x[i][j]} (x) na posição ({i}, {j})")
    return 1 - (errors/(len(x)*len(x[0])))


def digits(opening, areas, region):
    mask = np.zeros_like(opening)
    min_area, max_area = select(areas)
    boxes = {}
    count = 0
    for obj in region:
        area = obj["area"]

        if (area > min_area and area < max_area):
            bbox = obj.bbox
            mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] |= (obj.filled_image)
            boxes[obj.label] = {}
            boxes[obj.label]['mask'] = (
                opening[bbox[0]:bbox[2], bbox[1]:bbox[3]] & (obj.filled_image))
            boxes[obj.label]['clean'] = (
                boxes[obj.label]['mask'] | (~obj.filled_image))

            # Porcentagem de pixeis preenchidos em cada região
            boxes[obj.label]['filled'] = obj.area/obj.filled_area
            boxes[obj.label]['bbox'] = list(bbox)
            count += 1

    return boxes, mask


def sort(x):
    return sorted(x, key=lambda x: x[2])


def compute(filename):
    basedir = os.path.join(os.getcwd(), 'sudoku')
    img = load(filename+'.png')
    img_clahe = contrast(img)
    clean = threshold(img_clahe)
    opening = ~morphology.remove_small_objects(~clean, min_size=30)
    label = morphology.label(opening, connectivity=2)
    region = measure.regionprops(label)
    areas = []
    for obj in region:
        area = obj["area"]
        if area > 20:
            areas.append(area)
    boxes, mask = digits(opening, areas, region)
    points = [[x] + boxes[x]['bbox'] + [boxes[x]['filled']]
              for x in boxes.keys()]
    points = sorted(points, key=lambda y: y[1])

    if len(points) != 81:
        return None

    chunks = np.asarray(points).reshape(9, 9, 6)

    matrixM = map(sort, chunks)
    matrix = list(matrixM)
    x = predict(matrix, boxes, 'mnist.h5')

    try:
        with open(os.path.join(basedir, f'{filename}.txt')) as file:
            y = file.read().split()
    except Exception as e:
        return None

    return {"digits": compare(x, y), "x": x, "blanks": compare(x, y, blank=True)}


sudoku = os.path.join(os.getcwd(), 'sudoku')
files = [os.path.splitext(x) for x in os.listdir(sudoku)]


results = {}
for f in files:
    if f[1] == '.png':
        try:
            res = compute(f[0])
        except:
            res = None

        if res:
            results[f[0]] = res


if pprint:
    pprint.pprint(results)
else:
    print(results)
