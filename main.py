import cv2
import numpy as np
import pandas as pd



# DATA_PATH = 'C:/Users/ABC/Desktop/유동주 폴더/vision/Splited/train'
# picture_list = os.listdir(DATA_PATH)


def preprocess(src):
    image = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    width, height = image.shape[:2]
    pattern_value_list = []
    for i in range(height - 2):
        for j in range(width - 2):
            roi = image[i:i + 3, j:j + 3]
            threshold = roi[1, 1]
            roi[roi <= threshold] = 0
            roi[roi > threshold] = 1
            pattern_value_list.append(get_hist_num(roi))
    res = get_histogram(pattern_value_list)
    return res


def get_hist_num(roi):
    return roi[0, 1] * 128 + roi[0, 0] * 64 + roi[1, 0] * 32 + roi[2, 0] * 16 + roi[2, 1] * 8 + roi[2, 2] * 4 + roi[
        1, 2] * 2 + roi[0, 2] * 1


def get_histogram(pattern_value_list):
    counter = {x: 0 for x in range(0, 256)}
    for v in pattern_value_list:
        counter[v] += 1
    return np.array(list(counter.values()))



df = pd.read_csv('train.txt')


file_names = df['file_name']
labels = df['label']


print(file_names)
print(labels)