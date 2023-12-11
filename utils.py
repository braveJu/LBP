import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

HP = {
    'learning_rate': 0.001,
    'batch_size': 1,
    'epochs': 5,
    'output_size' : 11,
}

LABEL_DICT = {
    0: 'KTH_aluminkum_foil',
    1: 'KTH_brown_bread',
    2: 'KTH_corduroy',
    3: 'KTH_cork',
    4: 'KTH_cotton',
    5: 'KTH_cracker',
    6: 'KTH_linen',
    7: 'KTH_orange_peel',
    8: 'KTH_sponge',
    9: 'KTH_styrofoam',
    10: 'KTH_wool',
}


epsilon = 0.00000000001

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
    return np.array(list(counter.values())).astype(np.float16)


def onehot_encoding(idx, length=11):
    arr = np.zeros(length)
    arr[idx] = 1.
    if idx >= length:
        raise "idx가 length를 넘어갑니다!"
    return torch.Tensor(arr)


def string_to_numpy_array(array_string):
    # Remove brackets and split the string by spaces
    array_elements = array_string.strip('[]').split()

    # Convert the string elements to floats and create a numpy array
    return np.array([float(element) for element in array_elements])

def get_KTH():
    # df = pd.read_csv('train.txt')
    # X = []
    # for f in tqdm(df['file_name']):
    #     X.append(preprocess(f))
    # y = df['label']
    # df2 = pd.DataFrame({'X':X, 'Y':y})
    # df2.to_csv('KTH_df.csv')
    df = pd.read_csv('KTH_df.csv')
    df = df.drop('Unnamed: 0', axis=1)
    X = [string_to_numpy_array(x)+epsilon for x in df['X']]
    y = df['Y']
        
    return train_test_split(X,y, test_size=0.3, random_state=42, shuffle=True)