import cv2
import numpy as np
from copy import deepcopy

image = cv2.imread("images/flip_test.jpg", cv2.IMREAD_COLOR)
if image is None: raise Exception("영상 파일 읽기 오류 발생")  # 예외 처리


def flip(image, flip_code):
    h, w = image.shape[:2]
    image_cp = deepcopy(image)

    for i in range(h):
        for j in range(w):
            if flip_code == 'x':
                image_cp[i, j] = image[i, w - j - 1]
            elif flip_code == 'y':
                image_cp[i, j] = image[h - i - 1, j]
            elif flip_code == 'xy':
                image_cp[i, j] = image[h - i - 1, w - j - 1]
            else:
                raise Exception('제대로된 Flip Code를 입력해주세요')
    return image_cp


cv2.waitKey(0)
# x_axis = cv2.flip(image, 0)  # x축 기준 상하 뒤집기
# y_axis = cv2.flip(image, 1)  # y축 기준 좌우 뒤집기
# xy_axis = cv2.flip(image, -1)  # x,y 축으로 뒤집기
# rep_image = cv2.repeat(image, 1, 2)  # 반복 복사
# trans_image = cv2.transpose(image)  # 행렬 전치
#
# print(image.shape)
#
# # 각 행렬을 영상으로 표시
# titles = ['image', 'x_axis', 'y_axis', 'xy_axis', 'rep_image', 'trans_image']
# for title in titles:
#     cv2.imshow(title, eval(title))
import numpy as np
import cv2


def divide_channel(img):
    print(f"해당 이미지의 형상 = {img.shape}")  # 형상 확인
    width, height, channel = img.shape  # ( width, height, channel) 의 튜플로 저장되어 있음

    shape = ['width', 'height', 'channel']

    for s in shape:
        print(f"{s} : {eval(s)}")  # width, height, channel 출력

    return cv2.split(img)  # 각각 채널 나눠서 리턴


def remove_interpolation(channel, color):
    array = np.zeros((channel.shape[0], channel.shape[1]), np.uint8)

    if color == "blue":  # blue인 경우 홀수 행, 홀수 열의 값을 반환
        for row in range(1, channel.shape[0], 2):
            for col in range(1, channel.shape[1], 2):
                array[row][col] = channel[row][col]

    elif color == "green": # green의 경우 짝수 행 홀수열
        for row in range(0, channel.shape[0], 2):
            for col in range(1, channel.shape[1], 2):
                array[row][col] = channel[row][col]

        for row in range(1, channel.shape[0], 2): # 홀수행 짝수열
            for col in range(0, channel.shape[1], 2):
                array[row][col] = channel[row][col]

    elif color == "red": # red의 경우 짝수행 짝수열
        for row in range(0, channel.shape[0], 2):
            for col in range(0, channel.shape[1], 2):
                array[row][col] = channel[row][col]

    return array # 각각 보간을 제거한 행렬 리턴


if __name__ == "__main__":
    image = cv2.imread("images/color.jpg", cv2.IMREAD_COLOR)
    if image is None: raise Exception("영상 파일 읽기 에러")

    blue_channel, green_channel, red_channel = divide_channel(image)

    channels = ['blue_channel', 'green_channel', 'red_channel']

    print(" - " * 20 + " 채널 출력 " + " - " * 20)
    for channel in channels:
        print(f"{channel}\n{eval(channel)}")

    print("- " * 20 + " 보간 제거한 채널 " + " - " * 20)

    inter_blue = remove_interpolation(blue_channel, "blue")
    inter_green = remove_interpolation(green_channel, "green")
    inter_red = remove_interpolation(red_channel, "red")

    inters = ["inter_blue", "inter_green", "inter_red"]

    for inter in inters:
        print(f"{inter}\n{eval(inter)}")

    print("- " * 20 + " 보간 제거한 채널들 합치기 " + " - " * 20)
    bayer = cv2.merge((inter_blue, inter_green, inter_red))
    print(f"합친 영상의 형상 : {bayer.shape}")
    print(bayer)

    cv2.imwrite("images/last.jpg", bayer)

    cv2.namedWindow("BAYER IMAGE", cv2.WINDOW_NORMAL)
    cv2.namedWindow("NORMAL IMAGE", cv2.WINDOW_NORMAL)
    cv2.imshow("BAYER IMAGE", bayer)
    cv2.imshow("NORMAL IMAGE", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()