import copy
import numpy as np
import cv2
import openslide
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3000000000



class ImageProcess():
    def __init__(self):
        pass

    # 定义图片展示函数
    @staticmethod
    def img_show(name, img, ShowImageType=1):
        if ShowImageType == 0:
            cv2.imshow(name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            plt.imshow(img)
            plt.title(name)
            plt.show()


    # 灰度处理
    @staticmethod
    def gray_process(np_img):
        # 灰度处理
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        return gray


    # 二值化处理
    @staticmethod
    def binary_process(np_gray, cutoff=100):
        # 转二值黑白图片腐蚀
        ret, binary = cv2.threshold(np_gray, cutoff, 255, cv2.THRESH_BINARY)
        return binary


    # 膨胀图像与腐蚀图像
    # process_control是一个2维的list，保存处理方式
    # 如：process_control = [['dilate',9],['erode',19],['dilate',7],['erode',20],['dilate',27]]
    @staticmethod
    def dilate_erode(np_binary, process_control, kernel_size=7):
        # 图片膨胀与腐蚀
        # 定义核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        curr_np_img = np_binary.copy()
        for pc in process_control:
            # 膨胀图像
            if pc[0] == 'dilate':
                curr_np_img = cv2.dilate(curr_np_img, kernel, iterations=pc[1])
            # 腐蚀图像
            else:
                curr_np_img = cv2.erode(curr_np_img, kernel, iterations=pc[1])
        # return
        return curr_np_img



if __name__ == '__main__':
    # 读取图片信息
    img = cv2.imread('d:/delete/flusplit/ori_0_0.tif')
    ImageProcess.img_show('img', img)

    gray = ImageProcess.gray_process(img)
    ImageProcess.img_show('gray', gray)

    binary = ImageProcess.binary_process(gray, cutoff=100)
    ImageProcess.img_show('binary', binary)

    process_control = [['dilate', 9], ['erode', 19], ['dilate', 7], ['erode', 20], ['dilate', 27]]
    dilate_erode = ImageProcess.dilate_erode(binary, process_control)
    ImageProcess.img_show('dilate_erode',dilate_erode)

    pass


