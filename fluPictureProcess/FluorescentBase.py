import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3000000000



# 定义图片展示函数
ShowImageType = 1
def img_show(name, img):
    if ShowImageType == 0:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        plt.imshow(img)
        plt.title(name)
        plt.show()



class FluorescentBase():
    def __init__(self):
        pass


    # Picture enhancement
    # 图片增强处理
    ###### black-white为性线变换
    # black：比black小的都为0
    # white：比white大的都为255
    # black-white间进行线性变换，即black以下为0，white以上为255，中间值X的变换值=255*(X-2)/(white-black)
    ###### gamma为伽玛变换，为非线性变换
    # 像素值X的变换值=(X/255)**(1/gamma)*255
    # gamma大于1会让较暗的值进行增强
    # 处理灰度，即只处理一个通道
    @staticmethod
    def gray_enhance(gray, black=2, white=70, gamma=1.69):
        gray = np.where(gray<black, black, gray)
        gray = np.where(gray>white, white, gray)
        gray = np.power( (gray - black)/(white-black), 1.0/gamma)
        gray = np.round(gray*255.0)
        gray = np.array(gray,dtype='uint8')
        return gray
    # 处理RGB，即处理3个通道
    @staticmethod
    def RGB_enhance(img, black=[2, 2, 2], white=[70, 70, 70], gamma=[1.69, 1.69, 1.69]):
        b, g, r= cv2.split(img)
        b = FluorescentBase.gray_enhance(b, black=black[0], white=white[0], gamma=gamma[0])
        g = FluorescentBase.gray_enhance(g, black=black[1], white=white[1], gamma=gamma[1])
        r = FluorescentBase.gray_enhance(r, black=black[2], white=white[2], gamma=gamma[2])
        merged = cv2.merge([b, g, r])
        return merged
        pass

    # 线性增强处理
    # 处理灰度，即只处理一个通道
    @staticmethod
    def gray_times_enhance(gray, black=2, times=20, gamma=1.69):
        gray = np.where(gray < black, 0, gray)
        gray = np.power(gray, gamma) * times
        gray = np.where(gray >= 255.0, 255.0, gray)
        gray = np.round(gray)
        gray = np.array(gray, dtype='uint8')
        return gray

    # 处理RGB，即处理3个通道
    @staticmethod
    def RGB_times_enhance(img, black=[2, 2, 2], times=[20, 20, 20], gamma=[1.69, 1.69, 1.69]):
        b, g, r = cv2.split(img)
        b = FluorescentBase.gray_times_enhance(b, black=black[0], times=times[0], gamma=gamma[0])
        g = FluorescentBase.gray_times_enhance(g, black=black[1], times=times[1], gamma=gamma[1])
        r = FluorescentBase.gray_times_enhance(r, black=black[2], times=times[2], gamma=gamma[2])
        merged = cv2.merge([b, g, r])
        return merged
        pass


    pass

    # 处理灰度，即只处理一个通道
    @staticmethod
    def gray_enhance_times(gray, black=2, white=70, gamma=1.69, time=10):
        gray = np.where(gray < black, black, gray)
        gray = np.where(gray > white, white, gray)
        gray = np.power((gray - black) / (white - black), 1.0 / gamma)
        gray = np.round(gray * 255.0 * time)
        gray = np.where(gray >= 255.0, 255.0, gray)
        gray = np.round(gray)
        gray = np.array(gray, dtype='uint8')
        return gray

    # 处理RGB，即处理3个通道
    @staticmethod
    def RGB_enhance_times(img, black=[2, 2, 2], white=[70, 70, 70], gamma=[1.69, 1.69, 1.69], times=[10,10,10]):
        b, g, r = cv2.split(img)
        b = FluorescentBase.gray_enhance_times(b, black=black[0], white=white[0], gamma=gamma[0], time=times[0])
        g = FluorescentBase.gray_enhance_times(g, black=black[1], white=white[1], gamma=gamma[1], time=times[1])
        r = FluorescentBase.gray_enhance_times(r, black=black[2], white=white[2], gamma=gamma[2], time=times[2])
        merged = cv2.merge([b, g, r])
        return merged
        pass



    # binary enhancement
    # 二值化增强
    # 即低于某个值的为0，高于或等于的为255
    @staticmethod
    def gray_binary_enhance(gray, cutoff):
        gray = np.where(gray<cutoff, 0, 255)
        gray = np.array(gray, dtype='uint8')
        return gray
    @staticmethod
    def RGB_binary_enhance(img, cutoff=[2, 2, 2]):
        b, g, r= cv2.split(img)
        b = FluorescentBase.gray_binary_enhance(b, cutoff=cutoff[0])
        g = FluorescentBase.gray_binary_enhance(g, cutoff=cutoff[1])
        r = FluorescentBase.gray_binary_enhance(r, cutoff=cutoff[2])
        merged = cv2.merge([b, g, r])
        return merged
    pass

    @staticmethod
    def quantile_threshold(color, quantile=99.9):
        cutoff = np.percentile(color, quantile)
        cutoff = int(cutoff)
        return cutoff

    @staticmethod
    def color_channel_pixel_hist(gray_img, color=[186, 85, 211]):
        # 像素计算
        val_count = np.zeros(256,dtype=int)
        for i in range(256):
            val_count[i] = np.sum(np.where(gray_img==i,1,0))
        # print(val_count)
        # 生成矩形图
        histImg = np.zeros([256, 256, 3], np.uint8)
        hpt = int(0.9 * 256);
        maxVal = np.max(val_count)

        for h in range(256):
            intensity = int(val_count[h] * hpt / maxVal)
            cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

        return histImg
        pass

    @staticmethod
    # 在X/Y轴方向对图片像素值进行求和
    # axis = 0: w方法
    # axis = 1: h方法
    def pixel_mapping(gray_img, axis, sliding_window_size=3):
        # X/Y映射求和
        pixel_sum = np.sum(gray_img, axis=axis)
        w_h = len(pixel_sum)
        # 滑动窗口法进行平滑处理
        win_sum = np.zeros(w_h, dtype='float32')
        for i in range(w_h):
            start = max(0, i - sliding_window_size)
            end = min(w_h - 1, i + sliding_window_size)
            win_sum[i] = np.sum(pixel_sum[start:(end + 1)])
        # 保存
        return win_sum
        pass





if __name__ == '__main__':
    # 读取图片信息
    img = cv2.imread('d:/delete/flusplit/ori_0_0.tif')
    # np_img = np.array(img,dtype='uint8')
    # img_en = FluorescentImgBase.RGB_enhance(img,black=[8,8,8],white=[30,30,30],gamma=[1.69,1.69,1.69])
    img_en = FluorescentBase.RGB_enhance(img,black=[8,8,8],white=[40,40,40],gamma=[2,2,2])
    # img_en = FluorescentImgBase.RGB_binary_enhance(img, cutoff=[9,9,9])
    img_show("img_en",img_en)
    img_show("img",img)
    pass


