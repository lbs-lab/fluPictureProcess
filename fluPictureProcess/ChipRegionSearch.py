import copy
import numpy as np
import cv2
import openslide
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3000000000
import re

from fluorescent.fluPictureProcess.MRXSBase import MRXSBase
from fluorescent.fluPictureProcess.ImageProcess import ImageProcess
from fluorescent.fluPictureProcess.FluorescentBase import FluorescentBase

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




class ChipRegionSearch():
    def __init__(self, mrxs_file):
        self.mrxs_base = MRXSBase(slide_file=mrxs_file)
        self.search_level = 3

        # init 5 img
        self.img_origin = None
        self.img_enhance = None
        self.img_gray = None
        self.img_binary = None
        self.img_dilate_erode = None
        self.img_chip_box = None

        # 提取图片
        nbox = [0, 0, self.mrxs_base.level_w_h[self.search_level, 0], self.mrxs_base.level_w_h[self.search_level, 1]]
        self.img_origin = self.mrxs_base.mrxs_crop(nbox, self.search_level)
        self.img_origin = self.img_origin[:,:,0:3]
        pass

    def search_chip_by_auto(self):
        self.search_set = self.auto_generate_search_set()
        self.reg_box = self.search_chip_by_semi_auto(search_set=self.search_set)
        return self.reg_box
        pass

    def search_chip_by_semi_auto(self, search_set):
        # 图片变换
        self.img_enhance = FluorescentBase.RGB_enhance(self.img_origin,black=search_set['blacks'],white=search_set['whites'],gamma=search_set['gammas'])
        self.img_gray = ImageProcess.gray_process(self.img_enhance)
        self.img_binary = ImageProcess.binary_process(self.img_gray, cutoff=search_set['binary_cutoff'])
        if isinstance(search_set['process_control_str'],list):
            self.img_dilate_erode = ImageProcess.dilate_erode(self.img_binary, process_control=search_set['process_control_str'])
        else:
            process_control = ChipRegionSearch.str_2_process_control(search_set['process_control_str'])
            self.img_dilate_erode = ImageProcess.dilate_erode(self.img_binary, process_control=process_control)

        # 识别芯片区域
        # 获取最小距形区域的4个顶点坐标
        reg_box = ChipRegionSearch.cal_min_rect_by_cv2(self.img_dilate_erode)
        print(f"reg_box:{reg_box.tolist()}")
        # 绘制最小外接矩形
        self.img_chip_box = self.img_enhance.copy()
        cv2.drawContours(self.img_chip_box, [reg_box], 0, (255, 255, 0), 3)

        # 保存并返回
        self.search_set = search_set
        # np array 2 list
        reg_box = reg_box*np.power(2,self.search_level)
        reg_box = list(map(lambda x: [int(x[0]),int(x[1])], reg_box))
        self.reg_box = reg_box
        return reg_box
        pass

    @staticmethod
    # 从预处理后的照片中提取最大的轮廓，然后基于最大轮廓计算最小的距形
    def cal_min_rect_by_cv2(binary_img):
        # 识别轮廓
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 获取最大的轮廓
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        # 获取最小的距形轮廓
        min_rect = cv2.minAreaRect(cnts[0])
        # print("min_rect:", min_rect)
        # 提取4个顶点，并转成整数
        rect_points = cv2.boxPoints(min_rect)
        rect_points = np.int0(rect_points)

        # 返回
        return rect_points
        pass

    def auto_generate_search_set(self):
        # init
        search_set = {}
        # black white and gamma
        search_set['gammas'] = [1.69, 1.69, 1.69]
        b, g, r= cv2.split(self.img_origin)
        blacks = [2, 2, 2]
        whites = [100, 100, 100]
        blacks[1] = FluorescentBase.quantile_threshold(g, quantile=10)
        blacks[2] = FluorescentBase.quantile_threshold(r, quantile=10)
        whites[1] = FluorescentBase.quantile_threshold(g, quantile=99.9)
        whites[2] = FluorescentBase.quantile_threshold(r, quantile=99.9)
        search_set['blacks'] = blacks
        search_set['whites'] = whites
        # img process
        search_set['binary_cutoff'] = 100
        search_set['process_control_str'] = [['dilate', 9], ['erode', 19], ['dilate', 7], ['erode', 20], ['dilate', 33]]
        #
        print('search_set:',search_set)
        return search_set
        pass

    @staticmethod
    def str_2_process_control(pc_str):
        process_control = []
        pc_str_all = pc_str.split(',')
        for pc in pc_str_all:
            if not re.search(':', pc):
                continue
            p, c = pc.split(':')
            process_control.append([p, int(c)])
        return process_control



    def show_img(self, type='chip_box', is_show=True):
        img = None
        if type == 'chip_box':
            img = self.img_chip_box
        elif type == 'dilate_erode':
            img = self.img_dilate_erode
        elif type == 'enhance':
            img = self.img_enhance
        elif type == 'binary':
            img = self.img_binary
        elif type == 'origin':
            img = self.img_origin
        else:
            img = self.img_gray
        # show
        if is_show:
            plt.imshow(img)
            plt.title(str(type))
            plt.show()

        pass







if __name__ == '__main__':
    # 读取图片信息
    flu_file = 'D:\\delete\\flu\\flu_8889_bc1_40X.mrxs'
    flu_file = 'D:\\delete\\flu\\flu_two_8889_bc1_40X.mrxs'



    # create
    crs = ChipRegionSearch(mrxs_file=flu_file)

    reg_box = crs.search_chip_by_auto()
    print('reg_box:', reg_box)

    img_show('img_enhance', crs.img_enhance)
    img_show('img_gray', crs.img_gray)
    img_show('img_binary', crs.img_binary)
    img_show('img_dilate_erode', crs.img_dilate_erode)
    crs.show_img(type='chip_box')

    pass


