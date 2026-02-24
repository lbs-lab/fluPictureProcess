import copy
import numpy as np
import cv2
import numpy.linalg as lg
import openslide
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3000000000

from fluorescent.fluPictureProcess.MRXSRegionExtract import MRXSRegionExtract
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


# 芯片简介
# 一个芯片有46*46个分区，定义为subarea
# 每个分区内有30*35个微珠，定义为microBead



class SplitFluImg():
    def __init__(self, np_img, wh_interval=[906, 913]):
        # 常量初始化
        # 每个小区域的大小, w表示水平方向，即31个中心孔距，h表示竖直方向，即36行的行高
        # numpy中axis=0表示numpy的y轴方向，即图片的w方向；axis=1表示numpy的x轴方向，即图片的h方向
        self.wh_interval = wh_interval
        # 滑动窗口大小，在向X/Y轴求和映射后，用滑动窗口法进行平滑处理
        # Sliding window size : 21个像素=10+1+10
        self.sliding_window_size = 10
        # 局部搜索最小值时的搜索半径，即搜索窗口大小为51像素=25+1+25
        self.local_search_radius = 25

        # 外界参数初始化
        # np_img为原始的荧光图片，未进行过任何处理
        self.np_img = np_img
        # 图片中第一个完整分区在芯片上的坐标
        # 坐标按图片的坐标系来定义，[x,y]，x为W方向, y为H方向
        self.subarea_leftTop = [0, 0]

        # 调用其它初始化函数
        # init
        # 荧光值校正：过滤噪音，信号强化等
        # black white and gamma
        gammas = [1.69, 1.69, 1.69]
        blacks = [2, 2, 2]
        whites = [100, 100, 100]
        blacks[1] = FluorescentBase.quantile_threshold(self.np_img[:,:,1], quantile=92.5)
        blacks[2] = FluorescentBase.quantile_threshold(self.np_img[:,:,2], quantile=92.5)
        whites[1] = FluorescentBase.quantile_threshold(self.np_img[:,:,1], quantile=99.9)
        whites[2] = FluorescentBase.quantile_threshold(self.np_img[:,:,2], quantile=99.9)
        print("blacks:",blacks)
        print("whites:",whites)
        # self.np_img_adjust = SplitFluImg.adjust_brightness(self.np_img,[2, 8, 8], [1.0, 1.0, 1.0], [20, 20, 20])
        self.np_img_adjust = SplitFluImg.adjust_brightness(self.np_img,blacks, gammas, whites)
        # 荧光图片进行灰度化与二值化处理
        self.__image_preprocessing()
        # X/Y求和映射
        # w方向
        self.w_pixel_sum = self.__pixel_mapping(axis=0)
        # h方向
        self.h_pixel_sum = self.__pixel_mapping(axis=1)
        # 在W/H搜索分割点
        # W方向
        self.w_split_pos = self.__search_split_pos(self.w_pixel_sum, self.wh_interval[0])
        # H方向
        self.h_split_pos = self.__search_split_pos(self.h_pixel_sum, self.wh_interval[1])
        # 计算中心孔距
        # 从W与H两个方向都求一下中心孔距，并取平均值
        self.center_dist = np.mean(np.diff(self.w_split_pos)) / 31.0
        self.center_dist += np.mean(np.diff(self.h_split_pos)) / 36.0 / (np.sqrt(3.0) / 2.0)
        self.center_dist = self.center_dist / 2.0
        pass


    def set_subarea_leftTop(self, subarea_leftTop):
        self.subarea_leftTop = subarea_leftTop

    def split_flu_img_into_dir(self, dir):
        # dir处理，去掉末尾的空白与斜线
        dir = dir.strip().strip('/').strip('\\')
        # 循环截取每个分区
        for w_i in range(len(self.w_split_pos)-1):
            for h_i in range(len(self.h_split_pos)-1):
                # box
                w_start = self.w_split_pos[w_i]
                w_end = self.w_split_pos[w_i+1]
                h_start = self.h_split_pos[h_i]
                h_end = self.h_split_pos[h_i+1]
                # 当前分区在chip上的位置
                chip_index_w = self.subarea_leftTop[0] + w_i
                chip_index_h = self.subarea_leftTop[1] + h_i
                # 输出原始荧光图片
                cut = self.np_img[h_start:h_end,w_start:w_end]
                outfile = f"{dir}/ori_{chip_index_w}_{chip_index_h}.tif"
                cv2.imwrite(outfile, cut)
                # 输出简单校正的荧光图片
                if chip_index_w%3==0 and chip_index_h%3==0:
                    cut = self.np_img_adjust[h_start:h_end, w_start:w_end]
                    outfile = f"{dir}/adjust_{chip_index_w}_{chip_index_h}.tif"
                    cv2.imwrite(outfile, cut)
        pass


    # 局部比较最小值
    # 注意先会指定最小值位置，这个值一般就是最小位置，只有当两边有值比这个更小时才更新，==是不会进行更新的
    def __local_min_val(self, vals, start_pos):
        w_h = len(vals)
        min_pos = start_pos
        min_val = vals[start_pos]
        search_width = self.local_search_radius
        for loc_i in range(max(0,start_pos - search_width), min(w_h, start_pos + search_width + 1), 1):
            if min_val > vals[loc_i]:
                min_val = vals[loc_i]
                min_pos = loc_i
        return min_pos
    def __search_split_pos(self, vals, step):
        # init
        w_h = len(vals)
        global_best_poss = []

        # 从中间求最小值的位置
        min_pos = round(0.8*step) + np.argmin(vals[round(0.8*step):(w_h-round(0.8*step))])
        global_best_poss.append(min_pos)

        # 向后进行搜索
        # 移动到下个位置附近
        pos = round(global_best_poss[len(global_best_poss) - 1] + step)
        while pos < w_h:
            # 局部搜索最小值
            min_pos = self.__local_min_val(vals,pos)
            # 保存
            global_best_poss.append(min_pos)
            # 移动到下个位置附近
            pos = round(global_best_poss[len(global_best_poss) - 1] + step)

        # 向前进行搜索
        # 移动到下个位置附近
        pos = round(global_best_poss[0] - step)
        while pos > 0:
            # 局部搜索最小值
            min_pos = self.__local_min_val(vals, pos)
            # 保存
            global_best_poss.insert(0, min_pos)
            # 移动到下个位置附近
            pos = round(global_best_poss[0] - step)

        # 修正最前面一个位置与最后一个位置
        global_best_poss = self.__correct_start_and_end_pos(vals,global_best_poss)

        return global_best_poss
        pass
    def __correct_start_and_end_pos(self, vals, global_best_poss):
        # init
        w_h = len(vals)
        # 对最后一个位置和最前面一个位置进行修正
        # 计算最优间距
        optimal_distance = round(np.mean(np.diff(global_best_poss[1:-1])))
        # 最后一个位置修正
        pos = global_best_poss[-1]
        if pos == w_h - 1:
            global_best_poss = global_best_poss[0:-1]
        else:
            if vals[pos] < vals[pos - 1] and vals[pos] < vals[pos + 1]:
                pass
            else:
                pos = global_best_poss[-2] + optimal_distance + 1
                if pos < w_h:
                    global_best_poss[-1] = pos
        # 最前面一个位置修正
        pos = global_best_poss[0]
        if pos == 0:
            global_best_poss = global_best_poss[1:]
        else:
            if vals[pos] < vals[pos - 1] and vals[pos] < vals[pos + 1]:
                pass
            else:
                pos = global_best_poss[1] - optimal_distance - 1
                if pos >= 0:
                    global_best_poss[0] = pos
        # return
        return global_best_poss

    # 在X/Y轴方向对图片像素值进行求和
    # axis = 0: w方法
    # axis = 1: h方法
    def __pixel_mapping(self, axis):
        # X/Y映射求和
        pixel_sum = np.sum(self.np_img_adjust_binary, axis=axis)
        w_h = len(pixel_sum)
        # 滑动窗口法进行平滑处理
        win_sum = np.zeros(w_h, dtype='float32')
        for i in range(w_h):
            start = max(0, i - self.sliding_window_size)
            end = min(w_h - 1, i + self.sliding_window_size)
            win_sum[i] = np.sum(pixel_sum[start:(end + 1)])
        # 保存
        return win_sum
        pass

    @staticmethod
    def pixel_hist_add_line(histImg, poss, color=[255, 255, 0]):
        histImg2 = copy.deepcopy(histImg)
        for pos in poss:
            cv2.line(histImg2, (pos, 256), (pos, 0), color)
        return histImg2
    @staticmethod
    def pixel_hist(val_sum, color=[186,85,211]):
        histImg = np.zeros([256, len(val_sum), 3], np.uint8)
        hpt = int(0.9 * 256);
        maxVal = max(val_sum)

        for h in range(len(val_sum)):
            intensity = int(val_sum[h] * hpt / maxVal)
            cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

        return histImg
        pass


    def __image_preprocessing(self):
        # 灰度处理
        gray = cv2.cvtColor(self.np_img_adjust, cv2.COLOR_RGB2GRAY)
        # 转二值黑白图片腐蚀
        ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # store
        self.np_img_adjust_binary = binary
        pass


    # def low_memory_adjust(color, black, white, gamma):
    #     color_uint8 = np.zeros(color.shape, dtype='uint8')
    #     for i in range(color.shape[0]):
    #         c = color[i,:]
    #         c = np.where(c < black, 0, c)
    #         c = np.power(c * 1.0 * white, gamma)
    #         c = np.minimum(c, 255)
    #         c = c.astype(np.uint8)
    #         color_uint8[i,:] = c
    #     return color_uint8
    #
    #     pass
    @staticmethod
    def low_memory_adjust(color, black, white, gamma, img, index):
        for i in range(color.shape[0]):
            c = color[i,:]
            c = np.where(c < black, 0, c)
            c = np.power(c * 1.0 * white, gamma)
            c = np.minimum(c, 255)
            c = c.astype(np.uint8)
            img[i,:,index] = c

    @staticmethod
    def adjust_brightness(img, black=[2, 2, 2], gamma=[1.69, 1.69, 1.69], white=[30, 30, 30]):
        print("split start")
        b, g, r, a = cv2.split(img)
        # r = np.where(r<black[0],0,r)
        # color = np.where(g < black[1], 0, g)
        # color = np.power(color * 1.0 * white[1], gamma[1])
        # color = np.minimum(color, 255)
        # color = color.astype(np.uint8)
        # g = color
        print("new start")
        new_np_img = np.zeros([r.shape[0], r.shape[1], 3], dtype='uint8')
        print("g start")
        SplitFluImg.low_memory_adjust(g,black=black[1], white=white[1], gamma=gamma[1], img=new_np_img, index=1)
        print("r start")
        SplitFluImg.low_memory_adjust(r,black=black[2], white=white[2], gamma=gamma[2], img=new_np_img, index=2)
        print("OVER")

        return new_np_img
        pass

    def calc_next_boxs(self, img_wh):
        w_len = len(self.w_split_pos)
        h_len = len(self.h_split_pos)
        # start index
        start_indexs = [[w_len-1, 0], [0, h_len-1], [w_len-1,h_len-1]]
        # next box
        w_sp = int(self.w_split_pos[-1] - self.wh_interval[0]/2.0)
        h_sp = int(self.h_split_pos[-1] - self.wh_interval[1]/2.0)
        next_boxs = [[w_sp, 0, img_wh[0]-1, int(img_wh[1]/2)], [0, h_sp, int(img_wh[0]/2), img_wh[1]-1], [w_sp, h_sp, img_wh[0]-1, img_wh[1]-1]]
        # return
        return start_indexs,next_boxs
        pass

    # @staticmethod
    # def cal_subarea_boxs(w_split_points, h_split_points):
    #     subarea_boxs = []
    #     # 左上角
    #     w_len = len(w_split_points)
    #     h_len = len(h_split_points)
    #     for w_i in range(w_len):
    #         for h_i in range(h_len):
    #
    #     pass

    @staticmethod
    def order_points(pts):
        # 初始化矩形4个顶点的坐标
        rect = np.zeros((4, 2), dtype='float32')
        # 坐标点求和 x+y
        s = pts.sum(axis=1)
        # np.argmin(s) 返回最小值在s中的序号
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # diff就是后一个元素减去前一个元素  y-x
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # 返回矩形有序的4个坐标点
        return rect

    @staticmethod
    def calc_next_boxs_16(img_wh, box_sets, start_indexs, w_splits, h_splits, curr_n, split_num):
        # init
        wh_interval = [906, 913]

        # calc wi hi
        w_i = curr_n%split_num
        h_i = int(curr_n/split_num)

        # calc wwwww
        if w_i == 0:
            w_start = 0
            w_end = int(img_wh[0] / split_num)
            index_w = 0
        else:
            w_start = int(box_sets[-1][0] + w_splits[-1][-1] - wh_interval[0]/2.0)
            w_end = int(img_wh[0] / split_num * (w_i+1))
            index_w = start_indexs[-1][0] + len(w_splits[-1])-1

        # calc hhhhh
        if h_i == 0:
            h_start = 0
            h_end = int(img_wh[1] / split_num)
            index_h = 0
        else:
            h_start = int(box_sets[-split_num][1] + h_splits[-split_num][-1] - wh_interval[1]/2.0) # -split_num为取上一行
            h_end = int(img_wh[1] / split_num * (h_i+1))
            index_h = start_indexs[-split_num][1] + len(h_splits[-split_num]) - 1 # -split_num为取上一行

        # update
        n_box = [w_start, h_start, w_end, h_end]
        n_index = [index_w, index_h]
        n_box = list(map(lambda x: int(x), n_box))
        n_index = list(map(lambda x: int(x), n_index))
        box_sets.append(n_box)
        start_indexs.append(n_index)
        pass

    @staticmethod
    def calc_center_dist(w_splits_sets, h_splits_sets):
        center_dist = []
        for spl in w_splits_sets:
            curr_dist = 1.0 * (spl[-2] - spl[1]) / (len(spl)-3) / 31.0
            center_dist.append(curr_dist)
        for spl in h_splits_sets:
            curr_dist = 1.0 * (spl[-2] - spl[1]) / (len(spl) - 3) / 36.0 / np.power(3,1/2) * 2.0
            center_dist.append(curr_dist)
        print('center_dist:',center_dist)
        return np.mean(center_dist)
        pass


def run_SplitFluImg(mrxs_file, chip_region, outdir):
    # init
    start_box_sets = []
    start_indexs_sets = []
    w_splits_sets = []
    h_splits_sets = []

    # 依据芯片区域提取图片
    print("init openslide")
    slide = openslide.OpenSlide(mrxs_file)
    print("new MRXSRegionExtract")
    rect = np.array(chip_region, dtype=int)
    rect = SplitFluImg.order_points(rect)
    rect = np.array(rect, dtype=int)
    mre = MRXSRegionExtract(rect, slide)

    wh_interval = [int(mre.img_wh[0]/46.0), int(mre.img_wh[1]/46.0)]
    print('wh_interval:',wh_interval)

    split_num = 4
    for i in range(split_num*split_num):
        # 获取下一个小块的位置
        SplitFluImg.calc_next_boxs_16(mre.img_wh, start_box_sets, start_indexs_sets, w_splits_sets, h_splits_sets, i, split_num)
        # 提取小块区域
        print(f"###{i}:extract one region")
        np_img = mre.extract_mrxs_region(start_box_sets[i])
        # 分割分区，并保存到目录中
        print(f"{i}:new SplitFluImg")
        sfi = SplitFluImg(np_img,wh_interval=wh_interval)
        print(f"{i}:split img")
        sfi.set_subarea_leftTop(start_indexs_sets[i])
        sfi.split_flu_img_into_dir(outdir)
        # 保存分割点信息
        sfi.w_split_pos = list(map(lambda x: int(x), sfi.w_split_pos))
        sfi.h_split_pos = list(map(lambda x: int(x), sfi.h_split_pos))
        w_splits_sets.append(sfi.w_split_pos)
        h_splits_sets.append(sfi.h_split_pos)
        print("start_box_sets",start_box_sets)
        print("start_indexs_sets",start_indexs_sets)
        print("w_splits_sets",w_splits_sets)
        print("h_splits_sets",h_splits_sets)

        w_hist = SplitFluImg.pixel_hist(sfi.w_pixel_sum)
        w_hist_line = SplitFluImg.pixel_hist_add_line(w_hist, sfi.w_split_pos)
        outfile = f"{outdir}/w_hist_line_{i}.tif"
        cv2.imwrite(outfile, w_hist_line)
        # print("sfi.w_split_pos=", sfi.w_split_pos)
        # img_show("w_hist_line", w_hist_line)

        h_hist = SplitFluImg.pixel_hist(sfi.h_pixel_sum)
        h_hist_line = SplitFluImg.pixel_hist_add_line(h_hist, sfi.h_split_pos)
        outfile = f"{outdir}/h_hist_line_{i}.tif"
        cv2.imwrite(outfile, h_hist_line)
        # print("sfi.h_split_pos=", sfi.h_split_pos)
        # img_show("h_hist_line", h_hist_line)

    # 构建字典并返回
    split_dict = {}
    split_dict['start_box_sets'] = start_box_sets
    split_dict['start_indexs_sets'] = start_indexs_sets
    split_dict['w_splits_sets'] = w_splits_sets
    split_dict['h_splits_sets'] = h_splits_sets
    split_dict['center_dist'] = SplitFluImg.calc_center_dist(w_splits_sets,h_splits_sets)
    return split_dict




if __name__ == '__main__':
    # 读取图片信息
    flu_file = "D:/delete/flu/chip_one_cycle_4.mrxs"
    # create
    rect = np.array([[2824,3840],[52424,3728],[52536,53608],[2944,53720]])
    # run
    # dict_vals = run_SplitFluImg(flu_file, rect, 'd:/delete/flusplit/')
    dict_vals = run_SplitFluImg(flu_file, rect, 'd:/delete/flusplit3/')
    print('dict_vals:', dict_vals)


    pass


