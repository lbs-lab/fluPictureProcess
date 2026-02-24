import copy

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image,ImageDraw
Image.MAX_IMAGE_PIXELS = 3000000000

from fluorescent.fluPictureProcess.FluorescentBase import FluorescentBase
from fluorescent.fluPictureProcess.SplitFluImg import SplitFluImg


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

'''
荧光解码-荧光信号识别
1 文件格式定义
文件一共46X46X30X35行：46X46即芯片一共有46*46个区块，30X35即每个区块有35行每行有30个点。(顺序从左到右，从上到下)
每一行有18个数字，位置编码分别为0,1,2-17，即18位是荧光解码信息。
荧光编码：分别是0123456，即0-CY3，1-CY5，2-黑色，3无法判断但有可能是0/1，4无法判断但有可能是0/2，5无法判断但有可能是1/2，6无法判断但有可能是0/1/2。
文件名后缀:*.flu
2 文件案例
101221111012111000
123222111100005110
000011112621111000
222114100101010100
......
002011010122222222
'''


# 已知直线方程：y=kx+b,并已知一个点A(x0,y0)
# 求直线上距离点A为L的点的坐标B(xn,yn)
# xn = x0+-L/sqrt(1+k^2)
# yn = y0+-L|k|/sqrt(1+k^2)
# 注意k的正负号，如果k>0，则xn与yn同号，反之，则一个加，另一个减
class MicroBeadMaskDist():
    def __init__(self,center_dist=29.2):
        # 初始化microbead的属性
        self.center_dist = center_dist
        self.bead_radius = self.center_dist/4.8*2.7/2.0
        self.row_height = self.center_dist * np.sqrt(3.0)/2.0
        # microbead一共有35行，每行有30个
        self.std_cols = 30
        self.std_rows = 35
        # 设置mask的半径
        self.mask_radius = round(self.bead_radius * 1.1)

        # init
        self.calc_microbead_mask()
        # img_show("img",self.microbead_mask)

        # 单个点的信号提取mask
        self.single_mask = self.single_signal_mask()
        pass

    def calc_microbead_mask(self):
        self.mask_width = round(self.center_dist * (31.0 + 3.0))
        self.mask_height = round(self.row_height * (36.0 + 3.0))
        np_img_mask = np.zeros([self.mask_height, self.mask_width], dtype='uint8')
        cent = np.zeros([self.std_rows, self.std_cols, 2],dtype='float32')
        # 处理W
        for h_i in range(self.std_rows):
            h_pos = (h_i+2)*self.row_height
            for w_i in range(self.std_cols):
                w_pos = 0
                if h_i%2 == 0:
                    w_pos = (w_i*2 + 4.5)*self.center_dist/2.0
                else:
                    w_pos = (w_i*2 + 3.5) * self.center_dist / 2.0
                cent[h_i, w_i, :] = [h_pos, w_pos]
                cv2.circle(np_img_mask, (round(w_pos), round(h_pos)), self.mask_radius, 1, -1)

        self.cent_points = cent
        self.microbead_mask = np_img_mask
        self.microbead_mask_neg = 1- np_img_mask
        # print(self.microbead_mask[50:100,50:100])
        # print(self.microbead_mask_neg[50:100,50:100])
        pass

    def __search_perfect_mask_pos(self, np_img, w_start, w_range, h_start, h_range, step):
        max_val = -999999999999.9
        best_pos = [0, 0]
        curr_img = np.zeros([self.mask_height, self.mask_width], dtype='float32')
        for w_i in range(w_start, w_range, step):
            for h_i in range(h_start, h_range, step):
                curr_img.fill(0)
                curr_img[h_i:(h_i+np_img.shape[0]), w_i:(w_i+np_img.shape[1])] = np_img
                # curr_val = np.sum(curr_img*self.microbead_mask)-np.sum(curr_img*self.microbead_mask_neg)
                # curr_val = np.sum(np.power(curr_img,2.0)*self.microbead_mask)
                curr_val = np.sum(curr_img*self.microbead_mask)
                if max_val < curr_val:
                    max_val = curr_val
                    best_pos = [w_i, h_i]
        return best_pos,max_val
    def __disturbance_perfect_mask_pos(self, np_img, perfect_pos, perfect_val):
        dist_wh = [[-1,0],[1,0],[-0.5,-1],[0,-1],[0.5,-1],[-0.5,1],[0,1],[0.5,1]]
        # print("self.center_dist",self.center_dist)
        # print("perfect_pos,perfect_val",perfect_pos,perfect_val)
        for w_i, h_i in dist_wh:
            # print("wi,hi:",w_i, h_i)
            # print("perfect_pos", perfect_pos)
            w_pos = perfect_pos[0]+round(self.center_dist*w_i)
            h_pos = perfect_pos[1]+round(self.row_height*h_i)
            # print("wpos,hpos:",w_pos,h_pos)
            if w_pos < 0 or w_pos+np_img.shape[1]>=self.mask_width:
                continue
            if h_pos < 0 or h_pos+np_img.shape[0]>=self.mask_height:
                continue
            best_pos,max_val = self.__search_perfect_mask_pos(np_img, w_pos, w_pos+1, h_pos, h_pos+1, 1)
            # print("best_pos,max_val:",best_pos,max_val)
            if max_val > perfect_val:
                perfect_pos = best_pos
                perfect_val = max_val
        return perfect_pos
        pass
    def search_perfect_mask_pos(self, np_img):
        # first
        best_pos,perfect_val = self.__search_perfect_mask_pos(np_img,0,round(self.center_dist*1.5),0,round(self.row_height*1.5),3)
        # local
        w_start = max(0, best_pos[0] - 3)
        w_range = best_pos[0] + 3
        h_start = max(0, best_pos[1]-3)
        h_range = best_pos[1] + 3
        best_pos,perfect_val = self.__search_perfect_mask_pos(np_img, w_start, w_range, h_start, h_range, 1)
        best_pos = self.__disturbance_perfect_mask_pos(np_img, best_pos,perfect_val)
        return best_pos
    def search_perfect_mask_pos_linerearch(self, np_img, h_best_pos):
        # print("h_best_pos:",h_best_pos)
        # print("just pos:",2*self.row_height-h_best_pos)
        h_best_pos = round(2*self.row_height-h_best_pos-1)
        # first
        best_pos,perfect_val = self.__search_perfect_mask_pos(np_img,0,round(self.center_dist*1.5),h_best_pos,h_best_pos+1,3)
        # local
        w_start = max(0, best_pos[0] - 3)
        w_range = best_pos[0] + 3
        h_start = max(0, best_pos[1]-3)
        h_range = best_pos[1] + 3
        best_pos,perfect_val = self.__search_perfect_mask_pos(np_img, w_start, w_range, h_start, h_range, 1)
        # print("best_pos:",best_pos)
        best_pos_d = self.__disturbance_perfect_mask_pos(np_img, best_pos, perfect_val)
        if best_pos_d[0] != best_pos[0] or best_pos_d[1] != best_pos[1]:
            best_pos_d,perfect_val = self.__search_perfect_mask_pos(np_img, best_pos_d[0]-1, best_pos_d[0]+2, best_pos_d[1]-1, best_pos_d[1]+2, 1)
        # print("best_pos:",best_pos)
        return best_pos_d
    def search_perfect_pos(self,np_img):
        h_pixel_sum = FluorescentBase.pixel_mapping(np_img, axis=1, sliding_window_size=3)
        row_height = self.center_dist * np.sqrt(3) / 2.0
        h_len = len(h_pixel_sum)
        max_val = -1
        max_index = -1
        for i in range(int(h_len-row_height*34-5)):
            curr_val = 0
            for h_index in range(35):
                curr_val += h_pixel_sum[round(i+h_index*row_height)]
            if curr_val >= max_val:
                max_val = curr_val
                max_index = i
        #-----------------
        # h_hist = SplitFluImg.pixel_hist(h_pixel_sum)
        # color = [255, 255, 0]
        # for h_index in range(35):
        #     pos = round(max_index+h_index*row_height)
        #     cv2.line(h_hist, (pos, 256), (pos, 0), color)
        # outfile = f"d:/delete/flusplit5/h_hist.tif"
        # cv2.imwrite(outfile, h_hist)
        # print("max_index;",max_index)
        # -------------------
        # exit()
        return max_index
        pass


    def draw_perfect_mask_pos_gray(self, np_img, best_pos):
        new_img = np.zeros([self.mask_height, self.mask_width],dtype='uint8')
        new_img[best_pos[1]:(best_pos[1]+np_img.shape[0]), best_pos[0]:(best_pos[0]+np_img.shape[1])] = np_img
        for h_i in range(self.std_rows):
            for w_i in range(self.std_cols):
                [h_pos, w_pos] = self.cent_points[h_i,w_i,:]
                cv2.circle(new_img, (round(w_pos), round(h_pos)), self.mask_radius+1, 255, 1)
        return new_img
        pass
    def draw_perfect_mask_pos_rgb(self, np_img, best_pos):
        new_img = np.zeros([self.mask_height, self.mask_width, 3],dtype='uint8')
        new_img[best_pos[1]:(best_pos[1]+np_img.shape[0]), best_pos[0]:(best_pos[0]+np_img.shape[1]), :] = np_img
        for h_i in range(self.std_rows):
            for w_i in range(self.std_cols):
                [h_pos, w_pos] = self.cent_points[h_i,w_i,:]
                cv2.circle(new_img, (round(w_pos), round(h_pos)), self.mask_radius+1, [255,255,0], 1)
        return new_img
        pass

    @staticmethod
    def calc_auto_binary_cutoff(img):
        cutoff = [2,2,2]
        cutoff[1] = FluorescentBase.quantile_threshold(img[:,:,1], quantile=92.5)
        cutoff[2] = FluorescentBase.quantile_threshold(img[:,:,2], quantile=92.5)
        return cutoff
        pass

    @staticmethod
    def calc_auto_BGR_cutoff(img):
        blacks = [2, 2, 2]
        blacks[1] = FluorescentBase.quantile_threshold(img[:, :, 1], quantile=92.5)
        blacks[2] = FluorescentBase.quantile_threshold(img[:, :, 2], quantile=92.5)
        whites = [100,100,100]
        whites[1] = FluorescentBase.quantile_threshold(img[:, :, 1], quantile=99.9)
        whites[2] = FluorescentBase.quantile_threshold(img[:, :, 2], quantile=99.9)
        gammas = [1.69,1.69,1.69]
        return blacks,whites,gammas
        pass


    def single_signal_mask(self):
        self.single_mask_radius = int(self.mask_radius)+1
        mask = np.zeros((self.single_mask_radius*2+1,self.single_mask_radius*2+1), dtype='uint8')
        cv2.circle(mask, (self.single_mask_radius, self.single_mask_radius), self.single_mask_radius, 1, -1)
        # img_show("single", mask)
        # print(np.sum(mask))
        return mask
    def signal_extract(self, ori_img, best_pos, out_dir, area_cutoff):
        # 构建最佳位置的图片
        new_img = np.zeros([self.mask_height, self.mask_width, 3], dtype='uint8')
        new_img[best_pos[1]:(best_pos[1] + ori_img.shape[0]), best_pos[0]:(best_pos[0] + ori_img.shape[1]), :] = ori_img[:,:,0:3]
        # init一张局部点图，后面都用这个内存，不用得复申请
        local_img = np.zeros((self.single_mask_radius*2+1, self.single_mask_radius*2+1), dtype='uint8')
        # 迭代每个信号点，并进行信号提取
        cy5_cy3 = []
        for h_i in range(self.std_rows):        # 35 row
            for w_i in range(self.std_cols):    # 30 col
                cent = self.cent_points[h_i,w_i]
                h_pos = round(cent[0])
                w_pos = round(cent[1])
                h_start = h_pos - self.single_mask_radius
                h_end = h_pos + self.single_mask_radius + 1
                w_start = w_pos - self.single_mask_radius
                w_end = w_pos + self.single_mask_radius + 1
                # cy5
                local_img[:,:] = new_img[h_start:h_end,w_start:w_end,1]
                cy5 = np.sum(local_img*self.single_mask)
                cy5_area = np.sum(np.where(local_img*self.single_mask>area_cutoff[0],1,0))
                # cy3
                local_img[:, :] = new_img[h_start:h_end, w_start:w_end, 2]
                cy3 = np.sum(local_img * self.single_mask)
                cy3_area = np.sum(np.where(local_img*self.single_mask>area_cutoff[1],1,0))
                cy5_cy3.append([cy5,cy3,cy5_area,cy3_area])
                # save
                # outfile = f"{out_dir}/small_{h_i}_{w_i}.tif"
                # cv2.imwrite(outfile, new_img[h_start:h_end,w_start:w_end,0:3])

        return cy5_cy3
        pass

    # 保存信号到文件中
    @staticmethod
    def store_flu_signal_2_file(outfile, flu_signal):
        with open(outfile, "w") as f:
            subarea_count = 0
            for sets in flu_signal:
                sub_h_i = int(subarea_count / 46)
                sub_w_i = subarea_count % 46
                subarea_count += 1
                count = 0
                for single_sig in sets:
                    h_i = int(count / 30)
                    w_i = count % 30
                    f.write(f"{sub_h_i}\t{sub_w_i}\t{h_i}\t{w_i}")
                    for val in single_sig:
                        f.write(f"\t{val}")
                    f.write(f"\n")
                    count += 1

    '''
    荧光解码-荧光信号识别
    1 文件格式定义
    文件一共46X46X30X35行：46X46即芯片一共有46*46个区块，30X35即每个区块有35行每行有30个点。(顺序从左到右，从上到下)
    每一行有18个数字，位置编码分别为0,1,2-17，即18位是荧光解码信息。
    荧光编码：分别是0123456，即0-CY3，1-CY5，2-黑色，3无法判断但有可能是0/1，4无法判断但有可能是0/2，5无法判断但有可能是1/2，6无法判断但有可能是0/1/2。
    文件名后缀:*.flu
    2 文件案例
    101221111012111000
    123222111100005110
    000011112621111000
    222114100101010100
    ......
    002011010122222222
    '''
    # 整合18轮荧光解码信号到最终解码结果中
    @staticmethod
    def merge_18_cycle_signal(cycle_signal_dirs, outfile):
        # init
        merge_signals = np.zeros((46*46*30*35,18),dtype='uint8')
        min_signal_area = 100
        min_diff_times = 2.0
        # 迭代每个荧光信号文件
        for cycle_i in range(18):
            # 打开并读取一个文件
            sig_dir = cycle_signal_dirs[cycle_i]
            curr_sig_infile = f"{sig_dir}/signal_en.txt"
            with open(curr_sig_infile, 'r') as fp:
                list_sig = fp.readlines()
            # 逐条处理荧光信号
            for line_i in range(0, len(list_sig)):
                vals = list_sig[line_i].strip('\n').split('\t')
                cy5 = int(vals[-2])
                cy3 = int(vals[-1])
                # 识别荧光信号
                # 荧光编码：分别是0123456，
                # 即0-CY3，1-CY5，2-黑色，3无法判断但有可能是0/1，4无法判断但有可能是0/2，5无法判断但有可能是1/2，6无法判断但有可能是0/1/2
                sig_val = 0
                if cy3 >= min_signal_area:
                    if cy3/(cy5+0.1) >= min_diff_times: # add 0.1 是处理除以0的情况发生
                        sig_val = 0
                    else:
                        sig_val = 3
                else:
                    if cy5 >= min_signal_area:
                        if cy5/(cy3+0.1) >= min_diff_times: # add 0.1 是处理除以0的情况发生
                            sig_val = 1
                        else:
                            sig_val = 3
                    else:
                        sig_val = 2
                # store
                merge_signals[line_i,cycle_i] = sig_val
        # 保存识别后的荧光信号到文件中
        with open(outfile, 'w') as fp:
            for line_i in range(46*46*30*35):
                for cycle_i in range(18):
                    fp.write(f"{merge_signals[line_i,cycle_i]}")
                fp.write(f"\n")






def run_one_subarea_signal_regnize(center_dist,infile,outfile):
    mbm = MicroBeadMaskDist(center_dist)
    # 读取图片
    img = cv2.imread(infile)
    # 自动识别参数，并进行荧光信息加强
    blacks, whites, gammas = MicroBeadMaskDist.calc_auto_BGR_cutoff(img)
    img_en = FluorescentBase.RGB_binary_enhance(img, cutoff=blacks)

    # 灰度处理
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 查找最合适位置
    h_best_pos = mbm.search_perfect_pos(img_gray)
    # print("h_best_pos", h_best_pos)
    while((2*mbm.row_height-h_best_pos-1)<0):
        h_best_pos = round(h_best_pos-mbm.row_height)
        print(f'{"=" * 50}\nh_best_pos has error:{infile}\n{"=" * 50}\n')
    # print("h_best_pos", h_best_pos)
    # if round(2 * mbm.row_height - h_best_pos - 1) < 0:
    #     print("h_best_pos", h_best_pos)
    #     start = int(h_best_pos - mbm.row_height)
    #     img = img[start:, :]
    #     img_en = img_en[start:, :]
    #     img_gray = img_gray[start:, :]
    #     h_best_pos = int(mbm.row_height)
    #     print(f'{"=" * 50}\nh_best_pos has error:{infile}\n{"=" * 50}\n')
    best_pos = mbm.search_perfect_mask_pos_linerearch(img_gray, h_best_pos=h_best_pos)
    # print("best_pos2:", best_pos)
    # 生成最合适位置的可视化图片，并进行保存
    if True:
        best_img_flu = mbm.draw_perfect_mask_pos_rgb(img_en, best_pos)
        cv2.imwrite(outfile, best_img_flu)

    # 提取信号信息

    pass




def run_flu_signal_regnize(center_dist, split_dir, out_dir):
    w_h = 46
    mbm = MicroBeadMaskDist(center_dist)
    ori_signal_sets = []
    en_signal_sets = []
    for i in range(w_h):
        print(f"processing: w_i={i}")
        for j in range(w_h):
            # print(f"processing: w_i={i}, h_i={j}")
            # 读取图片
            infile = f"{split_dir}/ori_{i}_{j}.tif"
            img = cv2.imread(infile)
            # 自动识别参数，并进行荧光信息加强
            blacks,whites,gammas = MicroBeadMaskDist.calc_auto_BGR_cutoff(img)
            # print(f"black:{blacks},whites:{whites}")
            # img_en = FluorescentBase.RGB_enhance_times(img,black=blacks,white=whites,gamma=gammas,times=[100,100,100])
            img_en = FluorescentBase.RGB_binary_enhance(img,cutoff=blacks)

            # 灰度处理
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # img_gray = cv2.cvtColor(img_en, cv2.COLOR_RGB2GRAY)
            # 查找最合适位置
            # print("start")
            # best_pos = mbm.search_perfect_mask_pos(img_gray)
            # print("best_pos1:", best_pos)
            h_best_pos = mbm.search_perfect_pos(img_gray)
            while ((2 * mbm.row_height - h_best_pos - 1) < 0):
                h_best_pos = round(h_best_pos - mbm.row_height)
                print(f'{"=" * 50}\nh_best_pos has error:{infile}\n{"=" * 50}\n')
            # if round(2*mbm.row_height-h_best_pos-1)<0:
            #     print("h_best_pos",h_best_pos)
            #     start = int(h_best_pos-mbm.row_height)
            #     img = img[start:, :]
            #     img_en = img_en[start:,:]
            #     img_gray = img_gray[start:,:]
            #     h_best_pos = int(mbm.row_height)
            #     print(f'{"="*50}\nh_best_pos has error:i={i},j={j}\n{"="*50}\n')
            img_gray_float = np.power(img_gray,2.0)
            # best_pos = mbm.search_perfect_mask_pos_linerearch(img_gray,h_best_pos=h_best_pos)
            best_pos = mbm.search_perfect_mask_pos_linerearch(img_gray_float,h_best_pos=h_best_pos)
            # print("best_pos2:", best_pos)
            # 生成最合适位置的可视化图片，并进行保存
            # if i%3==0 and j%3==0:
            if True:
                best_img_flu = mbm.draw_perfect_mask_pos_rgb(img_en, best_pos)
                outfile = f"{out_dir}/enhance_{i}_{j}.tif"
                cv2.imwrite(outfile, best_img_flu)

            # 提取信号信息
            ori_cy5_cy3 = mbm.signal_extract(img,best_pos,out_dir,area_cutoff=[blacks[1],blacks[2]])
            en_cy5_cy3 = mbm.signal_extract(img_en,best_pos,out_dir,area_cutoff=[100,100])
            ori_signal_sets.append(ori_cy5_cy3)
            en_signal_sets.append(en_cy5_cy3)

    # 将提取到的荧光信号信息保存到文件中
    outfile = f"{out_dir}/signal_ori.txt"
    MicroBeadMaskDist.store_flu_signal_2_file(outfile=outfile, flu_signal=ori_signal_sets)
    outfile = f"{out_dir}/signal_en.txt"
    MicroBeadMaskDist.store_flu_signal_2_file(outfile=outfile, flu_signal=en_signal_sets)







if __name__ == '__main__':

    run_flu_signal_regnize(34.674919995646974, 'D:/delete/chip_four_project/cycle_4/', 'd:/delete/flusplit5/')



