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



class SubareaBeadsLocation():
    def __init__(self, chip_subarea_split):
        self.start_box_sets = chip_subarea_split['start_box_sets']
        self.start_indexs_sets = chip_subarea_split['start_indexs_sets']
        self.w_splits_sets = chip_subarea_split['w_splits_sets']
        self.h_splits_sets = chip_subarea_split['h_splits_sets']
        self.center_dist = chip_subarea_split['center_dist']

        # init 46*46*4的box
        self.abs_boxs = np.zeros([46,46,4],dtype=int)

        # 处理一个BLOCK
        for i in range(len(self.start_box_sets)):
            self.process_one_block(i)

        # init 搜索状态
        self.reset_research_status()
        # init 分区的长与度
        self.subarea_wh = [int(self.center_dist*31), int(self.center_dist*np.power(3,0.5)/2*36)]
        pass

    def process_one_block(self, block_i):
        curr_box = self.start_box_sets[block_i]
        curr_index = self.start_indexs_sets[block_i]
        curr_w_splits = self.w_splits_sets[block_i]
        curr_h_splits = self.h_splits_sets[block_i]

        for w_i in range(1,len(curr_w_splits),1):
            for h_i in range(1, len(curr_h_splits), 1):
                # subarea的左上角与右下角坐标
                # 局部的坐标
                left_top = [curr_w_splits[w_i-1],curr_h_splits[h_i-1]]
                right_bottom = [curr_w_splits[w_i], curr_h_splits[h_i]]
                # 全局的坐标
                left_top = [left_top[0]+curr_box[0], left_top[1]+curr_box[1]]
                right_bottom = [right_bottom[0]+curr_box[0], right_bottom[1]+curr_box[1]]
                # 全局的index
                g_wi = curr_index[0]+w_i-1
                g_hi = curr_index[1]+h_i-1
                # 保存
                self.abs_boxs[g_wi,g_hi,:] = [left_top[0],left_top[1],right_bottom[0],right_bottom[1]]
                pass
        pass

    # 重置搜索的状态
    def reset_research_status(self):
        self.is_search = np.zeros([46,46],dtype='uint8')
        self.perfect_pos = np.zeros([46,46,2],dtype=int)
        pass
    # 获取某个分区的当前搜索状态
    def subarea_curr_status(self, pos):
        return self.is_search[pos[0],pos[1]]
        pass

    # 生成随机的subarea位置
    def get_random_subarea(self):
        r = [np.random.randint(1,46-1), np.random.randint(1,46-1)]
        return r
        pass
    # 生产随机的subarea位置，但是要求is_search的状态必须是0，即没有被搜索过
    def get_random_subarea_unsearch(self):
        while(1):
            r_pos = [np.random.randint(1,46-1), np.random.randint(1,46-1)]
            status = self.subarea_curr_status(r_pos)
            if status == 0:
                return r_pos
        pass
    # 生成随机的2*2的相邻分区,注意其不包含最外的一圈分区
    def get_random_subarea_2_2(self):
        r = [np.random.randint(1, 46 - 2), np.random.randint(1, 46 - 2)]
        return [[r[0], r[1]], [r[0]+1, r[1]], [r[0]+1, r[1]+1], [r[0], r[1]+1]]
        pass



    # 生成下一个没有被搜索的subarea的位置，但是其相阾subarea已经被搜索过
    # 1表示有未搜索的值
    # 0表示无未搜索的值，即搜索可以结束了
    def next_adjacent_subarea_unsearch(self):
        for w_i in range(46):
            for h_i in range(46):
                if self.is_search[w_i,h_i] == 1:
                    continue
                for local_w_i in range(max(0,w_i-1),min(45,w_i+1)+1):
                    for local_h_i in range(max(0,h_i-1),min(45,h_i+1)+1):
                        if self.is_search[local_w_i,local_h_i] == 1:
                            self.next_unsearch = [w_i,h_i]
                            self.next_adjacent = [local_w_i,local_h_i]
                            return 1
        return 0
        pass
    # 与上面的配合使用
    # 根据邻近的已知的位置来计算当前分区的最佳位置
    def calc_pos_by_adjacent_subarea(self):
        curr_loc = self.next_unsearch
        adja_loc = self.next_adjacent
        adja_perfect_pos = self.perfect_pos[adja_loc[0],adja_loc[1],:]
        curr_perfect_pos = [0, 0]
        curr_box = self.abs_boxs[curr_loc[0],curr_loc[1],:]
        adja_box = self.abs_boxs[adja_loc[0],adja_loc[1],:]
        # 计算邻近分区的最优位置的绝对坐标
        adja_abs_perfect_pos = [adja_box[0]+adja_perfect_pos[0], adja_box[1]+adja_perfect_pos[1]]
        curr_abs_perfect_pos = [0, 0]
        # for width
        if curr_loc[0] > adja_loc[0]:
            curr_abs_perfect_pos[0] = adja_abs_perfect_pos[0] + self.subarea_wh[0]
        elif curr_loc[0] == adja_loc[0]:
            curr_abs_perfect_pos[0] = adja_abs_perfect_pos[0]
        else:
            curr_abs_perfect_pos[0] = adja_abs_perfect_pos[0] - self.subarea_wh[0]
        # for height
        if curr_loc[1] > adja_loc[1]:
            curr_abs_perfect_pos[1] = adja_abs_perfect_pos[1] + self.subarea_wh[1]
        elif curr_loc[1] == adja_loc[1]:
            curr_abs_perfect_pos[1] = adja_abs_perfect_pos[1]
        else:
            curr_abs_perfect_pos[1] = adja_abs_perfect_pos[1] - self.subarea_wh[1]
        # 计算当前处理的分区的相对偏移中的最优位置
        curr_perfect_pos[0] = curr_abs_perfect_pos[0] - curr_box[0]
        curr_perfect_pos[1] = curr_abs_perfect_pos[1] - curr_box[1]
        return curr_perfect_pos
        pass
    # 搜索整个芯片的分区
    def search_whole_subareas(self, bsearch, split_dir):
        is_ok = self.first_four_subarea_search(bsearch, split_dir)
        if is_ok == 0:
            print("ERROR: first_four_subarea_search is failed!")
            return 0
        # 启动迭代搜索
        count = 0
        while(True):
            count += 1
            if count%50==0:
                print(f"processing {count}")
            # 选取下一个分区，并检测分区是否已经遍历完
            has_next = self.next_adjacent_subarea_unsearch()
            if has_next == 0:
                break
            # 计算最优位置并保存
            perfect_pos = self.calc_pos_by_adjacent_subarea()
            curr_loc = self.next_unsearch
            infile = f"{split_dir}/ori_{curr_loc[0]}_{curr_loc[1]}.tif"
            perfect_pos2 = bsearch.search_perfect_pos_by_adjacent(infile, perfect_pos)
            self.is_search[curr_loc[0],curr_loc[1]] = 1
            self.perfect_pos[curr_loc[0], curr_loc[1], :] = perfect_pos2
        pass

    # -------------------------------------------------------
    # 搜索第一个位置，2*2一共4个分区
    # -------------------------------------------------------
    # 分别依据4个分区的图片，进行每个分区的最优位置的搜索
    def calc_four_perfect_pos(self, bsearch, split_dir, four_subareas):
        four_perfect_pos = [None, None, None, None]
        for i in range(4):
            subarea = four_subareas[i]
            infile = f"{split_dir}/ori_{subarea[0]}_{subarea[1]}.tif"
            best_pos = bsearch.search_perfect_pos_by_traverse_win(infile)
            four_perfect_pos[i] = best_pos
        return four_perfect_pos
        pass
    # 检查4个最优位置坐标是不是在3个像素的偏差以内
    # 如果是，即返回1， 否则是0
    def check_first_four_pos(self, four_subareas, four_perfect_pos):
        # 计算绝对位置
        four_abs_pos = [None, None, None, None]
        for i in range(4):
            curr_pos = four_perfect_pos[i]
            curr_subarea = four_subareas[i]
            curr_abs_pos = self.calc_abs_pos(curr_subarea, curr_pos)
            four_abs_pos[i] = curr_abs_pos
        print("four_abs_pos:",four_abs_pos)
        print("self.subarea_wh:", self.subarea_wh)
        # check
        # for width
        if abs(four_abs_pos[1][0]-four_abs_pos[0][0]-self.subarea_wh[0]) > 3:
            return 0
        if abs(four_abs_pos[2][0]-four_abs_pos[0][0]-self.subarea_wh[0]) > 3:
            return 0
        if abs(four_abs_pos[3][0]-four_abs_pos[0][0]) > 3:
            return 0
        # for height
        if abs(four_abs_pos[1][1] - four_abs_pos[0][1]) > 3:
            return 0
        if abs(four_abs_pos[2][1] - four_abs_pos[0][1] - self.subarea_wh[1]) > 3:
            return 0
        if abs(four_abs_pos[3][1] - four_abs_pos[0][1] - self.subarea_wh[1]) > 3:
            return 0
        # is ok
        return 1
        pass
    # 依据最优位置计算绝对坐标
    def calc_abs_pos(self, subarea, pos):
        abs_box = self.abs_boxs[subarea[0], subarea[1], :]
        print("abs_pos:",abs_box)
        return [abs_box[0]-pos[0], abs_box[1]-pos[1]]
        pass
    # 随机生成2*2的小区域
    # 对每个分区进行最优位置的检查，如果相对偏差都在3个像素以内，则生成OK，否则生成失败
    # 随机过程会进行10次，只要有一次是OK的，就会直接结束
    def first_four_subarea_search(self, bsearch, split_dir):
        count = 0
        while(True):
            # 随机生成一个2*2的小区域，并进行相对位置检查
            four_subareas = self.get_random_subarea_2_2()
            print("four_subareas:", four_subareas)
            four_perfect_pos = self.calc_four_perfect_pos(bsearch, split_dir, four_subareas)
            print("four_perfect_pos:", four_perfect_pos)
            is_ok = self.check_first_four_pos(four_subareas, four_perfect_pos)
            # 如果2*2的分区是完整的，并且最优位置的偏差相互小于3，则进行保存并退出
            if is_ok == 1:
                for i in range(4):
                    subarea = four_subareas[i]
                    self.perfect_pos[subarea[0],subarea[1],:] = four_perfect_pos[i]
                    self.is_search[subarea[0], subarea[1]] = 1
                return 1
            # 最多进行10次随机搜索，如果不成功则退出
            count += 1
            if count==10:
                return 0
        pass
















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
class BeadSearch():
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
        pass

    def __search_perfect_mask_pos(self, np_img, w_start, w_range, h_start, h_range, step):
        max_val = 0.0
        best_pos = [0, 0]
        curr_img = np.zeros([self.mask_height, self.mask_width], dtype='float32')
        for w_i in range(w_start, w_range, step):
            for h_i in range(h_start, h_range, step):
                curr_img.fill(0)
                curr_img[h_i:(h_i+np_img.shape[0]), w_i:(w_i+np_img.shape[1])] = np_img
                curr_val = np.sum(curr_img*self.microbead_mask)
                if max_val < curr_val:
                    max_val = curr_val
                    best_pos = [w_i, h_i]
        return best_pos
    def search_perfect_mask_pos(self, np_img):
        # first
        best_pos = self.__search_perfect_mask_pos(np_img,0,round(self.center_dist*1.5),0,round(self.row_height*1.5),3)
        # local
        w_start = max(0, best_pos[0] - 3)
        w_range = best_pos[0] + 3
        h_start = max(0, best_pos[1]-3)
        h_range = best_pos[1] + 3
        best_pos = self.__search_perfect_mask_pos(np_img, w_start, w_range, h_start, h_range, 1)
        return best_pos
    def search_perfect_mask_pos_linerearch(self, np_img, h_best_pos):
        # print("h_best_pos:",h_best_pos)
        # print("just pos:",2*self.row_height-h_best_pos)
        h_best_pos = round(2*self.row_height-h_best_pos-1)
        # first
        best_pos = self.__search_perfect_mask_pos(np_img,0,round(self.center_dist*1.5),h_best_pos,h_best_pos+1,3)
        # local
        w_start = max(0, best_pos[0] - 3)
        w_range = best_pos[0] + 3
        h_start = max(0, best_pos[1]-3)
        h_range = best_pos[1] + 3
        best_pos = self.__search_perfect_mask_pos(np_img, w_start, w_range, h_start, h_range, 1)
        return best_pos
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
        return max_index
        pass

    def search_perfect_pos_by_traverse_win(self, infile):
        img = cv2.imread(infile)
        # 灰度处理
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        h_best_pos = self.search_perfect_pos(img_gray)
        # if round(2 * self.row_height - h_best_pos - 1) < 0:
        #     print("h_best_pos", h_best_pos)
        #     start = int(h_best_pos - mbm.row_height)
        #     img = img[start:, :]
        #     img_en = img_en[start:, :]
        #     img_gray = img_gray[start:, :]
        #     h_best_pos = int(mbm.row_height)
        best_pos = self.search_perfect_mask_pos_linerearch(img_gray, h_best_pos=h_best_pos)
        return best_pos

    def search_perfect_pos_by_adjacent(self, infile, start_pos):
        img = cv2.imread(infile)
        # 灰度处理
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # local
        local_range = 3
        w_start = max(0, start_pos[0] - local_range)
        w_range = start_pos[0] + local_range
        h_start = max(0, start_pos[1] - local_range)
        h_range = start_pos[1] + local_range
        best_pos = self.__search_perfect_mask_pos(img_gray, w_start, w_range, h_start, h_range, 1)
        return best_pos



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
        print(np.sum(mask))
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







def run_flu_signal_regnize_by_adjacent(ps_obj):
    # init
    chip_subarea_split = ps_obj.chip_subarea_splits[ps_obj.active_cycle]
    split_dir = ps_obj.cycle_work_dirs[ps_obj.active_cycle]
    out_dir = ps_obj.cycle_signal_dirs[ps_obj.active_cycle]
    # BeadSearch and SubareaBeadsLocation
    bsearch = BeadSearch(chip_subarea_split['center_dist'])
    sbl = SubareaBeadsLocation(chip_subarea_split)
    # 搜索整个分区
    sbl.search_whole_subareas(bsearch=bsearch,split_dir=split_dir)
    print("sbl.perfect_pos[3,3,:]: ", sbl.perfect_pos[3,3,:])

    # 输出图片
    w_h = 46
    for w_i in range(w_h):
        for h_i in range(w_h):
            print(f"processing: w_i={w_i}, h_i={h_i}")
            # 读取图片
            infile = f"{split_dir}/ori_{w_i}_{h_i}.tif"
            img = cv2.imread(infile)
            # 自动识别参数，并进行荧光信息加强
            blacks, whites, gammas = BeadSearch.calc_auto_BGR_cutoff(img)
            # print(f"black:{blacks},whites:{whites}")
            # img_en = FluorescentBase.RGB_enhance_times(img,black=blacks,white=whites,gamma=gammas,times=[100,100,100])
            img_en = FluorescentBase.RGB_binary_enhance(img, cutoff=blacks)
            best_pos = sbl.perfect_pos[w_i,h_i,:]
            best_img_flu = bsearch.draw_perfect_mask_pos_rgb(img_en, best_pos)
            outfile = f"{out_dir}/her_{w_i}_{h_i}.tif"
            cv2.imwrite(outfile, best_img_flu)
    ori_signal_sets = []
    en_signal_sets = []

    pass



def run_flu_signal_regnize(center_dist, split_dir, out_dir, sbl_obj):
    w_h = 46
    mbm = BeadSearch(center_dist)
    ori_signal_sets = []
    en_signal_sets = []
    for i in range(w_h):
        print(f"processing: w_i={i}")
        for j in range(w_h):
            print(f"processing: w_i={i}, h_i={j}")
            # 读取图片
            infile = f"{split_dir}/ori_{i}_{j}.tif"
            img = cv2.imread(infile)
            # 自动识别参数，并进行荧光信息加强
            blacks,whites,gammas = BeadSearch.calc_auto_BGR_cutoff(img)
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
            if round(2*mbm.row_height-h_best_pos-1)<0:
                print("h_best_pos",h_best_pos)
                start = int(h_best_pos-mbm.row_height)
                img = img[start:, :]
                img_en = img_en[start:,:]
                img_gray = img_gray[start:,:]
                h_best_pos = int(mbm.row_height)
            best_pos = mbm.search_perfect_mask_pos_linerearch(img_gray,h_best_pos=h_best_pos)
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
    BeadSearch.store_flu_signal_2_file(outfile=outfile, flu_signal=ori_signal_sets)
    outfile = f"{out_dir}/signal_en.txt"
    BeadSearch.store_flu_signal_2_file(outfile=outfile, flu_signal=en_signal_sets)







if __name__ == '__main__':
    import json
    json_file = 'D:/delete/chip_four_project/test.json'

    # 读取配置参数值
    from fluorescent.fluProjectSet.ProjectSetting import ProjectSetting
    with open(json_file, 'r') as fp:
        load_dict = json.load(fp)
        ps_obj = ProjectSetting.json_2_self(load_dict)
    print("ps_obj.chip_subarea_splits[0]:",ps_obj.chip_subarea_splits[0])
    ps_obj.active_cycle = 0
    run_flu_signal_regnize_by_adjacent(ps_obj=ps_obj)
    pass



