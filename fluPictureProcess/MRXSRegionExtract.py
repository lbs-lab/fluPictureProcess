import copy
import numpy as np
import cv2
import numpy.linalg as lg
import openslide
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

'''
###### 公式
MA=B
M:变换矩阵
A:原图上的坐标，如[x,y]，则A=[[x],[y],[1]]
B:A点在变换后图片中的坐，如[x',y'], 则B=[[x'],[y'],[1]]
###### 背景
在MRXS图片中有一个倾斜的矩形区域，其通过透视变换得到了正立的矩形图，图片的长宽保持不变
由于MRXS图片太大，无法直接读取出来，并进行变换，只能通过这个类来间接的访问局部图
###### 功能说明
MRXS图：即原图
透视变换图（N）：即从MRXS中截取变换后的图片
给定N上的一个矩形区域，即可从MRXS图中变换生成这个区域的图
###### 注意
由于N图比较大，无法直接一次全部提取出来，只能一部分一部分的局部访问
###### 功能实现方法
MRXS图片中矩形的4个顶点坐标: mrxs_rect = [pt1, pt2, pt3, pt4]
MRXS图片中矩形的W与H（即N图的W与H）：img_wh, w与h为mrxs_rect的w与h，小数位用round()进行四舍五入
N图最张的四个顶点坐标：nimg_rect = [[0,0],[img_wh[0],0],[img_wh[0],img_wh[1]],[0,img_wh[1]]]
变换矩阵：M
N图上的一个矩形区域：nimg_box = [x, y, x+w, y+h], nimg_box_wh = [w,h]
nimg_box平移到[0,0]处的坐标：nimg_box_ori_rect = [[0,0],[nimg_box_wh[0],0],[nimg_box_wh[0],nimg_box_wh[1]],[0,nimg_box_wh[1]]]
nimg_box对应4个顶点: nimg_box_rect = [pt1,pt2,pt3,pt4]
nimg_box_rect通过M逆变换得到MRXS图上4个顶点坐标：nimg_mrxs_rect_neg = [pt1,pt2,pt3,pt4]
通过求nimg_mrxs_rect_neg的最小正矩形（box），对其坐标以最小正矩形的左上角为(0,0)进行变换，得到move_rect
最后从原始大图片中读取图片时，就不用全部读取出来了，只需要把最小正矩形（box）中的部分读出来即可，这样速度与内存都可以保证
最后通过再一次的透视变换，得到最终的局部图片: move_rect ==> nimg_box_ori_rect, 图片大小为nimg_box_wh
'''
class MRXSRegionExtract():
    def __init__(self, mrxs_rect, slide):
        # init slide
        self.init_slide(slide)

        # init
        self.mrxs_rect = copy.deepcopy(mrxs_rect)
        self.img_wh = [0, 0]
        self.img_wh[0] = round(MRXSRegionExtract.two_point_dist(self.mrxs_rect[0], self.mrxs_rect[1]))
        self.img_wh[1] = round(MRXSRegionExtract.two_point_dist(self.mrxs_rect[0], self.mrxs_rect[3]))
        self.nimg_rect = [[0,0],[self.img_wh[0]-1,0],[self.img_wh[0]-1,self.img_wh[1]-1],[0,self.img_wh[1]-1]]

        # 计算M变换矩阵
        self.M = self.M_matrix(self.mrxs_rect, self.nimg_rect)
        self.M_inv = lg.inv(self.M)

        pass

    def extract_mrxs_region(self, nimg_box):
        # init
        nimg_box_wh = [nimg_box[2]-nimg_box[0], nimg_box[3]-nimg_box[1]]
        nimg_box_ori_rect = np.array([[0,0],[nimg_box_wh[0],0],[nimg_box_wh[0],nimg_box_wh[1]],[0,nimg_box_wh[1]]], dtype='float32')
        nimg_box_rect = [[nimg_box[0],nimg_box[1]], [nimg_box[2],nimg_box[1]], [nimg_box[2],nimg_box[3]], [nimg_box[0],nimg_box[3]]]
        nimg_box_rect = np.array(nimg_box_rect, dtype='float32')
        nimg_mrxs_rect_neg = self.cal_inv_pos(nimg_box_rect)

        # move rect
        box, move_rect = self.cal_min_rect_box(nimg_mrxs_rect_neg)
        # print("nimg_box_wh:",nimg_box_wh)
        # print("nimg_box_ori_rect:", nimg_box_ori_rect)
        # print("nimg_box_rect:", nimg_box_rect)
        # print("nimg_mrxs_rect_neg:", nimg_mrxs_rect_neg)
        # print("box:", box)

        # crop img
        tile = np.array(self.slide.read_region((box[0], box[1]), 0, (box[2], box[3])))

        # 变换矩阵
        M = self.M_matrix(move_rect, nimg_box_ori_rect)
        # 透视变换
        warped = cv2.warpPerspective(tile, M, (nimg_box_wh[0], nimg_box_wh[1]))
        return warped
        pass

    # MA = B
    # 逆运算
    # M-1MA = M-1B, 即A = M-1B
    # 注意：A,B,的矩阵值为[[x], [y], [1]]
    def cal_inv_pos(self, rect):
        rect_neg = []
        for pt in rect:
            t = np.array([[pt[0]], [pt[1]], [1]], dtype='float32')
            mt = np.matmul(self.M_inv, t)
            rect_neg.append([round(mt[0][0]), round(mt[1][0])])
        return np.array(rect_neg, dtype='float32')

    def M_matrix(self, scr_rect, dst_rect):
        rect = np.array(scr_rect, dtype='float32')
        dst = np.array(dst_rect, dtype='float32')
        # 变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        # return
        return M

    @staticmethod
    def two_point_dist(pt1, pt2):
        pow1 = np.power(pt2[0] - pt1[0], 2.0)
        pow2 = np.power(pt2[1] - pt1[1], 2.0)
        print(pt1,pt2,pow1,pow2)
        return np.sqrt(pow1 + pow2)

    def cal_min_rect_box(self, rect):
        min_x = min(rect[:,0])
        min_y = min(rect[:,1])
        max_x = max(rect[:,0])
        max_y = max(rect[:,1])
        box = [self.slide_x+min_x, self.slide_y+min_y, max_x-min_x, max_y-min_y]
        box = list(map(int, box))
        move_rect = rect - np.array([min_x,min_y])
        return box,move_rect
        pass

    def init_slide(self, slide):
        self.slide = slide
        self.slide_x = int(slide.properties['openslide.bounds-x'])
        self.slide_y = int(slide.properties['openslide.bounds-y'])
        self.slide_width = int(slide.properties['openslide.bounds-width'])
        self.slide_height = int(slide.properties['openslide.bounds-height'])


    pass




# 对荧光图片的荧光值进行增强处理
def adjust_brightness(img,black=[2,2,2],gamma=[1.69,1.69,1.69],white=[30,30,30]):
    r, g, b, a = cv2.split(img)
    # r = np.where(r<black[0],0,r)
    g = np.where(g<black[1],0,g)
    b = np.where(b<black[2],0,b)
    # r = np.power(r*1.0*white[0],gamma[0])
    g = np.power(g*1.0*white[1],gamma[1])
    b = np.power(b*1.0*white[2],gamma[2])
    r = np.minimum(r, 255)
    g = np.minimum(g, 255)
    b = np.minimum(b, 255)
    r=r.astype(np.uint8)
    g=g.astype(np.uint8)
    b=b.astype(np.uint8)
    print("r.min:",r.max())
    print("g.min:",g.max())
    print("b.min:",b.max())

    new_img = Image.fromarray(np.dstack((b,g,r)))
    return np.array(new_img)
    pass


if __name__ == '__main__':
    # 读取图片信息
    flu_file = 'D:\\delete\\flu\\flu_8889_bc1_40X.mrxs'
    # flu_file = 'D:\\delete\\flu\\flu_two_8889_bc1_40X.mrxs'
    slide = openslide.OpenSlide(flu_file)
    # box = [3318, 200626, 3018, 3017]
    # tile = np.array(slide.read_region((box[0], box[1]), 0, (box[2], box[3])))
    # img_show("tile", tile)
    # exit()

    # create
    rect = np.array([[834, 832], [11281, 770], [11343, 11288], [896, 11350]])*4
    mre = MRXSRegionExtract(rect,slide)

    # 提取一个图片
    nbox = [0,0,10000,10000]
    np_img = mre.extract_mrxs_region(nbox)
    print(np_img.shape)
    img_show("img", np_img)

    fluimg = adjust_brightness(np_img, [2, 8, 8], [1.0, 1.0, 1.0], [20, 20, 20])
    img_show("fluimg", fluimg)


    pass


