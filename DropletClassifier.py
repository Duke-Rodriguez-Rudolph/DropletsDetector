import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import os

class Classifier:
    # 构造函数
    def __init__(self,src_path):
        self.__src_img=cv2.imread(src_path)

    # 找寻同类型的所有文件
    def __readAllFile(self,path, out_type=-1):
        # 储存所有文件的名字与类别
        file_dict = {}
        # 遍历路径下所有文件
        all_file = os.listdir(path)
        # 根据文件后缀进行分类
        for file in all_file:
            key = file.split('.')[-1]
            value = file_dict.get(key, [])
            value.append(file)
            file_dict[key] = value
        # 根据输入的要求决定输出的内容
        if out_type == -1:
            return file_dict
        else:
            return file_dict.get(out_type, [])

    # 计算hist值
    def __histCaculate(self,droplet):
        histb = cv2.calcHist([droplet], [0], None, [256], [0, 255])
        histg = cv2.calcHist([droplet], [1], None, [256], [0, 255])
        histr = cv2.calcHist([droplet], [2], None, [256], [0, 255])
        for i in range(0, 20):
            histb[i][0] = 0
            histg[i][0] = 0
            histr[i][0] = 0
        b, g, r = 0, 0, 0
        allb, allg, allr = 0, 0, 0
        n = 50
        for i in range(0, 256):
            if histb[i][0] > n:
                b += (histb[i][0] - n) * i
                allb += (histb[i][0] - n)
            if histg[i][0] > n:
                g += (histg[i][0] - n) * i
                allg += (histg[i][0] - n)
            if histr[i][0] > n:
                r += (histr[i][0] - n) * i
                allr += (histr[i][0] - n)
        if allb == 0 or allg == 0 or allr == 0:
            return -1
        hist = (b / allb, g / allg, r / allr)
        # hist=(np.where(histb==np.max(histb))[0][0],np.where(histg==np.max(histg))[0][0],np.where(histr==np.max(histr))[0][0])

        # plt.plot(histb, color="b")
        # plt.plot(histg, color="g")
        # plt.plot(histr, color="r")

        return hist

    # 运用k聚类的分类
    def __kSort(self,rgbs):
        data = np.array(rgbs)
        data = data.reshape((len(data), 3))
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        print(len(data))
        # 绘制散点图
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z)

        # 添加坐标轴(顺序是Z, Y, X)
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        # plt.show()
        kmeans = KMeans(n_clusters=9)
        res = kmeans.fit(data)
        print('聚类中心们：', '\n', kmeans.cluster_centers_)
        self.__clusters=kmeans.cluster_centers_
        return kmeans

    # 单张照片分析
    def __single(self,img):
        results = []
        nucleus = 9
        while (True):
            # 高斯模糊
            blur = cv2.GaussianBlur(img, (nucleus, nucleus), 0)
            # canny算子查找边缘
            canny = cv2.Canny(blur, 3, 25)
            # 找轮廓
            contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 所有结果

            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                # 画出所有图形
                # img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

                '''#轮廓面积限制
                if area<30:
                    print('area')
                    continue
                '''

                # 矩形面积限制
                if w * h < (img.shape[0] * img.shape[1] * 0.07) or w * h > (img.shape[0] * img.shape[1] * 0.26):
                    # print('size')
                    continue

                # 长宽比限制
                if w / h > 1.2 or h / w > 1.2:
                    # print('scale')
                    continue

                # 居中
                if abs(y + h / 2 - img.shape[0] / 2) > 10:
                    continue

                # 结果都存储起来
                results.append((x, y, w, h))
                # 画出所有通过的矩形
                # img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
                # 画出目标圆形
                # img=cv2.circle(img,(int(x+w/2),int(y+h/2)),int(min(w/2,h/2))-2,(255,0,0),1)
            if len(results) != 0 or nucleus < 3:
                break
            nucleus = nucleus - 2

        # 展示过程图
        # cv2.imshow('canny',canny)
        # cv2.imshow('img',img)
        # 进行rgb统计
        for result in results:
            x, y, w, h = result
            droplet = img.copy()
            droplet = droplet[y:y + h, x:x + w]
            # 直方图处理
            hist = self.__histCaculate(droplet)
            if hist == -1:
                continue
            # 排除那些空白的液滴
            # if hist[0]>142 and hist[1]<158 and hist[1]>139 and hist[1]<155 and hist[2]>138 and hist[2]<151:
            # return -1
            # print(hist)
            # plt.show()
            return hist

        # cv2.waitKey(0)
        return -1

    # 扫描所有已分割的文件
    def scan(self, droplets):
        droplets_dict={}
        rgbs = []
        belong = {}
        nones = []
        for col, droplet in enumerate(droplets):
            for row,droplet_small in enumerate(droplet):
                [x, y, w, h, area] = droplet_small
                img = self.__src_img[y:y + h, x:x + w]
                hist = self.__single(img)
                jpg=str(col)+'_'+str(row)
                if hist == -1:
                    nones.append(jpg)
                    continue
                else:
                    rgbs.append(hist)
                belong[jpg] = hist
        droplets_dict['none']=nones
        print('获取颜色完毕')
        # 进行k聚类
        kmeans = self.__kSort(rgbs)
        for jpg in belong:
            hist = belong[jpg]
            droplet_type = kmeans.predict([hist])[0]

            droplet_type_dict=droplets_dict.get(droplet_type,[])
            droplet_type_dict.append(jpg)
            droplets_dict[droplet_type]=droplet_type_dict

        # 合并none
        min_num=10000
        min_index=None
        for droplet_type in droplets_dict:
            if len(droplets_dict[droplet_type])<min_num:
                min_index=droplet_type
        droplets_dict['none'].extend(droplets_dict[min_index])
        del droplets_dict[min_index]
        print('颜色分类完毕')
        self.__droplets_dict=droplets_dict

    # 得到相关记录
    def getDropletsDict(self):
        return self.__droplets_dict

    #设置相关记录
    def setDropletsDict(self,droplets_dict):
        self.__droplets_dict=droplets_dict

    # 记录全图
    def cut(self,droplets,droplets_dict,save_path):
        for droplet_type in droplets_dict:
            # 建立文件夹
            os.mkdir(save_path+str(droplet_type))
            droplet_type_dict=droplets_dict[droplet_type]
            for jpg in droplet_type_dict:
                # 拆分
                col,row=jpg.split('_')
                [x, y, w, h, area]=droplets[int(col)][int(row)]
                img = self.__src_img[y:y + h, x:x + w]
                cv2.imwrite(save_path+str(droplet_type)+'/'+jpg+'.jpg',img)

    # 得到颜色
    def getColor(self):
        colors={}
        for i,color in enumerate(self.__clusters):
            b,g,r=color
            colors[i]=(b,g,r)
        return colors

    # 得到聚类中心
    def getClusters(self):
        return self.__clusters