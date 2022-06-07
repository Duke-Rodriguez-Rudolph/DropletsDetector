import cv2
import numpy as np
from sklearn.cluster import KMeans
import random

class Separator:
    # 构造函数
    def __init__(self):
        self.__ranColor()

    # 设置图像(图像路径)
    def setImg(self,src_path):
        # 读取原始图片
        self.__src_img=cv2.imread(src_path)
        # 复制一份
        self.__draw_img=cv2.imread(src_path)

    # 设置自适应参数(block_size,C,液滴区域矩形面积上下限)
    def setThresh(self,block_size,C,area_list):
        self.__block_size=block_size
        self.__C=C
        self.__area_list=area_list

    # 图像预处理
    def __imgProcess(self):
        # 灰度图
        self.__gray = cv2.cvtColor(self.__src_img, cv2.COLOR_BGR2GRAY)
        # 二值图
        self.__binary = cv2.adaptiveThreshold(self.__gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              self.__block_size, self.__C)
        #其他处理
        # blur=cv2.GaussianBlur(self.__img,(9,9),0)
        # self.__binary=cv2.Canny(blur,20,25)

    #随机颜色
    def __ranColor(self):
        colors=[]
        for i in range(201):
            b=random.randint(0,255)
            g=random.randint(0,255)
            r=random.randint(0,255)
            colors.append((b,g,r))
        self.__color=colors

    # 粗查找后的变换格式与排序
    def __box2droplet(self):
        # k聚类的中心计算
        boxes = self.__boxes
        data = np.array(boxes)
        # 取x为聚类数据，计算200个列的x值
        x_middle = data[:, 0] + data[:, 2] / 2 # x+w/2
        x_middle = x_middle.reshape(-1, 1)
        kmeans = KMeans(n_clusters=200)
        res = kmeans.fit(x_middle)
        droplets = {} # 存储液滴信息{‘col_index’:[[x1,y1,w1,h1,area1],...]},后来弄成了np类型
        # 按列分类
        for box in boxes:
            box = np.int0(box) #修改数据类型为int
            [x, y, w, h, area] = box
            col_index = kmeans.predict([[x + w / 2]])[0]
            temp_list = droplets.get(col_index, [])
            temp_list.append([x, y, w, h, area])
            droplets[col_index] = temp_list
        # 由列表弄成np矩阵，然后排序写入字典
        for col_index in droplets:
            col_droplets = droplets[col_index]
            col_droplets = np.array(col_droplets)
            # 排序
            droplets[col_index] = col_droplets[np.argsort(col_droplets[:, 1])]

        # 上面的droplets字典没有对列进行排序，下面对列排序并赋值给成员变量
        self.__cluster_list = np.argsort(kmeans.cluster_centers_[:, 0])
        new_sort = []
        for num in range(len(self.__cluster_list)):
            for i in range(len(self.__cluster_list)):
                if num == self.__cluster_list[i]:
                    new_sort.append(i)
                    continue
        self.__cluster_list = new_sort
        sorted_droplets = []
        for index in range(0, 200):
            for i, sort_index in enumerate(self.__cluster_list):
                if sort_index == index:
                    droplet = droplets[i]
                    sorted_droplets.append(droplet)
                    continue

        # 输入变量中
        self.__droplets = sorted_droplets

    # 线性回归
    def __liner(self,droplet):
        x = droplet[:, 0] + droplet[:, 2] / 2
        y = droplet[:, 1] + droplet[:, 3] / 2
        mean_w = np.mean(droplet[:, 2])
        mean_h = np.mean(droplet[:, 3])
        mean_x = np.mean(x)

        # 对歪过头的进行处理
        droplet = [list(droplet[i]) for i in range(len(x)) if abs(x[i] - mean_x) < (mean_w / 2 + 10)]
        y = np.array([y[i] for i in range(len(x)) if abs(x[i] - mean_x) < (mean_w / 2 + 10)]).reshape(-1, 1)
        x = np.array([x[i] for i in range(len(x)) if abs(x[i] - mean_x) < (mean_w / 2 + 10)]).reshape(-1, 1)
        y = y[np.argsort(y[:, 0])].reshape(-1, 1)
        x = x[np.argsort(y[:, 0])].reshape(-1, 1)
        # 线性拟合
        if ((x[-1][0] - x[0][0]) == 0):
            coef = 99999
        else:
            coef = (y[-1][0] - y[0][0]) / (x[-1][0] - x[0][0])
        intercept = y[0][0] - coef * x[0][0]

        return [coef, intercept, mean_w, mean_h]

    # 计算两直线交点
    def __line_intersection(self,line1, line2):
        coef, inter = line1
        k, xn, yn = line2
        b = yn - k * xn
        x = (b - inter) / (coef - k)
        y = k * x + b
        return x, y

    #计算两点之间坐标值
    def __distance(self,pointA,pointB):
        x2=(pointA[0]-pointB[0])**2
        y2=(pointA[1]-pointB[1])**2
        return (x2+y2)**0.5

    # 精细处理
    def __fineSearch(self):
        # 取出液滴
        droplets = self.__droplets
        index = 0
        is_find = 0
        while True:
            # 判断是否为1000
            if index == 200:
                # 200列查完先判断是否每个都是50个
                is_ok = 1 # 先置1，后面不是再变回0
                for i, droplet in enumerate(droplets):
                    # 但凡有一个不是50个，则置0
                    if droplet.shape[0] != 50:
                        is_ok = 0
                        break
                # 如果检查通过，则退出循环，不然则重置参数
                if is_ok:
                    break
                else:
                    index = 0
                    is_find = 0

            # 下面正式单个循环开始，首先寻找目标
            now = droplets[index] #now是现在的液滴

            # 不为50的直接跳过
            if now.shape[0] != 50:
                index += 1
                continue

            # 如果满足了50个，才有以下判断：对前部进行判断
            if index == 0 or is_find == 1:
                pass
            else:
                # 获取斜率
                before = droplets[index - 1]
                min_k = 999
                for now_droplet in now:
                    xn, yn, w, h, area = now_droplet
                    for before_droplet in before:
                        xb, yb, w, h, area = before_droplet

                        # 判断奇偶
                        if (index + 1) % 2 == 1:
                            # 奇数操作
                            if ((yb - yn) / (xb - xn)) < 0:
                                continue
                            if abs((yb - yn) / (xb - xn)) < abs(min_k):
                                min_k = (yb - yn) / (xb - xn)
                        else:
                            # 偶数操作
                            if ((yb - yn) / (xb - xn)) > 0:
                                continue
                            if abs((yb - yn) / (xb - xn)) < abs(min_k):
                                min_k = (yb - yn) / (xb - xn)

                # 获取线
                coef, intercept, mean_w, mean_h = self.__liner(before)
                # 获取点集
                points_list = []
                for now_droplet in now:
                    xn, yn, w, h, area = now_droplet
                    x, y = self.__line_intersection([coef, intercept], [min_k, xn + w / 2, yn + h / 2])
                    points_list.append([x, y])
                    # cv2.line(self.__draw_img,(int(xn+w/2),int(yn+h/2)),(int(x),int(y)), (0,0,255), 3,4)
                correct_list = []
                # 查看点集合理性
                distance_thold = 25
                # print('distance_thold:',distance_thold)
                for point in points_list:
                    is_locate = 0
                    for before_droplet in before:
                        xb, yb, w, h, area = before_droplet
                        distance = self.__distance(point, [xb + w / 2, yb + h / 2])
                        if distance < distance_thold:
                            is_locate = 1
                            break
                    if is_locate:
                        correct_list.append([int(xb), int(yb), int(w), int(h), int(area)])
                    else:
                        correct_list.append(
                            [int((point[0] - mean_w / 2)), int(point[1] - mean_h / 2), int(mean_w), int(mean_h),
                             int(mean_w * mean_h)])

                # print('before_think:',len(correct_list),before.shape[0])
                correct_list = np.array(correct_list)
                before = correct_list[np.argsort(correct_list[:, 1])]
                droplets[index - 1] = before
                is_find = 1
            # 对后部进行判断
            if index == 199:
                pass
            else:
                # 获取斜率
                after = droplets[index + 1]
                if (after.shape[0] == 50):
                    index += 1
                    continue
                min_k = 999
                for now_droplet in now:
                    xn, yn, w, h, area = now_droplet
                    for after_droplet in after:
                        xa, ya, w, h, area = after_droplet

                        # 判断奇偶
                        if (index + 1) % 2 == 1:
                            # 奇数操作
                            if ((ya - yn) / (xa - xn)) > 0:
                                continue
                            if abs((ya - yn) / (xa - xn)) < abs(min_k):
                                min_k = (ya - yn) / (xa - xn)
                        else:
                            # 偶数操作
                            if ((ya - yn) / (xa - xn)) < 0:
                                continue
                            if abs((ya - yn) / (xa - xn)) < abs(min_k):
                                min_k = (ya - yn) / (xa - xn)

                # 获取线
                coef, intercept, mean_w, mean_h = self.__liner(after)
                # 获取点集
                points_list = []
                for now_droplet in now:
                    xn, yn, w, h, area = now_droplet
                    x, y = self.__line_intersection([coef, intercept], [min_k, xn + w / 2, yn + h / 2])
                    points_list.append([x, y])
                    # cv2.line(self.__draw_img,(int(xn+w/2),int(yn+h/2)),(int(x),int(y)), (0,0,255), 3,4)
                correct_list = []
                # 查看点集合理性
                distance_thold = 25
                for point in points_list:
                    is_locate = 0
                    for after_droplet in after:
                        xa, ya, w, h, area = after_droplet
                        distance = self.__distance(point, [xa + w / 2, ya + h / 2])
                        if distance < distance_thold:
                            is_locate = 1
                            break
                    if is_locate:
                        correct_list.append([int(xa), int(ya), int(w), int(h), int(area)])
                    else:
                        correct_list.append(
                            [int((point[0] - mean_w / 2)), int(point[1] - mean_h / 2), int(mean_w), int(mean_h),
                             int(mean_w * mean_h)])

                correct_list = np.array(correct_list)
                after = correct_list[np.argsort(correct_list[:, 1])]
                droplets[index + 1] = after

            index += 1

    # 粗略查找液滴，使用找轮廓的方法
    def __roughSearch(self):
        # 寻找轮廓
        contours, hierarchy = cv2.findContours(self.__binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 寻找子轮廓所在层
        index = 0 # index为第一个液滴的序号
        for i in hierarchy[0]:
            child_num = 0
            first_child = i[2]
            if first_child == -1:
                continue
            while first_child != -1:
                child_num += 1
                first_child = hierarchy[0][first_child][0]
            if child_num >= 5000:
                index = i[2]
                # print("遍历中的第一个液滴的轮廓层级序号：",index)

        # print("第一个液滴的轮廓层级序号：",index)
        boxes = []
        # 利用层级结构遍历每一个液滴
        while index != -1:
            index = hierarchy[0][index][0]
            cnt = contours[index]
            # 矩形拟合
            rect = cv2.boundingRect(cnt)
            [x, y, w, h] = rect
            area = w * h
            # print("一个液滴的面积：",area)
            # 如果液滴的面积不在阈值内，则跳过
            if self.__area_list[0] >= area or self.__area_list[1] <= area:
                continue
            # 如果液滴的长宽比不符合要求，则跳过
            if w / h <= 1.2 or w / h >= 4:
                continue
            boxes.append([x, y, w, h, area])

        self.__boxes = boxes
        self.__box2droplet()

    #开始分割
    def doCut(self):
        self.__imgProcess()
        print('图片预处理完毕')
        self.__roughSearch()
        print('粗搜寻完毕')
        self.__fineSearch()
        print('细搜寻完毕')
        num = 0
        for droplet in (self.__droplets):
            num += len(droplet)
        print("总液滴数:", num)

    # 画图
    def drawRect(self,droplets,save_path):
        for i, droplet in enumerate(droplets):
            for droplet_small in droplet:
                [x, y, w, h, area] = droplet_small
                cv2.rectangle(self.__draw_img, (int(x), int(y)), (int(x + w), int(y + h)), self.__color[i], 4)
                cv2.putText(self.__draw_img, str(i), (int(x), int(y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(save_path,self.__draw_img)

    # 剪切成很小
    def cutSmall(self,droplets,save_path):
        for i,droplet in enumerate(droplets):
            for j,droplet_small in enumerate(droplet):
                [x,y,w,h,area]=droplet_small
                name=str(i)+'_'+str(j)+'.jpg'
                cut_path=save_path+name
                background=self.__src_img[y:y+h,x:x+w]
                cv2.imwrite(cut_path,background)
        print('分割器剪切完毕！')

    # 剪切成很小
    def cutExample(self,droplet,jpg,save_path):
        [x,y,w,h,area]=droplet
        name=jpg+'.jpg'
        cut_path=save_path+name
        background=self.__src_img[y:y+h,x:x+w]
        cv2.imwrite(cut_path,background)
        
    # 获取液滴
    def getDroplets(self):
        return self.__droplets

    def getB(self,path):
        self.__imgProcess()
        cv2.imwrite(path,self.__binary)
        return 

'''
text=Separator()
text.setImg('./16.jpg')
text.setThresh(99,15,[1400,2400])
text.getB('./result.jpg')
'''
