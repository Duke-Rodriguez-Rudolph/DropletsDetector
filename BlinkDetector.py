import cv2
import numpy as np
import os

class BlinkDetector:
    # 构造函数
    def __init__(self, src_path,value):
        self.__src_img = cv2.imread(src_path)
        self.__value=value

    def blinkCheck(self,droplets):
        blink_candidate = {}
        blink_list = [] #存储所有发光
        none_list=[]    #存储所有不发光
        all_np_list = []
        np_list = []
        for col, droplet in enumerate(droplets):
            for row,droplet_small in enumerate(droplet):
                [x, y, w, h, area] = droplet_small
                img = self.__src_img[y:y + h, x:x + w]
                is_blink, mean_v = self.__single(img)
                check_path=str(col)+'_'+str(row)
                if is_blink:
                    blink_candidate[check_path] = mean_v
                    np_list.append(mean_v)
                else:
                    none_list.append(check_path)
                all_np_list.append(mean_v)
        np_list = np.array(np_list)
        all_np_list = np.array(all_np_list)
        print('np_mean:', np.mean(np_list), 'np_var:', np.var(np_list))
        print('all_np_mean:', np.mean(all_np_list), 'all_np_var:', np.var(all_np_list))

        for check_path in blink_candidate:
            if blink_candidate[check_path] > (np.mean(np_list) - 18):
                blink_list.append(check_path)
            else:
                none_list.append(check_path)

        self.__blink_list=blink_list
        self.__none_list=none_list

    #取出结果
    def getBlink(self):
        return self.__blink_list

    def getNone(self):
        return self.__none_list

    # 检测亮
    def __single(self,img):
        # 转为hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 获得掩码
        lower = np.array([0, 0, self.__value])
        higher = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, higher)
        # 寻找轮廓
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        is_blink = 0
        mean_v = -1
        for cnt in contours:
            # 矩形拟合
            rect = cv2.boundingRect(cnt)
            [x, y, w, h] = rect
            # area=w*h
            area = cv2.contourArea(cnt)
            if w / h > 1.3 or h / w > 1.3:
                continue
            if x + w / 2 > img.shape[1] / 2:
                continue
            if area / (w * h) > 0.5 and w * h > img.shape[0] * img.shape[1] * 0.1:
                # cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),(0,0,255), 4)
                # print('area:',area)
                # print('img.shape[0]*img.shape[1]*0.25:',img.shape[0]*img.shape[1]*0.3)
                is_blink = 1
                # 拆分通道
                all_v = 0
                H, S, V = cv2.split(hsv)
                for row in range(y + int(h / 4), y + +int(h / 4 * 3)):
                    for col in range(x + int(w / 4), x + int(w / 4 * 3)):
                        all_v += V[row, col]
                mean_v = all_v / (w * h)
                break
        # cv2.imshow('mask',mask)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        return is_blink, mean_v

    #存储
    def save(self,save_path,droplets,blink_list,none_list):
        #建立文件夹
        os.mkdir(save_path+'blink/')
        os.mkdir(save_path + 'none/')

        for blink in blink_list:
            path=save_path+'blink/'+blink+'.jpg'
            col,row=blink.split('_')
            [x, y, w, h, area] = droplets[int(col)][int(row)]
            img = self.__src_img[y:y + h, x:x + w]
            cv2.imwrite(path,img)

        for none in none_list:
            path = save_path+'none/'+none+'.jpg'
            col,row=none.split('_')
            [x, y, w, h, area] = droplets[int(col)][int(row)]
            img = self.__src_img[y:y + h, x:x + w]
            cv2.imwrite(path,img)
