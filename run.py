import os
import time
import cv2
from DropletSeparator import Separator
from DropletClassifier import Classifier
from BlinkDetector import BlinkDetector
from FileProcessor import FileProcessor

# 路径
src1_path='src1.jpg'
src2_path='src2.jpg'
src3_path='src3.jpg'
detect_list=['fold2']
# 参数
block_size=99
C=14
area_list=[14000,23000]
value=120
# 开关
# 是否输出检测图
is_src1=1
is_src2=1
is_src3=1
# 是否输出小图
is_cut1=0
is_cut2=0
is_cut3=0
# 是否输出颜色分类图
is_color=0
# 是否输出荧光分类
is_blink=0
# 是否输出对比图
is_contrast=1
# 是否输出一个样例
is_example=1


def run(src_path,chip):
    # 实例化一个文件处理器
    fileprocessor=FileProcessor(src_path+'_result/'+chip+'/record.xlsx')
    # 先分割两个图片
    # 实例化两个分割器
    separator1=Separator()
    separator2=Separator()
    # 设置两个分割器的路径
    separator1.setImg(src_path+'/'+chip+'/'+src1_path)
    separator2.setImg(src_path+'/'+chip+'/'+src2_path)
    # 设置两个分割器的参数
    separator1.setThresh(block_size,20,area_list)
    separator2.setThresh(block_size,C,area_list)
    # 开始分割
    separator1.doCut()
    separator2.doCut()
    # 记录数据
    fileprocessor.writeLocation(separator1.getDroplets(), 0)
    fileprocessor.writeLocation(separator2.getDroplets(), 1)

    # 再对结果1进行颜色分类
    classifier=Classifier(src_path+'/'+chip+'/'+src1_path)
    classifier.scan(separator1.getDroplets())
    # 记录数据
    fileprocessor.writeColor(classifier.getDropletsDict())
    fileprocessor.writeClusters(classifier.getClusters())

    # 再对结果
    blink_detector=BlinkDetector(src_path+'/'+chip+'/'+src3_path,value)
    blink_detector.blinkCheck(separator2.getDroplets())
    # 记录数据
    fileprocessor.writeBlink(blink_detector.getBlink(),blink_detector.getNone())

    # 制作结果
    print('制作结果中')
    fileprocessor.doResult()
    print('制作结果完毕')
    print('输出图像')
    makePicture(src_path, chip, separator1.getDroplets(), separator2.getDroplets(), classifier.getDropletsDict(),
                blink_detector.getBlink(), blink_detector.getNone(), classifier.getColor())
    print('输出图像完毕')

def makePicture(src_path,chip,droplets1,droplets2,droplets_dict,blink_list,none_list,colors):
    separator = Separator()
    if is_src1:
        separator.setImg(src_path+'/'+chip+'/'+src1_path)
        separator.drawRect(droplets1,src_path+'_result/'+chip+'/'+src1_path)

    if is_src2:
        separator.setImg(src_path+'/'+chip+'/'+src2_path)
        separator.drawRect(droplets2,src_path+'_result/'+chip+'/'+src2_path)

    if is_src3:
        separator.setImg(src_path+'/'+chip+'/'+src3_path)
        separator.drawRect(droplets2,src_path+'_result/'+chip+'/'+src3_path)

    if is_cut1:
        path=src_path+'_result/'+chip+'/small1/'
        os.mkdir(path)
        separator.setImg(src_path + '/' + chip + '/' + src1_path)
        separator.cutSmall(droplets1,path)

    if is_cut2:
        path=src_path+'_result/'+chip+'/small2/'
        os.mkdir(path)
        separator.setImg(src_path + '/' + chip + '/' + src2_path)
        separator.cutSmall(droplets2,path)

    if is_cut3:
        path=src_path+'_result/'+chip+'/small3/'
        os.mkdir(path)
        separator.setImg(src_path + '/' + chip + '/' + src3_path)
        separator.cutSmall(droplets2,path)

    if is_color:
        classifier = Classifier(src_path + '/' + chip + '/' + src1_path)
        path = src_path + '_result/' + chip + '/color/'
        os.mkdir(path)
        classifier.cut(droplets1, droplets_dict, path)

    if is_blink:
        blink_detector = BlinkDetector(src_path + '/' + chip + '/' + src3_path, value)
        path = src_path + '_result/' + chip + '/blink/'
        os.mkdir(path)
        blink_detector.save(path,droplets2,blink_list,none_list)

    if is_example:
        separator.setImg(src_path + '/' + chip + '/' + src1_path)
        path = src_path + '_result/' + chip + '/example/'
        os.mkdir(path)
        for droplets_type in droplets_dict:
            if droplets_type=='none':
                continue
            for i in range(0,5):
                col,row=droplets_dict[droplets_type][i].split('_')
                jpg=str(droplets_type)+'_'+str(i)
                droplet=droplets1[int(col)][int(row)]
                separator.cutExample(droplet,jpg,path)
        print('样板剪切完毕！')
                
    if is_contrast:
        img1 = cv2.imread(src_path + '/' + chip + '/' + src1_path)
        img3 = cv2.imread(src_path + '/' + chip + '/' + src3_path)
        for droplet_type in droplets_dict:
            if droplet_type == 'none':
                continue
            droplet_type_dict = droplets_dict[droplet_type]
            for jpg in droplet_type_dict:
                if (jpg in blink_list)==False:
                    continue
                col,row=jpg.split('_')
                [x, y, w, h, area] = droplets1[int(col)][int(row)]
                cv2.rectangle(img1, (int(x), int(y)), (int(x + w), int(y + h)), colors[droplet_type], 10)
                [x, y, w, h, area] = droplets2[int(col)][int(row)]
                cv2.rectangle(img3, (int(x), int(y)), (int(x + w), int(y + h)), colors[droplet_type], 10)
        cv2.imwrite(src_path + '_result/' + chip + '/contrast1.jpg', img1)
        cv2.imwrite(src_path + '_result/' + chip + '/contrast3.jpg', img3)

    
        
        
if __name__=="__main__":
   
    total_start = time.time() # 开始计时（总）

    for detect_path in detect_list:
        fold_start = time.time() # 开始计时
        chip_list = os.listdir('./' + detect_path) # 浏览文件夹
        # 如果存在则跳过，如果不存在则建立
        if os.path.exists('./' + detect_path + '_result'):
            pass
        else:
            os.mkdir('./' + detect_path + '_result')
        # 遍历一个fold里的chip
        for chip in chip_list:
            chip_start = time.time()
            result_path ='./' + detect_path + '_result/' + chip
            os.mkdir(result_path)
            try:
                print(detect_path, chip, '开始')
                run('./' + detect_path, chip)
            except:
                print(detect_path, chip, '有问题，跳过')
                continue
            else:
                print(detect_path, chip, '已完成')
                chip_end = time.time()
                print(chip, "运行时间为", chip_end - chip_start, '秒')

        fold_end = time.time()
        print(detect_path, "运行时间为", fold_end - fold_start, '秒')

    print('已全部完成')
    total_end = time.time()
    print("总运行时间为", total_end - total_start, '秒')
    '''
    fileprocessor = FileProcessor('./fold16_result/16-Folder/record.xlsx')
    droplets1,droplets2,droplets_dict,blink_list,none_list,colors=fileprocessor.readInformation(1)
    makePicture('./fold16','16-Folder',droplets1,droplets2,droplets_dict,blink_list,none_list,colors)
     '''
