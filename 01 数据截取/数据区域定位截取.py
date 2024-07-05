import cv2
import numpy as np
from PIL import Image, ImageFilter
import os

# 读取图像和模板
image = cv2.imread('data_image.png')
template1 = cv2.imread('sensor_template.png')
template2 = cv2.imread('wave_template.png')

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
gray_template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)

# 获取模板的尺寸
h1, w1 = gray_template1.shape
h2, w2 = gray_template2.shape

# 匹配模板
res1 = cv2.matchTemplate(gray_image, gray_template1, cv2.TM_CCOEFF_NORMED)
res2 = cv2.matchTemplate(gray_image, gray_template2, cv2.TM_CCOEFF_NORMED)

# 确定最佳匹配位置
min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)

# 截取数据图片
path_list = next(os.walk('.'))[1]
path_list = path_list[1:]
dir_cnt = len(path_list)
print(path_list)
print(dir_cnt)
for i in range(len(path_list)):

    #   -------------截取图片固定区域-------------   #
    for f in os.listdir(path_list[i]):
        if not os.path.exists('crops-{}'.format(path_list[i])):
            os.mkdir('crops-{}'.format(path_list[i]))
        if f.endswith('.png'):
            im = Image.open(path_list[i] + '/' + f)
            fn, fext = os.path.splitext(f)
            cropim = im.crop((max_loc2[0] - 51, max_loc2[1] + 71, max_loc2[0] + 31, max_loc2[1] + 88))  # 图片中截取的位置
            cropim.save('crops-{}/'.format(path_list[i]) + '{}-Channel 1.png'.format(fn))  # 截图存储：位置/命名

            cropim2 = im.crop((max_loc2[0] + 231, max_loc2[1] + 71, max_loc2[0] + 231 + 82, max_loc2[1] + 88))
            cropim2.save('crops-{}/'.format(path_list[i]) + '{}-Channel 2.png'.format(fn))  # 截图存储：位置/命名

            cropim3 = im.crop((max_loc2[0] + 511, max_loc2[1] + 71, max_loc2[0] + 511 + 82, max_loc2[1] + 88))  # 图片中截取的位置
            cropim3.save('crops-{}/'.format(path_list[i]) + '{}-Channel 3.png'.format(fn))  # 截图存储：位置/命名

            cropim4 = im.crop((max_loc1[0] - 21, max_loc1[1] + 52, max_loc1[0] + 264, max_loc1[1] + 128))
            cropim4.save('crops-{}/'.format(path_list[i]) + '{}-Force.png'.format(fn))  # 截图存储：位置/命名




'''
im = Image.open('data_image.png')
fn, fext = os.path.splitext(f)

cropim = im.crop((max_loc2[0] - 51, max_loc2[1] + 71, max_loc2[0] + 31, max_loc2[1] + 88))  # 图片中截取的位置
cropim.save('crops-{}/'.format(path_list[i]) + '{}-Channel 1.png'.format(fn))  # 截图存储：位置/命名

cropim2 = im.crop((max_loc2[0] + 231, max_loc2[1] + 71, max_loc2[0] + 231 + 82, max_loc2[1] + 88))
cropim2.save('crops-{}/'.format(path_list[i]) + '{}-Channel 2.png'.format(fn))  # 截图存储：位置/命名

cropim3 = im.crop((max_loc2[0] + 511, max_loc2[1] + 71, max_loc2[0] + 511 + 82, max_loc2[1] + 88))  # 图片中截取的位置
cropim3.save('crops-{}/'.format(path_list[i]) + '{}-Channel 3.png'.format(fn))  # 截图存储：位置/命名

cropim4 = im.crop((max_loc1[0] - 21, max_loc1[1] + 52, max_loc1[0] + 264, max_loc1[1] + 128))
cropim4.save('crops-{}/'.format(path_list[i]) + '{}-Force.png'.format(fn))  # 截图存储：位置/命名

# 绘制矩形框
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image, top_left, bottom_right, 255, 2)

# 显示结果
cv2.imshow('Matched Window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(top_left)
'''