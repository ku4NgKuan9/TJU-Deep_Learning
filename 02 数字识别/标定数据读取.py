import os
import pytesseract
import standard as standard
from PIL import Image
import xlwt
import xlsxwriter as xw
from PIL import Image, ImageFilter

#  ！！输入通道数量！！  （允许范围：1~5）
cntt2 = 4   #数据的列数
def xw_toExcel(cntt2,data, fileName):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['序号', 'CH-1','CH-2','CH-3','Force']  # 设置表头
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    i = 2  # 从第二行开始写入数据
    for j in range(len(data)):
        row = 'A' + str(i)
        worksheet1.write_row(row, data[j])
        i += 1
    workbook.close()  # 关闭表

value = []
cnt = 1
num = 250  #  ！！输入大于每组图片数量的数！！
jj = 1
str_list = []

for f in os.listdir('crops-轴向'):
    if f.endswith('.png'):
        s = 'crops-轴向/' + f
        img = Image.open(s)
        # 图像预处理
        img = img.convert('L')
        img = img.filter(ImageFilter.SMOOTH_MORE)
        # 图像识别
        text = pytesseract.image_to_string(img, config = r'--oem 1 --psm 6 -l eng tessedit_char_whitelist = 0123456789').replace(',', '.')  #Neural nets LSTM engine only
        # 识别结果后处理
        text = text.replace(' ', '')
        text = text.replace('.', '')
        text = text[:4] + '.' + text[4:8]
        text = text.rstrip('\n')  # 右侧去除多余字符
        text = text.rstrip('.')
        # 列表
        str_list = str_list + [text]
        jj = jj + 1
        if jj == cntt2 + 1:
            jj = 1
            str_list = [cnt] + str_list
            value.append(str_list)
            str_list = []
            cnt = cnt + 1

print(cnt)

fileName = '轴向-数据识别.xlsx'

xw_toExcel(cntt2, value, fileName)
