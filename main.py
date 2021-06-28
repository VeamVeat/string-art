from PIL import Image, ImageOps, ImageDraw,ImageEnhance,ImageColor
import math
from numba import jit
from PIL import ImageFilter
import cv2
from bresenham import bresenham
from skimage.color import rgb2gray
from skimage.draw import line
import numpy as np
import matplotlib.pyplot as plt
import sys


#Исходное изображение квадратное
#Дописать обработку изображения для прямоугольного

// закомментировать блок после обработки изображения
basewidth = 2000
image = Image.open(r"D:\Разработки\String-atrt\дева_анфаз.jpg")
npImage=np.array(image)
h,w=image.size

# Create same size alpha layer with circle
alpha = Image.new('L', image.size,0)
draw = ImageDraw.Draw(alpha)
draw.pieslice([0,0,h,w],0,360,fill=255)


# Convert alpha Image to numpy array
npAlpha=np.array(alpha)

# Add alpha layer to RGB
img=np.dstack((npImage,npAlpha))

img= Image.fromarray(img)

#изменение размера
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))

img = img.resize((basewidth, hsize), Image.ANTIALIAS)

img = img.filter(ImageFilter.SHARPEN)

#оттенки серого
img = img.convert('LA')

# #добавление резкости
img = ImageEnhance.Sharpness(img)
img = img.enhance(15.0)
#
# #увеличение контрастности серого
enhactor = ImageEnhance.Contrast(img)
factor = 0.64
img = enhactor.enhance(factor)
img.save('дева_анфаз.png')
// закоментить всё что сверху после преобразования изображения в оттенки серого




#количество гвоздей по окружности
inputs = np.array([[round((i * 0.01) - 1,3) for k in range(1)] for i in range(201)])
print(inputs[200])

NPOINTS = 300

#исходное обработанное изображение (2000*2000)
image = Image.open(r"D:\Методы интелектуального анализа данных\Второйсиместр\String-art\дева_анфаз.png")
width,height = image.size

ArrayImage = np.array(image) #ArrayImage[,][0], один канал цвета
drawOriginal = ImageDraw.Draw(image)

#создаём холст для рисования
img = Image.new( mode = "RGB", size = (width, height) ,color=(255,255,255))
draw = ImageDraw.Draw(img)

radius = int(width/2)

#рисуем окружность
bbox = (width/2 - width/2, height/2 - height/2, width/2 + width/2 - 1, height/2 + height/2 - 1)
draw.ellipse(bbox,outline=0)

#массив точек по окружности
ListPointsCercle = [[0 for j in range(2)] for i in range(NPOINTS)]

shag = float(90/(NPOINTS/4))
k = 0

for i in np.arange(0.0,360.0,shag):
    #создаём координаты
    x = int(radius*np.cos(math.radians(i)))
    y = int(radius*np.sin(math.radians(i)))

    # сохраняем координаты
    if x + radius == 2000:
        ListPointsCercle[k][0] = x + radius - 1
    else:
        ListPointsCercle[k][0] = x + radius
    if y + radius == 2000:
        ListPointsCercle[k][1] = y + radius - 1
    else:
        ListPointsCercle[k][1] = y + radius

    #рисуем точки по окружности
    bbox = (x-4+radius, y-4+radius, x+4+radius,y+4+radius)
    draw.ellipse(bbox,  fill=ImageColor.getcolor("#c67c40", "RGB"))
    # draw.point((ListPointsCercle[k][0], ListPointsCercle[k][1]), (34, 253, 57, 0))
    k+=1

print(ListPointsCercle[0][0], ListPointsCercle[0][1])

#получение массива всех пикселей линии(переписать, что бы заполняло каждый пиксель)
def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

ITERATION = 2800

#начальная точка линии
StartPointX = ListPointsCercle[47][0]
StartPointY = ListPointsCercle[47][1]

#конечная точка линии
EndPointX = 0
EndPointY = 0

ArrayImageWhite = np.array(img)

for k in range(ITERATION):
    ListSumColor = [[0 for j in range(5)] for i in range(np.shape(ListPointsCercle)[0] - 1)]
    IndexColor = 0

    for i in range(0,np.shape(ListPointsCercle)[0]): #проход по всем линиям
        if StartPointX != ListPointsCercle[i][0] or StartPointY != ListPointsCercle[i][1]:

            ListPoint = get_line(StartPointX, StartPointY, ListPointsCercle[i][0],
                                ListPointsCercle[i][1])  # коодинаты всех точек одной линии

            CountPixel = np.shape(ListPoint)[0]
            ArrayColor = [0 for k in range(CountPixel)]

            for j in range(CountPixel):
                ArrayColor[j] = ArrayImage[ListPoint[j][0], ListPoint[j][1]][0]  # картеж [x,y][0]
            # if k == 2:
            #     print(ArrayColor,"Интенсивность пикселя")
            ListSumColor[IndexColor][0] = int(255 - (sum(ArrayColor) / CountPixel))
            ListSumColor[IndexColor][1] = StartPointX
            ListSumColor[IndexColor][2] = StartPointY
            ListSumColor[IndexColor][3] = ListPointsCercle[i][0]  # X
            ListSumColor[IndexColor][4] = ListPointsCercle[i][1]  # Y
            IndexColor+=1

    MinElement = ListSumColor[0][0]
    # print(MinElement)
    IndexMinElement = 0

    for i in range(0,np.shape(ListSumColor)[0]):
        if ListSumColor[i][0] > MinElement and ListSumColor[i][0] != 0:
            MinElement = ListSumColor[i][0]
            IndexMinElement = i

    NewLine = get_line(int(ListSumColor[IndexMinElement][1]), int(ListSumColor[IndexMinElement][2]),
                      int(ListSumColor[IndexMinElement][3]), int(ListSumColor[IndexMinElement][4]))

    for j in range(np.shape(NewLine)[0]):
        if NewLine[j][0] != 1999 and NewLine[j][1] != 1999:
            UpColor = ArrayImageWhite[NewLine[j][0], NewLine[j][1] - 1][0] - 4
            DownColor = ArrayImageWhite[NewLine[j][0], NewLine[j][1] + 1][0] - 4
            CenterColor = int(ArrayImageWhite[NewLine[j][0], NewLine[j][1]][0]/3)
            LeftColor = ArrayImageWhite[NewLine[j][0] - 1, NewLine[j][1]][0] - 4
            RightColor = ArrayImageWhite[NewLine[j][0] + 1, NewLine[j][1]][0] - 4

            for g in range(3):#rgb
                ArrayImageWhite[NewLine[j][0], NewLine[j][1] - 1][g] = np.where(UpColor<0,0,UpColor)
                ArrayImageWhite[NewLine[j][0], NewLine[j][1] + 1][g] = np.where(DownColor<0,0,DownColor)
                ArrayImageWhite[NewLine[j][0], NewLine[j][1]][g] = CenterColor
                ArrayImageWhite[NewLine[j][0] - 1, NewLine[j][1]][g] = np.where(LeftColor<0,0,LeftColor)
                ArrayImageWhite[NewLine[j][0] + 1, NewLine[j][1]][g] = np.where(RightColor<0,0,RightColor)

        ArrayImage[NewLine[j][0], NewLine[j][1]][0] = 255  # обнуляем точки в оригинальном изображении

    StartPointX = int(ListSumColor[IndexMinElement][3])
    StartPointY = int(ListSumColor[IndexMinElement][4])

    if k % 200 == 0:
        print("Итерация-", k, "\n")

Img = Image.fromarray(ArrayImageWhite, 'RGB')
Img.show()
