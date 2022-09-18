import cv2 as cv
#import numpy as np
import matplotlib.pyplot as plt
from rembg import remove

img= cv.imread('./1.webp')
img_resize=cv.resize(img,(1000,700))
img_rgb=cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)

crop=cv.selectROI("Draw a rectangle over an object inside the image", img_rgb)

img_crop=img_resize[int(crop[1]): int(crop[1]+crop[3]), int(crop[0]):int(crop[0]+crop[2])]
img_rgb2=cv.cvtColor(img_crop, cv.COLOR_BGR2RGB)

#plt.imshow(img_rgb2)
#plt.show(img_rgb2)


cv.imshow('bg_before', img_rgb2)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite('img3.jpg', img_rgb2)

#plt.imsave('./img3', img_rgb2)

with open('./img3.jpg', 'rb') as i:
    with open('./img4.jpg', 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)


output=cv.imread('./img4.jpg')

img_rgb3=cv.cvtColor(output, cv.COLOR_BGR2RGB)
cv.imshow('bg_after', img_rgb3)
cv.waitKey()
cv.destroyAllWindows()

#before_border=cv.findContours(output,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#final_border=cv.drawContours(output, before_border, -1, 255,3)

#image = cv.imread('img4.jpg')
#image = cv.copyMakeBorder(image,20,20,20,20, cv.BORDER_CONSTANT, 255)
#img_rgb4=cv.cvtColor(image, cv.COLOR_BGR2RGB)
#cv.imwrite('img5.jpg', img_rgb4)

border = cv.copyMakeBorder(img_rgb3,10,10,10,10, cv.BORDER_CONSTANT)

#cv.imshow('image', img_rgb3)
#cv.imshow('bottom', bottom)
cv.imshow('border', border)

#cv.imshow('final', img_rgb4)
cv.waitKey()
cv.destroyAllWindows()