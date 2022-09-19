import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove

font = cv.FONT_HERSHEY_COMPLEX
def show(name):
    cv.imshow('', name)
    cv.waitKey()
    cv.destroyAllWindows()

choice='Yes'
if (choice=='Yes'):

        img1= cv.imread('./1.webp')
        img_resize1=cv.resize(img1,(1000,700))

        crop = cv.selectROI("Draw a rectangle over an object inside the image", img_resize1)
        img_crop = img_resize1[int(crop[1]): int(crop[1] + crop[3]), int(crop[0]):int(crop[0] + crop[2])]
        show(img_crop)
        cv.imwrite('img1.jpg', img_crop)

        with open('./img1.jpg', 'rb') as i:
            with open('./img2.jpg', 'wb') as o:
                input = i.read()
                output = remove(input)
                o.write(output)
        output=cv.imread('./img2.jpg')
        show(output)


        blank = np.zeros(img_resize1.shape, dtype='uint8')

        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        blur = cv.blur(gray, (10, 10))
        ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_OTSU)

        contours, heirarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(blank, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
        show(blank)

        for cnt in contours:
            approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)
            cv.drawContours(blank, [approx], 0, (0, 0, 0), 1)
            n = approx.ravel()
            i = 0
            for j in n:
                if (i % 2 == 0):
                            x = n[i]
                            y = n[i + 1]
                            string = str(x) + " " + str(y)
                            if (i == 0):
                                cv.putText(blank, "tip", (x, y),font,
                                            0.5, (0, 255, 0),1)
                            else:
                                cv.putText(blank, string, (x, y),font,
                                            0.5, (0, 255, 0),1)
                i = i + 1
        show(blank)



