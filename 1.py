import cv2 as cv
import numpy as np
from rembg import remove

font = cv.FONT_HERSHEY_COMPLEX
def show(name):
    cv.imshow('', name)
    cv.waitKey()
    cv.destroyAllWindows()

choice=input(print("Do you want to continue? Yes/No (press 'q' for No)"))
if (choice=='Yes'):

        img1= cv.imread('./1.webp')
        img_resize1=cv.resize(img1,(1000,700))

        crop = cv.selectROI("Draw a rectangle over an object inside the image", img_resize1)
        print('Selected bounding boxes: {}'.format(crop))
        n=(crop[0], crop[2])
        print(n)

        img_crop = img_resize1[int(crop[1]): int(crop[1] + crop[3]), int(crop[0]):int(crop[0] + crop[2])]
        show("Select an area to crop the image",img_crop)
        cv.imwrite('img1.jpg', img_crop)

        with open('./img1.jpg', 'rb') as i:
            with open('./img2.jpg', 'wb') as o:
                inp = i.read()
                output = remove(inp)
                o.write(output)
        output=cv.imread('./img2.jpg')
        show("Cropped Image",output)


        blank = np.zeros(img_resize1.shape, dtype='uint8')

        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        blur = cv.blur(gray, (10, 10))
        ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_OTSU)

        contours, heirarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        #cv.drawContours(img_resize1,contours, contourIdx=-1, color=(255, 0, 0), thickness=1)
        #show(img_resize1)
        print("contour(0,0)", contours[0][0])
        contours[0][0]=contours[0][0]+crop[0]
        print("contour(0,0)",contours[0][0])
        contours[0][1]=contours[0][1]+crop[2]

        img_1=img_crop.copy()
        for cnt in contours:

            approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)

            cv.drawContours(img_1, [approx], -1, (255, 0, 0), 1)
            n = approx.ravel()
            i = 0
            for j in n:
                if (i % 2 == 0):

                            x = n[i]
                            y = n[i + 1]
                            string = str(x) + " " + str(y)
                            #if (i == 0):
                             #   cv.putText(blank, "tip", (x+crop[0], y+crop[2]),font,
                              #              0.5, (0, 255, 0),1)
                            #else:
                            cv.putText(img_1, string, (x+crop[0], y+crop[2]),font,0.5, (0, 255, 0),1)
                i = i + 1
        show(img_1)
        cv.imwrite('img77.jpg', img_1)
        above = cv.imread('./img77.jpg')
        below = cv.imread('./1.webp')
        x_offset = crop[0]
        y_offset = crop[1]
        img_resize1[y_offset:y_offset + img_1.shape[0], x_offset:x_offset + img_1.shape[1]] = img_1
        #added_image = cv.add(below, above)
        show(img_resize1)

        #c=input("")
        #if(c=='c'):
         #   show(img_resize1)

if (choice == 'q'):
    print()












    '''
        #crop2 = cv.selectROI("Draw a rectangle over an object inside the image", img_rgb3)
        #img_crop2 = img_rgb3[int(crop2[1]): int(crop2[1] + crop2[3]), int(crop2[0]):int(crop2[0] + crop2[2])]
        #cv.imwrite('result2.png', img_crop2)
        #show(img_crop2)
        #above = cv.imread('./result.png')
        #below = cv.imread('./1.webp')
        #added_image = cv.add(below, above)
        #show(added_image)

        #blank = np.zeros(img_resize.shape, dtype='uint8')
        #print(f'Number of contours: {len(contours)}')

        cv.drawContours(img_rgb3, contours, -3, (255,0,0), 10, 1)
        show(img_rgb3)
        cv.imwrite('imgO.jpg', img_rgb3)
        overlay = cv.imread('./imgO.jpg')
        overlay = cv.resize(overlay, (100, 100))
        background = cv.imread('./1.webp')
        background = cv.resize(background, (512, 512))
        added_image = cv.add(background, overlay)
        cv.imwrite('combined.png', added_image)
        show(added_image)

    background = cv.imread('./1.webp')
        background=cv.resize(background, (512,512))
        overlay = cv.imread('./imgO.jpg')
        overlay = cv.resize(overlay, (512, 512))
        added_image = cv.add(background, overlay)
        cv.imwrite('combined.png', added_image)
        show(added_image)

        h, w = overlay.shape[:2]
        added_image = cv.add(background, overlay)
        shapes = np.zeros_like(background, np.uint8)
        shapes[background.shape[1] - h:, background.shape[1] - w:] = overlay
        cv.imwrite('combined.png', added_image)
        show(added_image)

'''
