import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove

font = cv.FONT_HERSHEY_COMPLEX
def show(name):
    cv.imshow('', name)
    cv.waitKey()
    cv.destroyAllWindows()

#choice=input(print("Do you want to continue? Yes/No (press 'q' for No)"))
choice='Yes'
if (choice=='Yes'):
        img= cv.imread('./1.webp')
        img_resize=cv.resize(img,(1000,700))

        crop=cv.selectROI("Draw a rectangle over an object inside the image", img_resize)
        img_crop=img_resize[int(crop[1]): int(crop[1]+crop[3]), int(crop[0]):int(crop[0]+crop[2])]
        img_rgb2=cv.cvtColor(img_crop, cv.COLOR_BGR2RGB)
        show(img_rgb2)

        cv.imwrite('img3.jpg', img_rgb2)

        with open('./img3.jpg', 'rb') as i:
            with open('./img4.jpg', 'wb') as o:
                input = i.read()
                output = remove(input)
                o.write(output)

        output=cv.imread('./img4.jpg')
        img_rgb3=cv.cvtColor(output, cv.COLOR_BGR2RGB)
        show(img_rgb3)

        blank = np.zeros(img_rgb3.shape, dtype='uint8')
        cv.imshow('Blank', blank)

        gray = cv.cvtColor(img_rgb3, cv.COLOR_BGR2GRAY)
        show(gray)
        blur = cv.blur(gray, (10,10))
        show(blur)


        ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_OTSU)
        show(thresh)
        contours, heirarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        print(f'Number of countours: {len(contours)}')
        cv.drawContours(blank, contours=contours, contourIdx=-1, color=(0, 255, 255), thickness=2)



        for i, c in enumerate(contours):
                mask = np.zeros(thresh.shape, np.uint8)
                cv.drawContours(mask, [c], 0, (0, 255, 0), 3)
                mean, _, _, _ = cv.mean(thresh, mask=mask)
                cv.drawContours(img_rgb3, [c], 0, (255, 0, 0), 3)

        # Going through every contours found in the image.
        for cnt in contours:

            approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)

            # draws boundary of contours.
            cv.drawContours(img_rgb3, [approx], 0, (0, 0, 255), 5)

            # Used to flatted the array containing
            # the co-ordinates of the vertices.
            n = approx.ravel()
            i = 0

            for j in n:
                if (i % 2 == 0):
                    x = n[i]
                    y = n[i + 1]

                    # String containing the co-ordinates.
                    string = str(x) + " " + str(y)

                    if (i == 0):
                        # text on topmost co-ordinate.
                        cv.putText(img_rgb3, "tip", (x, y),font,
                                    0.5, (255, 0, 0))
                    else:
                        # text on remaining co-ordinates.
                        cv.putText(img_rgb3, string, (x, y),font,
                                     0.5, (0, 255, 0))
                i = i + 1

        # Showing the final image.

        show(blank)

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
