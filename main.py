import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
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

        gray = cv.cvtColor(img_rgb3, cv.COLOR_BGR2GRAY)
        blur = cv.blur(gray, (10,10))


        ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_OTSU)
        contours, heirarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Iterate over all contours
        for i, c in enumerate(contours):
            # Find mean colour inside this contour by doing a masked mean
            mask = np.zeros(thresh.shape, np.uint8)
            cv.drawContours(mask, [c], 0, (255, 0, 0), 3)
            # DEBUG: cv.imwrite(f"mask-{i}.png",mask)
            mean, _, _, _ = cv.mean(thresh, mask=mask)
            # DEBUG: print(f"i: {i}, mean: {mean}")

            # Get appropriate colour for this label
            #label = 2 if mean > 1.0 else 1
            #colour = RGBforLabel.get(label)
            # DEBUG: print(f"Colour: {colour}")
            # Outline contour in that colour on main image, line thickness=1
            cv.drawContours(img_resize, [c], 0, (255, 0, 0), 3)

        # Save result
        cv.imwrite('result.png', img_resize)
        show(img_resize)










'''
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
if(choice=='q'):
    print()