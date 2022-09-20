import cv2 as cv
from rembg import remove

font = cv.FONT_HERSHEY_COMPLEX
def show(name):
    cv.imshow('', name)
    cv.waitKey()
    cv.destroyAllWindows()

while True:

        img1= cv.imread('./2.jpg')
        img_resize1=cv.resize(img1,(1000,700))

        crop = cv.selectROI("Draw a rectangle over an object inside the image", img_resize1)
        img_crop = img_resize1[int(crop[1]): int(crop[1] + crop[3]), int(crop[0]):int(crop[0] + crop[2])]
        show(img_crop)
        cv.imwrite('img1.jpg', img_crop)

        with open('./img1.jpg', 'rb') as i:
            with open('./img2.jpg', 'wb') as o:
                inp = i.read()
                output = remove(inp)
                o.write(output)
        output=cv.imread('./img2.jpg')
        show(output)

        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        blur = cv.blur(gray, (10, 10))
        ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_OTSU)
        contours, heirarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        img_1=img_crop.copy()
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, 0 * cv.arcLength(cnt, True), True)
            cv.drawContours(img_1, [approx], -1, (255, 0, 0), 2)
            n = approx.ravel()
            i = 0
            for j in n:
                if (i % 2 == 0):
                            x = n[i]
                            y = n[i + 1]
                            string = str(x) + " " + str(y)
                            cv.putText(img_1, string, (x+crop[0], y+crop[2]),font,0.5, (0, 255, 0),1)
                i = i + 1
        cv.imwrite('img77.jpg', img_1)
        above = cv.imread('./img77.jpg')
        below = cv.imread('./1.webp')
        x_offset = crop[0]
        y_offset = crop[1]
        img_resize1[y_offset:y_offset + img_1.shape[0], x_offset:x_offset + img_1.shape[1]] = img_1
        show(img_resize1)
        print()

        x=input("Enter 'q' to exit OR 'y' to continue: ")
        if(x == "q"):
           break




