import imutils
import numpy as np
import cv2


def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture("rtsp://admin:admin@192.168.0.182/user=admin_password=admin_channel=1_stream=0.sdp")  # Rtsp Адрес при работе на камере в аудитории 221 вставить "rtsp://admin:admin@192.168.0.182/user=admin_password=admin_channel=1_stream=0.sdp"
fgbg = cv2.createBackgroundSubtractorMOG2()  # Обнаружение движения

detects = []  #Обнаружение

posL = 350 # Размещение синей линии
offset = 200   # Смещение линии границ (голубых)

xy1 = (300, posL)  # Начальная точка линий относитель левой стороны окна
xy2 = (950, posL)  #Начальная точка линий относитель правой стороны окна

total = 0 # Итог

up = 0
down = 0

while 1:
    ret, frame = cap.read()
    #frame = cv2.GaussianBlur(frame, (51, 51), 0)
    frame = imutils.rotate(frame,angle=270)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)



    fgmask = fgbg.apply(gray)

    fgmask = fgmask * (fgmask > 210)
    fgmask = cv2.GaussianBlur(fgmask, (3,3), 0)
    fgmask = fgmask * (fgmask > 210)

    cv2.imshow("fgmask", fgmask)
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    th = cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("th", th)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    # cv2.imshow("opening", opening)

    dilation = cv2.dilate(opening, kernel, iterations=8)
    # cv2.imshow("dilation", dilation)

    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=8)
    #cv2.imshow("closing", closing)

    cv2.line(frame, xy1, xy2, (255, 0, 0), 3)

    cv2.line(frame, (xy1[0], posL - offset), (xy2[0], posL - offset), (255, 255, 0), 2)

    cv2.line(frame, (xy1[0], posL + offset), (xy2[0], posL + offset), (255, 255, 0), 2)

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)

        if int(area) > 20000:
            centro = center(x, y, w, h)

            cv2.putText(frame, str(i), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(frame, centro, 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if len(detects) <= i:
                detects.append([])
            if centro[1] > posL - offset and centro[1] < posL + offset:
                detects[i].append(centro)
            else:
                detects[i].clear()
            i += 1

    if i == 0:
        detects.clear()

    i = 0

    if len(contours) == 0:
        detects.clear()

    else:

        for detect in detects:
            for (c, l) in enumerate(detect):

                if detect[c - 1][1] < posL and l[1] > posL:
                    detect.clear()
                    up += 1
                    #total += 1
                    cv2.line(frame, xy1, xy2, (0, 255, 0), 5)
                    continue

                if detect[c - 1][1] > posL and l[1] < posL:
                    detect.clear()
                    down += 1
                    #total += 1
                    cv2.line(frame, xy1, xy2, (0, 0, 255), 5)
                    continue

                if c > 0:
                    cv2.line(frame, detect[c - 1], l, (0, 0, 255), 1)

    alltotal = up - down
    cv2.putText(frame, ("В помещении : " + str(alltotal)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Вошло: " + str(up), (10, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Вышло: " + str(down), (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()