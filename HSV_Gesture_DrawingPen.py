# Import Module
import cv2
import numpy as np

# ===== Function ===== #
# HSV 필터 범위 사용자 설정
def Get_Filter_Value():
    def nothing(x):
        pass
    if __name__ == '__main__':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3,640)
        cap.set(4,360)
        cv2.namedWindow("HSV Value")
        cv2.createTrackbar("H MIN", "HSV Value", 0, 255, nothing)
        cv2.createTrackbar("S MIN", "HSV Value", 0, 255, nothing)
        cv2.createTrackbar("V MIN", "HSV Value", 0, 255, nothing)
        cv2.createTrackbar("H MAX", "HSV Value", 255, 255, nothing)
        cv2.createTrackbar("S MAX", "HSV Value", 255, 255, nothing)
        cv2.createTrackbar("V MAX", "HSV Value", 255, 255, nothing)

        button = 'y'
        [Y_lower_range, Y_upper_range,
         G_lower_range, G_upper_range,
         R_lower_range, R_upper_range,
         Skin_lower_range, Skin_upper_range] = np.array([[0, 0, 0], [255, 255, 255]] * 4)


    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("H MIN", "HSV Value")
        s_min = cv2.getTrackbarPos("S MIN", "HSV Value")
        v_min = cv2.getTrackbarPos("V MIN", "HSV Value")
        h_max = cv2.getTrackbarPos("H MAX", "HSV Value")
        s_max = cv2.getTrackbarPos("S MAX", "HSV Value")
        v_max = cv2.getTrackbarPos("V MAX", "HSV Value")

        lower_value = np.array([h_min, s_min, v_min])
        upper_value = np.array([h_max, s_max, v_max])

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 121:
            button = 'y'
        elif key == 103:
            button = 'g'
        elif key == 114:
            button = 'r'
        elif key == 115:
            button = 's'

        if button == 's':
            Skin_lower_range = lower_value
            Skin_upper_range = upper_value
            skinRegionHSV = cv2.inRange(hsv, Skin_lower_range, Skin_upper_range)
            skin_blur = cv2.blur(skinRegionHSV, (2, 2))
            ret, mask = cv2.threshold(skin_blur, 0, 255, cv2.THRESH_BINARY)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.putText(result, 'Skin_Fliter', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (177, 206, 251), 2)

        else:
            mask = cv2.inRange(hsv, lower_value, upper_value)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            if button == 'y':
                Y_lower_range = lower_value
                Y_upper_range = upper_value
                cv2.putText(result, 'Yellow_Fliter', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif button == 'g':
                G_lower_range = lower_value
                G_upper_range = upper_value
                cv2.putText(result, 'Green_Fliter', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif button == 'r':
                R_lower_range = lower_value
                R_upper_range = upper_value
                cv2.putText(result, 'Red_Fliter', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("HSV Value", result)
        cv2.imshow("Mask", mask)

    range_value = [Y_lower_range, Y_upper_range,
                   G_lower_range, G_upper_range,
                   R_lower_range, R_upper_range,
                   Skin_lower_range, Skin_upper_range]
    print(range_value)
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(range_value)
# Hand Detecting ROI 설정
def Set_ROI(event, x, y, flags, param):
    global isDragging, x0, y0, x1, y1, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = (x - 640) * 2
        y0 = y * 2
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            cv2.rectangle(frame, (x0, y0), (x, y), [255,0,0], 2)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            x1 = (x - 640) * 2
            y1 = y * 2
            w = x1 - x0
            h = y1 - y0

            if w > 0 and h > 0:
                roi = frame[y0:y0 + h, x0:x0 + w]
            else:
                print('drag should start from left-top side')
# Yellow, Green, Red 검출
def Color_Mask(lower_range, upper_range):
    global hsv
    global frame
    kernel = np.ones((5, 5), np.uint8)
    hsv_mask = cv2.inRange(hsv, lower_range, upper_range)
    hsv_mask = cv2.erode(hsv_mask, kernel, iterations=1)
    hsv_mask = cv2.dilate(hsv_mask, kernel, iterations=2)
    res_hsv = cv2.bitwise_and(frame, frame, mask=hsv_mask)
    return hsv_mask
# Hand 검출
def Skin_Mask(img):
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skinRegionHSV = cv2.inRange(hsvim, Skin_lower_range, Skin_upper_range)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    return thresh
# Hand Hull 검출
def Get_Hull(mask_img):
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(contours)
    return contours, hull
# Hand Defects 검출
def Get_Defects(contours):
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull)
    return defects

# ====== Main ====== #
# 전역변수 초기화 / 프로그램 초기값 설정

# HSV 필터값 조절 #
Custom_Range = False #True : 조절, False : 디폴트값
if Custom_Range:
    filter_value = Get_Filter_Value()
else:
    filter_value = np.array([[ 15,  65, 165], [29, 255, 255], [ 32,  72, 115], [ 76, 255, 255], [ 82,  92, 156], [255, 255, 255], [ 0,  33, 136], [188, 255, 255]])

# HSV 필터값 설정
Y_lower_range, Y_upper_range, G_lower_range, G_upper_range,R_lower_range, R_upper_range, Skin_lower_range, Skin_upper_range = filter_value

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)
canvas = None
clear_cavas = np.zeros_like(cap)


#컬러펜 변수 설정
xr, yr = 0, 0
xg, yg = 0, 0
xy, yy = 0, 0
y_btn = 0
g_btn = 0
r_btn = 0

#ROI 변수 설정
isDragging = False
x0, y0, x1, y1 = -1, -1, -1, -1

#제스쳐 인식 변수 설정
cnt_temp_array = np.array([0,0,0])
cnt_filtered = 0
n = 0

# 프로그램 실행
while (1):
    # ROI 설정
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_roi = frame[y0:y1, x0:x1]
    cv2.setMouseCallback('VIRTUAL PEN', Set_ROI) #마우스 이벤트 콜백 함수

    if not isDragging:
        frame = cv2.rectangle(frame, (x0+1280, y0), (x1+1280,y1), color=(255, 255, 255), thickness=1, lineType=cv2.LINE_4,
                              shift=None)

    # Convex Hull을 통한 제스쳐 인식
    try:
        # 마스크를통한 핸드 디텍팅
        Skin_mask = Skin_Mask(frame_roi)
        contours, hull = Get_Hull(Skin_mask)
        cv2.drawContours(frame_roi, [contours], -1, (255, 255, 0), 2)
        cv2.drawContours(frame_roi, [hull], -1, (0, 255, 255), 2)
        defects = Get_Defects(contours)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):  # 손가락 사이각 계산
                s, e, f, d = defects[i][0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                #손가락 사이 삼각형의 각 변의 길이 계산
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                # 제2코사인 법칙
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                # 90도 보다 작을 시 손가락으로 판정
                if angle <= np.pi / 2:
                    cnt += 1
                    cv2.circle(frame_roi, far, 4, [0, 0, 255], -1)
            if cnt > 0:
                cnt = cnt + 1
            if cnt > 5:
                cnt = 5
            # Moving Average Filter 손가락 튀는값 제어
            cnt_temp_array[n] = cnt
            n += 1
            n = n%3
            cnt_filtered = np.floor(cnt_temp_array.mean()) #평균 계산
            cv2.putText(frame_roi, str(cnt_filtered), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    except:
        pass

    if canvas is None:
        canvas = np.zeros_like(frame)
    # HSV 필터
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Y_mask = Color_Mask(Y_lower_range, Y_upper_range)
    R_mask = Color_Mask(R_lower_range, R_upper_range)
    G_mask = Color_Mask( G_lower_range, G_upper_range)
    # Virtual Pen
    # Yellow
    contours, hierarchy = cv2.findContours(Y_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for con_pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (0, 255, 255), 2)

            cv2.putText(frame, "Yellow Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255))
            if y_btn == 2:
                xy, yy = x,y
                y_btn = 0
            else:
                # 캔버스에 그림 그리기
                if cnt_filtered < 2:
                    canvas = cv2.line(canvas, (xy, yy), (x, y), [0, 255, 255], 30)
                    xy, yy = x, y
                    y_btn = 0
                if cnt_filtered == 3 or cnt_filtered == 4:
                    canvas = cv2.line(canvas, (xy, yy), (x, y), [0, 0, 0], 30)
                    xy, yy = x, y
                    y_btn = 0

    # yellow countour가 2회 보이지 않을경우 좌표 초기화
    y_btn += 1
    if y_btn > 2:
        y_btn = 2

    #Red
    contours, hierarchy = cv2.findContours(R_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for con_pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(frame, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255))
            if r_btn == 2:
                xr, yr = x,y
                r_btn = 0
            else:
                if cnt_filtered < 2:
                    canvas = cv2.line(canvas, (xr, yr), (x, y), [0, 0,255], 30)
                    xr, yr = x, y
                    r_btn = 0
                if cnt_filtered == 3 or cnt_filtered == 4:
                    canvas = cv2.line(canvas, (xr, yr), (x, y), [0, 0, 0], 30)
                    xr, yr = x, y
                    r_btn = 0
    r_btn += 1
    if r_btn > 2:
        r_btn = 2

    #Green
    contours, hierarchy = cv2.findContours(G_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for con_pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (0, 255, 0), 2)

            cv2.putText(frame, "Green Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0))
            if g_btn == 2:
                xg, yg = x,y
                g_btn = 0
            else:
                if cnt_filtered < 2:
                    canvas = cv2.line(canvas, (xg, yg), (x, y), [0, 255, 0], 30)
                    xg, yg = x, y
                    g_btn = 0
                if cnt_filtered == 3 or cnt_filtered == 4:
                    canvas = cv2.line(canvas, (xg, yg), (x, y), [0, 0, 0], 30)
                    xg, yg = x, y
                    g_btn = 0
    g_btn += 1
    if g_btn > 2:
        g_btn = 2

    frame = cv2.add(frame, canvas)
    stacked = np.vstack((frame,canvas))
    cv2.imshow('VIRTUAL PEN', cv2.resize(stacked, None, fx=0.5, fy=0.5))

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if cnt_filtered == 5:
            canvas = None

#Close Window
cv2.destroyAllWindows()
cap.release()



