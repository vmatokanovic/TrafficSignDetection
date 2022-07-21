import cv2
import numpy as np
import os
 
path = 'RefImages'

def imgRedMask(mImg):
    img_blur = cv2.GaussianBlur(mImg, (5, 5), 0)
    hsv_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([125, 50, 30]) #zadnja vrijednost je default 50
    upper_red1 = np.array([180, 255, 255])

    lower_red2 = np.array([0, 50, 50])
    upper_red2 = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask= cv2.bitwise_or(red_mask1, red_mask2)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    return red_mask

def imgBlueMask(mImg):
    img_blur = cv2.GaussianBlur(mImg, (5, 5), 0)
    hsv_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 170, 60])
    upper_blue = np.array([125, 255, 255])
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    return blue_mask

def getRedContours(mImg):
    shape = -1
    contours,hierarchy = cv2.findContours(mImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>90000:
            ##################NOVO DODANO#################
            mask = np.zeros((hh,ww), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, (255,255,255), cv2.FILLED)
            mask_inv = 255 - mask
            bckgnd = np.full_like(img, (255,255,255))
            image_masked = cv2.bitwise_and(img, img, mask=mask)
            bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)
            result = cv2.add(image_masked, bckgnd_masked)
            cv2.imwrite('shapes_inverted_mask.jpg', mask_inv)
            cv2.imwrite('shapes_masked.jpg', image_masked)
            cv2.imwrite('shapes_bckgrnd_masked.jpg', bckgnd_masked )
            cv2.imwrite('shapes_result_red.jpg', result)
            #####################
            cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
 
            if objCor == 3: 
                shape = 3
            elif objCor == 4:
                shape = 4
            elif 4<= objCor <=8: 
                shape = 0
            else:shape=-1
            print("objCor: " + str(objCor))
 
            #cv2.putText(imgContour,str(shape),
            #            (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
            #            (0,0,0),2)   
    return shape

def getBlueContours(mImg):
    shape = -1
    contours,hierarchy = cv2.findContours(mImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>90000:
            
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            #print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
 
            if objCor == 3:
                ##################NOVO DODANO#################
                mask = np.zeros((hh,ww), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], 0, (255,255,255), cv2.FILLED)
                mask_inv = 255 - mask
                bckgnd = np.full_like(img, (255,255,255))
                image_masked = cv2.bitwise_and(img, img, mask=mask)
                bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)
                result = cv2.add(image_masked, bckgnd_masked)
                cv2.imwrite('shapes_inverted_mask.jpg', mask_inv)
                cv2.imwrite('shapes_masked.jpg', image_masked)
                cv2.imwrite('shapes_bckgrnd_masked.jpg', bckgnd_masked )
                cv2.imwrite('shapes_result_blue.jpg', result)
                #####################
                cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)
                shape = 3
            elif objCor == 4:
                ##################NOVO DODANO#################
                mask = np.zeros((hh,ww), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], 0, (255,255,255), cv2.FILLED)
                mask_inv = 255 - mask
                bckgnd = np.full_like(img, (255,255,255))
                image_masked = cv2.bitwise_and(img, img, mask=mask)
                bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)
                result = cv2.add(image_masked, bckgnd_masked)
                cv2.imwrite('shapes_inverted_mask.jpg', mask_inv)
                cv2.imwrite('shapes_masked.jpg', image_masked)
                cv2.imwrite('shapes_bckgrnd_masked.jpg', bckgnd_masked )
                cv2.imwrite('shapes_result_blue.jpg', result)
                #####################
                cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)
                shape = 4
            elif objCor == 8:
                ##################NOVO DODANO#################
                mask = np.zeros((hh,ww), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], 0, (255,255,255), cv2.FILLED)
                mask_inv = 255 - mask
                bckgnd = np.full_like(img, (255,255,255))
                image_masked = cv2.bitwise_and(img, img, mask=mask)
                bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)
                result = cv2.add(image_masked, bckgnd_masked)
                cv2.imwrite('shapes_inverted_mask.jpg', mask_inv)
                cv2.imwrite('shapes_masked.jpg', image_masked)
                cv2.imwrite('shapes_bckgrnd_masked.jpg', bckgnd_masked )
                cv2.imwrite('shapes_result_blue.jpg', result)
                #####################
                cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)
                shape = 0
            #else:shape=-1
 
            print("objCor: " + str(objCor))
 
            #cv2.putText(imgContour,str(shape),
            #            (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
            #            (0,0,0),2)
    return shape


orb = cv2.ORB_create(nfeatures=1000)
sift = cv2.SIFT_create()


def findDes_ORB(images):    #ORB deskriptori referentnih slika
    desList=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findDes_SIFT(images):   #SIFT deskriptori referentnih slika
    desList=[]
    for img in images:
        kp,des = sift.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findID_ORB(img,desList, thres = 20):    #trazenje podudaranja izmedju testne slike i svih referentnih slika pomocu ORB
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING) 
    matchList=[]
    finalValue = -1 #Mora bit -1 jer je indeks 0 prvi znak u folderu
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance: #default je bilo 0.75
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    print(matchList)
    if len(matchList)!=0:
        if max(matchList) > thres:     #50 je threshold, ispod koje vrijednosti da ne detektira nikakav znak, ili je znak nepoznat
            finalValue = matchList.index(max(matchList)) #Dobijemo indeks maksimalne vrijednosti matcha
    return finalValue

def findID_SIFT(img,desList, thres = 15):   #trazenje podudaranja izmedju testne slike i svih referentnih slika pomocu SIFT
    kp2, des2 = sift.detectAndCompute(img, None)
    bf = cv2.BFMatcher() 
    matchList=[]
    finalValue = -1 #Mora bit -1 jer je indeks 0 prvi znak u folderu
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.6*n.distance: #default je bilo 0.75
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    print(matchList)
    if len(matchList)!=0:
        if max(matchList) >= thres:     #30 je threshold, ispod koje vrijednosti da ne detektira nikakav znak, ili je znak nepoznat
            finalValue = matchList.index(max(matchList)) #Dobijemo indeks maksimalne vrijednosti matcha
    return finalValue




img = cv2.imread("Djeca na cesti4.jpg")
hh, ww = img.shape[:2]
red_mask = imgRedMask(img)
blue_mask = imgBlueMask(img)
cv2.imshow("Crvena maska", cv2.resize(red_mask, None, fx=0.6,fy=0.6))
imgContour = img.copy()



shapeOfRedMask = getRedContours(red_mask)


print("Shape red mask: " + str(shapeOfRedMask))



if shapeOfRedMask != -1:   #Provjera je li naslo plavu ili crvenu masku u odredjenom obliku
    img_testna = cv2.imread("shapes_result_red.jpg", cv2.IMREAD_GRAYSCALE)
    ##Importanje slika
    images = []
    classNames = []
    myList = os.listdir(path)
    print('Total Classes Detected', len(myList))
    for cl in myList:
        imgCurrent = cv2.imread(f'{path}/{cl}',cv2.IMREAD_GRAYSCALE)
        images.append(imgCurrent)
        classNames.append(os.path.splitext(cl)[0])

    print(classNames)

    desList_ORB = findDes_ORB(images)
    desList_SIFT = findDes_SIFT(images)
    print(len(desList_ORB))
    print(len(desList_SIFT))
    id_ORB = findID_ORB(img_testna,desList_ORB)
    id_SIFT = findID_SIFT(img_testna,desList_SIFT)
    if id_ORB != -1:
        print("---ORB--- Prepoznat je znak: " + classNames[id_ORB])
        ####################NOVO DODANO ZA PRIKAZ KEYPOINTOVA###########
        orb_referentna = classNames[id_ORB]+".png"
        img_orbRef = cv2.imread(f'{path}/{orb_referentna}',cv2.IMREAD_GRAYSCALE)

        #cv2.imshow("Referentna slika s kojom usporedjuje", img_siftRef)
        kp1, des1 = orb.detectAndCompute(img_orbRef, None)
        kp2, des2 = orb.detectAndCompute(img_testna, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING) 
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7*n.distance: #default je bilo 0.75
                good_matches.append(m)
        result = cv2.drawMatches(img_orbRef,kp1,img_testna,kp2,good_matches,None,flags=2)
        cv2.imshow("ORB Keypoints", cv2.resize(result, None, fx=0.6,fy=0.6))
        ###############################################################
    else:
        print("---ORB--- Znak nije prepoznat ili ne postoji u bazi podataka!")
    if id_SIFT != -1:
        print("---SIFT--- Prepoznat je znak: " + classNames[id_SIFT])
        ####################NOVO DODANO ZA PRIKAZ KEYPOINTOVA###########
        sift_referentna = classNames[id_SIFT]+".png"
        img_siftRef = cv2.imread(f'{path}/{sift_referentna}',cv2.IMREAD_GRAYSCALE)

        #cv2.imshow("Referentna slika s kojom usporedjuje", img_siftRef)
        kp1, des1 = sift.detectAndCompute(img_siftRef, None)
        kp2, des2 = sift.detectAndCompute(img_testna, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m,n in matches:
            if m.distance < 0.6*n.distance: #default je bilo 0.75
                good_matches.append(m)
        result = cv2.drawMatches(img_siftRef,kp1,img_testna,kp2,good_matches,None,flags=2)
        cv2.imshow("SIFT Keypoints", cv2.resize(result, None, fx=0.6,fy=0.6))
        ###############################################################
    else:
        print("---SIFT--- Znak nije prepoznat ili ne postoji u bazi podataka!")

    cv2.imshow("Prepoznavanje znaka pomocu orba", cv2.resize(img_testna, None, fx=0.6,fy=0.6))
    

else:
    shapeOfBlueMask = getBlueContours(blue_mask)
    print("Shape blue mask: " + str(shapeOfBlueMask))
    if shapeOfBlueMask != -1:
        img_testna = cv2.imread("shapes_result_blue.jpg", cv2.IMREAD_GRAYSCALE)
        ##Importanje slika
        images = []
        classNames = []
        myList = os.listdir(path)
        print('Total Classes Detected', len(myList))
        for cl in myList:
            imgCurrent = cv2.imread(f'{path}/{cl}',cv2.IMREAD_GRAYSCALE)
            images.append(imgCurrent)
            classNames.append(os.path.splitext(cl)[0])

        print(classNames)

        desList_ORB = findDes_ORB(images)
        desList_SIFT = findDes_SIFT(images)
        print(len(desList_ORB))
        print(len(desList_SIFT))
        id_ORB = findID_ORB(img_testna,desList_ORB)
        id_SIFT = findID_SIFT(img_testna,desList_SIFT)
        if id_ORB != -1:
            print("---ORB--- Prepoznat je znak: " + classNames[id_ORB])
            ####################NOVO DODANO ZA PRIKAZ KEYPOINTOVA###########
            orb_referentna = classNames[id_ORB]+".png"
            img_orbRef = cv2.imread(f'{path}/{orb_referentna}',cv2.IMREAD_GRAYSCALE)

            #cv2.imshow("Referentna slika s kojom usporedjuje", img_siftRef)
            kp1, des1 = orb.detectAndCompute(img_orbRef, None)
            kp2, des2 = orb.detectAndCompute(img_testna, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING) 
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m,n in matches:
                if m.distance < 0.7*n.distance: #default je bilo 0.75
                    good_matches.append(m)
            result = cv2.drawMatches(img_orbRef,kp1,img_testna,kp2,good_matches,None,flags=2)
            cv2.imshow("ORB Keypoints", cv2.resize(result, None, fx=0.8,fy=0.8))
            ###############################################################
        else:
            print("---ORB--- Znak nije prepoznat ili ne postoji u bazi podataka!")
        if id_SIFT != -1:
            print("---SIFT--- Prepoznat je znak: " + classNames[id_SIFT])
            ####################NOVO DODANO ZA PRIKAZ KEYPOINTOVA###########
            sift_referentna = classNames[id_SIFT]+".png"
            img_siftRef = cv2.imread(f'{path}/{sift_referentna}',cv2.IMREAD_GRAYSCALE)

            #cv2.imshow("Referentna slika s kojom usporedjuje", img_siftRef)
            kp1, des1 = sift.detectAndCompute(img_siftRef, None)
            kp2, des2 = sift.detectAndCompute(img_testna, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m,n in matches:
                if m.distance < 0.6*n.distance: #default je bilo 0.75
                    good_matches.append(m)
            result = cv2.drawMatches(img_siftRef,kp1,img_testna,kp2,good_matches,None,flags=2)
            cv2.imshow("SIFT Keypoints", cv2.resize(result, None, fx=0.8,fy=0.8))
            ###############################################################
        else:
            print("---SIFT--- Znak nije prepoznat ili ne postoji u bazi podataka!")

        cv2.imshow("Prepoznavanje izdvojenog znaka pomocu ORB/SIFT", img_testna)
    else:
        print("Znak nije uopce prepoznat na slici!")


cv2.imshow("Detekcija prometnog znaka", cv2.resize(imgContour, None, fx=0.6,fy=0.6))

 
cv2.waitKey(0)
cv2.destroyAllWindows()
