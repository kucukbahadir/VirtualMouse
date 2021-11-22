import cv2
from HandTrackingModule import handDetector
import numpy as np
import autopy

camWidth, camHeight = 640,480
frameReduction = 50
smootheningValue = 6
previousLocX = 0
previousLocY = 0
currentLocX = 0
currentLocY = 0
previousTime = 0
screenWidth , screenHeight = autopy.screen.size()

def main():

    global previousLocX, previousLocY, previousTime
    cap = cv2.VideoCapture(0)
    cap.set(3, camWidth)
    cap.set(4, camHeight)

    detector = handDetector(maxHands = 1)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            # Getting Location of the Tap Landmarks of the index finger
            x1, y1 = lmList[8][1:]

            fingers = detector.fingersUp()

            # Virtual Screen Frame
            cv2.rectangle(img, (frameReduction,frameReduction), (camWidth - frameReduction, camHeight - frameReduction), (255, 255, 255), 2)

            # Detecting only Index Finger Up
            if fingers[1] == 1 and fingers[2] == 0:
                # Converting cam size to screen size
                x3 = np.interp(x1, (frameReduction,camWidth - frameReduction), (0,screenWidth))
                y3 = np.interp(y1, (frameReduction,camHeight - frameReduction), (0,screenHeight))

                # Smoothening the movement
                currentLocX = previousLocX + (x3 - previousLocX) / smootheningValue
                currentLocY = previousLocY + (y3 - previousLocY) / smootheningValue

                # Mouse movement
                autopy.mouse.move(screenWidth - currentLocX, currentLocY)

                previousLocX, previousLocY = currentLocX, currentLocY

                # Circle on index finger
                cv2.circle(img, (x1, y1), 15, (0,0,0), cv2.FILLED )

            # Clicking by closing Index and Middle finger
            if fingers[1] == 1 and fingers[2] == 1:
                lengthFingers, img, lineInfo = detector.findDistance(8,12,img)

                # Clicking
                if lengthFingers < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

        # Get FPS
        fps, previousTime = detector.getFps(previousTime)

        # To see FPS on the screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
