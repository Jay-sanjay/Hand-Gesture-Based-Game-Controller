import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time
from pynput.keyboard import Controller, Key


color1 = (42, 42, 165)
color2 = (42, 42, 165)
color3 = (42, 42, 165)
color4 = (42, 42, 165)
width=0
height=0
newD=0
x = [300,245,200,170,154,130,112,103,93,87,80,75,70,67,62,59,57]
y = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coff = np.polyfit(x,y,2)

kb = Controller()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=1,min_detection_confidence=0.65,min_tracking_confidence=0.65,max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            break  # If loading a video, use 'break' instead of 'continue'.

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            for hand_landmarks in results.multi_hand_landmarks:
                # Get 3D coordinates of point 5 (thumb tip)
                x5, y5, z5 = hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z
                image_height, image_width, _ = image.shape
                x_pixel5, y_pixel5 = int(x5 * image_width), int(y5 * image_height)

                # Get 3D coordinates of point 17 (index finger tip)
                x17, y17, z17 = hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z
                x_pixel17, y_pixel17 = int(x17 * image_width), int(y17 * image_height)
                
                # making the rectangle
                x_min, y_min = int(min(p.x * image.shape[1] for p in hand_landmarks.landmark)), \
                           int(min(p.y * image.shape[0] for p in hand_landmarks.landmark))
                x_max, y_max = int(max(p.x * image.shape[1] for p in hand_landmarks.landmark)), \
                           int(max(p.y * image.shape[0] for p in hand_landmarks.landmark))
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
               
                dist = int(math.sqrt((x_pixel5-x_pixel17)**2 + (y_pixel5-y_pixel17)**2))
                a,b,c=coff
                newD =(a * dist**2 + b * dist + c)
                cv2.putText(image, f'Distance: {round(newD, 2)} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#                 # Display 3D coordinates on the image
#                 cv2.putText(image, f'Thumb Tip: ({x_pixel5}, {y_pixel5}, {z5})', (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
#                 cv2.putText(image, f'Index Finger Tip: ({x_pixel17}, {y_pixel17}, {z17})', (10, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

                width = image.shape[1]
                height = image.shape[0]

                r1 = image[0:height//2,0:width//2]
                r2 = image[0:height//2,(width//2)+1:width]

                b1 = image[3*height//4:height,0:width//3]
                b2 = image[3*height//4:height,2*width//3:width]

                
                
                if newD < 85:
                    #color1 = (200, 200, 0)
                    if (0 < x_min < width//2-50 and 0 < y_min < height // 3) or (0 < x_max < width//2-50 and 0 < y_max < height // 3):
                        color1 = (200, 200, 0)
                        kb.press('A')
                        kb.release('A')
                    if (width // 2 + 50 < x_min < width//2 + 600  and 0 < y_min < height//3) or (width // 2 + 50 < x_max < width//2 + 600  and 0 < y_max < height//3):
                        color2 = (200, 200, 0)
                        kb.press('D')
                        kb.release('D')
                    if (x_min < 50 < x_max and y_min < height-50 < y_max):
                        color3 = (200, 200, 0)
                        kb.press('S')
                        kb.release('S')
                    if (x_min < width -50 < x_max and y_min < height-50 < y_max):
                        color4 = (200, 200, 0)
                        kb.press(Key.space)
                        kb.release(Key.space)

                else:
                    color1 = (42, 42, 165)
                    color2 = (42, 42, 165)
                    color3 = (42, 42, 165)
                    color4 = (42, 42, 165)






        image = cv2.rectangle(image, (0, 0), (width // 2 - 50, height // 3), color1, 3)  # left press
        cv2.putText(image, 'Left', (110, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        image = cv2.rectangle(image, (width // 2 + 50, 0), (width // 2 + 600, height // 3), color2, 3)  # right press
        cv2.putText(image, 'Right', (width // 2 + 110, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        drift_center = (width // 4 - 10, 2 * height // 3 + 100)
        nitro_center = (2 * width // 4 + 10, 2 * height // 3 + 100)

                # ...

                # Calculate the center coordinates for the circles near the corners
        drift_center = (50 + 50, height - 50 - 50)
        nitro_center = (width - 50 - 50, height - 50 - 50)

                # ...

                # Draw the circles
        image = cv2.circle(image, drift_center, 100, color3, 5)  # drift press
        cv2.putText(image, 'Drift', (drift_center[0] - 30, drift_center[1] + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        image = cv2.circle(image, nitro_center, 100,color4 , 5)  # nitro press
        cv2.putText(image, 'Nitro', (nitro_center[0] - 30, nitro_center[1] + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Palm-Distance', image)
       # cv2.resizeWindow('Palm-Distance', 2400, 2400)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

