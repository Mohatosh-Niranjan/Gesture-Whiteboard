import cv2
import mediapipe as mp
import pygame
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture-Controlled Whiteboard")
screen.fill((255, 255, 255))  # White background

# Colors and drawing settings
DRAW_COLOR = (0, 0, 0)  # Black color for drawing
ERASE_COLOR = (255, 255, 255)  # White color for erasing
BRUSH_SIZE = 5
ERASE_SIZE = 20

# Modes
MODE_DRAW = "DRAW"
MODE_ERASE = "ERASE"
MODE_CLEAR = "CLEAR"
MODE_NONE = "NONE"

mode = MODE_NONE
last_pos = None

# Gesture recognition function
def recognize_gesture(hand_landmarks):
    # Get landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Calculate distances
    index_thumb_dist = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
    index_middle_dist = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)

    # Recognize gestures
    if index_thumb_dist < 0.05:  # Thumb and index finger pinched
        return MODE_ERASE
    elif index_tip.y < middle_tip.y:  # Index finger up
        return MODE_DRAW
    elif index_tip.y > middle_tip.y and index_tip.y > thumb_tip.y:  # Fist (clear board)
        return MODE_CLEAR
    else:
        return MODE_NONE

# Main loop
cap = cv2.VideoCapture(0)  # Open webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more natural feel
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize gesture
            gesture = recognize_gesture(hand_landmarks)

            # Map index finger tip to canvas coordinates
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * WIDTH), int(index_tip.y * HEIGHT)

            # Perform actions based on gesture
            if gesture == MODE_DRAW:
                mode = MODE_DRAW
                if last_pos:
                    pygame.draw.line(screen, DRAW_COLOR, last_pos, (x, y), BRUSH_SIZE)
                last_pos = (x, y)
            elif gesture == MODE_ERASE:
                mode = MODE_ERASE
                if last_pos:
                    pygame.draw.circle(screen, ERASE_COLOR, (x, y), ERASE_SIZE)
                last_pos = (x, y)
            elif gesture == MODE_CLEAR:
                screen.fill((255, 255, 255))  # Clear the board
                last_pos = None
            else:
                last_pos = None

    # Display the whiteboard
    pygame.display.flip()

    # Show the webcam feed
    cv2.imshow("Hand Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()