import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe pose and drawing utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def main():
    cap = cv2.VideoCapture(0)
    counter, threshold = 0, 5
    stage, track = None, ""
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Preprocess the frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make pose detection
            results = pose.process(image)

            # Revert image for OpenCV display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check if landmarks are detected
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get shoulder, elbow, wrist points
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate elbow angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Display angle
                elbow_coords = tuple(np.multiply(elbow, [640, 480]).astype(int))
                cv2.putText(image, f'{int(angle)}', elbow_coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                elif angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                    track = "Good job!" if counter == threshold else ""

            # Draw the status box
            cv2.rectangle(image, (0, 0), (250, 140), (245, 117, 16), -1)

            # Display counter
            cv2.putText(image, 'REPS', (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, str(counter), (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            # Display stage
            cv2.putText(image, 'STAGE', (120, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, stage if stage else "", (120, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            # Display encouragement track
            if track:
                cv2.putText(image, track, (15, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Display the frame
            cv2.imshow('Mediapipe Feed', image)

            # Break loop on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
