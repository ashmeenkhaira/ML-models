import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    Args:
        a: first point [x, y]
        b: mid point [x, y]
        c: end point [x, y]
    Returns:
        angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables
counter = 0 
stage = None
# Define threshold array for milestones
thresholds = [5, 10, 15, 20, 25]  # Display message at 5, 10, 15, 20, and 25 reps
track = ""
form_feedback = ""
form_feedback_timer = 0
left_stage = None
right_stage = None

# Form check variables
ideal_raise_angle = 90  # Ideally arms should be raised to shoulder level (90 degrees)
acceptable_angle_deviation = 15  # Allow 15 degrees of deviation
elbow_straight_min = 160  # Minimum angle for straight elbow

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Convert to RGB and process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for posture analysis
            # Left side points
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            
            # Right side points
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            
            # Calculate relevant angles
            left_raise_angle = calculate_angle(left_hip, left_shoulder, left_wrist)
            right_raise_angle = calculate_angle(right_hip, right_shoulder, right_wrist)
            
            # Calculate elbow angles to check if arms are straight
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Calculate neck/shoulder alignment
            left_neck_angle = calculate_angle(left_ear, left_shoulder, left_hip)
            right_neck_angle = calculate_angle(right_ear, right_shoulder, right_hip)
            
            # Display angles
            cv2.putText(image, f"L: {str(int(left_raise_angle))}", 
                       tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, f"R: {str(int(right_raise_angle))}", 
                       tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Track state of each arm separately
            # Left arm logic
            if left_raise_angle < 30:
                left_stage = "down"
            elif left_raise_angle > 80 and left_stage == 'down':
                left_stage = "up"
                
            # Right arm logic
            if right_raise_angle < 30:
                right_stage = "down"
            elif right_raise_angle > 80 and right_stage == 'down':
                right_stage = "up"
            
            # Form checking logic
            if stage == "up" or (left_raise_angle > 60 and right_raise_angle > 60):
                form_issues = []
                
                # Check if arms are raised to proper height
                if abs(left_raise_angle - ideal_raise_angle) > acceptable_angle_deviation:
                    if left_raise_angle < ideal_raise_angle - acceptable_angle_deviation:
                        form_issues.append("Raise left arm higher")
                    elif left_raise_angle > ideal_raise_angle + acceptable_angle_deviation:
                        form_issues.append("Lower left arm slightly")
                
                if abs(right_raise_angle - ideal_raise_angle) > acceptable_angle_deviation:
                    if right_raise_angle < ideal_raise_angle - acceptable_angle_deviation:
                        form_issues.append("Raise right arm higher")
                    elif right_raise_angle > ideal_raise_angle + acceptable_angle_deviation:
                        form_issues.append("Lower right arm slightly")
                
                # Check if elbows are straight
                if left_elbow_angle < elbow_straight_min:
                    form_issues.append("Straighten left elbow")
                if right_elbow_angle < elbow_straight_min:
                    form_issues.append("Straighten right elbow")
                
                # Check shoulder alignment (shrugging)
                if left_neck_angle < 150:  # Shoulders should be down, not shrugged up
                    form_issues.append("Relax left shoulder down")
                if right_neck_angle < 150:
                    form_issues.append("Relax right shoulder down")
                
                # Balance check - compare left and right arm heights
                if abs(left_raise_angle - right_raise_angle) > 15:
                    form_issues.append("Balance arm heights")
                
                # Provide form feedback
                if form_issues:
                    form_feedback = "Form: " + " & ".join(form_issues[:2])  # Show max 2 issues at once
                    form_feedback_timer = 30  # Show feedback for next 30 frames
                elif form_feedback_timer <= 0:
                    form_feedback = "Form: Good technique!"
                    form_feedback_timer = 30
            
            # Count a rep only when both arms complete the movement
            if left_stage == "up" and right_stage == "up":
                stage = "up"
                # Reset arm stages to avoid double counting
                left_stage = "reset"
                right_stage = "reset"
                counter += 1
                print(counter)
                
                # Check all thresholds to see if we hit any milestone
                track = ""
                for threshold in thresholds:
                    if counter == threshold:
                        track = f"Good job! {threshold} reps!"
                        break
                        
            elif left_stage == "down" and right_stage == "down":
                stage = "down"
                
            # Decrease feedback timer
            if form_feedback_timer > 0:
                form_feedback_timer -= 1
                    
        except:
            pass
        
        # Draw UI elements
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, form_feedback, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(image, (0, 40), (225, 163), (245, 117, 16), -1)
        
        # Rep counter
        cv2.putText(image, 'RAISES', (15, 62), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Stage display
        cv2.putText(image, 'STAGE', (65, 62), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage if stage else "None", 
                    (60, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Feedback display
        cv2.putText(image, track,
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                                 
        # Display the frame
        cv2.imshow('Lateral Raise Form Checker', image)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()