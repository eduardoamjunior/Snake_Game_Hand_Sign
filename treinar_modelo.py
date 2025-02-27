import cv2
import numpy as np
import mediapipe as mp
import os
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands with higher detection confidence
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create directories for the dataset if they don't exist
def setup_directories():
    gestures = ['up', 'down', 'left', 'right', 'start', 'stop']
    if not os.path.exists('gesture_data'):
        os.makedirs('gesture_data')
    for gesture in gestures:
        if not os.path.exists(f'gesture_data/{gesture}'):
            os.makedirs(f'gesture_data/{gesture}')
    return gestures

# Extract specific finger pointing features based on the described gestures
def extract_finger_pointing_features(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    features = []
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw landmarks on the frame for visualization
        mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Extract key landmarks for finger pointing detection
        wrist = hand_landmarks.landmark[0]
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # Finger extension features
        # 1. Calculate vectors from wrist to each fingertip
        thumb_vec = [thumb_tip.x - wrist.x, thumb_tip.y - wrist.y, thumb_tip.z - wrist.z]
        index_vec = [index_tip.x - wrist.x, index_tip.y - wrist.y, index_tip.z - wrist.z]
        middle_vec = [middle_tip.x - wrist.x, middle_tip.y - wrist.y, middle_tip.z - wrist.z]
        
        # 2. Normalized vectors (direction only)
        def normalize(v):
            norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            if norm == 0:
                return v
            return [v[0]/norm, v[1]/norm, v[2]/norm]
        
        thumb_vec_norm = normalize(thumb_vec)
        index_vec_norm = normalize(index_vec)
        middle_vec_norm = normalize(middle_vec)
        
        # 3. Angle between fingers (crucial for distinguishing gestures)
        def dot_product(v1, v2):
            return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        
        thumb_index_angle = np.arccos(np.clip(dot_product(thumb_vec_norm, index_vec_norm), -1.0, 1.0))
        index_middle_angle = np.arccos(np.clip(dot_product(index_vec_norm, middle_vec_norm), -1.0, 1.0))
        
        # 4. Direction features (specifically for pointing)
        # For "up" - index finger should be pointing up (negative y direction)
        # For "down" - index finger should be pointing down (positive y direction)
        # For "left" - index finger should be pointing left (negative x direction)
        # For "right" - thumb should be pointing right (positive x direction)
        index_pointing_up = -index_vec_norm[1]  # Negative y is up
        index_pointing_down = index_vec_norm[1]  # Positive y is down
        index_pointing_left = -index_vec_norm[0]  # Negative x is left
        thumb_pointing_right = thumb_vec_norm[0]  # Positive x is right
        
        # 5. Key finger positions relative to each other
        thumb_index_x_diff = thumb_tip.x - index_tip.x
        thumb_index_y_diff = thumb_tip.y - index_tip.y
        
        # 6. Absolute positions in the image (less important but still useful)
        all_positions = []
        for landmark in hand_landmarks.landmark:
            all_positions.extend([landmark.x, landmark.y, landmark.z])
        
        # Combined set of features specifically designed for these gestures
        features = [
            # Pointing direction features
            index_pointing_up,
            index_pointing_down,
            index_pointing_left,
            thumb_pointing_right,
            
            # Angles between fingers
            thumb_index_angle,
            index_middle_angle,
            
            # Relative positions
            thumb_index_x_diff,
            thumb_index_y_diff,
            
            # Normalized direction vectors
            *thumb_vec_norm,
            *index_vec_norm,
            *middle_vec_norm,
            
            # Include all absolute positions as additional context
            *all_positions
        ]
        
        # Display feature data for debugging
        cv2.putText(frame, f"Index up: {index_pointing_up:.2f}", 
                    (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Index left: {index_pointing_left:.2f}", 
                    (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Thumb right: {thumb_pointing_right:.2f}", 
                    (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
    return features, frame

# Record gesture data with visual guidance for specific gestures
def record_gestures():
    gestures = setup_directories()
    
    cap = cv2.VideoCapture(0)
    current_gesture_index = 0
    current_gesture = gestures[current_gesture_index]
    recording = False
    frame_counter = 0
    sample_counter = 0
    last_key_time = time.time()
    
    cv2.namedWindow('Hand Gesture Recording')
    
    print(f"Ready to record gesture: {current_gesture}")
    print("Press 'r' to start/pause recording, 'n' for next gesture, 'q' to quit, 'd' to delete last sample")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for intuitive mirror view
        #frame = cv2.flip(frame, 1)
            
        # Show instructions on frame
        status = "RECORDING" if recording else "PAUSED"
        cv2.putText(frame, f"Gesture: {current_gesture} - Status: {status}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Samples: {sample_counter}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw a gesture guide with specific instructions for each gesture
        guide_img = np.zeros((180, 180, 3), dtype=np.uint8)
        guide_text = ""
        
        if current_gesture == "up":
            cv2.line(guide_img, (90, 140), (90, 40), (0, 255, 0), 3)
            cv2.circle(guide_img, (90, 40), 10, (0, 255, 0), -1)
            guide_text = "Point UP with INDEX finger"
        elif current_gesture == "down":
            cv2.line(guide_img, (90, 40), (90, 140), (0, 255, 0), 3)
            cv2.circle(guide_img, (90, 140), 10, (0, 255, 0), -1)
            guide_text = "Point DOWN with INDEX finger"
        elif current_gesture == "left":
            cv2.line(guide_img, (140, 90), (40, 90), (0, 255, 0), 3)
            cv2.circle(guide_img, (40, 90), 10, (0, 255, 0), -1)
            guide_text = "Point LEFT with INDEX finger"
        elif current_gesture == "right":
            # Thumb pointing right
            cv2.ellipse(guide_img, (90, 90), (40, 30), 0, 180, 270, (0, 255, 0), 3)
            cv2.line(guide_img, (90, 60), (130, 90), (0, 255, 0), 3)
            guide_text = "Vertical hand, THUMB right"
        elif current_gesture == "start":
            cv2.putText(guide_img, "START", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            guide_text = "Your START gesture"
        elif current_gesture == "stop":
            cv2.rectangle(guide_img, (40, 40), (140, 140), (0, 0, 255), 3)
            guide_text = "Your STOP gesture"
        
        # Add text guidance below the diagram
        cv2.putText(guide_img, guide_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Overlay the guide on the main frame
        frame[10:190, frame.shape[1]-190:frame.shape[1]-10] = guide_img
        
        features, frame = extract_finger_pointing_features(frame)
        
        # Record data if in recording mode and hand is detected
        if recording and features:
            frame_counter += 1
            if frame_counter % 5 == 0:  # Save every 5th frame to reduce redundancy
                sample_counter += 1
                file_path = f'gesture_data/{current_gesture}/{current_gesture}_{sample_counter}.pkl'
                with open(file_path, 'wb') as f:
                    pickle.dump(features, f)
                
                # Visual feedback for recording
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        
        cv2.imshow('Hand Gesture Recording', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        if current_time - last_key_time > 0.2:  # Debounce time of 200ms
            if key == ord('r'):  # Toggle recording
                recording = not recording
                last_key_time = current_time
                print(f"Recording {'started' if recording else 'paused'}")
            
            elif key == ord('n'):  # Next gesture
                current_gesture_index = (current_gesture_index + 1) % len(gestures)
                current_gesture = gestures[current_gesture_index]
                sample_counter = len(os.listdir(f'gesture_data/{current_gesture}'))
                last_key_time = current_time
                print(f"Switched to gesture: {current_gesture} (Current samples: {sample_counter})")
            
            elif key == ord('d'):  # Delete last sample
                if sample_counter > 0:
                    file_to_delete = f'gesture_data/{current_gesture}/{current_gesture}_{sample_counter}.pkl'
                    if os.path.exists(file_to_delete):
                        os.remove(file_to_delete)
                        sample_counter -= 1
                        print(f"Deleted last sample. Remaining: {sample_counter}")
                        last_key_time = current_time
            
            elif key == ord('q'):  # Quit
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Recording complete!")

# Load the recorded gesture data
def load_gesture_data():
    gestures = ['up', 'down', 'left', 'right', 'start', 'stop']
    X = []
    y = []
    sample_counts = {}
    
    for i, gesture in enumerate(gestures):
        gesture_dir = f'gesture_data/{gesture}'
        sample_counts[gesture] = 0
        
        if os.path.exists(gesture_dir):
            files = os.listdir(gesture_dir)
            for file in files:
                file_path = os.path.join(gesture_dir, file)
                try:
                    with open(file_path, 'rb') as f:
                        features = pickle.load(f)
                        if len(features) > 0:  # Ensure we have valid features
                            X.append(features)
                            y.append(i)
                            sample_counts[gesture] += 1
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    print("Samples per gesture:")
    for gesture, count in sample_counts.items():
        print(f"  - {gesture}: {count} samples")
    
    return np.array(X), np.array(y)

# Advanced model training with feature selection and hyperparameter tuning
def train_model():
    print("Loading gesture data...")
    X, y = load_gesture_data()
    
    if len(X) == 0:
        print("No gesture data found. Please record gestures first.")
        return None
    
    print(f"Data loaded: {len(X)} samples")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    
    # Train a more complex Random Forest classifier optimized for these gestures
    model = RandomForestClassifier(
        n_estimators=200,        # More trees for better accuracy
        max_depth=30,            # Allow deeper trees to capture complex patterns
        min_samples_split=5,     # Min samples required to split a node
        min_samples_leaf=2,      # Min samples required at a leaf node
        bootstrap=True,          # Use bootstrap samples
        class_weight='balanced', # Handle any class imbalance
        random_state=42,
        n_jobs=-1                # Use all available cores
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Visualize the confusion matrix
    gestures = ['up', 'down', 'left', 'right', 'start', 'stop']
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()
    
    # Feature importance analysis
    feature_importance = model.feature_importances_
    print("\nTop 10 most important features:")
    indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(indices):
        print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Problem analysis - check for specific misclassifications
    if len(y_test) > 0:
        right_idx = gestures.index('right')
        left_idx = gestures.index('left')
        up_idx = gestures.index('up')
        
        # Check for right vs left confusion
        right_as_left = np.sum((y_test == right_idx) & (y_pred == left_idx))
        left_as_right = np.sum((y_test == left_idx) & (y_pred == right_idx))
        
        if right_as_left > 0 or left_as_right > 0:
            print("\nPotential confusion between RIGHT and LEFT gestures:")
            print(f"  - RIGHT classified as LEFT: {right_as_left} times")
            print(f"  - LEFT classified as RIGHT: {left_as_right} times")
            print("  → Try making these gestures more distinct in your recordings")
    
    # Save the model
    with open('hand_gesture_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as 'hand_gesture_model.pkl'")
    return model

# Test the model with detailed feedback and stability filtering
def test_model():
    try:
        with open('hand_gesture_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return
    
    gestures = ['up', 'down', 'left', 'right', 'start', 'stop']
    
    cap = cv2.VideoCapture(0)
    
    print("Testing model. Press 'q' to quit.")
    
    # For tracking prediction stability
    prediction_history = []
    stability_threshold = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally
        #frame = cv2.flip(frame, 1)
        
        features, frame = extract_finger_pointing_features(frame)
        
        if features:
            # Make a prediction
            prediction = model.predict([features])[0]
            
            # Get prediction probabilities
            probabilities = model.predict_proba([features])[0]
            top_prob = np.max(probabilities) * 100
            
            # Add to history for stability
            prediction_history.append(prediction)
            if len(prediction_history) > stability_threshold:
                prediction_history.pop(0)
                
            # Only show stable predictions
            if len(prediction_history) == stability_threshold and all(p == prediction_history[0] for p in prediction_history):
                stable_prediction = prediction_history[0]
                gesture = gestures[stable_prediction]
                
                # Display the prediction with confidence
                color = (0, 255, 0) if top_prob > 75 else (0, 165, 255)
                cv2.putText(frame, f"Gesture: {gesture} ({top_prob:.1f}%)", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Show troubleshooting info for problem gestures
                if gesture in ["right", "left"]:
                    cv2.putText(frame, f"For RIGHT: Vertical hand, thumb pointing right", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, f"For LEFT: Index finger pointing left", 
                                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Show top 3 predictions with probabilities
                sorted_indices = np.argsort(probabilities)[::-1]
                for i in range(3):
                    idx = sorted_indices[i]
                    cv2.putText(frame, f"{i+1}. {gestures[idx]}: {probabilities[idx]*100:.1f}%", 
                                (10, 120 + 25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
            else:
                cv2.putText(frame, "Stabilizing prediction...", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        cv2.imshow('Gesture Recognition Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    while True:
        print("\nFinger Pointing Gesture Recognition System")
        print("1. Record Gestures (with specialized guides)")
        print("2. Train Model (with conflict analysis)")
        print("3. Test Model")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            record_gestures()
        elif choice == '2':
            train_model()
        elif choice == '3':
            test_model()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
