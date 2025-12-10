---
sidebar_label: Robotics Applications
---

# Robotics Applications

Robotics is an interdisciplinary field that integrates mechanical engineering, electrical engineering, computer science, and artificial intelligence to design, construct, operate, and apply robots. This section explores the various applications of AI in robotics and their impact on different industries.

## Overview of AI in Robotics

AI enables robots to perceive their environment, make decisions, learn from experience, and interact with humans and other machines. The integration of AI with robotics has led to more autonomous, flexible, and capable robotic systems.

## Types of Robotic Systems

### 1. Industrial Robots
Automated machines designed to perform manufacturing tasks in industrial settings.

```python
class IndustrialRobot:
    def __init__(self, robot_id, robot_type):
        self.robot_id = robot_id
        self.robot_type = robot_type  # articulated, SCARA, delta, cartesian
        self.position = [0, 0, 0]  # x, y, z coordinates
        self.orientation = [0, 0, 0]  # roll, pitch, yaw
        self.payload = 0  # maximum payload in kg
        self.status = "idle"
        self.task_queue = []
        
    def move_to_position(self, x, y, z, orientation=None):
        """Move robot to specified position"""
        self.position = [x, y, z]
        if orientation:
            self.orientation = orientation
        self.status = "moving"
        print(f"Robot {self.robot_id} moving to ({x}, {y}, {z})")
        
    def pick_object(self, weight):
        """Simulate picking up an object"""
        if weight <= self.payload:
            self.status = "carrying"
            print(f"Robot {self.robot_id} picked up object weighing {weight} kg")
            return True
        else:
            print(f"Object too heavy for robot {self.robot_id}")
            return False
            
    def place_object(self, x, y, z):
        """Place object at specified location"""
        self.status = "placing"
        self.move_to_position(x, y, z)
        self.status = "idle"
        print(f"Robot {self.robot_id} placed object at ({x}, {y}, {z})")
        
    def add_task(self, task):
        """Add task to robot's queue"""
        self.task_queue.append(task)
        
    def execute_tasks(self):
        """Execute queued tasks"""
        for task in self.task_queue:
            print(f"Executing task: {task}")
            # Execute task logic here
        self.task_queue = []

# Example: Assembly line robot
assembly_robot = IndustrialRobot("AR-001", "articulated")
assembly_robot.payload = 5  # 5kg payload
assembly_robot.add_task("Pick component A from bin 1")
assembly_robot.add_task("Place component A on assembly station")
assembly_robot.add_task("Tighten screws")
assembly_robot.execute_tasks()
```

### 2. Service Robots
Designed to assist humans in various environments like homes, hospitals, and offices.

```python
import datetime

class ServiceRobot:
    def __init__(self, robot_name, environment):
        self.name = robot_name
        self.environment = environment  # "hospital", "home", "office"
        self.battery_level = 100
        self.location = "charging_station"
        self.is_operational = True
        self.task_log = []
        
    def navigate(self, destination):
        """Navigate to destination using path planning"""
        print(f"{self.name} navigating to {destination}")
        # Simulate navigation
        self.location = destination
        self.battery_level -= 2  # Navigation consumes power
        
    def perform_service_task(self, task, location=None):
        """Perform a service task like cleaning, delivery, etc."""
        if not self.is_operational:
            print(f"{self.name} is not operational")
            return False
            
        if self.battery_level < 10:
            print(f"{self.name} battery too low, returning to charging station")
            self.return_to_charging_station()
            return False
            
        if location:
            self.navigate(location)
            
        print(f"{self.name} performing task: {task}")
        
        # Log the task
        task_record = {
            'task': task,
            'location': self.location,
            'timestamp': datetime.datetime.now(),
            'battery_after': self.battery_level
        }
        self.task_log.append(task_record)
        
        # Performing task consumes battery
        self.battery_level -= 3
        
        return True
        
    def return_to_charging_station(self):
        """Return to charging station when battery is low"""
        self.navigate("charging_station")
        print(f"{self.name} returned to charging station")
        
    def charge_battery(self, charge_amount):
        """Charge the robot's battery"""
        self.battery_level = min(100, self.battery_level + charge_amount)
        print(f"{self.name}'s battery charged to {self.battery_level}%")

# Example: Hospital service robot
hospital_robot = ServiceRobot("MediBot-01", "hospital")
hospital_robot.perform_service_task("Deliver medication", "Room 205")
hospital_robot.perform_service_task("Transport medical supplies", "Lab A")
hospital_robot.perform_service_task("Disinfect Room 205", "Room 205")

print(f"\nTask log for {hospital_robot.name}:")
for i, task in enumerate(hospital_robot.task_log, 1):
    print(f"{i}. {task['task']} at {task['location']} - "
          f"{task['timestamp'].strftime('%H:%M:%S')}")
```

### 3. Mobile Robots
Robots capable of moving around in their environment.

```python
import numpy as np
import heapq

class MobileRobot:
    def __init__(self, start_pos, goal_pos, environment_map):
        self.position = np.array(start_pos, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.environment = environment_map  # 2D grid: 0=free, 1=obstacle
        self.path = []
        self.battery = 100
        self.max_speed = 1.0  # m/s
        self.sensors = {
            'lidar': True,
            'camera': True,
            'imu': True,
            'gps': True
        }
        
    def detect_obstacles(self):
        """Detect obstacles using sensor data"""
        # This would interface with real sensors in a real implementation
        # For simulation, we'll use the environment map
        grid_x, grid_y = int(self.position[0]), int(self.position[1])
        
        # Simple obstacle detection in adjacent cells
        adjacent_positions = [
            (grid_x-1, grid_y), (grid_x+1, grid_y),
            (grid_x, grid_y-1), (grid_x, grid_y+1)
        ]
        
        obstacles = []
        for x, y in adjacent_positions:
            if (0 <= x < self.environment.shape[0] and 
                0 <= y < self.environment.shape[1] and 
                self.environment[x, y] == 1):
                obstacles.append((x, y))
                
        return obstacles
        
    def heuristic(self, a, b):
        """Heuristic function for path planning"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
    def plan_path(self):
        """Plan a path to the goal using A* algorithm"""
        start = tuple(self.position.astype(int))
        goal = tuple(self.goal.astype(int))
        
        # A* implementation
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current):
                if self.environment[next_pos] == 1:  # Obstacle
                    continue
                    
                new_cost = cost_so_far[current] + 1  # Simple cost model
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # Reconstruct path
        current = goal
        path = [current]
        while current != start:
            current = came_from.get(current)
            if current is None:
                return []  # No path found
            path.append(current)
        path.reverse()
        
        self.path = path
        return path
        
    def get_neighbors(self, pos):
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = [
            (x-1, y), (x+1, y), (x, y-1), (x, y+1),  # 4-directional
            (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)  # Diagonal
        ]
        valid_neighbors = [
            n for n in neighbors 
            if 0 <= n[0] < self.environment.shape[0] and 
               0 <= n[1] < self.environment.shape[1]
        ]
        return valid_neighbors
        
    def move_along_path(self):
        """Move robot along the planned path"""
        if not self.path:
            print("No path to follow. Planning path...")
            self.plan_path()
            
        if self.path:
            next_pos = self.path.pop(0)
            self.position = np.array(next_pos, dtype=float)
            print(f"Moved to position: {self.position}")
            self.battery -= 0.5  # Moving consumes battery
            return True
        else:
            print("No path available to the goal")
            return False

# Example: Create a mobile robot in a simple environment
environment = np.zeros((10, 10))  # 10x10 grid
# Add some obstacles
environment[3, 3:7] = 1  # Horizontal obstacle
environment[5:8, 5] = 1  # Vertical obstacle

mobile_robot = MobileRobot(start_pos=(1, 1), goal_pos=(8, 8), environment_map=environment)

print(f"Starting position: {mobile_robot.position}")
print(f"Goal position: {mobile_robot.goal}")

# Plan path
path = mobile_robot.plan_path()
print(f"Planned path: {path[:10]}...")  # Show first 10 steps

# Move robot along path (first few steps)
for _ in range(5):
    if mobile_robot.move_along_path():
        print(f"Current position: {mobile_robot.position}")
    else:
        break
```

## Perception and Computer Vision in Robots

Robots need to perceive and understand their environment to operate effectively.

```python
import cv2
import numpy as np

class RobotVisionSystem:
    def __init__(self):
        self.object_detection_model = None
        self.calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def calibrate_camera(self, calibration_images):
        """Calibrate camera using chessboard images"""
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane
        
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        if len(objpoints) > 0:
            ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            self.calibrated = True
            print("Camera calibrated successfully")
        else:
            print("Could not calibrate camera - no chessboards found in images")
    
    def detect_objects(self, image):
        """Detect objects in an image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple color-based object detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (example)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small objects
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                center_x, center_y = x + w//2, y + h//2
                
                objects.append({
                    'type': 'red_object',
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area
                })
        
        return objects
    
    def estimate_depth(self, left_image, right_image):
        """Estimate depth using stereo vision"""
        # Create stereo matcher
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        
        # Convert to grayscale
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity map
        disparity = stereo.compute(gray_left, gray_right)
        
        # Convert to depth (simplified)
        # In practice, you'd use the camera parameters for accurate depth
        depth_map = disparity.astype(np.float32) / 16.0
        
        return depth_map

# Example: Robot vision system
vision_system = RobotVisionSystem()

# Create a sample image for demonstration
sample_image = 255 * np.random.rand(480, 640, 3)
sample_image = sample_image.astype(np.uint8)

# Add a red rectangle (simulated object)
cv2.rectangle(sample_image, (100, 100), (200, 200), (0, 0, 255), -1)

# Detect objects
detected_objects = vision_system.detect_objects(sample_image)
print(f"Detected {len(detected_objects)} objects:")
for obj in detected_objects:
    print(f"  {obj['type']} at {obj['center']} with area {obj['area']:.0f}")
```

## Robot Learning and Adaptation

Robots can learn and adapt to new situations using machine learning techniques.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

class RobotLearner:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.training_data = []
        self.performance_history = []
        
    def collect_training_data(self, state, action, reward, next_state):
        """Collect experience for learning"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        }
        self.training_data.append(experience)
    
    def train_model(self, target_variable='action_outcome'):
        """Train the robot's learning model"""
        if len(self.training_data) < 10:
            print("Not enough training data to train the model")
            return False
        
        # Prepare features (state + action) and target (reward/outcome)
        features = []
        targets = []
        
        for exp in self.training_data:
            # Combine state and action as features
            state_action = exp['state'] + [exp['action']]
            features.append(state_action)
            
            # Use reward as target variable
            if target_variable == 'reward':
                targets.append(exp['reward'])
            else:  # Default to a combination of state transition quality
                # Calculate how much closer to goal the action made the robot
                current_distance = np.linalg.norm(np.array(exp['state'][:2]))
                next_distance = np.linalg.norm(np.array(exp['next_state'][:2]))
                distance_improvement = current_distance - next_distance
                targets.append(distance_improvement)
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Model trained successfully!")
        print(f"Training Score: {train_score:.4f}")
        print(f"Testing Score: {test_score:.4f}")
        
        self.is_trained = True
        return True
    
    def predict_outcome(self, state, action):
        """Predict the outcome of taking an action in a given state"""
        if not self.is_trained:
            return 0  # Default prediction if not trained
        
        # Combine state and action
        state_action = np.array(state + [action]).reshape(1, -1)
        
        # Predict outcome
        prediction = self.model.predict(state_action)[0]
        return prediction
    
    def learn_from_experience(self, state, action, reward, next_state):
        """Complete learning cycle: collect data and potentially retrain"""
        # Collect the new experience
        self.collect_training_data(state, action, reward, next_state)
        
        # After collecting enough experiences, consider retraining
        if len(self.training_data) % 20 == 0 and len(self.training_data) >= 20:
            print("Retraining model with new experiences...")
            self.train_model()
            
            # Store performance
            recent_experiences = self.training_data[-10:]  # Last 10 experiences
            avg_reward = np.mean([exp['reward'] for exp in recent_experiences])
            self.performance_history.append(avg_reward)
    
    def save_model(self, filepath):
        """Save the trained model to file"""
        if self.is_trained:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'training_data': self.training_data,
                    'performance_history': self.performance_history
                }, f)
            print(f"Model saved to {filepath}")
        else:
            print("No trained model to save")
    
    def load_model(self, filepath):
        """Load a previously trained model from file"""
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.model = saved_data['model']
        self.training_data = saved_data['training_data']
        self.performance_history = saved_data['performance_history']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

# Example: Robot learning to navigate
learner = RobotLearner()

# Simulate robot experiences
for episode in range(50):
    # Simulate a simple navigation task
    # State: [x_pos, y_pos, battery_level, obstacle_distance]
    # Action: movement direction (0-3 for N, E, S, W)
    # Reward: based on distance to goal and energy efficiency
    
    state = [np.random.uniform(-10, 10), np.random.uniform(-10, 10), 
             np.random.uniform(20, 100), np.random.uniform(0, 5)]
    action = np.random.randint(0, 4)
    reward = np.random.uniform(-1, 1)  # Can be positive or negative
    next_state = [s + np.random.uniform(-1, 1) for s in state]  # Small change
    
    # Learn from this experience
    learner.learn_from_experience(state, action, reward, next_state)

# Make a prediction
sample_state = [5, 5, 80, 2]  # At position (5,5), 80% battery, 2m from obstacle
sample_action = 1  # Move East
predicted_outcome = learner.predict_outcome(sample_state, sample_action)
print(f"Predicted outcome of moving East from {sample_state[:2]}: {predicted_outcome:.4f}")
```

## Human-Robot Interaction

Robots need to effectively communicate and collaborate with humans.

```python
import speech_recognition as sr
import pyttsx3
import datetime
import json

class HumanRobotInterface:
    def __init__(self):
        self.speech_recognizer = sr.Recognizer()
        self.text_to_speech = pyttsx3.init()
        
        # Set TTS properties
        self.text_to_speech.setProperty('rate', 150)  # Speed of speech
        self.text_to_speech.setProperty('volume', 0.9)  # Volume level
        
        # Conversation history
        self.conversation_history = []
        self.robot_name = "Robo-Assist"
        
    def listen(self):
        """Listen to user voice input"""
        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.speech_recognizer.adjust_for_ambient_noise(source)
                audio = self.speech_recognizer.listen(source, timeout=5)
                
            # Recognize speech using Google's speech recognition
            text = self.speech_recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("No speech detected")
            return ""
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""
    
    def speak(self, text):
        """Speak text to user"""
        print(f"{self.robot_name}: {text}")
        self.text_to_speech.say(text)
        self.text_to_speech.runAndWait()
        
    def process_command(self, command):
        """Process natural language commands"""
        command = command.lower()
        
        # Time request
        if "time" in command:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The current time is {current_time}"
            
        # Date request
        elif "date" in command:
            current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            response = f"Today is {current_date}"
            
        # Greeting
        elif any(greeting in command for greeting in ["hello", "hi", "hey"]):
            response = f"Hello! I'm {self.robot_name}, your robotic assistant. How can I help you today?"
            
        # Name inquiry
        elif "your name" in command:
            response = f"My name is {self.robot_name}. I'm here to assist you."
            
        # Help request
        elif "help" in command:
            response = ("I can help with various tasks. You can ask me about the time, date, "
                       "or just have a conversation. What would you like to know?")
            
        # Default response
        else:
            response = "I'm sorry, I didn't understand that command. How else can I help you?"
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': command,
            'robot_response': response
        })
        
        return response
    
    def initiate_conversation(self):
        """Start a conversation with the user"""
        self.speak(f"Hello! I'm {self.robot_name}, your robotic assistant.")
        
        while True:
            user_input = self.listen()
            
            if user_input:
                if "stop" in user_input.lower() or "bye" in user_input.lower():
                    self.speak("Goodbye! Have a great day!")
                    break
                else:
                    response = self.process_command(user_input)
                    self.speak(response)

# Example: Voice interaction system
hri_system = HumanRobotInterface()

# Process a sample command (without actually listening to microphone)
sample_command = "What time is it?"
response = hri_system.process_command(sample_command)
print(f"Command: {sample_command}")
print(f"Response: {response}")

# Another sample command
sample_command = "Hello Robo-Assist"
response = hri_system.process_command(sample_command)
print(f"Command: {sample_command}")
print(f"Response: {response}")
```

## Multi-Robot Coordination

Multiple robots can work together to accomplish complex tasks more efficiently.

```python
import threading
import time
import random

class MultiRobotCoordinator:
    def __init__(self, num_robots=3):
        self.robots = []
        self.tasks = []
        self.task_assignments = {}
        self.global_map = {}  # Shared map of environment
        self.communication_channel = []  # Shared communication
        self.lock = threading.Lock()  # For thread safety
        
        # Create robots
        for i in range(num_robots):
            robot = {
                'id': f'Robot-{i}',
                'position': [random.uniform(0, 10), random.uniform(0, 10)],
                'status': 'idle',
                'battery': 100,
                'current_task': None
            }
            self.robots.append(robot)
    
    def add_task(self, task_id, location, priority='normal', task_type='delivery'):
        """Add a task to the system"""
        task = {
            'id': task_id,
            'location': location,
            'priority': priority,
            'type': task_type,
            'status': 'pending',
            'assigned_robot': None
        }
        self.tasks.append(task)
        self._broadcast_message(f"New task {task_id} added at {location}")
        return task
    
    def assign_tasks(self):
        """Assign tasks to robots based on proximity and availability"""
        # Filter unassigned tasks
        unassigned_tasks = [task for task in self.tasks if task['status'] == 'pending']
        
        # Sort by priority then by task ID
        priority_order = {'high': 3, 'normal': 2, 'low': 1}
        unassigned_tasks.sort(key=lambda t: priority_order.get(t['priority'], 0), reverse=True)
        
        for task in unassigned_tasks:
            # Find the closest available robot
            available_robots = [r for r in self.robots if r['status'] == 'idle']
            
            if available_robots:
                # Calculate distances and assign to closest robot
                min_distance = float('inf')
                closest_robot = None
                
                for robot in available_robots:
                    distance = self._calculate_distance(robot['position'], task['location'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_robot = robot
                
                if closest_robot:
                    # Assign task
                    closest_robot['current_task'] = task['id']
                    closest_robot['status'] = 'busy'
                    task['assigned_robot'] = closest_robot['id']
                    task['status'] = 'assigned'
                    
                    self.task_assignments[task['id']] = closest_robot['id']
                    
                    self._broadcast_message(
                        f"Task {task['id']} assigned to {closest_robot['id']}"
                    )
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    def _broadcast_message(self, message):
        """Add message to shared communication channel"""
        with self.lock:
            self.communication_channel.append({
                'timestamp': time.time(),
                'message': message
            })
    
    def execute_task(self, robot_id, task_id):
        """Simulate robot executing a task"""
        robot = next(r for r in self.robots if r['id'] == robot_id)
        task = next(t for t in self.tasks if t['id'] == task_id)
        
        robot['status'] = 'executing'
        
        # Simulate task execution time based on distance and task complexity
        distance = self._calculate_distance(robot['position'], task['location'])
        execution_time = distance * 0.5  # 0.5 seconds per unit distance
        
        # Simulate battery consumption
        battery_consumption = execution_time * 2  # 2% per second
        robot['battery'] = max(0, robot['battery'] - battery_consumption)
        
        # Update position to task location
        robot['position'] = task['location']
        
        print(f"{robot_id} executing task {task_id} at location {task['location']}")
        
        # Simulate execution time
        time.sleep(min(execution_time, 2))  # Cap execution time for demo
        
        # Mark task as completed
        task['status'] = 'completed'
        robot['status'] = 'returning'
        
        print(f"{robot_id} completed task {task_id}")
        
        # Return to base or charging station
        robot['status'] = 'idle'
        robot['current_task'] = None
        
        self._broadcast_message(f"Task {task_id} completed by {robot_id}")
    
    def run_coordinated_operation(self):
        """Run a coordinated multi-robot operation"""
        # Add tasks
        self.add_task("T001", [5, 5], "high", "delivery")
        self.add_task("T002", [8, 2], "normal", "inspection")
        self.add_task("T003", [1, 9], "low", "maintenance")
        
        print(f"Initial robot positions: {[(r['id'], r['position']) for r in self.robots]}")
        
        # Assign tasks
        self.assign_tasks()
        
        # Execute tasks in a coordinated manner
        for task in self.tasks:
            if task['status'] == 'assigned':
                robot_id = task['assigned_robot']
                # In a real system, this would be a separate thread for each robot
                self.execute_task(robot_id, task['id'])
        
        # Print final status
        print("\nFinal robot status:")
        for robot in self.robots:
            print(f"  {robot['id']}: {robot['status']}, Battery: {robot['battery']:.1f}%, "
                  f"Position: {robot['position']}")
        
        # Print task status
        print(f"\nTask Status:")
        for task in self.tasks:
            print(f"  {task['id']}: {task['status']}")

# Example: Coordinated multi-robot operation
coordinator = MultiRobotCoordinator(num_robots=3)
coordinator.run_coordinated_operation()
```

## Challenges in Robotic Applications

### 1. Environmental Uncertainty
Robots must operate in unpredictable environments with changing conditions.

### 2. Safety and Reliability
Ensuring robots operate safely around humans and other systems.

### 3. Real-time Processing
Making decisions and executing actions within strict time constraints.

### 4. Power Management
Optimizing battery life and energy efficiency for autonomous operation.

### 5. Human-Robot Collaboration
Designing robots that can effectively work alongside humans.

## Future Directions

- **Soft Robotics**: Creating robots with flexible, adaptable components
- **Swarm Robotics**: Large groups of simple robots working together
- **Bio-inspired Robotics**: Learning from biological systems
- **Cloud Robotics**: Robots leveraging cloud-based AI and computation
- **Humanoid Robots**: More human-like robots for complex interactions

Robotics continues to evolve with advances in AI, materials science, and mechanical engineering, enabling increasingly sophisticated applications across manufacturing, healthcare, agriculture, and service industries.