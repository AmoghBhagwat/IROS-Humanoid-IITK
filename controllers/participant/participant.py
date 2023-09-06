from controller import Robot
import sys
sys.path.append('..')
from myutils.motion_library import MotionLibrary

# Eve's locate_opponent() is implemented in this module:
from myutils.image_processing import ImageProcessing as IP
from myutils.fall_detection import FallDetection
from myutils.gait_manager import GaitManager
from myutils.camera import Camera
from myutils.camera2 import Camera2
from myutils.finite_state_machine import FiniteStateMachine
from myutils.ellipsoid_gait_generator import EllipsoidGaitGenerator
from myutils.triangulation import Triangulation

import torch
import cv2
from torchvision import transforms
import numpy as np
import time
import threading

class Sultaan (Robot):
    SMALLEST_TURNING_RADIUS = 0.1 #0.1
    SAFE_ZONE = 0.75
    TIME_BEFORE_DIRECTION_CHANGE = 60   # 80
    k=0
    is_bot_visible = True
    
    def __init__(self):
        Robot.__init__(self)
        self.fall = Falsed = 0
        
        self.time_step = int(self.getBasicTimeStep())
        self.library = MotionLibrary()

        self.camera = Camera(self)
        self.camera2 = Camera2(self)
        self.fall_detector = FallDetection(self.time_step, self)
        self.gait_manager = GaitManager(self, self.time_step)
        self.heading_angle = 3.14 / 2
        self.counter = 0
        self.library.add('Shove', './Shove.motion', loop = False)
        self.library.add('Punch', './Punch.motion', loop = False)
        self.leds = {
            'rightf': self.getDevice('Face/Led/Right'), 
            'leftf': self.getDevice('Face/Led/Left'), 
            'righte': self.getDevice('Ears/Led/Right'), 
            'lefte': self.getDevice('Ears/Led/Left'), 
            'chest': self.getDevice('ChestBoard/Led'), 
        }
        
        self.HeadPitch = self.getDevice("HeadPitch")
       
        self.previousPosition = 0.5
        self.is_bot_visible = True
        self.area = 0

        self.model_loaded = False
        #self.library.play('Cust')
        # for locking motor
       
    def run(self):
        k=0
        
        yolo_thread = threading.Thread(target=self.run_yolo)
        yolo_thread.start()
        while self.step(self.time_step) != -1:
            # We need to update the internal theta value of the gait manager at every step:
            #self.HeadPitch.setPosition(0)
            t = self.getTime()
            self.leds['rightf'].set(0xff0000)
            self.leds['leftf'].set(0xff0000)
            self.leds['righte'].set(0xff0000)
            self.leds['lefte'].set(0xff0000)
            self.leds['chest'].set(0xff0000)
            self.gait_manager.update_theta()
            #x, k, z, yaw = EllipsoidGaitGenerator.compute_leg_position(self, is_left = 'True', desired_radius=1e3, heading_angle=0)
            #print('x=' + str(x))
            
            if(self.fall_detector.detect_fall()): 
                self.fall = True
            if 0.3 < t < 5:
                self.start_sequence()
            elif t > 5:
            # else:
                # self.fall
                self.fall_detector.check()
                
                if(not self.fall):
                    d = self.getDistance()
                    if d == 1:
                        print("boundary overflow")
                        self.library.play('TurnLeft60')
                    else:
                        if (self.model_loaded == False):
                            self.walk()
                            return
                        
                        print(f"area = {self.area}")
                        if (self.area > 0.48):
                            print(f"area = {self.area}, shoving")
                            self.library.play('Punch')
                        else:
                            self.walk()
    
    def getDistance(self):          #we use bottom oriented image for edge detection
        import cv2
        import numpy as np
        image = self.camera2.get_image()
        m = 0
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = (0, 50, 50)
        upper_red = (10, 255, 255)
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # print('len(image.getbands()):',len(image.getbands()))
        # image_shape = image.shape

# Get the number of channels from the shape tuple
        # num_channels = image_shape[-1]

        # print('num_channels:', num_channels)
        
        rgb_image = image[:, :, :3]

# Get the shape of the RGB image
        rgb_image_shape = rgb_image.shape

# Get the number of channels from the shape tuple
        num_channels = rgb_image_shape[-1]
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            image_height, image_width = image.shape[:2]
            bottom_threshold = 0.92 * image_height
            
            for point in box:
                x, y = point
                if y >= bottom_threshold:
                    print("Point:", point)
                    print("Bottom Threshold:", bottom_threshold)

            points_below_threshold = sum(point[1] >= bottom_threshold for point in box)
            percentage_below_threshold = points_below_threshold / len(box)
            
            #if any(point[1] >= bottom_threshold for point in box):
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
            print('percentage_below_threshold: ', percentage_below_threshold)
            if percentage_below_threshold >= 0.5:    #print('point[1]: ', point)
                if cv2.contourArea(largest_contour) >= 200:
                    
                    m=1
        return m

    
    
    def run_yolo(self):
        # Load the YOLOv5 model
        model = torch.hub.load('yolov5/', 'custom', path='recent.pt', source='local')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device).eval()
        self.model_loaded = True
        
        reference_image = cv2.cvtColor(self.camera.get_image(), cv2.COLOR_BGR2RGB)
        boxes = model([reference_image]).xyxy[0]
        x_size = boxes[0][2].item() - boxes[0][0].item()
        y_size = boxes[0][3].item() - boxes[0][1].item()
        area = x_size * y_size
        triangulation = Triangulation(2.0, 0.5, area)
        # while True:
        # Capture the image from the camera
        while True:
            image = self.camera.get_image()

            # Remove alpha channel if present
            if image.shape[2] == 4:
                image = image[:, :, :3]

            # Convert image to RGB format
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = model([img])

            # Display the detections

            # Access individual detection attributes (e.g., bounding boxes, labels)
            boxes = results.xyxy[0]#.numpy()
            # labels = results.names[0]

            # Process the detection results as needed
            if len(boxes) == 0:
                self.is_bot_visible = False
            #    self.library.play('TurnLeft60')
            else:
                self.is_bot_visible = True
                
            # print("Bot visible: " + str(self.is_bot_visible))
            if self.is_bot_visible:
                # print("Box: " + str(boxes))
                # print("x = " + str((boxes[0][0].item()-80) / 80))

                x_size = boxes[0][2].item() - boxes[0][0].item()
                y_size = boxes[0][3].item() - boxes[0][1].item()
                self.area = x_size * y_size / (120*160)
                self.previousPosition = ((boxes[0][2].item()+boxes[0][0].item())/2-80)/80

                print(f"distance = {triangulation.distance_to_camera(self.area*120*160)}")

            # Sleep for a short duration to avoid excessive CPU usage
            time.sleep(0.1)
    
    
    
  
    def start_sequence(self):
        """At the beginning of the match, the robot walks forwards to move away from the edges."""
        self.gait_manager.command_to_motors(heading_angle=0)
        
    
    
    def walk(self):
        normalized_x = self.previousPosition
        desired_radius = (self.SMALLEST_TURNING_RADIUS / normalized_x) if abs(normalized_x) > 1e-3 else None
        
        if (self.is_bot_visible == False):
            print("not visible")
            self.heading_angle = -abs(self.previousPosition) * (3.14 / 3)
            self.counter = 0
            self.gait_manager.command_to_motors(desired_radius=desired_radius/2, heading_angle=self.heading_angle)
            return  
        
        # if(normalized_x > 0.7): 
        #     self.heading_angle = 3.14/4
        #     self.counter = 0;  
        # elif(normalized_x < -0.7): 
        #     self.heading_angle = -(3.14/4)
        #     self.counter = 0
        # else:
        #     self.heading_angle = 0
        #     self.counter = 0
        if (abs(normalized_x) > 0.5):
            self.gait_manager.update_radius_calibration(0)
        else:
            self.gait_manager.update_radius_calibration(0.93)
        self.counter += 1
        # print(f"turning with radius {desired_radius}, angle {self.heading_angle}")
        self.gait_manager.command_to_motors(desired_radius=desired_radius/2, heading_angle=self.heading_angle)
        #self.library.play('Khushi')

    def _get_normalized_opponent_x(self):
        """Locate the opponent in the image and return its horizontal position in the range [-1, 1]."""
        img = self.camera.get_image()
        _, _, horizontal_coordinate = IP.locate_opponent(img)
        if horizontal_coordinate is None:
            return 0
        
        normalized = horizontal_coordinate * 2 / img.shape[1] - 1

        return normalized

# create the Robot instance and run main loop
wrestler = Sultaan()
wrestler.run()



