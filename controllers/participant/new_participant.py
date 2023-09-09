from controller import Robot
import sys
sys.path.append('..')

from myutils.motion_library import MotionLibrary
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
    SMALLEST_TURNING_RADIUS = 0.1
    SAFE_ZONE = 0.75
    TIME_BEFORE_DIRECTION_CHANGE = 200  # 8000 ms / 40 ms

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())

        self.library = MotionLibrary()
        self.library.add('Shove', './Shove.motion', loop = False)
        self.library.add('Punch', './Punch.motion', loop = False)

        self.camera = Camera(self)
        self.camera2 = Camera2(self)
        self.fall_detector = FallDetection(self.time_step, self)
        self.gait_manager = GaitManager(self, self.time_step)
        self.heading_angle = 3.14 / 2
        # Time before changing direction to stop the robot from falling off the ring
        self.counter = 0
        
        self.leds = {
            'rightf': self.getDevice('Face/Led/Right'), 
            'leftf': self.getDevice('Face/Led/Left'), 
            'righte': self.getDevice('Ears/Led/Right'), 
            'lefte': self.getDevice('Ears/Led/Left'), 
            'chest': self.getDevice('ChestBoard/Led'), 
        }

        self.botVisible = False

    def run(self):
        yolo_thread = threading.Thread(target=self.yolo)
        yolo_thread.start()

        while self.step(self.time_step) != -1:
            # Turn on LEDS 
            self.leds['rightf'].set(0xff0000)
            self.leds['leftf'].set(0xff0000)
            self.leds['righte'].set(0xff0000)
            self.leds['lefte'].set(0xff0000)
            self.leds['chest'].set(0xff0000)

            self.fall = self.fall_detector.detect_fall()

            t = self.getTime()
            self.gait_manager.update_theta()
            if 0.3 < t < 2:
                self.start_sequence()
            elif t > 2:
                self.fall_detector.check()

                if (self.fall): # equivalent to if not self.fall
                    return
                
                if self.near_edge():
                    return # TODO

                if self.area > 0.5: # TODO find ideal threshold
                    self.library.play('Punch')
                    return
                
                self.walk()
                

    def start_sequence(self):
        """At the beginning of the match, the robot walks forwards to move away from the edges."""
        self.gait_manager.command_to_motors(heading_angle=0)

    def near_edge(self):
        image = self.camera2.get_image()
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = (0, 50, 50)
        upper_red = (10, 255, 255)
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if (len(contours) > 0):
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            width, height = image.shape[:2]
            bottom_threshold = 0.92 * height

            points_below_threshold = sum(point[1] >= bottom_threshold for point in box)
            percentage_below_threshold = points_below_threshold / len(box)

            if percentage_below_threshold >= 0.5 and cv2.contourArea(largest_contour) >= 0.5:
                return True
            
        return False

    def walk(self):
        if (self.botVisible == False):
            self.gait_manager.command_to_motors()
            return
        
        if (self.nearEdge()):
            pass
            
        normalized_x = self._get_normalized_opponent_x()
        
        desired_radius = abs(self.SMALLEST_TURNING_RADIUS / normalized_x) if abs(normalized_x) > 1e-3 else None

        rotate_right = 0
        if normalized_x > 0:
            rotate_right = 1
        else:
            rotate_right = -1

        if (abs(normalized_x) > 0.7):
            self.gait_manager.update_radius_calibration(0)
        else:
            self.gait_manager.update_radius_calibration(0.93)
            rotate_right = 1

        self.gait_manager.command_to_motors(desired_radius=desired_radius, heading_angle=self.heading_angle, rotate_right=rotate_right)


    def _get_normalized_opponent_x(self):
        return self.previousPosition

    def yolo(self):
        model = torch.hub.load('yolov5/', 'custom', path='recent.pt', source='local')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device).eval()
        self.model_loaded = True
        
        # Get reference image for triangulation
        reference_image = cv2.cvtColor(self.camera.get_image(), cv2.COLOR_BGR2RGB)
        boxes = model([reference_image]).xyxy[0]
        
        while (len(boxes) == 0):
            boxes = model([reference_image]).xyxy[0]
        
        x_size = boxes[0][2].item() - boxes[0][0].item()
        y_size = boxes[0][3].item() - boxes[0][1].item()
        area = x_size * y_size  
        triangulation = Triangulation(2.0, 0.5, area)

        while True: # run forever
            image = self.camera.get_image()
            if image.shape[2] == 4:
                image = image[:, :, :3]

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model([img])

            bounding_boxes = results.xyxy[0]

            if (len(bounding_boxes == 0)):
                self.botVisible = False
                continue

            self.botVisible = True

            x_size = boxes[0][2].item() - boxes[0][0].item()
            y_size = boxes[0][3].item() - boxes[0][1].item()

            self.previousPosition = ((boxes[0][2].item()+boxes[0][0].item())/2-80)/80
            self.botDistance = triangulation.distance_to_camera(x_size * y_size)

            time.sleep(0.1)