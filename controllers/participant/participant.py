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
    TIME_BEFORE_DIRECTION_CHANGE = 200  # 8000 ms /. 40 ms

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())

        self.library = MotionLibrary()
        self.library.add('Shove', './Shove.motion', loop = False)
        self.library.add('Punch', './Punch.motion', loop = False)
        self.library.add('Khushi2', './Khushi2.motion', loop = False)
        # self.library.add('First', './First.motion', loop = False)
        self.library.add('kinchit', './kinchit.motion', loop = False)

        self.camera = Camera(self)
        self.camera2 = Camera2(self)
        self.fall_detector = FallDetection(self.time_step, self)
        self.gait_manager = GaitManager(self, self.time_step)
        self.heading_angle = 0
        # Time before changing direction to stop the robot from falling off the ring
        self.counter = 0
        
        self.leds = {
            'rightf': self.getDevice('Face/Led/Right'), 
            'leftf': self.getDevice('Face/Led/Left'), 
            'righte': self.getDevice('Ears/Led/Right'), 
            'lefte': self.getDevice('Ears/Led/Left'), 
            'chest': self.getDevice('ChestBoard/Led'), 
        }

        self.RShoulderPitch = self.getDevice('RShoulderPitch')
        self.LShoulderPitch = self.getDevice('LShoulderPitch')
        self.LElbowYaw = self.getDevice('LElbowYaw')
        self.RElbowYaw = self.getDevice('RElbowYaw')

        self.head_pitch = self.getDevice("HeadPitch")
        self.right_foot_sensor = self.getDevice('RFsr')
        self.right_foot_sensor.enable(self.time_step)
        self.left_foot_sensor = self.getDevice('LFsr')
        self.left_foot_sensor.enable(self.time_step)
        self.rl = self.getDevice('RFoot/Bumper/Left')
        self.rr = self.getDevice('RFoot/Bumper/Right')
        self.ll = self.getDevice('LFoot/Bumper/Left')
        self.lr = self.getDevice('LFoot/Bumper/Right')
        self.rl.enable(self.time_step)
        self.rr.enable(self.time_step)
        self.ll.enable(self.time_step)
        self.lr.enable(self.time_step)

        # for locking motor
        joints = ['HipYawPitch', 'HipRoll', 'HipPitch', 'KneePitch', 'AnklePitch', 'AnkleRoll']
        self.L_leg_motors = []
        for joint in joints:
            motor = self.getDevice(f'L{joint}')
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(1)
            self.L_leg_motors.append(motor)

        self.R_leg_motors = []
        for joint in joints:
            motor = self.getDevice(f'R{joint}')
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(1)
            self.R_leg_motors.append(motor)

        self.botVisible = True
        self.area = 0
        self.previousPosition = 0

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
            
            if 0.3 < t < 5:
                self.start_sequence()
            elif t > 5:
                self.fall_detector.check()
                self.walk()
                

    def start_sequence(self):
        """At the beginning of the match, the robot walks forwards to move away from the edges."""
        self.library.play('Khushi2')
        self.gait_manager.command_to_motors(heading_angle=0)

    def foot_sensor(self):
        return self.rl.getValue() + self.rr.getValue() + self.lr.getValue() + self.ll.getValue()
    
    def on_ring(self):
        image = self.camera2.get_image()
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        img1 = hsv_image.copy()
        img2 = hsv_image.copy()
        
        colorr_low = np.array([193,62,35])
        colorr_high = np.array([205,107,65])
        colorf_low = np.array([83,62,42])
        colorf_high = np.array([154,110,70])
        mask1 = cv2.inRange(img1, colorr_low, colorr_high)
        mask2 = cv2.inRange(img2, colorf_low, colorf_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        res1 = cv2.bitwise_and(img1,img1,mask1)
        res2 =  cv2.bitwise_and(img2,img2,mask2)
        gray1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
        contours1, _ = cv2.findContours(gray1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(gray2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
        # Check if contours2 is non-zero before calculating its centroid
        
        cy1, cx1 = None, None
        if len(contours1) > 0:
            contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
            cy1, cx1 = IP.get_contour_centroid(contours1[0])
        # Check if contours2 is non-zero before calculating its centroid
        cy2, cx2 = None, None
        if len(contours2) > 0:
            contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
            cy2, cx2 = IP.get_contour_centroid(contours2[0])

        print("cy1 = ", cy1, ", cy2 = ", cy2)
        if len(contours1) > 0 and len(contours2) > 0:
            if cy1 > cy2:
                return False
            else:
                return True

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

            height, width    = image.shape[:2]
            bottom_threshold = 0.92 * height

            points_below_threshold = sum(point[1] >= bottom_threshold for point in box)
            percentage_below_threshold = points_below_threshold / len(box)
            if percentage_below_threshold >= 0.5 and cv2.contourArea(largest_contour) >= 200:
                return True
            
        return False

    def walk(self):
        normalized_x = self._get_normalized_opponent_x()
        
        desired_radius = abs(self.SMALLEST_TURNING_RADIUS / normalized_x) if abs(normalized_x) > 1e-3 else None

        if self.near_edge():
            # print("near edge")
            self.gait_manager.update_radius_calibration(0)
            self.gait_manager.command_to_motors(desired_radius=0, heading_angle=0)
            return
            
        self.gait_manager.update_radius_calibration(0.93)    
        
        if (self.botVisible == False):
            # print("bot not visible")
            # self.library.play('kinchit')
            self.gait_manager.command_to_motors(desired_radius=0, heading_angle=0)
            return

        # print(f"normalized x = {normalized_x}")
        self.library.play('Khushi2')
        angle = 0
        if abs(normalized_x) > 0.6:
            angle = 3.14 / 5
        
        if desired_radius == None:
            rad = None
        else:
            if normalized_x > 0:
                rad = desired_radius
            else:
                rad = -desired_radius

        self.gait_manager.command_to_motors(desired_radius=rad, heading_angle=angle)


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
        
        self.use_area = False

        while (len(boxes) == 0):
            print("still finding reference image")
            boxes = model([reference_image]).xyxy[0]
            if (self.getTime() > 5):
                print("could not find image, quitting")
                self.use_area = True
                break
        
        if not self.use_area:
            x_size = boxes[0][2].item() - boxes[0][0].item()
            y_size = boxes[0][3].item() - boxes[0][1].item()
            area = x_size * y_size  
            triangulation = Triangulation(2.0, 0.5, area)

        counter = 50
    
        while True: # run forever
            image = self.camera.get_image()
            if image.shape[2] == 4:
                image = image[:, :, :3]

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model([img])

            bounding_boxes = results.xyxy[0]

            if (len(bounding_boxes) == 0):
                counter = 0
                self.botVisible = False
                continue
            
            counter += 1
            if (counter < 10):
                self.botVisible = False
                continue

            self.botVisible = True

            x_size = bounding_boxes[0][2].item() - bounding_boxes[0][0].item()
            y_size = bounding_boxes[0][3].item() - bounding_boxes[0][1].item()

            self.previousPosition = ((bounding_boxes[0][2].item()+bounding_boxes[0][0].item())/2-80)/80
            if not self.use_area:
                self.botDistance = triangulation.distance_to_camera(x_size * y_size)

            # print(f"position = {self.previousPosition}, distance = {self.botDistance}")

            

    def red_slope(self):
        image = self.camera2.get_image()
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours_red, key=cv2.contourArea, reverse=True)[:1]
        rotated_rect = None

        if len(contours) > 0:
            for contour in contours:
                # Fit a rotated bounding rectangle around the contour
                rotated_rect = cv2.minAreaRect(contour)
                if(75 <= rotated_rect[2] <= 105):
                    print("Moving Straight")
                    return 2  # move straight
                elif(0 < rotated_rect[2] < 75):
                    #self.gait_manager.update_direction(-1, 1)
                    return 1  # move in anticlockwise direction
                else:
                    #self.gait_manager.update_direction(1, -1)
                    return 0  # move in clockwise direction
        else:
            return -1  # Return a code to indicate no contour found

wrestler = Sultaan()
wrestler.run()
