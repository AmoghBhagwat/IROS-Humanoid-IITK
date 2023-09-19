from controller import Robot
import sys
#sys.path.append('..')
from myutils.motion_library import MotionLibrary

# Eve's locate_opponent() is implemented in this module:.
from myutils.image_processing import ImageProcessing as IP
from myutils.fall_detection import FallDetection
from myutils.gait_manager import GaitManager
from mytils.camera import Camera
from myutils.finite_state_machine import FiniteStateMachine


class Sultaan (Robot):
    SMALLEST_TURNING_RADIUS = 0.1
    SAFE_ZONE = 0.75
    TIME_BEFORE_DIRECTION_CHANGE = 80  # 8000 ms / 40 ms/

    def __init__(self):
        Robot.__init__(self)
        self.fall = False
        self.time_step = int(self.getBasicTimeStep())
        self.library = MotionLibrary()

        self.camera = Camera(self)
        self.fall_detector = FallDetection(self.time_step, self)
        self.gait_manager = GaitManager(self, self.time_step)
        self.heading_angle = 3.14 / 2
        self.counter = 0
        self.library.add('New', './Khushi.motion', loop = True)
        self.leds = {
            'rightf': self.getDevice('Face/Led/Right'), 
            'leftf': self.getDevice('Face/Led/Left'), 
            'righte': self.getDevice('Ears/Led/Right'), 
            'lefte': self.getDevice('Ears/Led/Left'), 
            'chest': self.getDevice('ChestBoard/Led'), 
        }


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

    def run(self):
        while self.step(self.time_step) != -1:
            # We need to update the internal theta value of the gait manager at every step:
            t = self.getTime()
            self.leds['rightf'].set(0xff0000)
            self.leds['leftf'].set(0xff0000)
            self.leds['righte'].set(0xff0000)
            self.leds['lefte'].set(0xff0000)
            self.leds['chest'].set(0xff0000)
            self.gait_manager.update_theta()
            if(self.fall_detector.detect_fall()): 
                self.fall = True
            if 0.3 < t < 2:
                self.start_sequence()
            elif t > 2:
                self.fall_detector.check()
                if(not self.fall):
                    self.walk()

    def start_sequence(self):
        """At the beginning of the match, the robot walks forwards to move away from the edges."""
        self.gait_manager.command_to_motors(heading_angle=0)

    def walk(self):
        normalized_x = self._get_normalized_opponent_x() 
        desired_radius = (self.SMALLEST_TURNING_RADIUS / normalized_x) if abs(normalized_x) > 1e-3 else None
        if(normalized_x > 0): 
            self.heading_angle = 3.14/4
            self.counter = 0;  
        elif(normalized_x < 0): 
            self.heading_angle = -(3.14/4)
            self.counter = 0 
        elif(normalized_x == 0): 
            return  
        self.counter += 1
        self.gait_manager.command_to_motors(desired_radius=desired_radius, heading_angle=self.heading_angle)
        self.library.play('New')

    def _get_normalized_opponent_x(self):
        """Locate the opponent in the image and return its horizontal position in the range [-1, 1]."""
        img = self.camera.get_image()
        _, _, horizontal_coordinate = IP.locate_opponent(img)
        if horizontal_coordinate is None:
            return 0
        return horizontal_coordinate * 2 / img.shape[1] - 1

# create the Robot instance and run main loop
wrestler = Sultaan()
wrestler.run()



