import importlib
import argparse
from stand import stand
import time
from gait_logic.quadruped import Quadruped

def get_controller(controller_file):
    controller_lib = importlib.import_module(controller_file)
    return controller_lib.controller

def select_motor():
    motor_map = {
        0: "FR_SHOULDER",
        1: "FR_ELBOW",
        2: "FR_HIP",
        3: "FL_SHOULDER",
        4: "FL_ELBOW",
        5: "FL_HIP",
        6: "BR_SHOULDER",
        7: "BR_ELBOW",
        8: "BL_SHOULDER",
        9: "BL_ELBOW",
        10: "BL_HIP",
        11: "BR_HIP"
    }
    print("Select the motor you want to control:")
    for key, motor in motor_map.items():
        print(f"{key}: {motor}")
    
    try:
        motor_id = int(input("Enter the motor number (0-11): "))
        if motor_id not in motor_map:
            raise ValueError("Invalid motor number.")
        motor_name = motor_map[motor_id]
        print(f"Selected motor: {motor_name}")
        return motor_id
    except ValueError as e:
        print(e)
        return select_motor()

if __name__ == "__main__":
    try:
        stand()
        time.sleep(2)
        parser = argparse.ArgumentParser()
        parser.add_argument('--controller', default='controllers.network_receiver')
        args = parser.parse_args()

        controller = get_controller(args.controller)

        r = Quadruped()
        r.calibrate()

        # Select which motor to run
        motor_id = select_motor()
        
        # Input the desired angle for the selected motor
        angle = float(input(f"Enter the desired angle for {motor_id}: "))
        r.set_angle(motor_id, angle)

        r.move(controller)
    
    except KeyboardInterrupt:
        print("Program is closing")
    
    finally:
        stand()
