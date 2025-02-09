from gpiozero import LED
import time

class BrakeController:
    def __init__(self, brake_pin=27, cooldown_time=20):
        # Initialize the brake control system with a cooldown timer.
        self.brake_signal = LED(brake_pin)  # GPIO pin for brake activation
        self.brake_active = False  # Track brake state
        self.last_stop_time = None  # Timestamp when vehicle last stopped
        self.cooldown_time = cooldown_time  # Cooldown period in seconds

    def activate_brake(self):
        # Activate the brake if cooldown has passed.
        if not self.brake_active:
            current_time = time.time()

            # If vehicle was stopped recently, prevent immediate activation
            if self.last_stop_time and (current_time - self.last_stop_time < self.cooldown_time):
                print(f"⏳ Cooldown active: Brake won't activate until {self.cooldown_time} seconds have passed.")
                return  # Don't activate the brake yet
            
            self.brake_signal.on()  # 🛑 Activate brake
            self.brake_active = True
            print("🛑 Brake ACTIVATED!")

    def release_brake(self):
        # Deactivate the brake and start the cooldown timer.
        if self.brake_active:
            self.brake_signal.off()  # ✅ Release brake
            self.brake_active = False
            self.last_stop_time = time.time()  # Save timestamp of stop
            print("✅ Brake RELEASED! Cooldown timer started.")

    def update(self, self_speed, detection_triggered):
        # Check brake logic: Activate on detection, release when speed is 0.
        if detection_triggered:
            self.activate_brake()
        elif self_speed == 0:
            self.release_brake()
