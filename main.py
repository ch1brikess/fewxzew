import customtkinter as ctk
import rclpy
import threading
import math
import cv2
import numpy as np
from PIL import Image
import time

from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float64


KEY_MAP = {
    'w': 'w', 'ц': 'w',
    'a': 'a', 'ф': 'a',
    's': 's', 'ы': 's',
    'd': 'd', 'в': 'd',
}


class RobotBrain(Node):
    def __init__(self):
        super().__init__('gui_robot_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Pose2D, '/odom_pose2d', self.odom_callback, 10)
        self.cam_sub = self.create_subscription(RosImage, '/front_camera/image', self.camera_callback, 10)
        self.manip_pos_pub = self.create_publisher(Float64, '/manipulate_pos', 10)
        self.manip_vel_pub = self.create_publisher(Float64, '/manipulate_vel', 10)
        self.current_pose = Pose2D()
        self.latest_frame = None
        self.lock = threading.Lock()

    def odom_callback(self, msg):
        with self.lock:
            self.current_pose = msg

    def camera_callback(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = buf.reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "bgra8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        with self.lock:
            self.latest_frame = frame

    def send_velocity(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def send_manipulate_pos(self, pos):
        msg = Float64()
        msg.data = float(pos)
        self.manip_pos_pub.publish(msg)

    def send_manipulate_vel(self, vel):
        msg = Float64()
        msg.data = float(vel)
        self.manip_vel_pub.publish(msg)


class RobotControlApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Robot Control Panel")
        self.geometry("1000x750")
        self.resizable(False, False)

        self.tolerance_dist = 0.02
        self.tolerance_angle = 1.0
        self.linear_speed = 0.8
        self.angular_speed = 0.8
        self.cam_width = 640
        self.cam_height = 480
        self.update_period_ms = 50
        self.key_release_delay_ms = 100
        self.camera_update_period_ms = 100

        rclpy.init()
        self.ros_node = RobotBrain()
        self.ros_thread = threading.Thread(target=self.ros_spin, daemon=True)
        self.ros_thread.start()

        self.move_state = "IDLE"
        self.target_pose = Pose2D()
        self.keyboard_enabled = True

        self.key_w = False
        self.key_a = False
        self.key_s = False
        self.key_d = False

        self.timer_w = None
        self.timer_a = None
        self.timer_s = None
        self.timer_d = None

        self.control_lock = threading.Lock()
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

        self.setup_ui()

        self.bind_all('<KeyPress>', self.on_key_press)
        self.bind_all('<KeyRelease>', self.on_key_release)

        self.gui_update_loop()
        self.camera_update_loop()

    def ros_spin(self):
        rclpy.spin(self.ros_node)

    def control_loop(self):
        while True:
            with self.control_lock:
                state = self.move_state
                target_x = self.target_pose.x
                target_y = self.target_pose.y
                target_theta = self.target_pose.theta

            if state == "IDLE":
                with self.control_lock:
                    w = self.key_w
                    a = self.key_a
                    s = self.key_s
                    d = self.key_d

                x = 0.0
                theta = 0.0
                if w:
                    x = self.linear_speed
                elif s:
                    x = -1 * self.linear_speed
                if a:
                    theta = self.angular_speed * 0.2
                elif d:
                    theta = -1 * self.angular_speed * 0.2

                self.ros_node.send_velocity(x, theta)

            else:
                with self.ros_node.lock:
                    curr_x = self.ros_node.current_pose.x
                    curr_y = self.ros_node.current_pose.y
                    curr_theta = self.ros_node.current_pose.theta

                if state == "MOVING_X":
                    dx = target_x - curr_x
                    dy = target_y - curr_y
                    dist_to_target = math.sqrt(dx * dx + dy * dy)
                    print(f"[MOVE] target=({target_x:.2f}, {target_y:.2f}) current=({curr_x:.2f}, {curr_y:.2f}) dist={dist_to_target:.3f}")

                    if dist_to_target < self.tolerance_dist:
                        self.ros_node.send_velocity(0, 0)
                        with self.control_lock:
                            self.move_state = "IDLE"
                        print(f"[STOP] MOVE done")
                    else:
                        theta_rad = math.radians(curr_theta)
                        dir_x = math.cos(theta_rad)
                        dir_y = math.sin(theta_rad)
                        projection = dx * dir_x + dy * dir_y
                        speed = self.linear_speed if projection > 0 else -self.linear_speed
                        self.ros_node.send_velocity(speed, 0)

                elif state == "MOVING_THETA":
                    diff = target_theta - curr_theta
                    print(f"[THETA] target={target_theta:.2f} current={curr_theta:.2f} diff={diff:.2f} tol={self.tolerance_angle}")
                    if abs(diff) < self.tolerance_angle:
                        self.ros_node.send_velocity(0, 0)
                        with self.control_lock:
                            self.move_state = "IDLE"
                        print(f"[STOP] THETA done")
                    else:
                        speed = self.angular_speed if diff > 0 else -self.angular_speed
                        self.ros_node.send_velocity(0, speed)

            time.sleep(0.03)

    def setup_ui(self):
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.camera_label = ctk.CTkLabel(
            self.left_frame, text="Загрузка камеры...",
            width=self.cam_width, height=self.cam_height
        )
        self.camera_label.pack(pady=10)

        self.coords_label = ctk.CTkLabel(
            self.left_frame, text="X: 0.00 | Y: 0.00 | Theta: 0.00",
            font=("Arial", 16, "bold")
        )
        self.coords_label.pack(pady=10)

        self.right_frame = ctk.CTkFrame(self, width=320)
        self.right_frame.pack(side="right", fill="y", padx=10, pady=10)

        self.keyboard_switch = ctk.CTkSwitch(
            self.right_frame, text="Клавиатура (WASD/ЦФЫВ)",
            command=self.toggle_keyboard
        )
        self.keyboard_switch.select()
        self.keyboard_switch.pack(anchor="w", padx=10, pady=(15, 0))

        ctk.CTkLabel(self.right_frame, text="Линейная скорость:", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        speed_row = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        speed_row.pack(fill="x", padx=10, pady=5)
        self.entry_speed = ctk.CTkEntry(speed_row, width=80)
        self.entry_speed.insert(0, "0.8")
        self.entry_speed.pack(side="left", padx=(0, 5))
        self.entry_speed.bind('<FocusOut>', self.validate_linear_speed)
        self.entry_speed.bind('<Return>', self.validate_linear_speed)

        ctk.CTkLabel(self.right_frame, text="Угловая скорость:", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        angular_speed_row = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        angular_speed_row.pack(fill="x", padx=10, pady=5)
        self.entry_angular_speed = ctk.CTkEntry(angular_speed_row, width=80)
        self.entry_angular_speed.insert(0, "0.8")
        self.entry_angular_speed.pack(side="left", padx=(0, 5))
        self.entry_angular_speed.bind('<FocusOut>', self.validate_angular_speed)
        self.entry_angular_speed.bind('<Return>', self.validate_angular_speed)

        ctk.CTkLabel(self.right_frame, text="Точное перемещение", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 0))

        ctk.CTkLabel(self.right_frame, text="Дистанция (метры):").pack(anchor="w", padx=10)
        dist_row = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        dist_row.pack(fill="x", padx=10, pady=5)
        self.entry_dist = ctk.CTkEntry(dist_row, placeholder_text="1.5 или -0.5")
        self.entry_dist.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_dist.bind('<Return>', lambda e: self.start_move_x())
        ctk.CTkButton(dist_row, text="ок", width=50, command=self.start_move_x).pack(side="right")

        ctk.CTkLabel(self.right_frame, text="Повернуть на:").pack(anchor="w", padx=10)
        angle_rel_row = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        angle_rel_row.pack(fill="x", padx=10, pady=5)
        self.entry_angle_rel = ctk.CTkEntry(angle_rel_row, placeholder_text="значение")
        self.entry_angle_rel.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_angle_rel.bind('<Return>', lambda e: self.start_move_theta_relative())
        ctk.CTkButton(angle_rel_row, text="ок", width=50, command=self.start_move_theta_relative).pack(side="right")

        ctk.CTkLabel(self.right_frame, text="Повернуть до:").pack(anchor="w", padx=10)
        angle_abs_row = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        angle_abs_row.pack(fill="x", padx=10, pady=5)
        self.entry_angle_abs = ctk.CTkEntry(angle_abs_row, placeholder_text="значение")
        self.entry_angle_abs.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_angle_abs.bind('<Return>', lambda e: self.start_move_theta_absolute())
        ctk.CTkButton(angle_abs_row, text="ок", width=50, command=self.start_move_theta_absolute).pack(side="right")

        ctk.CTkLabel(self.right_frame, text="Клешня", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 0))

        ctk.CTkLabel(self.right_frame, text="Скорость клешни:").pack(anchor="w", padx=10)
        manip_vel_row = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        manip_vel_row.pack(fill="x", padx=10, pady=5)
        self.entry_manip_vel = ctk.CTkEntry(manip_vel_row, placeholder_text="скорость")
        self.entry_manip_vel.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.entry_manip_vel.bind('<Return>', lambda e: self.send_manip_vel())
        ctk.CTkButton(manip_vel_row, text="ок", width=50, command=self.send_manip_vel).pack(side="right")

        ctk.CTkLabel(self.right_frame, text="Сервоприводы:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        
        servo_frame = ctk.CTkFrame(self.right_frame)
        servo_frame.pack(fill="x", padx=10, pady=5)
        
        self.servo_entries = []
        for i in range(4):
            row = ctk.CTkFrame(servo_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=f"Серво {i+1}:", width=60).pack(side="left")
            entry = ctk.CTkEntry(row, width=80, placeholder_text="0")
            entry.pack(side="left", padx=5)
            btn = ctk.CTkButton(row, text="ок", width=40, command=lambda idx=i: self.send_servo_pos(idx))
            btn.pack(side="right")
            self.servo_entries.append(entry)

        ctk.CTkLabel(
            self.right_frame,
            text="Управление: WASD / ЦФЫВ",
            text_color="gray"
        ).pack(side="bottom", pady=20)

    def toggle_keyboard(self):
        self.keyboard_enabled = self.keyboard_switch.get()
        if not self.keyboard_enabled:
            with self.control_lock:
                self.key_w = False
                self.key_a = False
                self.key_s = False
                self.key_d = False

    def validate_linear_speed(self, event=None):
        val = self.entry_speed.get()
        try:
            v = float(val)
            if v < 0:
                v = 0.0
            self.linear_speed = v
            self.entry_speed.delete(0, "end")
            self.entry_speed.insert(0, str(v))
        except ValueError:
            self.entry_speed.delete(0, "end")
            self.entry_speed.insert(0, str(self.linear_speed))

    def validate_angular_speed(self, event=None):
        val = self.entry_angular_speed.get()
        try:
            v = float(val)
            if v < 0:
                v = 0.0
            self.angular_speed = v
            self.entry_angular_speed.delete(0, "end")
            self.entry_angular_speed.insert(0, str(v))
        except ValueError:
            self.entry_angular_speed.delete(0, "end")
            self.entry_angular_speed.insert(0, str(self.angular_speed))

    def on_key_press(self, event):
        if not self.keyboard_enabled:
            return

        focused = self.focus_get()
        if focused in (self.entry_dist, self.entry_angle_rel, self.entry_angle_abs, self.entry_speed, self.entry_angular_speed, self.entry_manip_vel) or focused in self.servo_entries:
            return

        key = event.keysym.lower()
        mapped = KEY_MAP.get(key)
        if not mapped:
            return

        with self.control_lock:
            if mapped == 'w':
                self.key_w = True
                if self.timer_w is not None:
                    self.after_cancel(self.timer_w)
                    self.timer_w = None
            elif mapped == 'a':
                self.key_a = True
                if self.timer_a is not None:
                    self.after_cancel(self.timer_a)
                    self.timer_a = None
            elif mapped == 's':
                self.key_s = True
                if self.timer_s is not None:
                    self.after_cancel(self.timer_s)
                    self.timer_s = None
            elif mapped == 'd':
                self.key_d = True
                if self.timer_d is not None:
                    self.after_cancel(self.timer_d)
                    self.timer_d = None

    def on_key_release(self, event):
        if not self.keyboard_enabled:
            return

        focused = self.focus_get()
        if focused in (self.entry_dist, self.entry_angle_rel, self.entry_angle_abs, self.entry_speed, self.entry_angular_speed, self.entry_manip_vel) or focused in self.servo_entries:
            return

        key = event.keysym.lower()
        mapped = KEY_MAP.get(key)
        if not mapped:
            return

        def release_w():
            with self.control_lock:
                self.key_w = False
            self.timer_w = None

        def release_a():
            with self.control_lock:
                self.key_a = False
            self.timer_a = None

        def release_s():
            with self.control_lock:
                self.key_s = False
            self.timer_s = None

        def release_d():
            with self.control_lock:
                self.key_d = False
            self.timer_d = None

        if mapped == 'w':
            self.timer_w = self.after(self.key_release_delay_ms, release_w)
        elif mapped == 'a':
            self.timer_a = self.after(self.key_release_delay_ms, release_a)
        elif mapped == 's':
            self.timer_s = self.after(self.key_release_delay_ms, release_s)
        elif mapped == 'd':
            self.timer_d = self.after(self.key_release_delay_ms, release_d)

    def start_move_x(self):
        with self.control_lock:
            if self.move_state != "IDLE":
                return
            dist = float(self.entry_dist.get())
            with self.ros_node.lock:
                current_x = self.ros_node.current_pose.x
                current_y = self.ros_node.current_pose.y
                current_theta = self.ros_node.current_pose.theta
                theta_rad = math.radians(current_theta)
                self.target_pose.x = current_x + dist * math.cos(theta_rad)
                self.target_pose.y = current_y + dist * math.sin(theta_rad)
            self.move_state = "MOVING_X"
            print(f"[START] MOVING: target=({self.target_pose.x:.2f}, {self.target_pose.y:.2f}) dist={dist}")

    def start_move_theta_relative(self):
        val = self.entry_angle_rel.get()
        try:
            delta = float(val)
        except ValueError:
            return
        with self.control_lock:
            if self.move_state != "IDLE":
                return
            with self.ros_node.lock:
                current_theta = self.ros_node.current_pose.theta
                self.target_pose.theta = current_theta + delta
            self.move_state = "MOVING_THETA"
            print(f"[START] MOVING_THETA relative: current={current_theta:.2f} delta={delta:.2f} target={self.target_pose.theta:.2f}")

    def start_move_theta_absolute(self):
        val = self.entry_angle_abs.get()
        try:
            target = float(val)
        except ValueError:
            return
        with self.control_lock:
            if self.move_state != "IDLE":
                return
            self.target_pose.theta = target
            self.move_state = "MOVING_THETA"
            print(f"[START] MOVING_THETA absolute: target={target:.2f}")

    def send_manip_vel(self):
        val = float(self.entry_manip_vel.get())
        self.ros_node.send_manipulate_vel(val)

    def send_servo_pos(self, idx):
        val = float(self.servo_entries[idx].get())
        self.ros_node.send_manipulate_pos(val)

    def gui_update_loop(self):
        with self.ros_node.lock:
            pose = self.ros_node.current_pose
            self.coords_label.configure(
                text=f"X: {pose.x:.2f} м  |  Y: {pose.y:.2f} м  |  Theta: {pose.theta:.2f}"
            )

        self.after(self.update_period_ms, self.gui_update_loop)

    def camera_update_loop(self):
        with self.ros_node.lock:
            frame = self.ros_node.latest_frame

        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.resize((self.cam_width, self.cam_height), Image.Resampling.LANCZOS)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(self.cam_width, self.cam_height))
            self.camera_label.configure(image=ctk_image, text="")
            self.camera_label.image = ctk_image

        self.after(self.camera_update_period_ms, self.camera_update_loop)

    def on_closing(self):
        self.ros_node.send_velocity(0, 0)
        self.ros_node.destroy_node()
        rclpy.shutdown()
        self.destroy()


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
app = RobotControlApp()
app.protocol("WM_DELETE_WINDOW", app.on_closing)
app.mainloop()