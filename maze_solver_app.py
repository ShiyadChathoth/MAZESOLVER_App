import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import serial
import serial.tools.list_ports
import threading
import time
from collections import deque

class MazeSolverApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas_width = 640
        self.canvas_height = 480
        
        self.main_frame = tk.Frame(window)
        self.main_frame.pack(padx=10, pady=10)

        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.LEFT)
        
        self.canvas = tk.Canvas(self.video_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack()
        
        self.maze_roi = [100, 40, 540, 440]

        self.control_frame = tk.Frame(self.main_frame, width=200)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.control_frame.pack_propagate(False)

        self.ser = None
        self.command_queue = []
        self.current_image = None
        self.solution_path = None
        self.grid_overlay = None
        self.show_webcam = True
        
        self.solution_string_var = tk.StringVar(value="Solution: N/A")

        self.setup_ui()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.update()
        self.window.mainloop()

    def on_closing(self):
        self.show_webcam = False
        if self.vid.isOpened():
            self.vid.release()
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.window.destroy()

    def setup_ui(self):
        bt_frame = tk.LabelFrame(self.control_frame, text="Bluetooth Connection", padx=10, pady=10)
        bt_frame.pack(pady=10, fill=tk.X)
        
        self.bt_ports = [port.device for port in serial.tools.list_ports.comports()]
        self.bt_ports.insert(0, "Select Port")
        self.bt_var = tk.StringVar(value=self.bt_ports[0])
        
        self.bt_menu = ttk.Combobox(bt_frame, textvariable=self.bt_var, values=self.bt_ports, state="readonly")
        self.bt_menu.pack(pady=5, fill=tk.X)
        
        self.connect_button = tk.Button(bt_frame, text="Connect", command=self.connect_bt)
        self.connect_button.pack(pady=5, fill=tk.X)

        maze_frame = tk.LabelFrame(self.control_frame, text="Maze Controls", padx=10, pady=10)
        maze_frame.pack(pady=10, fill=tk.X)

        self.solve_button = tk.Button(maze_frame, text="Analyze Webcam & Solve", command=self.analyze_and_solve)
        self.solve_button.pack(pady=5, fill=tk.X)

        self.load_image_button = tk.Button(maze_frame, text="Load Image & Solve", command=self.load_and_solve_image)
        self.load_image_button.pack(pady=5, fill=tk.X)

        self.webcam_button = tk.Button(maze_frame, text="Switch to Webcam", command=self.switch_to_webcam, state=tk.DISABLED)
        self.webcam_button.pack(pady=5, fill=tk.X)
        
        robot_frame = tk.LabelFrame(self.control_frame, text="Robot Control", padx=10, pady=10)
        robot_frame.pack(pady=10, fill=tk.X)
        
        solution_label = tk.Label(robot_frame, textvariable=self.solution_string_var, wraplength=180, justify=tk.LEFT)
        solution_label.pack(pady=(0, 5))
        
        self.start_button = tk.Button(robot_frame, text="Send Solution to Robot", command=self.start_robot_control, state=tk.DISABLED)
        self.start_button.pack(pady=5, fill=tk.X)

        self.status_var = tk.StringVar(value="Ready. Connect to robot or load a maze.")
        self.status_bar = tk.Label(self.window, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def connect_bt(self):
        port = self.bt_var.get()
        if port == "Select Port":
            messagebox.showerror("Connection Error", "Please select a valid COM port.")
            return

        try:
            self.ser = serial.Serial(port, 9600, timeout=5)
            self.status_var.set(f"Connected to {port}. Waiting for robot...")
            self.connect_button.config(text="Disconnect", command=self.disconnect_bt)
            if self.command_queue:
                self.start_button.config(state=tk.NORMAL)
            self.bt_menu.config(state=tk.DISABLED)
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to {port}.\nError: {e}")
            self.status_var.set("Connection failed.")

    def disconnect_bt(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.status_var.set("Disconnected. Ready to connect.")
        self.connect_button.config(text="Connect", command=self.connect_bt)
        self.start_button.config(state=tk.DISABLED)
        self.bt_menu.config(state="readonly")

    def load_and_solve_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        self.show_webcam = False
        self.solution_path = None
        self.grid_overlay = None
        self.solution_string_var.set("Solution: N/A")
        self.webcam_button.config(state=tk.NORMAL)
        self.solve_button.config(state=tk.DISABLED)

        try:
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Image Error", "Failed to load image.")
                self.switch_to_webcam()
                return

            h, w, _ = image.shape
            scale = min(self.canvas_width / w, self.canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h))

            frame = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            x_offset = (self.canvas_width - new_w) // 2
            y_offset = (self.canvas_height - new_h) // 2
            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
            
            self.current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.maze_roi = [x_offset, y_offset, x_offset + new_w, y_offset + new_h]
            
            self.status_var.set("Analyzing loaded image...")
            threading.Thread(target=self._solve_maze_thread_target, args=(frame,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not process image.\nError: {e}")
            self.switch_to_webcam()

    def switch_to_webcam(self):
        self.show_webcam = True
        self.solution_path = None
        self.grid_overlay = None
        self.current_image = None
        self.solution_string_var.set("Solution: N/A")
        self.webcam_button.config(state=tk.DISABLED)
        self.solve_button.config(state=tk.NORMAL)
        self.maze_roi = [100, 40, 540, 440]
        self.status_var.set("Switched to webcam feed.")

    def analyze_and_solve(self):
        if self.current_image is not None:
            self.show_webcam = False
            self.webcam_button.config(state=tk.NORMAL)
            self.solve_button.config(state=tk.DISABLED)
            self.status_var.set("Analyzing webcam image...")
            
            image_to_process = self.current_image.copy()
            frame_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)
            threading.Thread(target=self._solve_maze_thread_target, args=(frame_to_process,), daemon=True).start()

    def _update_solution_ui(self, results):
        self.status_var.set(results["status"])
        self.solve_button.config(state=tk.NORMAL)
        
        if results["success"]:
            self.solution_path = results["solution_path"]
            self.grid_overlay = results["grid_overlay"]
            self.command_queue = self.generate_commands(results["raw_path"], results["maze_grid"])
            
            solution_str = "".join(self.command_queue)
            self.solution_string_var.set(f"Solution: {solution_str}")
            
            if self.ser and self.ser.is_open:
                self.start_button.config(state=tk.NORMAL)
        else:
            self.solution_string_var.set("Solution: N/A")
            self.solution_path = None
            self.grid_overlay = None
            self.command_queue = []
            self.start_button.config(state=tk.DISABLED)

    def _solve_maze_thread_target(self, frame):
        results = { "success": False, "status": "Analysis failed." }
        try:
            x1, y1, x2, y2 = self.maze_roi
            maze_img = frame[y1:y2, x1:x2]

            gray = cv2.cvtColor(maze_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            grid_size = 20
            h, w = thresh.shape
            grid_w, grid_h = w // grid_size, h // grid_size
            
            grid_overlay_local = np.zeros_like(maze_img)
            maze_grid = np.zeros((grid_h, grid_w), dtype=int)

            for r in range(grid_h):
                for c in range(grid_w):
                    cell = thresh[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size]
                    if np.mean(cell) < 200:
                        maze_grid[r, c] = 0
                    else:
                        maze_grid[r, c] = 1
                        cv2.rectangle(grid_overlay_local, (c*grid_size, r*grid_size), 
                                      ((c+1)*grid_size, (r+1)*grid_size), (0, 0, 150), -1)

            contours, _ = cv2.findContours((255-thresh), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            squares = []
            min_area_threshold = (grid_size * 0.5) ** 2
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area_threshold: continue
                x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
                aspect_ratio = float(w_cnt) / h_cnt if h_cnt > 0 else 0
                if 0.8 <= aspect_ratio <= 1.2:
                    squares.append({'contour': cnt, 'area': area})
            
            if len(squares) < 2:
                results["status"] = f"Error: Found {len(squares)} squares. Please use disconnected squares."
                self.window.after(0, lambda: self._update_solution_ui(results))
                return

            squares.sort(key=lambda s: s['area'])
            start_square, end_square = squares[0], squares[-1]

            M_start = cv2.moments(start_square['contour'])
            start_pos = (int(M_start["m01"] / M_start["m00"]) // grid_size, int(M_start["m10"] / M_start["m00"]) // grid_size)
            
            M_end = cv2.moments(end_square['contour'])
            end_pos = (int(M_end["m01"] / M_end["m00"]) // grid_size, int(M_end["m10"] / M_end["m00"]) // grid_size)

            maze_grid[start_pos[0], start_pos[1]] = 0
            maze_grid[end_pos[0], end_pos[1]] = 0

            path = self.solve_maze(maze_grid, start_pos, end_pos)

            if path:
                solution_path_local = [(c * grid_size + grid_size//2, r * grid_size + grid_size//2) for r, c in path]
                results.update({
                    "success": True,
                    "status": f"Path found! {len(self.generate_commands(path, maze_grid))} commands ready.",
                    "solution_path": solution_path_local,
                    "grid_overlay": grid_overlay_local,
                    "raw_path": path,
                    "maze_grid": maze_grid
                })
            else:
                results["status"] = "No solution found for the maze."
        
        except Exception as e:
            results["status"] = f"Error during analysis: {e}"

        self.window.after(0, lambda: self._update_solution_ui(results))

    def solve_maze(self, grid, start, end):
        q = deque([(start, [start])])
        visited = {start}
        while q:
            (r, c), path = q.popleft()
            if (r, c) == end:
                return path
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and \
                   grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))
        return None

    def generate_commands(self, path, grid):
        if not path or len(path) < 2:
            return []

        # 1. Identify all critical points (junctions, corners, start, end)
        critical_indices = {0, len(path) - 1}
        for i in range(1, len(path) - 1):
            r, c = path[i]
            # Check for junction
            neighbors = 0
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
                    neighbors += 1
            if neighbors > 2:
                critical_indices.add(i)
                continue
            
            # Check for corner
            r_prev, c_prev = path[i-1]
            r_next, c_next = path[i+1]
            if (r - r_prev) != (r_next - r) or (c - c_prev) != (c_next - c):
                 critical_indices.add(i)

        # 2. Sort and filter nodes to create a clean list of decision points
        node_indices = sorted(list(critical_indices))
        filtered_nodes = []
        if node_indices:
            last_idx = -10
            min_dist = 3 # Heuristic distance to prevent 'fat' nodes
            for idx in node_indices:
                if idx > last_idx + min_dist:
                    filtered_nodes.append(idx)
                    last_idx = idx
            if len(path) - 1 not in filtered_nodes:
                 if filtered_nodes and (len(path) - 1) <= filtered_nodes[-1] + min_dist:
                     filtered_nodes[-1] = len(path) - 1
                 else:
                     filtered_nodes.append(len(path) - 1)

        node_indices = filtered_nodes
        
        if len(node_indices) < 2:
            return ['F']

        # 3. Generate commands
        commands = []
        # Directions: 0:Right(+c), 1:Down(+r), 2:Left(-c), 3:Up(-r)
        
        # Determine the robot's initial orientation from the first segment
        p_start = path[node_indices[0]]
        p_first_node = path[node_indices[1]]
        dr, dc = p_first_node[0] - p_start[0], p_first_node[1] - p_start[1]
        
        current_dir = -1
        if abs(dc) > abs(dr):
            current_dir = 0 if dc > 0 else 2
        else:
            current_dir = 1 if dr > 0 else 3
        
        # 4. Iterate through the segments, deciding the turn at the start of each one
        for i in range(len(node_indices) - 1):
            p_current_node = path[node_indices[i]]
            p_next_node = path[node_indices[i+1]]
            
            dr_out, dc_out = p_next_node[0] - p_current_node[0], p_next_node[1] - p_current_node[1]

            target_dir = -1
            if abs(dc_out) > abs(dr_out):
                target_dir = 0 if dc_out > 0 else 2
            else:
                target_dir = 1 if dr_out > 0 else 3
            
            # Determine turn needed at the current node to face the next segment
            if target_dir != current_dir:
                diff = (target_dir - current_dir + 4) % 4
                if diff == 1: commands.append('R')
                elif diff == 3: commands.append('L')
                elif diff == 2: commands.extend(['R', 'R'])
                current_dir = target_dir
            
            # Add the command to move forward to the next node
            commands.append('F')

        return commands


    def start_robot_control(self):
        if not self.command_queue:
            messagebox.showinfo("Wait", "No commands to send.")
            return
        if not self.ser or not self.ser.is_open:
            messagebox.showerror("Error", "Bluetooth is not connected.")
            return

        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self.send_commands_to_robot, daemon=True).start()

    def send_commands_to_robot(self):
        try:
            self.status_var.set("Waking up robot...")
            time.sleep(2)
            self.ser.flushInput()

            for i, cmd in enumerate(self.command_queue):
                self.status_var.set(f"Sending command {i+1}/{len(self.command_queue)}: '{cmd}'")
                self.ser.write(cmd.encode())
                
                ack = self.ser.read(1).decode()
                if not ack:
                    self.status_var.set("Error: Robot timed out.")
                    messagebox.showerror("Robot Error", "Robot did not respond.")
                    break
                
                self.status_var.set(f"Robot ack: '{ack}'. Command '{cmd}' complete.")
            else:
                self.status_var.set("Maze complete! All commands sent.")
        except serial.SerialException as e:
            self.status_var.set("Serial error during operation.")
            messagebox.showerror("Serial Error", f"An error occurred: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.start_button.config(state=tk.NORMAL)

    def update(self):
        if self.show_webcam:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                self.current_image = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
                cv2.putText(self.current_image, "Webcam feed not available", (50, self.canvas_height//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.current_image is not None:
            display_img = self.current_image.copy()
            x1, y1, x2, y2 = self.maze_roi

            if self.grid_overlay is not None:
                roi_slice = display_img[y1:y2, x1:x2]
                display_img[y1:y2, x1:x2] = cv2.addWeighted(roi_slice, 0.7, self.grid_overlay, 0.3, 0)

            if self.solution_path:
                for i in range(len(self.solution_path) - 1):
                    p1 = (self.solution_path[i][0] + x1, self.solution_path[i][1] + y1)
                    p2 = (self.solution_path[i+1][0] + x1, self.solution_path[i+1][1] + y1)
                    cv2.line(display_img, p1, p2, (255, 0, 0), 3)
            
            if self.show_webcam:
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update)

if __name__ == "__main__":
    app = MazeSolverApp(tk.Tk(), "Maze Solver Vision Control")

