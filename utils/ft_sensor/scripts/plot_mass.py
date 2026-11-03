import threading
import tkinter as tk
from tkinter import ttk
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class MassRosNode(Node):
    def __init__(self, callback):
        super().__init__('mass_plotter_node')
        self.subscription = self.create_subscription(
            Float32,
            '/estimated_mass',
            callback,
            10)

class MassGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("ROS 2 Estimated Mass Live Plot")

        self.t0 = time.time()
        self.times = []
        self.mass_history = []
        self.data_lock = threading.Lock()

        self.running = True

        self._build_gui()

        # Initialize ROS 2
        if not rclpy.ok():
            rclpy.init()

        self.ros_node = MassRosNode(self._ros_callback)

        # Daemon=True is a safety net, but we will join it properly
        self.ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self.ros_thread.start()

        self._update_plot()

    def _ros_spin(self):
        """Background thread for ROS processing."""
        while rclpy.ok() and self.running:
            rclpy.spin_once(self.ros_node, timeout_sec=0.05)

        # Explicit cleanup
        if rclpy.ok():
            self.ros_node.destroy_node()

    def _ros_callback(self, msg: Float32):
        if not self.running:
            return
        with self.data_lock:
            curr_time = time.time() - self.t0
            self.times.append(curr_time)
            self.mass_history.append(msg.data)
            if len(self.mass_history) > 100000:
                self.mass_history.pop(0)
            if len(self.times) > 100000:
                self.times.pop(0)

    def _build_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("Estimated Mass over Time")
        self.ax.set_ylabel("Mass [kg]")
        self.ax.set_xlabel("Time [s]")
        self.ax.grid(True)

        self.line, = self.ax.plot([], [], 'b-', label="Mass")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.reset_button = ttk.Button(btn_frame, text="Reset Data", command=self._on_reset)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.quit_button = ttk.Button(btn_frame, text="Quit", command=self._on_quit)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def _on_reset(self):
        with self.data_lock:
            self.times = []
            self.mass_history = []
            self.t0 = time.time()

    def _on_quit(self):
        """Forceful and immediate shutdown."""
        self.running = False

        # Shutdown ROS immediately to break any active spin_once calls
        if rclpy.ok():
            rclpy.shutdown()

        # Wait briefly for thread to acknowledge
        if self.ros_thread.is_alive():
            self.ros_thread.join(timeout=0.2)

        # Destroy GUI
        self.master.quit()
        self.master.destroy()

    def _update_plot(self):
        if not self.running:
            return

        try:
            with self.data_lock:
                if self.times:
                    self.line.set_data(self.times, self.mass_history)
                    self.ax.relim()
                    self.ax.autoscale_view()

            if self.master.winfo_exists():
                self.canvas.draw_idle()
                # Schedule next update
                self.master.after(50, self._update_plot)
        except (tk.TclError, RuntimeError):
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = MassGUI(root)
    # Ensure the window 'X' button triggers the same graceful quit
    root.protocol("WM_DELETE_WINDOW", app._on_quit)
    root.mainloop()