import threading
import time
import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ft300python.ft300_stream import FT300Stream

# Configuration
PORT = "/dev/ttyUSB0"
TIMEOUT = 1
ZERO_RESET_ON_START = False

class FT300FzGUI:
    def __init__(self, master, port):
        self.master = master
        self.master.title("FT300 6-Axis Live Plot")

        self.ft_stream = FT300Stream(port, timeout=TIMEOUT, zero_reset=ZERO_RESET_ON_START)

        self.t0 = time.time()
        # Storage for 6 axes: Fx, Fy, Fz, Tx, Ty, Tz
        self.times = []
        self.data_history = [[] for _ in range(6)] 
        
        self.reset_requested = False
        self.running = True

        self._build_gui()

        self.thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.thread.start()

        self._update_plot()

    def _build_gui(self):
        # Create a 3x2 grid of subplots
        # Left column (0): Forces | Right column (1): Torques
        self.fig, self.axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        titles = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        labels = ["Force [N]", "Force [N]", "Force [N]", "Torque [Nm]", "Torque [Nm]", "Torque [Nm]"]
        colors = ["r", "g", "b", "r", "g", "b"]
        
        self.lines = []
        
        # Flatten axs for easy iteration: [0,0], [0,1], [1,0], [1,1], [2,0], [2,1]
        flat_axs = [self.axs[0,0], self.axs[1,0], self.axs[2,0], self.axs[0,1], self.axs[1,1], self.axs[2,1]]
        
        for i, ax in enumerate(flat_axs):
            line, = ax.plot([], [], color=colors[i], label=titles[i])
            ax.set_ylabel(labels[i])
            ax.set_title(titles[i], fontsize=10)
            #ax.set_ylim(-30, 30)
            ax.grid(True)
            self.lines.append(line)
            
        self.axs[2,0].set_xlabel("Time [s]")
        self.axs[2,1].set_xlabel("Time [s]")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.reset_button = ttk.Button(btn_frame, text="Reset Data & Zero", command=self._on_reset)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.quit_button = ttk.Button(btn_frame, text="Quit", command=self._on_quit)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def _on_reset(self):
        self.reset_requested = True

    def _on_quit(self):
        self.running = False
        self.master.quit()

    def _acquisition_loop(self):
        while self.running:
            try:
                # ft_values returns [Fx, Fy, Fz, Tx, Ty, Tz]
                ft_values = self.ft_stream.get_force_torque()

                if self.reset_requested:
                    self.times = []
                    self.data_history = [[] for _ in range(6)]
                    self.t0 = time.time()
                    self.reset_requested = False
                    print('Resetting plots...')

                t = time.time() - self.t0
                self.times.append(t)
                for i in range(6):
                    new_val = ft_values[i]
                    self.data_history[i].append(new_val)

                time.sleep(0.005)

            except Exception as e:
                print("Read error:", e)
                time.sleep(0.01)

    def _update_plot(self):
        if self.times:
            # Update each line and rescale its specific axis
            for i, line in enumerate(self.lines):
                line.set_data(self.times, self.data_history[i])
                ax = line.axes
                ax.relim()
                ax.autoscale_view()

        self.canvas.draw_idle()
        self.master.after(50, self._update_plot)

if __name__ == "__main__":
    root = tk.Tk()
    app = FT300FzGUI(root, PORT)
    root.protocol("WM_DELETE_WINDOW", app._on_quit)
    root.mainloop()