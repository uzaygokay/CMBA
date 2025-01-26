import tkinter as tk
from tkinter import simpledialog, filedialog
import numpy as np

class MazeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Generator")
        
        self.root.withdraw()  # Hide the main window initially
        self.get_maze_size()  # Get maze size BEFORE showing the main window
        
        if not self.rows or not self.cols:
            self.root.quit()
            return
        
        self.start_set = False
        self.startPosRow = -1
        self.startPosCol = -1
        self.start_direction = "→"
        
    def initialize_gui(self):
        self.grid = []
        self.selected_tool = "S"
        self.tool_buttons = {}
        self.create_widgets()

    def get_maze_size(self):
        size_dialog = tk.Toplevel(self.root)
        size_dialog.title("Maze Size")
        size_dialog.grab_set()  # Make dialog modal
        size_dialog.focus_force()  # Ensure focus is on the dialog

        tk.Label(size_dialog, text="Rows:").grid(row=0, column=0)
        tk.Label(size_dialog, text="Columns:").grid(row=0, column=1)

        row_entry = tk.Entry(size_dialog)
        col_entry = tk.Entry(size_dialog)
        row_entry.insert(0, "5")
        col_entry.insert(0, "5")

        row_entry.focus_set()  # Auto-focus row input
        row_entry.grid(row=1, column=0)
        col_entry.grid(row=1, column=1)

        def submit():
            self.rows = int(row_entry.get())
            self.cols = int(col_entry.get())
            size_dialog.destroy()  # Close the dialog

        submit_btn = tk.Button(size_dialog, text="Submit", command=submit)
        submit_btn.grid(row=2, columnspan=2)

        size_dialog.wait_window()  # Wait until the dialog is closed

        # Ensure valid values before proceeding
        if not hasattr(self, 'rows') or not hasattr(self, 'cols'):
            self.root.quit()  # Exit if the dialog was closed without input
            return

        self.root.deiconify()  # Show the main window after input is received
        self.initialize_gui()  # Now initialize the maze grid
    
    def create_widgets(self):
        self.canvas = tk.Frame(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        for r in range(self.rows + 2):
            row = []
            for c in range(self.cols + 2):
                if r == 0 or c == 0 or r == self.rows + 1 or c == self.cols + 1:
                    btn = tk.Button(self.canvas, bg="lightgrey", width=2, height=1)
                else:
                    btn = tk.Button(self.canvas, bg="black", width=2, height=1, command=lambda x=r, y=c: self.on_click(x, y))
                btn.grid(row=r, column=c, sticky="nsew")
                row.append(btn)
            self.grid.append(row)
        
        for i in range(self.rows + 2):
            self.canvas.grid_rowconfigure(i, weight=1)
        for j in range(self.cols + 2):
            self.canvas.grid_columnconfigure(j, weight=1)
        
        self.create_controls()
    
    def create_controls(self):
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack()
        
        tools = [("Start (S)", "S"), ("Target (T)", "T"), ("Wall", "W"), ("Reward (R)", "R"), ("Danger (D)", "D"), ("Unset", "U")]
        for text, value in tools:
            btn = tk.Button(self.control_frame, text=text, command=lambda v=value, b=text: self.set_tool(v, b))
            btn.pack(side=tk.LEFT)
            self.tool_buttons[text] = btn
        
        export_btn = tk.Button(self.control_frame, text="Export Maze", command=self.export_maze)
        export_btn.pack(side=tk.LEFT)
        
    def set_tool(self, tool, button_text):
        self.selected_tool = tool
        for btn in self.tool_buttons.values():
            btn.config(relief=tk.RAISED)
        self.tool_buttons[button_text].config(relief=tk.SUNKEN)
    
    def on_click(self, r, c):
        colors = {"S": "green", "T": "red", "W": "lightgrey", "R": "blue", "D": "orange", "U": "black"}
        symbols = {"S": "→", "T": "T", "W": "", "R": "R", "D": "D", "U": ""}

        rotation_dict = {
            "↑": "→",
            "→":"↓",
            "↓":"←",
            "←":"↑",
        }

        if self.selected_tool == "S":
            if self.start_set == False:
                self.startPosRow = r
                self.startPosCol = c
                self.start_set = True
                self.grid[r][c].config(bg=colors.get(self.selected_tool, "black"))
                self.grid[r][c].config(text=symbols.get(self.selected_tool, ""))
            else:
                if r==self.startPosRow and c==self.startPosCol:
                    rotated_start_symbol = rotation_dict.get(self.start_direction, "→")
                    self.grid[r][c].config(text=rotated_start_symbol)
                    self.start_direction = rotated_start_symbol
                else:
                    self.grid[self.startPosRow][self.startPosCol].config(bg=colors.get("U", "black"))
                    self.startPosRow = r
                    self.startPosCol = c
                    self.grid[r][c].config(bg=colors.get(self.selected_tool, "black"))
                    self.grid[r][c].config(text=symbols.get(self.selected_tool, ""))
        else:
            self.grid[r][c].config(bg=colors.get(self.selected_tool, "black"))
            self.grid[r][c].config(text=symbols.get(self.selected_tool, ""))
    
    def export_maze(self):
        maze_array = np.full((self.rows + 2, self.cols + 2), "0", dtype=str)
        start_direction_dict = {
            "↑": "up",
            "→":"right",
            "↓":"down",
            "←":"left",
        }


        for r in range(self.rows + 2):
            for c in range(self.cols + 2):
                color = self.grid[r][c].cget("bg")
                if color == "lightgrey":
                    maze_array[r, c] = "W"
                elif color == "black":
                    maze_array[r, c] = "0"
                elif color == "red":
                    maze_array[r, c] = "T"
                elif color == "blue":
                    maze_array[r, c] = "R"
                elif color == "orange":
                    maze_array[r, c] = "D"
                elif color == "green":
                    maze_array[r, c] = "S"
        
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(f"Maze Size: {self.rows}, {self.cols}\n")  # First line
                f.write(f"Start Direction: {start_direction_dict.get(self.start_direction)}\n\n")  # Second line with dimensions
                
                np.savetxt(f, maze_array, fmt='%s')  # Write the NumPy array to the file
        
if __name__ == "__main__":
    root = tk.Tk()
    app = MazeGUI(root)
    root.mainloop()
