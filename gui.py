import os
import tkinter as tk
from tkinter import ttk
import subprocess
import sys

def run_signlanguage():
    try:
        script_path = "example.py"

        if not os.path.exists(script_path):
            output_label.config(text=f"Error: {script_path} not found.")
            return
        
        result = subprocess.run([sys.executable, script_path],
                                capture_output=True, text=True)

        # Display output or errors in the label
        if result.returncode == 0:
            output_label.config(text=f"Output:\n{result.stdout}")
        else:
            output_label.config(text=f"Error:\n{result.stderr}")

    except Exception as e:
        output_label.config(text=f"Exception: {e}")


def run_gesturecontrol():
    try:
        script_path1 = "example.py"

        if not os.path.exists(script_path1):
            output_label.config(text=f"Error: {script_path1} not found.")
            return
        
        result = subprocess.run([sys.executable, script_path1],
                                capture_output=True, text=True)

        # Display output or errors in the label
        if result.returncode == 0:
            output_label1.config(text=f"Output:\n{result.stdout}")
        else:
            output_label1.config(text=f"Error:\n{result.stderr}")

    except Exception as e:
        output_label1.config(text=f"Exception: {e}")




BASE_DIR = os.path.dirname(os.path.abspath(__file__))


main = tk.Tk()
main.title("Main Window")
main.config(bg="#292727")
main.geometry("700x400")
main.update_idletasks()

geometryX = 0
geometryY = 0

main.geometry("+%d+%d"%(geometryX, geometryY))


style = ttk.Style(main)
style.theme_use("clam")

menu = tk.Menu(main)
main.config(menu=menu)
menu_0 = tk.Menu(menu, tearoff=0)
menu_0.add_command(label="New", command=lambda: print("New clicked"))
menu_0.add_command(label="Open", command=lambda: print("Open clicked"))
menu.add_cascade(label="File", menu=menu_0)
menu_1 = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Edit", menu=menu_1)

style.configure("button.TButton", background="#c0d5f2", foreground="#000", font=("Verdana", 13, "bold"))
style.map("button.TButton", background=[("active", "#E4E2E2")], foreground=[("active", "#000")])

button = ttk.Button(master=main, text="Gesture Control", style="button.TButton", command=run_gesturecontrol)
button.place(x=259, y=76, width=180, height=85)

style.configure("button1.TButton", background="#c0d5f2", foreground="#000", font=("Verdana", 13, "bold"))
style.map("button1.TButton", background=[("active", "#E4E2E2")], foreground=[("active", "#000")])


style.configure("button2.TButton", background="#c0d5f2", foreground="#000", font=("Verdana", 13, "bold"))
style.map("button2.TButton", background=[("active", "#E4E2E2")], foreground=[("active", "#000")])

button2 = ttk.Button(master=main, text="Settings", style="button2.TButton")
button2.place(x=13, y=306, width=100, height=60)


button1 = ttk.Button(master=main, text="Sign Language", style="button1.TButton", command=run_signlanguage)
button1.place(x=259, y=169, width=180, height=85)

output_label = tk.Label(main, text="", wraplength=450, justify="left")
output_label.pack(pady=10)

output_label1 = tk.Label(main, text="", wraplength=450, justify="left")
output_label1.pack(pady=10)



main.mainloop()