"""MyFinancialAdvisor — Professional Launcher with Splash Screen."""
import logging
import sys, os, subprocess, time
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent
APP = ROOT / "src" / "app.py"
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
VENV_PYW = ROOT / "venv" / "Scripts" / "pythonw.exe"
SPLASH_IMG = ROOT / "splash.png"
ICON = ROOT / "icon.ico"
REQ = ROOT / "requirements.txt"

def show_splash():
    """Show splash screen while app loads."""
    import tkinter as tk
    from PIL import Image, ImageTk

    splash = tk.Tk()
    splash.overrideredirect(True)

    # Center on screen
    sw, sh = splash.winfo_screenwidth(), splash.winfo_screenheight()
    w, h = 500, 320
    x, y = (sw - w) // 2, (sh - h) // 2
    splash.geometry(f"{w}x{h}+{x}+{y}")
    splash.attributes("-topmost", True)

    # Set icon
    try:
        if ICON.exists():
            splash.iconbitmap(str(ICON))
    except Exception as e:
        logger.debug("splash iconbitmap: %s", e)

    # Load splash image
    if SPLASH_IMG.exists():
        img = Image.open(str(SPLASH_IMG))
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(splash, image=photo, borderwidth=0)
        label.pack()
    else:
        # Fallback: text splash
        splash.configure(bg="#09090f")
        tk.Label(splash, text="MyFinancialAdvisor", font=("Helvetica", 28, "bold"),
                fg="white", bg="#09090f").pack(pady=(60, 10))
        tk.Label(splash, text="Quantitative Investment Terminal", font=("Helvetica", 12),
                fg="#a1a1aa", bg="#09090f").pack(pady=5)
        tk.Label(splash, text="Loading...", font=("Helvetica", 10),
                fg="#52525b", bg="#09090f").pack(pady=(40, 0))

    # Status label at bottom
    status_var = tk.StringVar(value="Starting...")
    status = tk.Label(splash, textvariable=status_var, font=("Helvetica", 9),
                     fg="#52525b", bg="#0f0f1e", anchor="w")
    status.place(x=50, y=258, width=400)

    # Progress bar (simple)
    canvas = tk.Canvas(splash, width=400, height=3, bg="#18182b", highlightthickness=0)
    canvas.place(x=50, y=290)
    progress_bar = canvas.create_rectangle(0, 0, 0, 3, fill="#4f46e5", outline="")

    def update_progress(pct, msg=""):
        canvas.coords(progress_bar, 0, 0, pct * 4, 3)
        if msg: status_var.set(msg)
        splash.update()

    return splash, update_progress

def ensure_venv():
    """Create venv and install deps if needed."""
    if VENV_PY.exists():
        return True

    splash, progress = show_splash()
    progress(10, "Creating virtual environment...")

    # Create venv
    subprocess.run([sys.executable, "-m", "venv", str(ROOT / "venv")],
                  capture_output=True)
    progress(30, "Installing dependencies...")

    # Install requirements
    pip = str(ROOT / "venv" / "Scripts" / "pip.exe")
    subprocess.run([pip, "install", "-r", str(REQ)],
                  capture_output=True)
    progress(90, "Installation complete!")
    time.sleep(1)
    progress(100, "Launching...")
    time.sleep(0.5)
    splash.destroy()
    return True

def main():
    # First run: install
    if not VENV_PY.exists():
        ensure_venv()

    # Show splash while loading
    try:
        splash, progress = show_splash()
        progress(20, "Loading modules...")
        splash.update()
        time.sleep(0.3)
        progress(50, "Initializing database...")
        splash.update()
        time.sleep(0.3)
        progress(80, "Starting application...")
        splash.update()
        time.sleep(0.3)
        progress(100, "Ready!")
        splash.update()
        time.sleep(0.5)
        splash.destroy()
    except:
        pass

    # Launch the app
    python = str(VENV_PY)
    if VENV_PYW.exists():
        python = str(VENV_PYW)  # No console window

    os.execv(str(VENV_PY), [str(VENV_PY), str(APP)])

if __name__ == "__main__":
    main()
