"""MyFinancialAdvisor — EXE entry point.
Shows splash screen, then launches the main app.
Bundled by PyInstaller into a single .exe with custom icon.
"""
import sys, os, time, threading
from pathlib import Path

# PyInstaller sets _MEIPASS for bundled mode
if getattr(sys, 'frozen', False):
    BASE = Path(sys._MEIPASS)
    ROOT = Path(sys.executable).parent
else:
    BASE = Path(__file__).parent
    ROOT = BASE

# Ensure src/ is on path
sys.path.insert(0, str(BASE / "src"))
os.chdir(str(ROOT))

def show_splash():
    """Native Tkinter splash — no deps required."""
    import tkinter as tk
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.attributes("-topmost", True)
    
    sw, sh = splash.winfo_screenwidth(), splash.winfo_screenheight()
    w, h = 600, 380
    splash.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    
    # Try icon
    try:
        ico = ROOT / "icon.ico"
        if not ico.exists(): ico = BASE / "icon.ico"
        if ico.exists(): splash.iconbitmap(str(ico))
    except: pass
    
    # Try image splash
    splash_loaded = False
    try:
        from PIL import Image, ImageTk
        sp_path = ROOT / "splash.png"
        if not sp_path.exists(): sp_path = BASE / "splash.png"
        if sp_path.exists():
            img = Image.open(str(sp_path))
            photo = ImageTk.PhotoImage(img)
            tk.Label(splash, image=photo, bd=0).pack()
            splash._photo = photo  # prevent GC
            splash_loaded = True
    except: pass
    
    if not splash_loaded:
        splash.configure(bg="#0c0c19")
        tk.Label(splash, text="MyFinancialAdvisor", font=("Segoe UI", 28, "bold"),
                fg="white", bg="#0c0c19").pack(pady=(80,10))
        tk.Label(splash, text="Quantitative Investment Terminal", font=("Segoe UI", 12),
                fg="#a1a1aa", bg="#0c0c19").pack()
    
    # Status
    status = tk.StringVar(value="Loading...")
    tk.Label(splash, textvariable=status, font=("Segoe UI", 9),
            fg="#525265", bg="#0c0c19").place(x=42, y=335)
    
    # Progress
    cv = tk.Canvas(splash, width=520, height=3, bg="#1e1e30", highlightthickness=0)
    cv.place(x=40, y=320)
    bar = cv.create_rectangle(0,0,0,3, fill="#4f46e5", outline="")
    
    def progress(pct, msg=""):
        cv.coords(bar, 0, 0, int(pct*5.2), 3)
        if msg: status.set(msg)
        splash.update_idletasks()
        splash.update()
    
    return splash, progress

def main():
    splash, progress = show_splash()
    progress(10, "Initializing...")
    time.sleep(0.3)
    
    progress(25, "Loading modules...")
    import customtkinter  # Heaviest import
    progress(50, "Loading data layer...")
    from core import portfolio, data, model, agent
    progress(70, "Initializing database...")
    # Ensure db dir exists
    (ROOT / "db").mkdir(exist_ok=True)
    progress(85, "Building interface...")
    time.sleep(0.2)
    progress(100, "Ready!")
    time.sleep(0.4)
    splash.destroy()
    
    # Launch main app
    from app import AlphaRanker
    app = AlphaRanker()
    app.mainloop()

if __name__ == "__main__":
    main()
