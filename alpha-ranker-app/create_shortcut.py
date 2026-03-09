"""Create a Windows desktop shortcut for MyFinancialAdvisor.
Run once after install: python create_shortcut.py
"""
import os, sys
from pathlib import Path

def create_shortcut():
    ROOT = Path(__file__).parent
    desktop = Path(os.environ.get("USERPROFILE", "~")) / "Desktop"
    
    shortcut_path = desktop / "MyFinancialAdvisor.lnk"
    target = str(ROOT / "venv" / "Scripts" / "pythonw.exe")
    arguments = f'"{ROOT / "MyFinancialAdvisor.pyw"}"'
    icon = str(ROOT / "icon.ico")
    workdir = str(ROOT)

    try:
        # Use PowerShell to create shortcut (works on all Windows)
        import subprocess
        ps_cmd = f'''
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("{shortcut_path}")
$s.TargetPath = "{target}"
$s.Arguments = {arguments}
$s.WorkingDirectory = "{workdir}"
$s.IconLocation = "{icon}"
$s.Description = "MyFinancialAdvisor - Quantitative Investment Terminal"
$s.Save()
'''
        subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
        print(f"Shortcut created: {shortcut_path}")
        print(f"  Target: {target}")
        print(f"  Icon: {icon}")
        print()
        print("You can now launch MyFinancialAdvisor from your desktop!")

    except Exception as e:
        print(f"Could not create shortcut automatically: {e}")
        print()
        print("Manual steps:")
        print(f"  1. Right-click Desktop > New > Shortcut")
        print(f'  2. Target: {target} "{ROOT / "MyFinancialAdvisor.pyw"}"')
        print(f"  3. Change Icon > Browse > {icon}")
        print(f"  4. Name: MyFinancialAdvisor")

if __name__ == "__main__":
    create_shortcut()
