"""Auto-install Ollama + choice of model."""
import subprocess, os, sys, json, time, urllib.request, shutil
from pathlib import Path

OLLAMA_URL = "https://ollama.com/download/OllamaSetup.exe"
DOWNLOAD_DIR = Path(__file__).parent.parent.parent / "db"

MODELS = {
    "mistral-small": {"name": "Mistral Small 3 (24B)", "size": "~14 Go", "ram": "16 Go+", "desc": "Meilleur pour finance + francais. Recommande.", "cmd": "mistral-small"},
    "qwen2.5:14b": {"name": "Qwen 2.5 (14B)", "size": "~9 Go", "ram": "16 Go", "desc": "Excellent raisonnement quantitatif.", "cmd": "qwen2.5:14b"},
    "llama3.1": {"name": "Llama 3.1 (8B)", "size": "~4.7 Go", "ram": "8 Go", "desc": "Leger et polyvalent.", "cmd": "llama3.1"},
    "phi4": {"name": "Phi-4 (14B)", "size": "~9 Go", "ram": "16 Go", "desc": "Bon en raisonnement, Microsoft.", "cmd": "phi4"},
    "gemma3:12b": {"name": "Gemma 3 (12B)", "size": "~8 Go", "ram": "12 Go", "desc": "Google, bon equilibre.", "cmd": "gemma3:12b"},
}
DEFAULT_MODEL = "mistral-small"

def is_ollama_installed():
    if shutil.which("ollama"): return True
    for p in [Path(os.environ.get("LOCALAPPDATA","")) / "Programs" / "Ollama" / "ollama.exe",
              Path("C:/Program Files/Ollama/ollama.exe")]:
        if p.exists(): return True
    return False

def is_ollama_running():
    try:
        with urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/tags"), timeout=2) as r:
            return r.status == 200
    except: return False

def get_installed_models():
    try:
        with urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/tags"), timeout=3) as r:
            return [m["name"] for m in json.loads(r.read().decode()).get("models",[])]
    except: return []

def find_best_model():
    models = get_installed_models()
    for pref in ["mistral-small","qwen2.5:14b","llama3.1","phi4","gemma3:12b","mistral","llama3","qwen2.5"]:
        for m in models:
            if m.startswith(pref): return m
    return models[0] if models else None

def download_ollama(callback=None):
    dest = DOWNLOAD_DIR / "OllamaSetup.exe"
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    if dest.exists() and dest.stat().st_size > 50_000_000:
        if callback: callback("Installeur Ollama deja telecharge.")
        return str(dest)
    if callback: callback("Telechargement d'Ollama (~90 Mo)...")
    try:
        urllib.request.urlretrieve(OLLAMA_URL, str(dest))
        if callback: callback("Telechargement termine.")
        return str(dest)
    except Exception as e:
        if callback: callback(f"Erreur: {e}")
        return None

def install_ollama(callback=None):
    if is_ollama_installed():
        if callback: callback("Ollama deja installe.")
        return True
    exe = download_ollama(callback)
    if not exe: return False
    if callback: callback("Installation d'Ollama...")
    try:
        subprocess.run([exe, "/VERYSILENT", "/NORESTART"], capture_output=True, timeout=120)
        return True
    except:
        try:
            subprocess.run([exe], timeout=300)
            return True
        except Exception as e:
            if callback: callback(f"Erreur: {e}")
            return False

def start_ollama(callback=None):
    if is_ollama_running(): return True
    if callback: callback("Demarrage d'Ollama...")
    cmd = shutil.which("ollama")
    if not cmd:
        for p in [Path(os.environ.get("LOCALAPPDATA","")) / "Programs" / "Ollama" / "ollama.exe",
                  Path("C:/Program Files/Ollama/ollama.exe")]:
            if p.exists(): cmd = str(p); break
    if not cmd: return False
    try:
        subprocess.Popen([cmd, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        for _ in range(15):
            time.sleep(1)
            if is_ollama_running(): return True
    except: pass
    return False

def pull_model(model_key=DEFAULT_MODEL, callback=None):
    info = MODELS.get(model_key, {"cmd": model_key})
    cmd = info.get("cmd", model_key)
    models = get_installed_models()
    for m in models:
        if m.startswith(cmd):
            if callback: callback(f"Modele {cmd} deja installe.")
            return True
    if callback: callback(f"Telechargement de {cmd} ({info.get('size','?')})...")
    if callback: callback("Ca peut prendre 5-15 min selon ta connexion.")
    try:
        payload = json.dumps({"name": cmd, "stream": False}).encode("utf-8")
        req = urllib.request.Request("http://localhost:11434/api/pull", data=payload,
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            if callback: callback(f"Modele {cmd} pret!")
            return True
    except:
        try:
            ocmd = shutil.which("ollama") or "ollama"
            subprocess.run([ocmd, "pull", cmd], capture_output=True, timeout=900)
            return True
        except Exception as e:
            if callback: callback(f"Erreur: {e}")
            return False

def full_setup(model_key=DEFAULT_MODEL, callback=None):
    if not is_ollama_installed():
        if not install_ollama(callback): return False
        time.sleep(2)
    if not is_ollama_running():
        if not start_ollama(callback): return False
    if not pull_model(model_key, callback): return False
    if callback: callback("Pret! Agent IA local disponible.")
    return True
