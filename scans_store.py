"""Save Scan: named scans ko disk par JSON mein rakhta hai (Chartink ke 'Save Scan' jaisa).
Dhyan rahe: Streamlit Cloud ka disk app reboot/redeploy par khaali ho sakta hai, isliye
'Download backup' aur 'Backup se load karo' bhi diya hai — wahi sabse pakka tarika hai."""
import json
import os

STORE_PATH = "/tmp/saved_scans.json"


def _read():
    if not os.path.exists(STORE_PATH):
        return {}
    try:
        with open(STORE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _write(data):
    try:
        with open(STORE_PATH, "w") as f:
            json.dump(data, f)
        return True
    except Exception:
        return False


def list_scans():
    return sorted(_read().keys())


def save_scan(name, cfg):
    data = _read()
    data[name] = cfg
    return _write(data)


def load_scan(name):
    return _read().get(name)


def delete_scan(name):
    data = _read()
    data.pop(name, None)
    return _write(data)


def export_all():
    return json.dumps(_read(), indent=2)


def import_all(text, merge=True):
    incoming = json.loads(text)
    data = _read() if merge else {}
    data.update(incoming)
    return _write(data), len(incoming)
