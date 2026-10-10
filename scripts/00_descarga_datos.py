import stat
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = RAW / "ml-100k.zip"
DATA_DIR = RAW / "ml-100k"

# Archivos que usa el taller
ARCHIVOS = ["u.data", "u.item", "u.genre", "u.user", "README"]

registro = []

def log(msg=""):
    print(msg)
    registro.append(str(msg))

RAW.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# 1. Descarga (solo si no existe)
log("1. Descarga")
if ZIP_PATH.exists():
    log(f"   {ZIP_PATH.name} ya existe, no se descarga de nuevo")
else:
    log(f"   Descargando {URL}")
    urllib.request.urlretrieve(URL, ZIP_PATH)
log(f"   Tamaño del zip: {ZIP_PATH.stat().st_size / 1e6:.2f} MB")

# 2. Verificación del zip y extracción
log("\n2. Extracción")
with zipfile.ZipFile(ZIP_PATH) as z:
    corrupto = z.testzip()
    if corrupto is not None:
        raise RuntimeError(f"Archivo dañado dentro del zip: {corrupto}")
    log(f"   Zip íntegro, {len(z.namelist())} entradas")
    if DATA_DIR.exists():
        log(f"   {DATA_DIR.name}/ ya existe, no se extrae de nuevo")
    else:
        z.extractall(RAW)
        log(f"   Extraído en {DATA_DIR.relative_to(ROOT)}")

# 3. Archivos usados: tamaño y modo solo lectura
log("\n3. Archivos que usa el taller (solo lectura)")
for nombre in ARCHIVOS:
    ruta = DATA_DIR / nombre
    if not ruta.exists():
        raise FileNotFoundError(ruta)
    ruta.chmod(stat.S_IREAD)
    log(f"   {nombre:<8} {ruta.stat().st_size / 1e3:>9.1f} kB")

(REPORTS / "00_registro_descarga.txt").write_text("\n".join(registro), encoding="utf-8")
print("\nRegistro guardado en reports/00_registro_descarga.txt")