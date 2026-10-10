from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "ml-100k"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)

registro = []

def log(msg=""):
    print(msg)
    registro.append(str(msg))

# 1. Lectura
# u.data: separado por tabuladores. u.item y u.genre: por "|" y en latin-1.
valoraciones = pd.read_csv(
    DATA_DIR / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
)
generos = pd.read_csv(
    DATA_DIR / "u.genre", sep="|", names=["genero", "id"], encoding="latin-1"
)["genero"].tolist()
peliculas = pd.read_csv(
    DATA_DIR / "u.item",
    sep="|",
    names=["item_id", "titulo", "fecha_estreno", "fecha_video", "imdb_url"] + generos,
    encoding="latin-1",
)

log("1. Lectura")
log(f"   valoraciones: {valoraciones.shape[0]:,} filas x {valoraciones.shape[1]} columnas")
log(f"   peliculas:    {peliculas.shape[0]:,} filas x {peliculas.shape[1]} columnas")
log(f"   géneros:      {len(generos)}")

# 2. Valores nulos
log("\n2. Valores nulos")
log(f"   valoraciones: {int(valoraciones.isna().sum().sum())}")
nulos = peliculas.isna().sum()
for col, n in nulos[nulos > 0].items():
    log(f"   peliculas.{col}: {n}")

# 3. Integridad de las valoraciones
log("\n3. Integridad")
dup = valoraciones.duplicated(["user_id", "item_id"]).sum()
log(f"   Pares (usuario, película) repetidos: {dup}")
fuera = (~valoraciones["item_id"].isin(peliculas["item_id"])).sum()
log(f"   Valoraciones de películas que no están en u.item: {fuera}")
log(f"   Escala de rating: {sorted(valoraciones['rating'].unique().tolist())}")

# 4. Títulos repetidos con distinto item_id
log("\n4. Títulos repetidos")
ids_por_titulo = peliculas.groupby("titulo")["item_id"].apply(list)
repetidos = ids_por_titulo[ids_por_titulo.str.len() > 1]
log(f"   Títulos con más de un item_id: {len(repetidos)}")
for titulo, ids in repetidos.items():
    log(f"   {titulo}: {ids}")

# Son la misma película con dos códigos: se reemplaza el item_id mayor por el menor
mapa = {}
for ids in repetidos:
    ids = sorted(ids)
    mapa[ids[1]] = ids[0]

valoraciones["item_id"] = valoraciones["item_id"].replace(mapa)
peliculas = peliculas[~peliculas["item_id"].isin(mapa.keys())]

# Si un usuario valoró las dos versiones, se conserva su valoración más reciente
antes = len(valoraciones)
valoraciones = valoraciones.sort_values("timestamp")
valoraciones = valoraciones.drop_duplicates(["user_id", "item_id"], keep="last")
valoraciones = valoraciones.sort_values(["user_id", "item_id"])
log(f"   Valoraciones eliminadas por duplicado: {antes - len(valoraciones)}")
log(f"   Resultado: {len(peliculas):,} películas y {len(valoraciones):,} valoraciones")

# 5. Descripción de la matriz usuario-película
n_usuarios = valoraciones["user_id"].nunique()
n_peliculas = valoraciones["item_id"].nunique()
densidad = len(valoraciones) / (n_usuarios * n_peliculas)
por_usuario = valoraciones.groupby("user_id").size()
por_pelicula = valoraciones.groupby("item_id").size()

log("\n5. Matriz usuario-película")
log(f"   Usuarios: {n_usuarios:,}   Películas valoradas: {n_peliculas:,}")
log(f"   Densidad: {densidad:.4f}  (dispersión: {1 - densidad:.2%})")
log(f"   Valoraciones por usuario:  mín {por_usuario.min()}, mediana {por_usuario.median():.0f}, máx {por_usuario.max()}")
log(f"   Valoraciones por película: mín {por_pelicula.min()}, mediana {por_pelicula.median():.0f}, máx {por_pelicula.max()}")
log(f"   Películas con 1 sola valoración: {(por_pelicula == 1).sum()}")

log("\n   Distribución de ratings:")
dist = valoraciones["rating"].value_counts().sort_index()
for r, n in dist.items():
    log(f"   {r}: {n:>6,} ({n / len(valoraciones):.1%})")
log(f"   Media: {valoraciones['rating'].mean():.3f}")

# 6. Preparación de las tablas finales
valoraciones["fecha"] = pd.to_datetime(valoraciones["timestamp"], unit="s")
valoraciones = valoraciones.drop(columns="timestamp")
log(f"\n6. Periodo de las valoraciones: {valoraciones['fecha'].min():%d/%m/%Y} – {valoraciones['fecha'].max():%d/%m/%Y}")

peliculas["anio"] = pd.to_datetime(
    peliculas["fecha_estreno"], format="%d-%b-%Y", errors="coerce"
).dt.year.astype("Int64")
peliculas = peliculas.drop(columns=["fecha_estreno", "fecha_video", "imdb_url"])
peliculas["n_valoraciones"] = peliculas["item_id"].map(por_pelicula).fillna(0).astype(int)
peliculas["rating_medio"] = peliculas["item_id"].map(
    valoraciones.groupby("item_id")["rating"].mean()
).round(3)
peliculas = peliculas[["item_id", "titulo", "anio", "n_valoraciones", "rating_medio"] + generos]

log("\n   Películas por género (una película puede tener varios):")
for g, n in peliculas[generos].sum().sort_values(ascending=False).items():
    log(f"   {g:<12} {n:>4}")

valoraciones.to_csv(PROCESSED / "valoraciones.csv", index=False)
peliculas.to_csv(PROCESSED / "peliculas.csv", index=False)
log("\nGuardado: data/processed/valoraciones.csv y data/processed/peliculas.csv")

(REPORTS / "01_registro_preparacion.txt").write_text("\n".join(registro), encoding="utf-8")