from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, BaselineOnly, Dataset, KNNBasic, Reader
from surprise.model_selection import KFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

registro = []

def log(msg=""):
    print(msg)
    registro.append(str(msg))

# 1. Datos en el formato de surprise: (usuario, ítem, rating) y la escala del rating
valoraciones = pd.read_csv(PROCESSED / "valoraciones.csv")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(valoraciones[["user_id", "item_id", "rating"]], reader)

log("1. Datos")
log(f"   {len(valoraciones):,} valoraciones, {valoraciones['user_id'].nunique()} usuarios, "
    f"{valoraciones['item_id'].nunique():,} películas")

# 2. Modelos
modelos = {
    "Baseline": BaselineOnly(verbose=False),
    "KNN usuario-usuario": KNNBasic(sim_options={"user_based": True}, verbose=False),
    "KNN ítem-ítem": KNNBasic(sim_options={"user_based": False}, verbose=False),
    "SVD": SVD(random_state=42),
}
# 3. Validación cruzada: las mismas 5 particiones para todos los modelos
kf = KFold(n_splits=5, random_state=42)
filas = []
for nombre, modelo in modelos.items():
    print(f"Evaluando {nombre}...")
    res = cross_validate(modelo, data, measures=["RMSE", "MAE"], cv=kf)
    filas.append({
        "modelo": nombre,
        "RMSE": res["test_rmse"].mean(),
        "RMSE_std": res["test_rmse"].std(),
        "MAE": res["test_mae"].mean(),
        "entrenamiento_s": sum(res["fit_time"]) / 5,
        "prediccion_s": sum(res["test_time"]) / 5,
    })
resultados = pd.DataFrame(filas).sort_values("RMSE")

log("\n2. Validación cruzada (5 particiones, media por partición)")
log(f"   {'modelo':<22} {'RMSE':>7} {'± desv':>7} {'MAE':>7} {'entren. (s)':>12} {'pred. (s)':>10}")
for _, r in resultados.iterrows():
    log(f"   {r['modelo']:<22} {r['RMSE']:>7.4f} {r['RMSE_std']:>7.4f} {r['MAE']:>7.4f} "
        f"{r['entrenamiento_s']:>12.1f} {r['prediccion_s']:>10.1f}")

base = resultados.set_index("modelo").loc["Baseline", "RMSE"]
log("\n   Mejora de RMSE respecto al Baseline:")
for _, r in resultados.iterrows():
    if r["modelo"] != "Baseline":
        log(f"   {r['modelo']:<22} {(base - r['RMSE']) / base:>7.2%}")

resultados.to_csv(REPORTS / "04_resultados_modelos.csv", index=False)

# 4. Figura: RMSE por modelo
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(resultados["modelo"][::-1], resultados["RMSE"][::-1], xerr=resultados["RMSE_std"][::-1],
        color="tab:blue", capsize=4)
for i, (v, s) in enumerate(zip(resultados["RMSE"][::-1], resultados["RMSE_std"][::-1])):
    ax.text(v + s + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
ax.set_xlim(0.85, resultados["RMSE"].max() + 0.04)
ax.set_xlabel("RMSE (media de 5 particiones; menor es mejor)")
ax.set_title("Comparación de modelos con hiperparámetros por defecto")
fig.tight_layout()
fig.savefig(FIGURES / "06_comparacion_modelos.png", dpi=150)
plt.close(fig)

(REPORTS / "04_registro_recomendacion.txt").write_text("\n".join(registro), encoding="utf-8")
print("\nRegistro guardado en reports/04_registro_recomendacion.txt")