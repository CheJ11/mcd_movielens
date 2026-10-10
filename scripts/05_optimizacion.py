from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader
from surprise.model_selection import GridSearchCV, KFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

USUARIO = 1               # usuario para el que se generan las recomendaciones
MIN_VALORACIONES = 20     # solo se recomiendan películas con al menos 20 valoraciones

registro = []

def log(msg=""):
    print(msg)
    registro.append(str(msg))

valoraciones = pd.read_csv(PROCESSED / "valoraciones.csv")
peliculas = pd.read_csv(PROCESSED / "peliculas.csv").set_index("item_id")
data = Dataset.load_from_df(valoraciones[["user_id", "item_id", "rating"]], Reader(rating_scale=(1, 5)))
kf = KFold(n_splits=5, random_state=42)
rmse_defecto = pd.read_csv(REPORTS / "04_resultados_modelos.csv").set_index("modelo")["RMSE"]

# 1. Grid search para SVD
grilla_svd = {
    "n_factors": [50, 100, 150],
    "n_epochs": [20, 30],
    "lr_all": [0.005, 0.01],
    "reg_all": [0.02, 0.1],
    "random_state": [42],
}
print("Grid search SVD (24 combinaciones x 5 particiones)...")
gs_svd = GridSearchCV(SVD, grilla_svd, measures=["rmse", "mae"], cv=kf)
gs_svd.fit(data)

res_svd = pd.DataFrame(gs_svd.cv_results).sort_values("rank_test_rmse")
log("1. SVD: 5 mejores combinaciones de 24")
log(f"   {'n_factors':>9} {'n_epochs':>9} {'lr_all':>7} {'reg_all':>8} {'RMSE':>8}")
for _, r in res_svd.head(5).iterrows():
    p = r["params"]
    log(f"   {p['n_factors']:>9} {p['n_epochs']:>9} {p['lr_all']:>7} {p['reg_all']:>8} {r['mean_test_rmse']:>8.4f}")

# 2. Grid search para KNN ítem-ítem
grilla_knn = {
    "k": [20, 40, 60, 80],
    "sim_options": {"name": ["msd", "cosine", "pearson_baseline"], "user_based": [False]},
    "verbose": [False],
}
print("Grid search KNN ítem-ítem (12 combinaciones x 5 particiones)...")
gs_knn = GridSearchCV(KNNBasic, grilla_knn, measures=["rmse", "mae"], cv=kf)
gs_knn.fit(data)

res_knn = pd.DataFrame(gs_knn.cv_results)
res_knn["k"] = res_knn["params"].apply(lambda p: p["k"])
res_knn["similitud"] = res_knn["params"].apply(lambda p: p["sim_options"]["name"])
tabla_knn = res_knn.pivot(index="k", columns="similitud", values="mean_test_rmse")
log("\n2. KNN ítem-ítem: RMSE por k y medida de similitud")
log("   " + tabla_knn.round(4).to_string().replace("\n", "\n   "))

# 3. Comparación final
mejor_svd = gs_svd.best_score["rmse"]
mejor_knn = gs_knn.best_score["rmse"]
log("\n3. Comparación (RMSE, mismas 5 particiones)")
log(f"   {'modelo':<16} {'por defecto':>12} {'optimizado':>11} {'mejora':>8}   mejores parámetros")
for nombre, defecto, mejor, params in [
    ("SVD", rmse_defecto["SVD"], mejor_svd, gs_svd.best_params["rmse"]),
    ("KNN ítem-ítem", rmse_defecto["KNN ítem-ítem"], mejor_knn, gs_knn.best_params["rmse"]),
]:
    params = {k: v for k, v in params.items() if k not in ("random_state", "verbose")}
    log(f"   {nombre:<16} {defecto:>12.4f} {mejor:>11.4f} {(defecto - mejor) / defecto:>8.2%}   {params}")
log(f"   {'Baseline':<16} {rmse_defecto['Baseline']:>12.4f}")

# 4. Top-5 para un usuario con el mejor modelo, entrenado con todos los datos
if mejor_svd <= mejor_knn:
    nombre_mejor, modelo = "SVD", gs_svd.best_estimator["rmse"]
else:
    nombre_mejor, modelo = "KNN ítem-ítem", gs_knn.best_estimator["rmse"]
modelo.fit(data.build_full_trainset())

vistas = set(valoraciones.loc[valoraciones["user_id"] == USUARIO, "item_id"])
candidatas = peliculas[(peliculas["n_valoraciones"] >= MIN_VALORACIONES) & (~peliculas.index.isin(vistas))]
# clip=False: sin recortar a 5, para que las predicciones altas no empaten en 5.0
predicciones = pd.Series({i: modelo.predict(USUARIO, i, clip=False).est for i in candidatas.index})
top5 = predicciones.nlargest(5)

log(f"\n4. Recomendaciones para el usuario {USUARIO} (modelo: {nombre_mejor} optimizado)")
favoritas = valoraciones[(valoraciones["user_id"] == USUARIO) & (valoraciones["rating"] == 5)]
log(f"   Ha valorado {len(vistas)} películas; {len(favoritas)} con 5 estrellas. Algunas de ellas:")
for i in favoritas.merge(peliculas, left_on="item_id", right_index=True).nlargest(8, "n_valoraciones")["item_id"]:
    log(f"      {peliculas.loc[i, 'titulo']}")
log(f"\n   Películas candidatas (no vistas, con >= {MIN_VALORACIONES} valoraciones): {len(candidatas):,}")
log("   Top 5 recomendadas:")
generos = peliculas.columns[4:]
for i, est in top5.items():
    g = ", ".join(generos[peliculas.loc[i, generos] == 1])
    log(f"   {est:.2f}  {peliculas.loc[i, 'titulo']}  ({g}; {peliculas.loc[i, 'n_valoraciones']} val., "
        f"media {peliculas.loc[i, 'rating_medio']:.2f})")

# 5. Figura: RMSE por defecto y optimizado
fig, ax = plt.subplots(figsize=(7, 4))
x = range(2)
ax.bar([i - 0.2 for i in x], [rmse_defecto["SVD"], rmse_defecto["KNN ítem-ítem"]], width=0.4,
       label="Por defecto", color="tab:gray")
ax.bar([i + 0.2 for i in x], [mejor_svd, mejor_knn], width=0.4, label="Optimizado", color="tab:blue")
ax.axhline(rmse_defecto["Baseline"], color="tab:red", ls="--", lw=1, label="Baseline")
ax.set_xticks(list(x), ["SVD", "KNN ítem-ítem"])
ax.set_ylim(0.85, 1.0)
ax.set_ylabel("RMSE (5 particiones; menor es mejor)")
ax.set_title("Efecto del ajuste de hiperparámetros")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "07_optimizacion.png", dpi=150)
plt.close(fig)

(REPORTS / "05_registro_optimizacion.txt").write_text("\n".join(registro), encoding="utf-8")
print("\nRegistro guardado en reports/05_registro_optimizacion.txt")