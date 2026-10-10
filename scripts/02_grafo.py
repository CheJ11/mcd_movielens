from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

registro = []

def log(msg=""):
    print(msg)
    registro.append(str(msg))

valoraciones = pd.read_csv(PROCESSED / "valoraciones.csv")
peliculas = pd.read_csv(PROCESSED / "peliculas.csv")
titulos = peliculas.set_index("item_id")["titulo"]

# 1. Grafo bipartito
# Los ids de usuarios y películas se solapan (ambos empiezan en 1): se usan prefijos u y m
valoraciones["u"] = "u" + valoraciones["user_id"].astype(str)
valoraciones["m"] = "m" + valoraciones["item_id"].astype(str)
usuarios = valoraciones["u"].unique().tolist()
pelis = valoraciones["m"].unique().tolist()

G = nx.Graph()
G.add_nodes_from(usuarios, bipartite=0)
G.add_nodes_from(pelis, bipartite=1)
G.add_weighted_edges_from(zip(valoraciones["u"], valoraciones["m"], valoraciones["rating"]))

log("1. Grafo bipartito")
log(f"   Nodos: {G.number_of_nodes():,} ({len(usuarios)} usuarios + {len(pelis):,} películas)")
log(f"   Aristas: {G.number_of_edges():,}")
log(f"   ¿Bipartito?: {bipartite.is_bipartite(G)}")
log(f"   Componentes conexas: {nx.number_connected_components(G)}")

# 2. Métricas por nodo
# Intermediación: se aproxima con 500 nodos de origen (k) para reducir el tiempo de cálculo
# Agrupamiento: el coeficiente clásico cuenta triángulos, que no existen en un grafo
# bipartito; bipartite.clustering mide cuánto comparte un nodo sus vecinos con otros de su tipo
nodos = pd.DataFrame({
    "grado": dict(G.degree()),
    "centralidad_grado": bipartite.degree_centrality(G, usuarios),
    "intermediacion": nx.betweenness_centrality(G, k=500, seed=42),
    "agrupamiento": bipartite.clustering(G),
})
U = nodos.loc[usuarios].copy()
P = nodos.loc[pelis].copy()
U["rating_medio"] = valoraciones.groupby("u")["rating"].mean()
P["titulo"] = [titulos[int(n[1:])] for n in P.index]

# 3. Centralidad de grado
log("\n2. Centralidad de grado")
log("   Top 5 usuarios (más películas valoradas):")
log(f"   {'usuario':>8} {'películas':>10} {'centralidad':>12} {'rating medio':>13}")
for nodo, r in U.nlargest(5, "grado").iterrows():
    log(f"   {nodo[1:]:>8} {r['grado']:>10.0f} {r['centralidad_grado']:>12.3f} {r['rating_medio']:>13.2f}")

log("\n   Top 10 películas (más valoradas):")
for _, r in P.nlargest(10, "grado").iterrows():
    log(f"   {r['grado']:>4.0f}  {r['centralidad_grado']:.3f}  {r['titulo']}")

log(f"\n   Grado de usuarios:  mediana {U['grado'].median():.0f}, media {U['grado'].mean():.1f}, máx {U['grado'].max():.0f}")
log(f"   Grado de películas: mediana {P['grado'].median():.0f}, media {P['grado'].mean():.1f}, máx {P['grado'].max():.0f}")

# 4. Centralidad de intermediación
log("\n3. Centralidad de intermediación")
log("   Top 10 películas:")
top_grado = P.nlargest(10, "grado")["titulo"].tolist()
for _, r in P.nlargest(10, "intermediacion").iterrows():
    marca = "" if r["titulo"] in top_grado else "   <- no está en el top 10 por grado"
    log(f"   {r['intermediacion']:.4f}  ({r['grado']:>3.0f} val.)  {r['titulo']}{marca}")

# 5. Coeficiente de agrupamiento
log("\n4. Coeficiente de agrupamiento")
log(f"   Triángulos en el grafo: {sum(nx.triangles(G).values())} -> el coeficiente clásico es 0")
log(f"   Agrupamiento bipartito medio: usuarios {U['agrupamiento'].mean():.3f}, películas {P['agrupamiento'].mean():.3f}")

# 6. Figuras
# 6.1 Distribución de grado (eje x logarítmico: pocos nodos tienen grados muy altos)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, df, nombre, color in [(axes[0], U, "Usuarios", "tab:blue"), (axes[1], P, "Películas", "tab:orange")]:
    bins = np.logspace(0, np.log10(df["grado"].max()), 30)
    ax.hist(df["grado"], bins=bins, color=color, edgecolor="white")
    ax.axvline(df["grado"].median(), color="black", ls="--", lw=1, label=f"Mediana = {df['grado'].median():.0f}")
    ax.set_xscale("log")
    ax.set_title(f"{nombre}: número de valoraciones")
    ax.set_xlabel("Grado (escala log)")
    ax.set_ylabel(f"Número de {nombre.lower()}")
    ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "01_distribucion_grado.png", dpi=150)
plt.close(fig)

# 6.2 Películas más centrales según grado e intermediación
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, nombre in [(axes[0], "grado", "grado"), (axes[1], "intermediacion", "intermediación")]:
    top = P.nlargest(10, col).iloc[::-1]
    ax.barh(top["titulo"], top[col])
    ax.set_xlabel(nombre.capitalize())
    ax.set_title(f"Top 10 películas por {nombre}")
fig.tight_layout()
fig.savefig(FIGURES / "02_peliculas_centrales.png", dpi=150)
plt.close(fig)

# 6.3 Agrupamiento frente a grado
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(P["grado"], P["agrupamiento"], s=8, alpha=0.5, color="tab:orange", label="Películas")
ax.scatter(U["grado"], U["agrupamiento"], s=8, alpha=0.5, color="tab:blue", label="Usuarios")
ax.set_xscale("log")
ax.set_xlabel("Grado (escala log)")
ax.set_ylabel("Coeficiente de agrupamiento bipartito")
ax.set_title("Agrupamiento frente a grado")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "03_agrupamiento_vs_grado.png", dpi=150)
plt.close(fig)

(REPORTS / "02_registro_grafo.txt").write_text("\n".join(registro), encoding="utf-8")
print("\nRegistro guardado en reports/02_registro_grafo.txt")