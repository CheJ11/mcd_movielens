import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms import bipartite

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

N_PELICULAS = 50     # películas del subgrafo
K_VECINOS = 3        # cada película se conecta con sus 3 películas más similares
MAX_NIVELES = 15     # niveles de Girvan-Newman a evaluar

registro = []

def log(msg=""):
    print(msg)
    registro.append(str(msg))

valoraciones = pd.read_csv(PROCESSED / "valoraciones.csv")
peliculas = pd.read_csv(PROCESSED / "peliculas.csv")
titulos = peliculas.set_index("item_id")["titulo"]
generos = peliculas.columns[5:].tolist()

# 1. Grafo bipartito (igual que en 02_grafo.py)
valoraciones["u"] = "u" + valoraciones["user_id"].astype(str)
valoraciones["m"] = "m" + valoraciones["item_id"].astype(str)
usuarios = valoraciones["u"].unique().tolist()

B = nx.Graph()
B.add_nodes_from(usuarios, bipartite=0)
B.add_nodes_from(valoraciones["m"].unique(), bipartite=1)
B.add_edges_from(zip(valoraciones["u"], valoraciones["m"]))
grado = dict(B.degree())

def titulo(nodo):
    return titulos[int(nodo[1:])]

# 2. Proyección sobre las películas más valoradas
top_pelis = valoraciones["m"].value_counts().head(N_PELICULAS).index.tolist()
sub = B.subgraph(top_pelis + usuarios)
P = bipartite.overlap_weighted_projected_graph(sub, top_pelis, jaccard=True)

# La proyección es casi completa (todas comparten algún usuario): para que se vea una
# estructura, cada película conserva solo las aristas hacia sus K películas más similares
H = nx.Graph()
for n in P.nodes():
    vecinos = sorted(P[n].items(), key=lambda x: x[1]["weight"], reverse=True)[:K_VECINOS]
    for v, d in vecinos:
        H.add_edge(n, v, weight=d["weight"])

log("1. Subgrafo de películas")
log(f"   Películas: {N_PELICULAS} (mínimo {grado[top_pelis[-1]]} valoraciones)")
log(f"   Proyección completa: {P.number_of_edges()} aristas")
log(f"   Conservando los {K_VECINOS} vecinos más similares: {H.number_of_edges()} aristas, "
    f"{nx.number_connected_components(H)} componente(s)")

# 3. Girvan-Newman: se elige el nivel con mayor modularidad
log("\n2. Girvan-Newman")
log(f"   {'comunidades':>12} {'modularidad':>12}")
mejor, mejor_q = None, -1
for particion in itertools.islice(nx.community.girvan_newman(H), MAX_NIVELES):
    q = nx.community.modularity(H, particion)
    log(f"   {len(particion):>12} {q:>12.3f}")
    if q > mejor_q:
        mejor, mejor_q = particion, q
log(f"   Partición elegida: {len(mejor)} comunidades (modularidad {mejor_q:.3f})")

# 4. Descripción de las comunidades
log("\n3. Comunidades")
comunidad = {}
for c, nodos in enumerate(sorted(mejor, key=len, reverse=True), start=1):
    ids = [int(n[1:]) for n in nodos]
    perfil = peliculas.set_index("item_id").loc[ids, generos].mean().sort_values(ascending=False)
    log(f"\n   Comunidad {c} ({len(nodos)} películas). Géneros principales: "
        + ", ".join(f"{g} {p:.0%}" for g, p in perfil.head(3).items()))
    for n in sorted(nodos, key=grado.get, reverse=True):
        log(f"      {grado[n]:>3}  {titulo(n)}")
        comunidad[n] = c

# 5. Figura: subgrafo de películas coloreado por comunidad
fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(H, seed=42, k=0.6)
colores = [comunidad[n] for n in H.nodes()]
tamanos = [grado[n] for n in H.nodes()]
nx.draw_networkx_edges(H, pos, alpha=0.3, ax=ax)
nx.draw_networkx_nodes(H, pos, node_color=colores, cmap="tab10", vmin=1, vmax=10,
                       node_size=tamanos, alpha=0.9, ax=ax)
nx.draw_networkx_labels(H, pos, labels={n: titulo(n)[:25] for n in H.nodes()}, font_size=7, ax=ax)
ax.set_title(f"Comunidades (Girvan-Newman) entre las {N_PELICULAS} películas más valoradas\n"
             f"Color = comunidad, tamaño = número de valoraciones")
ax.axis("off")
fig.tight_layout()
fig.savefig(FIGURES / "04_comunidades.png", dpi=150)
plt.close(fig)

# 6. Figura: una parte del grafo bipartito original
# Usuarios al azar: los 15 más activos valoraron casi todas las películas populares
# y el grafo saldría completo
top_usuarios = pd.Series(usuarios).sample(15, random_state=42).tolist()
S = B.subgraph(top_usuarios + top_pelis[:15])
pos = nx.bipartite_layout(S, top_usuarios)
fig, ax = plt.subplots(figsize=(10, 8))
nx.draw_networkx_edges(S, pos, alpha=0.2, ax=ax)
nx.draw_networkx_nodes(S, pos, nodelist=top_usuarios, node_color="tab:blue", node_size=150, ax=ax, label="Usuarios")
nx.draw_networkx_nodes(S, pos, nodelist=top_pelis[:15], node_color="tab:orange", node_size=150, ax=ax, label="Películas")
nx.draw_networkx_labels(S, pos, labels={n: n[1:] for n in top_usuarios}, font_size=7, ax=ax)
for n in top_pelis[:15]:
    x, y = pos[n]
    ax.text(x + 0.03, y, titulo(n), fontsize=8, va="center")
ax.set_title(f"Grafo bipartito: 15 usuarios al azar y las 15 películas más valoradas\n"
             f"({S.number_of_edges()} de {15 * 15} aristas posibles)")
ax.legend(loc="lower left")
ax.axis("off")
fig.tight_layout()
fig.savefig(FIGURES / "05_grafo_bipartito_muestra.png", dpi=150)
plt.close(fig)

(REPORTS / "03_registro_comunidades.txt").write_text("\n".join(registro), encoding="utf-8")
print("\nRegistro guardado en reports/03_registro_comunidades.txt")