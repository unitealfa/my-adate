# -*- coding: utf-8 -*-
"""
main.py — Lanceur interactif "tout-en-un" pour le projet Cluster-First / Route-Second

 Ce que fait ce script :
- Fournit un MENU clair (1/2/3/4/5/6/7/0) pour :
  [1] Lister les instances détectées dans ./data
  [2] Lister les instances "recommandées"
  [3] Démo (1 run) — lancer le solveur sur une instance et afficher les 1ères routes
  [4] Tests rapides — vérifier la faisabilité (capacité + fenêtres) sur 1..n instances
  [5] Benchmarks — N runs/instance, statistiques, GAP (si .sol), graphiques
  [6] Explorateur visuel — générer/éditer un layout synthétique en direct
  [7] Changer les paramètres par défaut (seed, itérations Tabu, etc.)
  [0] Quitter

 Confort et robustesse :
- Résolution d’instance flexible : nom simple ("X-n101-k25.vrp") ou chemin relatif/absolu.
- Auto-remap des fichiers non standard à la racine vers la version "cvrplib" officielle.
  -> Exemple : "A-n32-k5.vrp" (racine) sera remplacé par "data/cvrplib/A-n32-k5.vrp" si dispo.
- Messages et descriptions détaillés à chaque étape pour bien comprendre les sorties :
  • "Feasible": True/False (toutes les contraintes respectées ?)
  • "Vehicles": nb de véhicules utilisés
  • "Cost": coût total (distance, etc. selon l’instance)
  • "Routes": premières séquences de clients desservies

 Prérequis :
- Lancer Python depuis la RACINE du projet (le dossier qui contient /solver, /data, etc.)
- Assurez-vous que solver/__init__.py existe (fichier vide suffit).
- Dépendances : vrplib, numpy, matplotlib (installées via pip)

Utilisation rapide :
    python .\main.py
"""

from __future__ import annotations
import os
import sys
import time
from statistics import mean, stdev
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))

# When this script is launched from outside the Cluster-First directory (e.g. from
# the repository root or by double-clicking the file on Windows), ensure the
# Cluster-First package is importable.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
    
DATA_DIR = os.path.join(HERE, "data")
IMAGES_DIR = os.path.join(HERE, "images")

# -------------------------------------------------------------------
# Imports du projet (assure-toi d'avoir solver/__init__.py)
# -------------------------------------------------------------------
try:
    from solver.data import Instance, load_vrplib
    from solver.solver import solve_vrp
    from solver.evaluator import eval_route
except Exception as e:
    print("❌ Impossible d'importer le package 'solver'.")
    print("   → LANCE Python depuis la RACINE du projet (le dossier qui contient /solver et /data).")
    print("   → Vérifie que 'solver/__init__.py' existe (même vide).")
    print("Détail de l'erreur :", e)
    sys.exit(1)


try:
    from experiments import interactive_layout
except Exception:
    interactive_layout = None
# -------------------------------------------------------------------
# Utilitaires — recherche d’instances et remap auto vers cvrplib
# -------------------------------------------------------------------
def list_instances_all(data_dir: str = DATA_DIR):
    """
    Liste TOUT ce qui ressemble à une instance (VRP/VRPTW/Solomon).
    """
    exts = {".vrp", ".txt", ".vrptw"}
    out = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    out.sort()
    return out

def list_instances_recommended(data_dir: str = DATA_DIR):
    """
    Liste "recommandée" (fichiers canoniques) :
    - cvrplib\... (CVRP canonique + VRPTW Solomon)
    - évite les copies non standard à la racine
    """
    all_items = list_instances_all(data_dir)
    rec = []
    for p in all_items:
        rp = os.path.relpath(p, data_dir)
        # On privilégie cvrplib/..., Solomon, Vrp-Set-X...
        if rp.lower().startswith("cvrplib") or "vrp-set" in rp.lower():
            rec.append(p)
    rec.sort()
    return rec

def resolve_instance(name_or_path: str, data_dir: str = DATA_DIR) -> str:
    """
    Résout un nom (ex: 'X-n101-k25.vrp') en chemin absolu :
      1) chemin tel quel s'il existe
      2) data_dir/name
      3) recherche récursive sous data/
    """
    # 1) Chemin tel quel
    if os.path.exists(name_or_path):
        return os.path.abspath(name_or_path)
    # 2) data_dir/name
    direct = os.path.join(data_dir, name_or_path)
    if os.path.exists(direct):
        return os.path.abspath(direct)
    # 3) recherche récursive
    fname = os.path.basename(name_or_path).lower()
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower() == fname:
                return os.path.abspath(os.path.join(root, f))
    raise FileNotFoundError(f"Instance introuvable: {name_or_path} (cherché sous {data_dir})")

def find_canonical_sibling(path_in_data: str):
    """
    Si path pointe vers un fichier à la RACINE de data/ (souvent non standard),
    tenter de retrouver la version CANONIQUE (même base name) sous data/cvrplib/**.
    Retourne le chemin canonique si trouvé, sinon None.
    """
    base = os.path.basename(path_in_data)
    cvrp = os.path.join(DATA_DIR, "cvrplib")
    if not os.path.isdir(cvrp):
        return None
    for root, _, files in os.walk(cvrp):
        for f in files:
            if f.lower() == base.lower():
                return os.path.join(root, f)
    return None

def try_load_instance(path: str):
    """
    Charge une instance avec robustesse :
    - Tente load_vrplib(path)
    - Si erreur typique (format non reconnu) ET que le fichier est à la racine,
      remap automatiquement vers la version "cvrplib" si disponible, et recharge.
    Retourne (instance, chemin_effectif).
    """
    try:
        inst = load_vrplib(path)
        return inst, path
    except Exception as e:
        # Si fichier à la racine (ou hors cvrplib), tenter remap
        rel = os.path.relpath(path, DATA_DIR)
        if not rel.lower().startswith("cvrplib"):
            cand = find_canonical_sibling(path)
            if cand and os.path.exists(cand):
                print("ℹ️  Format non standard détecté, remap auto vers la version canonique :")
                print("   ", rel, "→", os.path.relpath(cand, DATA_DIR))
                try:
                    inst = load_vrplib(cand)
                    return inst, cand
                except Exception as e2:
                    # On échoue encore : on remonte l'erreur originale + info remap
                    raise RuntimeError(
                        f"Echec de chargement même après remap vers cvrplib : {e2}"
                    ) from e2
        # Pas de remap possible : on propage l'erreur initiale explicite
        raise

def read_opt_cost_local(path_vrp: str):
    """
    Lit un coût de référence depuis un .sol voisin (si présent).
    Essaie via vrplib.read_solution, sinon extraction naïve du plus petit nombre.
    """
    path_sol = os.path.splitext(path_vrp)[0] + ".sol"
    if not os.path.exists(path_sol):
        return None
    try:
        import vrplib  # type: ignore
        sol = vrplib.read_solution(path_sol)
        if isinstance(sol, dict) and "cost" in sol:
            return float(sol["cost"])
    except Exception:
        pass
    # Fallback naïf
    try:
        with open(path_sol, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        nums = []
        for tok in txt.replace(",", " ").split():
            t = tok.strip()
            if t.replace(".", "", 1).isdigit():
                try:
                    nums.append(float(t))
                except Exception:
                    pass
        return min(nums) if nums else None
    except Exception:
        return None

# -------------------------------------------------------------------
# Paramètres modifiables via le menu [5]
# -------------------------------------------------------------------
DEFAULTS = {
    "seed": 11,
    "tabu_iter": 2000,
    "tabu_stall": 250,
    "show_routes": 0,  # 0 = toutes les tournées
    "runs": 20,
    "bench_instances": [
        # Instances "canonique" recommandées (stables)
        os.path.join("cvrplib", "A-n32-k5.vrp"),
        os.path.join("cvrplib", "Vrp-Set-X", "X", "X-n101-k25.vrp"),
        os.path.join("cvrplib", "Vrp-Set-X", "X", "X-n200-k36.vrp"),
    ],
}

# -------------------------------------------------------------------
# Aides affichage / explications
# -------------------------------------------------------------------
def explain_instance(inst_path: str):
    rp = os.path.relpath(inst_path, DATA_DIR)
    print("\n📦 Instance chargée :", rp)
    print("   • Format : CVRP (capacité) si .vrp ; VRPTW (fenêtres) si Solomon .txt")
    print("   • Données typiques : coordonnées (x,y), demande q_i, capacité Q, temps/Distances, fenêtres [a_i,b_i]…")
    print("   • Objectif : minimiser le coût total (distance/temps) sous contraintes (capacité, fenêtres, etc.)")

def explain_result(inst: Instance, res: dict, showk: int) -> None:
    print("\n🧾 Résultat détaillé :")
    feasible_txt = "Oui" if res.get("feasible") else "Non"
    print(f"   • Solution faisable ? : {feasible_txt} (Oui = toutes les contraintes sont respectées)")
    print(
        "   • Véhicules utilisés : "
        f"{res.get('used_vehicles', 0)} tournée(s) réellement effectuée(s)"
    )
    cost = res.get("cost", 0.0)
    distance = res.get("distance", cost)
    print(
        "   • Distance totale parcourue : "
        f"{distance:.2f} unité(s) de distance (ex. kilomètres dans les instances classiques)"
    )
    print(
        "   • Coût total optimisé     : "
        f"{cost:.2f} (identique à la distance si l'instance n'impose pas d'autres coûts)"
    )
    routes = res.get("routes", [])
    total_routes = len(routes)
    print(f"   • Nombre de tournées générées : {total_routes}")

    if total_routes == 0:
        print("   • Aucune tournée à afficher.")
        return

    if showk <= 0 or showk >= total_routes:
        print("   • Détails des tournées : (affichage complet)")
        routes_to_show = routes
    else:
        print(f"   • Détails des tournées : (premières {showk} sur {total_routes}, mets 0 pour tout afficher)")
        routes_to_show = routes[:showk]

    for idx, route in enumerate(routes_to_show, start=1):
        if not route:
            print(f"     - Tournée #{idx:02d} : aucun client desservi")
            continue
        path_txt = " → ".join(str(c) for c in route)
        print(f"     - Tournée #{idx:02d} ({len(route)} client(s)) : {path_txt}")

    if 0 < showk < total_routes:
        remaining = total_routes - showk
        print(f"     … {remaining} autre(s) tournée(s) masquée(s). Indique 0 pour tout afficher.")


def ask_int(prompt: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    """Demande un entier avec validation simple et rappel du défaut."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("   ↪️ Merci d'entrer un nombre ENTIER (ex : 42). Réessaie.")
                continue

        if min_value is not None and value < min_value:
            print(f"   ↪️ La valeur doit être ≥ {min_value}. Réessaie.")
            continue
        if max_value is not None and value > max_value:
            print(f"   ↪️ La valeur est plafonnée à {max_value}. Réessaie.")
            continue
        return value


def _sanitize_name_for_file(name: str) -> str:
    safe = [c if c.isalnum() or c in {"_", "-"} else "_" for c in name]
    return "".join(safe).strip("_") or "instance"


def show_routes_plot(inst: Instance, routes: List[List[int]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dépendances optionnelles
        print("⚠️ Impossible de créer le graphique (matplotlib indisponible) :", exc)
        return

    if not routes:
        print("⚠️ Graphique non généré : aucune tournée calculée.")
        return

    os.makedirs(IMAGES_DIR, exist_ok=True)
    slug = _sanitize_name_for_file(inst.name if hasattr(inst, "name") else "instance")
    save_path = os.path.join(IMAGES_DIR, f"plan_{slug}.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(f"Plan des tournées — {inst.name}")
    coords = inst.coords
    depot_x, depot_y = coords[0]
    ax.scatter([depot_x], [depot_y], c="black", s=120, marker="s", label="Dépôt")

    cmap = plt.cm.get_cmap("tab20", max(1, len(routes)))
    for idx, route in enumerate(routes, start=1):
        if not route:
            continue
        path = [0] + route + [0]
        xs = [coords[i][0] for i in path]
        ys = [coords[i][1] for i in path]
        color = cmap((idx - 1) % cmap.N)
        ax.plot(xs, ys, "-o", color=color, linewidth=2, label=f"Tournée {idx}")
        for client in route:
            ax.annotate(str(client), (coords[client][0], coords[client][1]),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xlabel("Coordonnée X")
    ax.set_ylabel("Coordonnée Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    fig.savefig(save_path, dpi=150)
    rel_path = os.path.relpath(save_path, HERE)
    print(f"   ↳ Plan sauvegardé : {rel_path}")

    backend = plt.get_backend().lower()
    if "agg" in backend:
        print("   ↳ Ouvre ce fichier PNG pour visualiser le plan (backend sans affichage direct).")
        plt.close(fig)
    else:  # pragma: no cover - dépend de l'environnement d'exécution
        plt.show()


def _animate_console(routes: List[List[int]]) -> None:
    print("🚚 Animation console (déplacements étape par étape) :")
    for idx, route in enumerate(routes, start=1):
        if not route:
            print(f"   - Camion {idx}: aucune visite (tournée vide).")
            continue
        full_path = [0] + route + [0]
        for step in range(1, len(full_path)):
            start = full_path[step - 1]
            end = full_path[step]
            print(f"   - Camion {idx}: {start} → {end}")
            time.sleep(0.3)


def animate_routes(inst: Instance, routes: List[List[int]]) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as exc:  # pragma: no cover - dépendances optionnelles
        print("⚠️ Animation graphique indisponible (matplotlib manquant) :", exc)
        _animate_console(routes)
        return

    if not routes:
        print("⚠️ Animation non générée : aucune tournée.")
        return

    backend = plt.get_backend().lower()
    if "agg" in backend:
        print("⚠️ Backend matplotlib non interactif. Affichage console à la place.")
        _animate_console(routes)
        return

    coords = inst.coords
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(f"Animation des tournées — {inst.name}")
    depot_x, depot_y = coords[0]
    ax.scatter([depot_x], [depot_y], c="black", s=120, marker="s", label="Dépôt")

    cmap = plt.cm.get_cmap("tab20", max(1, len(routes)))
    movers = []
    segments: List[tuple[int, List[float], List[float]]] = []

    for idx, route in enumerate(routes):
        if not route:
            continue
        path = [0] + route + [0]
        xs = [coords[i][0] for i in path]
        ys = [coords[i][1] for i in path]
        color = cmap((idx) % cmap.N)
        ax.plot(xs, ys, "-", color=color, linewidth=1.5)
        mover, = ax.plot([coords[0][0]], [coords[0][1]], marker="o", markersize=10,
                         color=color, alpha=0.9)
        movers.append((idx, mover))
        for start_idx, end_idx in zip(path, path[1:]):
            segments.append((idx, coords[start_idx], coords[end_idx]))

    if not segments:
        print("⚠️ Aucune arête à animer (tournées vides). Animation console utilisée.")
        plt.close(fig)
        _animate_console(routes)
        return

    ax.set_xlabel("Coordonnée X")
    ax.set_ylabel("Coordonnée Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)

    steps_per_segment = 25
    total_frames = len(segments) * steps_per_segment

    def update(frame: int):  # pragma: no cover - animation interactive
        seg_idx = min(frame // steps_per_segment, len(segments) - 1)
        progress = (frame % steps_per_segment) / (steps_per_segment - 1)
        route_idx, start_xy, end_xy = segments[seg_idx]
        x = start_xy[0] + (end_xy[0] - start_xy[0]) * progress
        y = start_xy[1] + (end_xy[1] - start_xy[1]) * progress
        for idx_mover, mover in movers:
            if idx_mover == route_idx:
                mover.set_data([x], [y])
        return [m for _, m in movers]

    FuncAnimation(fig, update, frames=total_frames, interval=80, blit=True, repeat=False)
    plt.show()


def offer_visualizations(inst: Instance, routes: List[List[int]]) -> None:
    if not routes:
        return

    choice_static = input("\n🎨 Voir un plan statique des tournées ? (o/n) [o] > ").strip().lower()
    if choice_static in {"", "o", "oui", "y", "yes"}:
        show_routes_plot(inst, routes)

    choice_live = input("🚛 Lancer une animation (camions en mouvement) ? (o/n) [n] > ").strip().lower()
    if choice_live in {"o", "oui", "y", "yes"}:
        animate_routes(inst, routes)
        
# -------------------------------------------------------------------
# Actions de menu
# -------------------------------------------------------------------
def action_list_all():
    print(f"\n📁 data_dir = {DATA_DIR}")
    items = list_instances_all(DATA_DIR)
    if not items:
        print("Aucune instance trouvée.")
    else:
        print("Liste complète (inclut racine et sous-dossiers) :")
        for p in items:
            rel = os.path.relpath(p, DATA_DIR)
            print(" -", rel)
    input("\n(Entrée) Retour au menu… ")

def action_list_recommended():
    print(f"\n📁 data_dir = {DATA_DIR}")
    items = list_instances_recommended(DATA_DIR)
    if not items:
        print("Aucune instance 'canonique' trouvée (cvrplib/...).")
    else:
        print("Instances recommandées (cvrplib / Solomon / Vrp-Set-X) :")
        for p in items:
            rel = os.path.relpath(p, DATA_DIR)
            print(" -", rel)
    print("\n💡 Conseil : privilégie ces fichiers 'canoniques' pour éviter les erreurs de format.")
    input("\n(Entrée) Retour au menu… ")

def action_demo():
    examples = [
        (
            os.path.join("cvrplib", "A-n32-k5.vrp"),
            "Instance CVRP compacte : 32 clients, flotte maximale de 5 véhicules. Idéale pour une exécution très rapide.",
        ),
        (
            os.path.join("cvrplib", "Vrp-Set-X", "X", "X-n101-k25.vrp"),
            "Instance CVRP moyenne : 101 clients (≈100) et jusqu'à 25 véhicules. Bon compromis entre taille et temps de calcul.",
        ),
        (
            os.path.join("cvrplib", "Vrp-Set-Solomon", "C101.txt"),
            "Instance VRPTW de Solomon : contraintes de capacité + fenêtres de temps serrées pour la série C100.",
        ),
    ]
    print("\n[Démo] Choisis une instance de démonstration parmi ces suggestions :")
    for idx, (path, desc) in enumerate(examples, start=1):
        print(f"  [{idx}] {path} — {desc}")
    print(
        "  Tape le NUMÉRO pour l'utiliser directement ou saisis un chemin personnalisé (relatif/absolu)."
    )

    default_inst = examples[1][0]
    default_inst_display = default_inst.replace(os.sep, "\\")
    inst_prompt = (
        "Nom/chemin instance ou numéro [défaut: 2 → "
        f"{default_inst_display}"
        "] > "
    )
    inst_in = input(inst_prompt).strip()

    if inst_in in {"1", "2", "3"}:
        inst_in = examples[int(inst_in) - 1][0]
    elif not inst_in:
        inst_in = default_inst
        
    # Paramètres
    seed = ask_int(
        (
            "\nGraine aléatoire :"
            "\n   → Saisis un ENTIER (même valeur = mêmes résultats aléatoires)."
            f"\n   → Laisse vide pour utiliser la valeur par défaut ({DEFAULTS['seed']})."
            "\nNombre choisi > "
        ),
        DEFAULTS["seed"],
    )
    iters = ask_int(
        (
            "\nItérations Tabu maximum :"
            "\n   → Nombre total de mouvements testés (1 à 2000)."
            f"\n   → Par défaut : {DEFAULTS['tabu_iter']} (recommandé pour un bon résultat)."
            "\nNombre choisi > "
        ),
        DEFAULTS["tabu_iter"],
        min_value=1,
        max_value=2000,
    )
    stall = ask_int(
        (
            "\nArrêt si pas d'amélioration :"
            "\n   → Combien d'itérations consécutives sans progrès avant de stopper (1 à 2000)."
            f"\n   → Valeur par défaut : {DEFAULTS['tabu_stall']}."
            "\nNombre choisi > "
        ),
        DEFAULTS["tabu_stall"],
        min_value=1,
        max_value=2000,
    )
    showk = ask_int(
        (
            "\nAffichage des tournées détaillées :"
            "\n   → Indique combien de tournées afficher en détail."
            "\n   → Tape 0 pour afficher TOUTES les tournées calculées."
            f"\n   → Valeur par défaut : {DEFAULTS['show_routes']}."
            "\nNombre choisi > "
        ),
        DEFAULTS["show_routes"],
        min_value=0,
    )

    # Chargement robuste + solve
    try:
        req = resolve_instance(inst_in, DATA_DIR)
        inst, eff = try_load_instance(req)
        explain_instance(eff)
        res = solve_vrp(inst, rng_seed=seed, tabu_max_iter=iters, tabu_no_improve=stall)
        explain_result(inst, res, showk)
        offer_visualizations(inst, res["routes"])
    except Exception as e:
        print("\n❌ Erreur de chargement/exécution :", e)
        print("   → Essaie avec une instance sous 'data/cvrplib/...'.")
    input("\n(Entrée) Retour au menu… ")

def action_tests():
    print("\n[Tests rapides] — Vérifie la faisabilité (capacité + fenêtres) sur 1..n instances.")
    print("💡 Laisse vide pour : cvrplib\\A-n32-k5.vrp  cvrplib\\Vrp-Set-X\\X\\X-n101-k25.vrp")
    raw = input("Instances (séparées par des espaces) > ").strip()
    if not raw:
        instances = [
            os.path.join("cvrplib", "A-n32-k5.vrp"),
            os.path.join("cvrplib", "Vrp-Set-X", "X", "X-n101-k25.vrp"),
        ]
    else:
        instances = [x for x in raw.split() if x.strip()]

    try:
        iters = int(input(f"Itérations Tabu max [défaut: 500] > ") or 500)
        stall = int(input(f"Arrêt si pas d'amélioration [défaut: 100] > ") or 100)
    except ValueError:
        print("⚠️ Entrée invalide, utilisation des valeurs par défaut (500/100).")
        iters, stall = 500, 100

    ok_all = True
    for item in instances:
        try:
            req = resolve_instance(item, DATA_DIR)
            inst, eff = try_load_instance(req)
            res = solve_vrp(inst, rng_seed=42, tabu_max_iter=iters, tabu_no_improve=stall)
            feas = res["feasible"]
            if not feas:
                ok_all = False
                print(f"❌ {os.path.basename(eff)} -> solution globale infaisable")
            else:
                # Double check route par route
                bad = []
                for r, k in zip(res["routes"], res["veh_types"]):
                    er = eval_route(inst, r, k)
                    if er.capacity_excess > 0 or er.tw_violation > 0:
                        bad.append((er.capacity_excess, er.tw_violation))
                if bad:
                    ok_all = False
                    print(f"❌ {os.path.basename(eff)} -> violations sur {len(bad)} routes")
                else:
                    print(f"✅ OK: {os.path.basename(eff)} | veh: {res['used_vehicles']} | cost: {res['cost']:.2f}")
        except Exception as e:
            ok_all = False
            print(f"❌ Erreur sur '{item}': {e}")

    if not ok_all:
        print("\nCertaines instances ne passent pas.")
        print("💡 Astuce : privilégie les fichiers sous 'data/cvrplib/...'.")
    input("\n(Entrée) Retour au menu… ")

def action_layout_explorer():
    print("\n[Explorateur visuel] — Crée/édite un layout synthétique en direct.")
    if interactive_layout is None:
        print("❌ Module indisponible : vérifie que 'experiments/interactive_layout.py' est présent et les dépendances installées.")
        input("\n(Entrée) Retour au menu… ")
        return

    print("💡 Ce mode ouvre une fenêtre Matplotlib dédiée (équivalent à 'python experiments/interactive_layout.py').")
    print("   Tu peux modifier le nombre de clients, routes, déplacer/ajouter/supprimer des points et exporter en CSV.")

    seed_raw = input("Graine aléatoire (entier, vide = aléatoire) > ").strip()
    seed = None
    if seed_raw:
        try:
            seed = int(seed_raw)
        except ValueError:
            print("⚠️ Entrée invalide, utilisation d'une graine aléatoire.")
            seed = None

    try:
        interactive_layout.launch(seed=seed)
    except Exception as exc:
        print("❌ Impossible de lancer l'explorateur :", exc)

    input("\n(Entrée) Retour au menu… ")
    
    
def action_bench():
    print("\n[Benchmarks] — N runs/instance, stats, GAP (si .sol), graphiques.")
    print("Instances par défaut :")
    for s in DEFAULTS["bench_instances"]:
        print("  -", s)
    raw = input("Changer la liste ? Chemins séparés par ';' (vide = garder par défaut) > ").strip()
    if raw:
        instances = [x.strip() for x in raw.split(";") if x.strip()]
    else:
        instances = DEFAULTS["bench_instances"]

    try:
        runs  = int(input(f"Nombre de runs [défaut: {DEFAULTS['runs']}] > ") or DEFAULTS["runs"])
        iters = int(input(f"Itérations Tabu max [défaut: {DEFAULTS['tabu_iter']}] > ") or DEFAULTS["tabu_iter"])
        stall = int(input(f"Arrêt si pas d'amélioration [défaut: {DEFAULTS['tabu_stall']}] > ") or DEFAULTS["tabu_stall"])
    except ValueError:
        print("⚠️ Entrée invalide, utilisation des valeurs par défaut.")
        runs, iters, stall = DEFAULTS["runs"], DEFAULTS["tabu_iter"], DEFAULTS["tabu_stall"]

    save = input("Sauver les graphes (PNG) ? (o/n) [n] > ").strip().lower() == "o"
    if save:
        outdir = input("Dossier de sortie [défaut: ./plots] > ").strip() or os.path.join(HERE, "plots")
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = None
    show = input("Afficher les graphes à l'écran ? (o/n) [o] > ").strip().lower()
    show_plots = (show != "n")

    # Config matplotlib
    import matplotlib
    import matplotlib.pyplot as plt
    if outdir and not show_plots:
        matplotlib.use("Agg")

    for item in instances:
        try:
            req = resolve_instance(item, DATA_DIR)
            inst, eff = try_load_instance(req)
            opt = read_opt_cost_local(eff)

            costs, times, feas = [], [], 0
            for r in range(runs):
                t0 = time.perf_counter()
                res = solve_vrp(inst, rng_seed=r, tabu_max_iter=iters, tabu_no_improve=stall)
                times.append(time.perf_counter() - t0)
                costs.append(res["cost"])
                feas += int(res["feasible"])

            avg_c = mean(costs)
            sd_c  = stdev(costs) if len(costs) > 1 else 0.0
            avg_t = mean(times)

            print(f"\n📦 {os.path.basename(eff)}")
            print(f"  ✅ Feasible runs : {feas}/{runs}")
            print(f"  💰 Cost         : mean={avg_c:.2f}  std={sd_c:.2f}")
            print(f"  ⏱️ Time         : mean={avg_t:.2f}s")

            if opt is not None:
                gaps = [100.0 * (c - opt) / opt for c in costs]
                print(f"  📉 GAP mean     : {mean(gaps):.2f}% (vs {os.path.basename(os.path.splitext(eff)[0]+'.sol')})")

            # Boxplot
            plt.figure()
            plt.boxplot(costs)
            plt.title(f"Boxplot des coûts — {os.path.basename(eff)}")
            plt.ylabel("Coût total")
            if outdir:
                fp = os.path.join(outdir, f"box_{os.path.basename(eff)}.png")
                plt.savefig(fp, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close()

            # Histogramme
            plt.figure()
            plt.hist(costs, bins=8)
            plt.title(f"Distribution des coûts — {os.path.basename(eff)}")
            plt.xlabel("Coût")
            plt.ylabel("Fréquence")
            if outdir:
                fp = os.path.join(outdir, f"hist_{os.path.basename(eff)}.png")
                plt.savefig(fp, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close()

        except Exception as e:
            print(f"❌ Erreur sur '{item}': {e}")

    print("\nℹ️ Interprétation rapide :")
    print("   • 'Feasible runs' = nombre de runs sans violation de contraintes.")
    print("   • 'Cost mean/std' = coût moyen et dispersion (stabilité de la métaheuristique).")
    print("   • 'GAP mean'      = écart moyen vs solution de référence (.sol) si disponible.")
    if outdir:
        print(f"   • Graphes PNG     = sauvegardés dans : {outdir}")
    input("\n(Entrée) Retour au menu… ")

def action_change_defaults():
    print("\n[Paramètres par défaut] — Appuie sur Entrée pour conserver la valeur actuelle.")
    try:
        DEFAULTS["seed"]        = int(input(f"seed (graine) [{DEFAULTS['seed']}] > ") or DEFAULTS["seed"])
        DEFAULTS["tabu_iter"]   = int(input(f"tabu_iter (itérations max) [{DEFAULTS['tabu_iter']}] > ") or DEFAULTS["tabu_iter"])
        DEFAULTS["tabu_stall"]  = int(input(f"tabu_stall (arrêt si stagnation) [{DEFAULTS['tabu_stall']}] > ") or DEFAULTS["tabu_stall"])
        DEFAULTS["show_routes"] = int(input(f"show_routes (nb routes affichées) [{DEFAULTS['show_routes']}] > ") or DEFAULTS["show_routes"])
        DEFAULTS["runs"]        = int(input(f"runs (bench par défaut) [{DEFAULTS['runs']}] > ") or DEFAULTS["runs"])
    except ValueError:
        print("⚠️ Entrée invalide, valeurs actuelles conservées.")
    print("\n✅ Paramètres mis à jour :", DEFAULTS)
    input("\n(Entrée) Retour au menu… ")

# -------------------------------------------------------------------
# Boucle du menu
# -------------------------------------------------------------------
def main_menu():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("==========================================================")
        print("  Cluster-First / Route-Second — Menu principal")
        print("  Dossier projet  :", HERE)
        print("  Dossier données :", DATA_DIR)
        print("==========================================================")
        print(" [1] Lister (COMPLET)         — tout ce qui est trouvé dans ./data (y compris racine)")
        print(" [2] Lister (RECOMMANDÉ)      — uniquement cvrplib / Vrp-Set-X / Solomon (formats stables)")
        print(" [3] Démo (1 run)             — exécution sur 1 instance + aperçu des routes")
        print(" [4] Tests rapides            — faisabilité (capacité + fenêtres) sur 1..n instances")
        print(" [5] Benchmarks               — N runs/instance + stats + GAP + graphes")
        print(" [6] Explorateur visuel       — générer/éditer un layout synthétique en direct")
        print(" [7] Changer paramètres       — seed, itérations Tabu, etc.")
        print(" [0] Quitter")
        chx = input("\nVotre choix [0-7] > ").strip()
        if   chx == "1": action_list_all()
        elif chx == "2": action_list_recommended()
        elif chx == "3": action_demo()
        elif chx == "4": action_tests()
        elif chx == "5": action_bench()
        elif chx == "6": action_layout_explorer()
        elif chx == "7": action_change_defaults()
        elif chx == "0":
            print("À bientôt 👋")
            return
        else:
            input("Choix invalide. (Entrée) ")

# -------------------------------------------------------------------
# Entrée
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Garde-fou : data/
    if not os.path.isdir(DATA_DIR):
        print(f"❌ data/ introuvable à {DATA_DIR}")
        sys.exit(1)

    # Garde-fou : solver/__init__.py
    init_path = os.path.join(HERE, "solver", "__init__.py")
    if not os.path.exists(init_path):
        try:
            os.makedirs(os.path.dirname(init_path), exist_ok=True)
            with open(init_path, "w", encoding="utf-8") as f:
                f.write("# package solver\n")
            print("ℹ️  Fichier 'solver/__init__.py' créé automatiquement.")
        except Exception:
            print("⚠️ Impossible de créer solver/__init__.py automatiquement. Vérifie les droits d’accès.")

    main_menu()
