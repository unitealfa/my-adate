# -*- coding: utf-8 -*-
"""
main.py — Lanceur interactif "tout-en-un" pour le projet Cluster-First / Route-Second

 Ce que fait ce script :
- Fournit un MENU clair (1/2/3/4/5/6/0) pour :
  [1] Lister les instances détectées dans ./data
  [2] Lister les instances "recommandées"
  [3] Démo (1 run) — lancer le solveur sur une instance et afficher les 1ères routes
  [4] Tests rapides — vérifier la faisabilité (capacité + fenêtres) sur 1..n instances
  [5] Benchmarks — N runs/instance, statistiques, GAP (si .sol), graphiques
  [6] Changer les paramètres par défaut (seed, itérations Tabu, etc.)
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
from math import isfinite
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))

# When this script is launched from outside the Cluster-First directory (e.g. from
# the repository root or by double-clicking the file on Windows), ensure the
# Cluster-First package is importable.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
    
DATA_DIR = os.path.join(HERE, "data")

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


def _format_number(value: float) -> str:
    """Formate un nombre : entier sans décimales sinon 2 décimales."""
    if not isfinite(value):
        if value > 0:
            return "+∞"
        if value < 0:
            return "-∞"
        return "NaN"
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.2f}"


def _safe_time_window(inst: Instance, client: int) -> Tuple[float, float] | None:
    """Récupère la fenêtre temporelle [a, b] si elle est disponible."""
    window_a = getattr(inst, "window_a", None)
    window_b = getattr(inst, "window_b", None)
    if window_a is None or window_b is None:
        return None

    try:
        open_time = float(window_a[client])
        close_time = float(window_b[client])
    except (IndexError, TypeError):
        return None

    return open_time, close_time


def _compute_waiting_segments(
    inst: Instance, route: List[int], veh_type: int
) -> List[Dict[str, Any]]:
    """Retourne les détails temporels (attente, fenêtres, service) d'une tournée."""
    if not route:
        return []

    seq = [0] + route
    current_time = getattr(inst, "depot_open", 0.0)
    waiting_info: List[Dict[str, Any]] = []

    service_times = getattr(inst, "service", None)

    for prev, client in zip(seq, seq[1:]):
        travel = float(inst.time[prev][client]) if hasattr(inst, "time") else 0.0
        arrival = current_time + travel
        window = _safe_time_window(inst, client)
        wait = 0.0
        if window is not None:
            wait = max(0.0, window[0] - arrival)

        raw_service = 0.0
        if service_times is not None:
            try:
                raw_service = float(service_times[client])
            except (IndexError, TypeError):
                raw_service = 0.0

        start_service = arrival + wait
        current_time = start_service + raw_service

        if client == 0:
            continue

        waiting_info.append(
            {
                "client": client,
                "wait": wait,
                "window": window,
                "service": raw_service,
                "arrival": arrival,
                "start_service": start_service,
            }
        )

    filtered: List[Dict[str, Any]] = []
    for info in waiting_info:
        window = info.get("window")
        has_window = False
        if window is not None:
            start, end = window
            has_window = abs(start) > 1e-9 or abs(end) > 1e-9
        wait = info.get("wait", 0.0)
        service = info.get("service", 0.0)
        if has_window or wait > 1e-9 or service > 1e-9:
            if not has_window:
                info["window"] = None
            filtered.append(info)

    return filtered

def _build_route_timeline(
    inst: Instance, route: List[int], veh_type: int
) -> List[Dict[str, Any]]:
    """Construit une chronologie (trajets + attentes) pour une tournée."""
    if not route:
        return []

    coords = inst.coords
    seq = [0] + route + [0]
    current_time = inst.depot_open
    timeline: List[Dict[str, Any]] = []

    for prev, client in zip(seq, seq[1:]):
        start_xy: Tuple[float, float] = (coords[prev][0], coords[prev][1])
        end_xy: Tuple[float, float] = (coords[client][0], coords[client][1])

        travel = float(inst.time[prev][client])
        timeline.append({
            "start": start_xy,
            "end": end_xy,
            "duration": max(0.0, travel),
        })

        arrival = current_time + travel
        wait = 0.0
        if client != 0:
            wait = max(0.0, inst.window_a[client] - arrival)
        if wait > 1e-9:
            timeline.append({
                "start": end_xy,
                "end": end_xy,
                "duration": wait,
            })

        service = inst.service[client] if client != 0 else 0.0
        current_time = arrival + wait + service

    return timeline



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
        
    veh_types = res.get("veh_types", [0 for _ in routes])

    for idx, route in enumerate(routes_to_show, start=1):
        if not route:
            print(f"     - Tournée #{idx:02d} : aucun client desservi")
            continue
        path_txt = " → ".join(str(c) for c in route)
        header_txt = f"     - Tournée #{idx:02d} ({len(route)} client(s)) : "
        print(f"{header_txt}{path_txt}")

        veh_idx = idx - 1
        veh_type = veh_types[veh_idx] if veh_idx < len(veh_types) else 0
        waiting_segments = _compute_waiting_segments(inst, route, veh_type)
        if waiting_segments:
            spacer = " " * len(header_txt)
            for details in waiting_segments:
                client_id = details["client"]
                parts: List[str] = []

                window = details.get("window")
                if window:
                    start, end = window
                    if abs(start) > 1e-9 or abs(end) > 1e-9:
                        parts.append(
                            f"fenêtre [{_format_number(start)}, {_format_number(end)}]"
                        )

                service = details.get("service", 0.0)
                if service > 1e-9:
                    parts.append(f"service {_format_number(service)}")

                wait = details.get("wait", 0.0)
                if wait > 1e-9:
                    parts.append(f"attente {_format_number(wait)}")

                if not parts:
                    continue

                print(f"{spacer}* client {client_id} — {', '.join(parts)}")

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


def show_routes_plot(inst: Instance, routes: List[List[int]], veh_types: List[int] | None = None) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button
    except Exception as exc:  # pragma: no cover - dépendances optionnelles
        print("⚠️ Impossible d'afficher le plan (matplotlib indisponible) :", exc)
        return

    if not routes:
        print("⚠️ Graphique non généré : aucune tournée calculée.")
        return

    backend = plt.get_backend().lower()
    if "agg" in backend:  # pragma: no cover - dépend du système local
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            try:
                plt.switch_backend("Qt5Agg")
            except Exception:
                print("⚠️ Backend matplotlib non interactif : impossible d'afficher le plan.")
                return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Essaye d'occuper automatiquement l'espace disponible dans la fenêtre GUI
    try:  # pragma: no cover - dépend du backend local
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "window"):
            try:
                manager.window.state("zoomed")  # TkAgg
            except Exception:
                try:
                    manager.window.showMaximized()  # Qt5Agg
                except Exception:
                    pass
        elif hasattr(manager, "frame"):
            try:
                manager.frame.Maximize(True)  # WxAgg
            except Exception:
                pass
    except Exception:
        pass
    
    ax.set_title(f"Plan des tournées — {inst.name}")
    coords = inst.coords
    depot_x, depot_y = coords[0]
    ax.scatter([depot_x], [depot_y], c="black", s=120, marker="s", label="Dépôt")

    try:
        cmap = mpl.colormaps.get_cmap("tab20", max(1, len(routes)))
    except Exception:  # pragma: no cover - compat anciennes versions
        cmap = plt.get_cmap("tab20", max(1, len(routes)))

    mover_artists: List[tuple[int, Any]] = []
    timed_segments_by_route: Dict[int, List[Dict[str, Any]]] = {}

    if veh_types is None:
        veh_types = [0 for _ in routes]

    for idx, route in enumerate(routes, start=1):
        route_key = idx - 1
        if not route:
            timed_segments_by_route[route_key] = []
            continue
        path = [0] + route + [0]
        xs = [coords[i][0] for i in path]
        ys = [coords[i][1] for i in path]
        color = cmap((idx - 1) % cmap.N)
        ax.plot(xs, ys, "-o", color=color, linewidth=2, label=f"Tournée {idx}")
        mover, = ax.plot([depot_x], [depot_y], marker="o", markersize=10,
                         color=color, alpha=0.9, visible=False)
        mover_artists.append((route_key, mover))
        veh_type = veh_types[route_key] if route_key < len(veh_types) else 0
        client_details = {
            info["client"]: info for info in _compute_waiting_segments(inst, route, veh_type)
        }
        for client in route:
            label_lines = [str(client)]
            details = client_details.get(client)
            if details:
                window = details.get("window")
                if window and (abs(window[0]) > 1e-9 or abs(window[1]) > 1e-9):
                    label_lines.append(
                        f"[{_format_number(window[0])}, {_format_number(window[1])}]"
                    )
                service = details.get("service", 0.0)
                if service > 1e-9:
                    label_lines.append(f"serv {_format_number(service)}")
            ax.annotate(
                "\n".join(label_lines),
                (coords[client][0], coords[client][1]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )
        timed_segments_by_route[route_key] = _build_route_timeline(inst, route, veh_type)

    ax.set_xlabel("Coordonnée X")
    ax.set_ylabel("Coordonnée Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.6,
        fontsize=8,
    )
    def _apply_layout(_event=None):  # pragma: no cover - interaction graphique
        try:
            fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
        except Exception:
            pass

    _apply_layout()
    fig.canvas.mpl_connect("resize_event", _apply_layout)

    steps_per_segment = 25
    frames_by_route: Dict[int, List[tuple[Dict[str, Any], int]]] = {}
    totals_by_route: Dict[int, int] = {}
    max_total_frames = 0

    for route_key, timeline in timed_segments_by_route.items():
        if not timeline:
            frames_by_route[route_key] = []
            totals_by_route[route_key] = 0
            continue

        durations = [seg.get("duration", 0.0) for seg in timeline if seg.get("duration", 0.0) > 0]
        avg_duration = sum(durations) / len(durations) if durations else 1.0

        frame_segments: List[tuple[Dict[str, Any], int]] = []
        total_route_frames = 0
        for seg in timeline:
            duration = max(0.0, float(seg.get("duration", 0.0)))
            if duration <= 0:
                frame_count = 1
            else:
                frame_count = max(1, int(round((duration / avg_duration) * steps_per_segment)))
            frame_segments.append((seg, frame_count))
            total_route_frames += frame_count

        frames_by_route[route_key] = frame_segments
        totals_by_route[route_key] = total_route_frames
        if total_route_frames > max_total_frames:
            max_total_frames = total_route_frames

    if max_total_frames == 0:
        print("⚠️ Aucune tournée non vide à animer.")

    total_frames = max(1, max_total_frames)
    anim_holder: dict[str, FuncAnimation | None] = {"anim": None}

    def init_positions():  # pragma: no cover - animation interactive
        for _, mover in mover_artists:
            mover.set_visible(True)
            mover.set_data([depot_x], [depot_y])
        return [m for _, m in mover_artists]

    def update(frame: int):  # pragma: no cover - animation interactive
        if max_total_frames == 0:
            return [m for _, m in mover_artists]

        for idx_mover, mover in mover_artists:
            frame_segments = frames_by_route.get(idx_mover, [])
            total_route_frames = totals_by_route.get(idx_mover, 0)
            if not frame_segments or total_route_frames == 0:
                mover.set_data([depot_x], [depot_y])
                continue

            if frame >= total_route_frames:
                last_seg = frame_segments[-1][0]
                last_xy = last_seg["end"]
                mover.set_data([last_xy[0]], [last_xy[1]])
                continue

            remaining = frame
            for seg, frame_count in frame_segments:
                if remaining < frame_count:
                    start_xy = seg["start"]
                    end_xy = seg["end"]
                    if frame_count == 1 or start_xy == end_xy:
                        mover.set_data([end_xy[0]], [end_xy[1]])
                    else:
                        progress = remaining / max(1, frame_count - 1)
                        x = start_xy[0] + (end_xy[0] - start_xy[0]) * progress
                        y = start_xy[1] + (end_xy[1] - start_xy[1]) * progress
                        mover.set_data([x], [y])
                    break
                remaining -= frame_count
        return [m for _, m in mover_artists]

    button_ax = fig.add_axes([0.72, 0.02, 0.25, 0.07])
    launch_button = Button(button_ax, "Lancer une animation", color="#e0e0e0", hovercolor="#d0d0d0")

    def on_click(_event):  # pragma: no cover - interaction utilisateur
        if max_total_frames == 0:
            return
        anim_holder["anim"] = FuncAnimation(
            fig,
            update,
            init_func=init_positions,
            frames=total_frames,
            interval=80,
            blit=True,
            repeat=False,
        )
        launch_button.label.set_text("Rejouer l'animation")
        fig.canvas.draw_idle()

    launch_button.on_clicked(on_click)

    plt.show()


def offer_visualizations(
    inst: Instance, routes: List[List[int]], veh_types: List[int] | None = None
) -> None:
    if not routes:
        return

    show_routes_plot(inst, routes, veh_types)
        
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
        offer_visualizations(inst, res["routes"], res.get("veh_types"))
    except Exception as e:
        print("\n❌ Erreur de chargement/exécution :", e)
        print("   → Essaie avec une instance sous 'data/cvrplib/...'.")
    input("\n(Entrée) Retour au menu… ")

def action_tests():
    print("\n[Tests rapides] — Vérifie la faisabilité (capacité + fenêtres) sur 1..n instances.")
    default_instances = [
        os.path.join("cvrplib", "A-n32-k5.vrp"),
        os.path.join("cvrplib", "Vrp-Set-X", "X", "X-n101-k25.vrp"),
    ]

    print("Choisis comment sélectionner les instances à tester :")
    print("  [1] Utiliser la sélection par défaut (2 instances recommandées)")
    print("  [2] Choisir dans la liste recommandée détectée dans ./data")
    print("  [3] Saisir manuellement un ou plusieurs chemins/noms d'instances")

    choice = input("Ton choix [1-3] (Entrée = 1) > ").strip() or "1"

    instances: List[str]
    if choice == "1":
        instances = default_instances
        print("\n🟢 Sélection par défaut :")
        for path in instances:
            print("   -", path)
    elif choice == "2":
        recommended = list_instances_recommended(DATA_DIR)
        if not recommended:
            print("⚠️ Aucune instance recommandée détectée sous ./data. Passe en saisie manuelle.")
            instances = default_instances
        else:
            print("\nListe recommandée :")
            for idx, path in enumerate(recommended, start=1):
                print(f"   [{idx:02d}] {os.path.relpath(path, DATA_DIR)}")

            while True:
                print("\nIndique les numéros à tester (ex : 1 4 5).")
                print("Laisse vide pour tout tester.")
                raw_idx = input("Ta sélection > ").strip()
                if not raw_idx:
                    instances = [os.path.relpath(p, DATA_DIR) for p in recommended]
                    break

                try:
                    indexes = [int(tok) for tok in raw_idx.split()]
                except ValueError:
                    print("↪️ Merci d'indiquer uniquement des numéros (1, 2, 3…).")
                    continue

                if any(idx < 1 or idx > len(recommended) for idx in indexes):
                    print(f"↪️ Les numéros doivent être compris entre 1 et {len(recommended)}.")
                    continue

                # Conversion en chemins relatifs pour cohérence avec le reste du programme
                instances = [os.path.relpath(recommended[idx - 1], DATA_DIR) for idx in indexes]
                break
    elif choice == "3":
        print("\nSaisis les chemins ou noms d'instances séparés par des espaces.")
        print("Exemples :")
        print("   - cvrplib/A-n32-k5.vrp")
        print("   - cvrplib/Vrp-Set-X/X/X-n101-k25.vrp")
        print("   - data/mon_instance_personnalisee.vrp")
        raw = input("Instances > ").strip()
        instances = [x for x in raw.split() if x.strip()]
        if not instances:
            print("↪️ Aucune instance saisie, retour à la sélection par défaut.")
            instances = default_instances
    else:
        print("⚠️ Choix inconnu, utilisation de la sélection par défaut.")
        instances = default_instances

    print("\nParamètres de Tabu Search :")
    iters = ask_int(
        "   - Itérations maximum (défaut 500) > ",
        500,
        min_value=1,
    )
    stall = ask_int(
        "   - Arrêt si pas d'amélioration après (défaut 100) > ",
        100,
        min_value=1,
    )
    
    showk = ask_int(
        "   - Détails des tournées (0 = toutes) [défaut 0] > ",
        DEFAULTS["show_routes"],
        min_value=0,
    )
    
    showk = ask_int(
        "   - Détails des tournées (0 = toutes) [défaut 0] > ",
        DEFAULTS["show_routes"],
        min_value=0,
    )

    ok_all = True
    for item in instances:
        try:
            req = resolve_instance(item, DATA_DIR)
            inst, eff = try_load_instance(req)
            explain_instance(eff)
            res = solve_vrp(inst, rng_seed=42, tabu_max_iter=iters, tabu_no_improve=stall)
            explain_result(inst, res, showk)
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
                    offer_visualizations(inst, res["routes"], res.get("veh_types"))
        except Exception as e:
            ok_all = False
            print(f"❌ Erreur sur '{item}': {e}")

    if not ok_all:
        print("\nCertaines instances ne passent pas.")
        print("💡 Astuce : privilégie les fichiers sous 'data/cvrplib/...'.")
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
        print(" [6] Changer paramètres       — seed, itérations Tabu, etc.")
        print(" [0] Quitter")
        chx = input("\nVotre choix [0-6] > ").strip()
        if   chx == "1": action_list_all()
        elif chx == "2": action_list_recommended()
        elif chx == "3": action_demo()
        elif chx == "4": action_tests()
        elif chx == "5": action_bench()
        elif chx == "6": action_change_defaults()
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
