from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import json
import random
from typing import Dict

from .data import Depot, Client, save_dataset
from .solver import solve
from .viz import plot_routes

DEFAULT_GENERATED_DIR = Path("data/generated")
INDEX_FILE = DEFAULT_GENERATED_DIR / "index.json"


def parse_args():
    p = argparse.ArgumentParser(description="VRPTW (k camions, TW dures, durée max)")
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--data", type=str, help="JSON existant (clients + TW)")
    mode.add_argument("--random", action="store_true", help="générer aléatoirement")

    p.add_argument("--interactive", action="store_true", help="Lancer l'assistant interactif")
    p.add_argument("--n-clients", type=int, default=50)
    p.add_argument("--k", type=int)
    p.add_argument("--shift-duration", type=float, default=None)
    p.add_argument("--time-limit", type=int, default=60)
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.interactive and args.k is None:
        p.error("--k est requis en mode non interactif")
    if not args.interactive and not (args.random or args.data):
        p.error("Merci de préciser --data ou --random en mode non interactif")
    if args.interactive and (args.random or args.data):
        p.error("--interactive ne peut pas être combiné avec --random ou --data")
    return args

def generate_random(n_clients: int, seed: int, out_dir: Path) -> Path:
    random.seed(seed)
    depot = Depot(0.0, 0.0, 0, 10**9, 0)
    clients = []
    for i in range(1, n_clients+1):
        x, y = random.uniform(-10,10), random.uniform(-10,10)
        width = random.randint(60, 180)
        e = random.randint(0, 600-width)
        l = e + width
        clients.append(Client(i, x, y, e, l, 5))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"vrptw_{ts}_{n_clients}c.json"
    save_dataset(depot, clients, str(path))
    return path


def load_saved_datasets() -> Dict[str, str]:
    if not INDEX_FILE.exists():
        return {}
    with INDEX_FILE.open("r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return {}


def register_dataset(name: str, dataset_path: Path) -> None:
    DEFAULT_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    index = load_saved_datasets()
    index[name] = str(dataset_path)
    with INDEX_FILE.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)


def prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = "[O/n]" if default else "[o/N]"
    while True:
        answer = input(f"{question} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in {"o", "oui", "y", "yes"}:
            return True
        if answer in {"n", "non", "no"}:
            return False
        print("Réponse invalide, merci de répondre par o/n.")


def prompt_int(question: str, min_value: int = 1) -> int:
    while True:
        try:
            value = int(input(question).strip())
        except ValueError:
            print("Veuillez entrer un nombre valide.")
            continue
        if value < min_value:
            print(f"Merci de saisir un entier supérieur ou égal à {min_value}.")
            continue
        return value


def choose_saved_dataset(saved: Dict[str, str]) -> Path | None:
    if not saved:
        print("Aucun scénario enregistré n'est disponible pour le moment.")
        return None
    names = list(saved.keys())
    print("Scénarios disponibles :")
    for idx, name in enumerate(names, start=1):
        print(f"  {idx}. {name}")
    while True:
        selection = input("Sélectionnez un numéro ou appuyez sur Entrée pour annuler : ").strip()
        if not selection:
            return None
        try:
            pos = int(selection)
        except ValueError:
            print("Merci d'entrer un numéro valide.")
            continue
        if 1 <= pos <= len(names):
            return Path(saved[names[pos - 1]])
        print("Numéro hors plage.")


def interactive_setup() -> tuple[int, Path]:
    print("=== Assistant VRPTW ===")
    k = prompt_int("Nombre de camions disponibles : ")
    saved = load_saved_datasets()
    while True:
        print("\nQue souhaitez-vous faire ?")
        print("  1. Générer une carte de clients aléatoire")
        print("  2. Charger un fichier de données existant")
        print("  3. Réutiliser un scénario enregistré")
        choice = input("Votre choix (1/2/3) : ").strip()

        if choice == "1":
            n_clients = prompt_int("Nombre de clients à générer : ")
            seed_input = input("Graine aléatoire (laisser vide pour défaut 42) : ").strip()
            seed = int(seed_input) if seed_input else 42
            dataset_path = generate_random(n_clients, seed, DEFAULT_GENERATED_DIR)
            print(f"Jeu de données généré : {dataset_path}")
            if prompt_yes_no("Souhaitez-vous enregistrer ce scénario pour le réutiliser plus tard ?"):
                while True:
                    name = input("Nom du scénario (laisser vide pour utiliser le nom de fichier) : ").strip()
                    if not name:
                        name = dataset_path.stem
                    if name in saved and not prompt_yes_no("Un scénario portant ce nom existe déjà. Écraser ?"):
                        continue
                    register_dataset(name, dataset_path)
                    saved[name] = str(dataset_path)
                    print(f"Scénario enregistré sous le nom '{name}'.")
                    break
            return k, dataset_path
        if choice == "2":
            path = Path(input("Chemin du fichier JSON : ").strip())
            if path.exists():
                return k, path
            print("Fichier introuvable, merci de réessayer.")
            continue
        if choice == "3":
            dataset_path = choose_saved_dataset(saved)
            if dataset_path is not None:
                return k, dataset_path
            continue
        print("Choix invalide, merci de sélectionner 1, 2 ou 3.")

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        k, dataset_path = interactive_setup()
    else:
        k = args.k
        if args.data:
            dataset_path = Path(args.data)
        else:
            dataset_path = generate_random(args.n_clients, args.seed, DEFAULT_GENERATED_DIR)

    data, best = solve(str(dataset_path), k=k, shift_duration=args.shift_duration, time_limit_s=args.time_limit)

    # sorties
    last_return = best.last_return
    gap = None
    if args.shift_duration is not None:
        gap = args.shift_duration - last_return

    msg = (
        f"Objective: {best.cost:.3f}"
        f" | Dist: {best.dist:.3f}"
        f" | TW_violation: {best.time_warp:.3f}"
        f" | Last truck duration: {last_return:.3f}"
    )
    if gap is not None:
        msg += f" | Gap to shift: {gap:.3f}"
    print(msg)
    png = out_dir / "routes.png"
    plot_routes(data, best, str(png))
    print(f"Plot écrit: {png}")

if __name__ == "__main__":
    main()
