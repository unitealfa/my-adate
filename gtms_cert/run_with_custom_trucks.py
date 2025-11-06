"""Helper to run the GTMS-Cert solver with a configurable vehicle count."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .io import read_input
from .main import solve_gtms_cert
from .visualize import TestResult, launch_visual_app


def _load_template(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Fichier d'entrée introuvable: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Fichier d'entrée invalide ({path}): {exc}") from exc


def _prepare_instance(template: dict, trucks: int) -> Path:
    vehicles = template.setdefault("vehicles", {})
    vehicles["k"] = trucks
    vehicles.setdefault("use_all", True)

    if "customers" in template:
        client_count = len(template["customers"])
    else:
        client_count = len(template.get("node_ids", [])) - 1
    if client_count < trucks:
        raise SystemExit(
            "Le nombre de camions ne peut pas dépasser le nombre de clients disponibles."
        )

    temp_file = tempfile.NamedTemporaryFile("w", suffix="_gtms_cert.json", delete=False)
    json.dump(template, temp_file, indent=2)
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exécuter GTMS-Cert avec un nombre de camions donné")
    default_template = Path(__file__).resolve().parent / "tests" / "sample_instance_200_clients.json"
    parser.add_argument(
        "--template",
        type=Path,
        default=default_template,
        help="Chemin vers le fichier JSON modèle (200 clients par défaut)",
    )
    parser.add_argument(
        "--trucks",
        type=int,
        required=True,
        help="Nombre de camions disponibles",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("solution_gtms_cert.json"),
        help="Chemin du fichier de sortie JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour le solveur",
    )
    parser.add_argument(
        "--cands",
        type=int,
        default=32,
        help="Nombre de candidats pour la recherche locale",
    )
    parser.add_argument(
        "--lb-iters",
        type=int,
        default=0,
        help="Itérations pour la borne inférieure (0 pour des tests rapides)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Afficher immédiatement la visualisation du graphe de tournées",
    )

    args = parser.parse_args(argv)

    if args.trucks <= 0:
        raise SystemExit("Le nombre de camions doit être strictement positif.")

    template = _load_template(args.template)
    temp_input = _prepare_instance(template, args.trucks)

    try:
        result = solve_gtms_cert(
            str(temp_input),
            str(args.output),
            seed=args.seed,
            cands=args.cands,
            lb_iters=args.lb_iters,
        )
        if args.show:
            data = read_input(temp_input, cands=args.cands)
            launch_visual_app(
                data,
                result,
                tests=[
                    TestResult(
                        name="Simulation personnalisée",
                        passed=True,
                        message="Visualisation directe",
                    )
                ],
            )
    finally:
        try:
            temp_input.unlink()
        except FileNotFoundError:
            pass
    print("Résultat du solveur :")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Résultat également enregistré dans {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())