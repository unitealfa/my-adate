# ADEME – Optimisation des tournées de livraison (VRP)

## Description

This project addresses the Vehicle Routing Problem (VRP) as part of the ADEME initiative for sustainable mobility. The goal is to determine the optimal delivery routes for a fleet of vehicles, minimizing total distance while respecting logistical constraints such as vehicle capacity.

## Features

- Formal mathematical modeling of the VRP
- Graph-based visualization of delivery networks using NetworkX and Matplotlib
- Interactive Matplotlib dashboard (`python -m gtms_cert.visualize`) to visualise test results and animate live truck routes
- Complexity analysis and references to foundational literature
- Jupyter notebooks for interactive exploration and demonstration

## Project Structure

- `Livrable_Modélisation.ipynb`: Main notebook with problem modeling, visualization, and analysis
- `testttt.ipynb`: Notebook for Python code experimentation
- `.vscode/`: VS Code configuration for C/C++ development (optional)
- `requirements.txt`: Python dependencies

## Installation

1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Create and activate a virtual environment (recommended):
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Open `Livrable_Modélisation.ipynb` in JupyterLab or VS Code and run the cells to visualize and analyze the VRP model.

### Lancement automatique du solveur (Windows)

Un fichier `gtms_cert\run_gtms_cert.bat` est fourni pour lancer automatiquement plusieurs exécutions du solveur GTMS-Cert sur une machine de type Intel i7 11ᵉ génération avec 16 Go de RAM. Il démarre jusqu’à six processus en parallèle avec des graines différentes, surveille les journaux et arrête tout dès qu’un `gap` ≤ 1 % est atteint.

1. Ouvrez l’Explorateur de fichiers et double-cliquez sur `gtms_cert\run_gtms_cert.bat` (ou exécutez-le depuis un terminal `cmd`).
2. Attendez que la fenêtre console indique la fin de la supervision. Les journaux détaillés sont stockés dans le dossier `runs\` à la racine du projet.
3. Relancez le script si vous souhaitez explorer de nouvelles graines.

Le script applique automatiquement les paramètres recommandés (5 camions, 200 clients, graines fixes) pour rechercher un `gap` inférieur à 1 % sans intervention supplémentaire.

## Dependencies

Key packages (see [`requirements.txt`](requirements.txt)):
- `networkx`
- `matplotlib`
- `numpy`
- `pandas`
- `jupyterlab`

## References

- Dantzig, G. B., & Ramser, J. H. (1959). *The Truck Dispatching Problem.*
- Toth, P., & Vigo, D. (2014). *Vehicle Routing: Problems, Methods, and Applications.*
- [VRPLIB Documentation](https://vrplib.readthedocs.io)

## License

Specify your license here (e.g., MIT, GPL).

## Contact

For questions or contributions, open an issue or contact the project maintainer.