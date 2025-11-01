"""Interactive client layout explorer.

This module provides a small interactive utility that lets you visualise and
modify synthetic customer layouts that can be used for VRP experiments.  It is
designed as a companion tool for the "Cluster-First / Route-Second" pipeline so
that during manual tests you can quickly inspect client locations, regenerate a
new layout, and tune key parameters in real time.

Running the script opens a Matplotlib window with the following features:

* 📍 Real-time scatter plot of the depot (⚑) and customer positions.
* 🧭 Automatic route sketching (customers sorted by angle and split per route).
* 🎚️ Sliders to change the number of customers and the number of routes.
* 🎲 "Regenerate" button to draw a brand new layout with the current settings.
* 🖱️ Direct manipulation: left-click moves the closest customer, right-click
  adds a new customer, and `Ctrl + right-click` removes the closest one.

The goal is not to deliver an exact VRP solution but to give testers an
immediate, visual playground comparable to an in-game "mod menu" for customer
placements and route sketches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider


@dataclass
class LayoutState:
    """Container for the mutable state of the layout."""

    area_size: float = 100.0
    n_clients: int = 25
    n_routes: int = 4
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.depot = np.array([self.area_size / 2.0, self.area_size / 2.0])
        self.clients = np.empty((0, 2))
        self.routes: List[np.ndarray] = []
        self.regenerate()

    # ------------------------------------------------------------------
    # Core state manipulation helpers
    # ------------------------------------------------------------------
    def regenerate(self) -> None:
        """Draw a brand new set of clients."""

        if self.n_clients <= 0:
            self.clients = np.empty((0, 2))
        else:
            self.clients = self._rng.random((self.n_clients, 2)) * self.area_size
        self._compute_routes()

    def set_client_count(self, value: int) -> None:
        self.n_clients = max(0, int(value))
        self.regenerate()

    def set_route_count(self, value: int) -> None:
        self.n_routes = max(1, int(value))
        self._compute_routes()

    def reset(self) -> None:
        """Reset to factory defaults and reseed the RNG."""

        self._rng = np.random.default_rng(self.seed)
        self.n_clients = 25
        self.n_routes = 4
        self.regenerate()

    def move_client(self, index: int, new_pos: np.ndarray) -> None:
        if 0 <= index < len(self.clients):
            self.clients[index] = new_pos
            self._compute_routes()

    def add_client(self, position: np.ndarray) -> None:
        self.clients = np.vstack([self.clients, position]) if len(self.clients) else np.array([position])
        self.n_clients = len(self.clients)
        self._compute_routes()

    def remove_client(self, index: int) -> None:
        if len(self.clients) == 0 or not (0 <= index < len(self.clients)):
            return
        self.clients = np.delete(self.clients, index, axis=0)
        self.n_clients = len(self.clients)
        self._compute_routes()

    def _compute_routes(self) -> None:
        """Build simple circular routes by angular sweep."""

        if len(self.clients) == 0:
            self.routes = []
            return

        vectors = self.clients - self.depot
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        order = np.argsort(angles)
        segments = np.array_split(order, min(self.n_routes, len(order)))
        routes = []
        for seg in segments:
            if len(seg) == 0:
                continue
            pts = self.clients[seg]
            route = np.vstack([self.depot, pts, self.depot])
            routes.append(route)
        self.routes = routes


class LayoutViewer:
    """Interactive Matplotlib viewer for :class:`LayoutState`."""

    def __init__(self, state: LayoutState) -> None:
        self.state = state
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(left=0.08, right=0.92, bottom=0.20)

        # Widgets area
        self.slider_clients = Slider(
            ax=self.fig.add_axes([0.1, 0.11, 0.8, 0.03]),
            label="Clients",
            valmin=0,
            valmax=200,
            valinit=self.state.n_clients,
            valstep=1,
        )
        self.slider_routes = Slider(
            ax=self.fig.add_axes([0.1, 0.06, 0.8, 0.03]),
            label="Routes",
            valmin=1,
            valmax=20,
            valinit=self.state.n_routes,
            valstep=1,
        )
        button_regen = Button(self.fig.add_axes([0.1, 0.015, 0.25, 0.035]), "Regénérer", color="#3f72af", hovercolor="#6fa8dc")
        button_reset = Button(self.fig.add_axes([0.4, 0.015, 0.25, 0.035]), "Réinitialiser", color="#999999", hovercolor="#bbbbbb")
        button_save = Button(self.fig.add_axes([0.7, 0.015, 0.25, 0.035]), "Exporter", color="#2a9d8f", hovercolor="#52b69a")

        self.slider_clients.on_changed(self._on_clients_changed)
        self.slider_routes.on_changed(self._on_routes_changed)
        button_regen.on_clicked(lambda _: self._on_regenerate())
        button_reset.on_clicked(lambda _: self._on_reset())
        button_save.on_clicked(lambda _: self._on_export())

        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_click)

        self._update_plot()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_clients_changed(self, value: float) -> None:
        self.state.set_client_count(int(value))
        self._update_plot()

    def _on_routes_changed(self, value: float) -> None:
        self.state.set_route_count(int(value))
        self._update_plot()

    def _on_regenerate(self) -> None:
        self.state.regenerate()
        self.slider_clients.set_val(self.state.n_clients)
        self._update_plot()

    def _on_reset(self) -> None:
        self.state.reset()
        self.slider_clients.set_val(self.state.n_clients)
        self.slider_routes.set_val(self.state.n_routes)
        self._update_plot()

    def _on_export(self) -> None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"layout-{timestamp}.csv"
        data = np.vstack([self.state.depot.reshape(1, 2), self.state.clients])
        header = "id,x,y,type\n"
        lines = [header]
        lines.append(f"0,{self.state.depot[0]:.3f},{self.state.depot[1]:.3f},depot\n")
        for idx, (x, y) in enumerate(self.state.clients, start=1):
            lines.append(f"{idx},{x:.3f},{y:.3f},client\n")
        with open(fname, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        print(f"✅ Layout exporté vers {fname} ({len(self.state.clients)} clients).")

    def _on_mouse_click(self, event) -> None:
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        pos = np.array([event.xdata, event.ydata])
        if len(self.state.clients) == 0:
            nearest_index = -1
        else:
            distances = np.linalg.norm(self.state.clients - pos, axis=1)
            nearest_index = int(np.argmin(distances))

        if event.button == 1:  # Left click → move closest client
            if nearest_index >= 0:
                self.state.move_client(nearest_index, pos)
        elif event.button == 3:  # Right click → add/remove client
            if event.key == "control" and nearest_index >= 0:
                self.state.remove_client(nearest_index)
                self.slider_clients.set_val(self.state.n_clients)
            else:
                self.state.add_client(pos)
                self.slider_clients.set_val(self.state.n_clients)
        else:
            return

        self._update_plot()

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _update_plot(self) -> None:
        self.ax.clear()
        self.ax.set_title(
            "Explorateur d'agencement des clients\n"
            "Glissez-déposez: clic gauche = déplacer | clic droit = ajouter | Ctrl+clic droit = supprimer"
        )
        self.ax.set_xlim(0, self.state.area_size)
        self.ax.set_ylim(0, self.state.area_size)
        self.ax.grid(True, color="#dddddd")

        # Depot
        self.ax.scatter(
            [self.state.depot[0]],
            [self.state.depot[1]],
            marker="*",
            c="#ff6f61",
            s=220,
            edgecolors="black",
            linewidths=1.5,
            label="Dépôt",
            zorder=5,
        )

        # Clients
        if len(self.state.clients):
            self.ax.scatter(
                self.state.clients[:, 0],
                self.state.clients[:, 1],
                c="#1f77b4",
                s=60,
                edgecolors="white",
                linewidths=0.8,
                label=f"Clients ({len(self.state.clients)})",
                zorder=4,
            )

        # Routes
        palette = plt.get_cmap("tab20")
        for idx, route in enumerate(self.state.routes):
            color = palette(idx % 20)
            self.ax.plot(route[:, 0], route[:, 1], "-o", color=color, alpha=0.7, linewidth=2.0)

        self.ax.legend(loc="upper right")
        self.ax.figure.canvas.draw_idle()


def launch(seed: Optional[int] = None) -> None:
    """Launch the interactive layout explorer."""

    viewer = LayoutViewer(LayoutState(seed=seed))
    print("ℹ️  Contrôles :")
    print("   • Slider 'Clients' : modifie le nombre de clients (avec régénération automatique)")
    print("   • Slider 'Routes'  : modifie le nombre de routes (segmentation angulaire)")
    print("   • Bouton 'Regénérer' : redessine un layout aléatoire avec les paramètres courants")
    print("   • Bouton 'Réinitialiser' : revient aux paramètres par défaut")
    print("   • Bouton 'Exporter' : enregistre les positions dans un CSV")
    print("   • Souris :")
    print("        - Clic gauche déplace le client le plus proche")
    print("        - Clic droit ajoute un client à l'emplacement visé")
    print("        - Ctrl + clic droit supprime le client le plus proche")
    plt.show()


if __name__ == "__main__":
    launch()