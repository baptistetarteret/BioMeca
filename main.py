import os

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.optimize import differential_evolution
from scipy.sparse import lil_matrix, csr_matrix
from Tissue import Tissue
from meshing_2d import Mesh2D
from efficiency import compute_efficiency_2d
import Physique

# --- Fonction de coût picklable (module-level requis par multiprocessing) ---

class _CostFunction:
    """Evalue le coût d'une combinaison (phases, intensités).

    Doit être au niveau module pour être sérialisable par Pool (workers).
    """

    def __init__(self, amplitudes_norm, prop_phases, target_idx, other_idx,
                 alpha, n_sources):
        self.amplitudes_norm = amplitudes_norm
        self.prop_phases = prop_phases
        self.target_idx = target_idx
        self.other_idx = other_idx
        self.alpha = alpha
        self.n_sources = n_sources

    def __call__(self, x):
        phases = x[:self.n_sources]
        intensities = x[self.n_sources:]
        scales = np.sqrt(intensities)[:, np.newaxis]
        field = np.sum(
            scales * self.amplitudes_norm
            * np.exp(1j * (self.prop_phases + phases[:, np.newaxis])),
            axis=0)
        intensity = np.abs(field) ** 2
        a = self.alpha
        return (- a[0] * np.mean(intensity[self.target_idx])
                + a[1] * np.mean(intensity[self.other_idx])
                - a[2] * 0.01 * np.mean(intensities))


class BioMecaController:

    def __init__(self):
        self.tissues = []
        self.mesh_2d = None

    def add_tissue(self, name, center, radius, absorption_coeff, refractive_index,
                   density=1040.0, specific_heat=3700.0, thermal_conductivity=0.50):
        tissue = Tissue(name, np.array(center, dtype=float), radius,
                        absorption_coeff, refractive_index,
                        density, specific_heat, thermal_conductivity)
        self.tissues.append(tissue)
        return tissue

    def create_mesh_2d(self, N):
        if len(self.tissues) < 2:
            raise ValueError("Au moins 2 tissus nécessaires pour le maillage 2D")
        self.mesh_2d = Mesh2D(N, self.tissues)
        return self.mesh_2d

    def propagate(self, source_pos, frequency, intensity):
        """Propage une onde depuis source_pos sur le maillage 2D."""
        if self.mesh_2d is None:
            raise ValueError("Maillage 2D non créé")
        distances = np.linalg.norm(self.mesh_2d.nodes - source_pos, axis=1)
        source_node = int(np.argmin(distances))
        return self.mesh_2d.compute_field_propagation(intensity, source_node, frequency)

    def propagate_multi_sources(self, n_sources, distance, frequency,
                                phases, intensities):
        """Propage n_sources sources avec phases et intensités individuelles.

        La superposition se fait en champ complexe : E = sqrt(I_k) * exp(i*phi).
        L'intensité résultante est |Σ E_i|² (interférence constructive/destructive).
        """
        if self.mesh_2d is None:
            raise ValueError("Maillage 2D non créé")

        n_nodes = len(self.mesh_2d.nodes)
        field_total = np.zeros(n_nodes, dtype=complex)

        source_positions = []
        for k in range(n_sources):
            angle = np.pi * k / n_sources
            pos = np.array([distance * np.cos(angle), distance * np.sin(angle)])
            source_positions.append(pos)

            I_k, phi_k = self.propagate(pos, frequency, intensities[k])
            phi_k = (phi_k + phases[k]) % (2 * np.pi)
            field_total += np.sqrt(I_k) * np.exp(1j * phi_k)

        intensity = np.abs(field_total) ** 2
        phase = np.angle(field_total) % (2 * np.pi)

        return intensity, phase, source_positions

    def _precompute_source_fields(self, n_sources, distance, frequency):
        """Pré-calcule les champs normalisés (intensité source = 1) pour chaque source.

        Le BFS est coûteux : on le fait une seule fois par source.
        Les champs normalisés sont ensuite mis à l'échelle par sqrt(I_k)
        lors de la superposition, ce qui permet d'optimiser phases ET intensités
        sans relancer la propagation.
        """
        n_nodes = len(self.mesh_2d.nodes)
        amplitudes_norm = np.zeros((n_sources, n_nodes))
        prop_phases = np.zeros((n_sources, n_nodes))
        positions = []

        for k in range(n_sources):
            angle = np.pi * k / n_sources
            pos = np.array([distance * np.cos(angle), distance * np.sin(angle)])
            positions.append(pos)

            I_k, phi_k = self.propagate(pos, frequency, 1.0)
            amplitudes_norm[k] = np.sqrt(I_k)
            prop_phases[k] = phi_k

        return amplitudes_norm, prop_phases, positions

    def _superpose(self, amplitudes_norm, prop_phases, phases, intensities):
        """Superposition vectorisée des champs pré-calculés.

        E_k = sqrt(I_k) * A_norm_k * exp(i * (phi_prop_k + phase_k))
        Opération matricielle : pas de boucle Python.
        """
        scales = np.sqrt(intensities)[:, np.newaxis]          # (n_sources, 1)
        phase_offsets = phases[:, np.newaxis]                  # (n_sources, 1)
        field = np.sum(
            scales * amplitudes_norm * np.exp(1j * (prop_phases + phase_offsets)),
            axis=0)
        return np.abs(field) ** 2, np.angle(field) % (2 * np.pi)

    def optimize(self, target_tissue_name, n_sources, distance,
                 frequency, I_max, alpha = (1.0, 2.0, 0.5)):
        """Optimise phases (et optionnellement intensités) pour focaliser sur la cible.

        Minimise : -mean(I_cible) + alpha * mean(I_hors_cible)

        Args:
            target_tissue_name: nom du tissu à cibler (ex: "tumor")
            I_max: intensité maximale par source
        Returns:
            (phases, intensités, résultat_scipy, positions_sources)
        """
        if self.mesh_2d is None:
            raise ValueError("Maillage 2D non créé")

        target_idx = np.array([i for i, t in enumerate(self.mesh_2d.node_tissue)
                               if t.name == target_tissue_name])
        other_idx = np.array([i for i, t in enumerate(self.mesh_2d.node_tissue)
                              if t.name != target_tissue_name])

        if len(target_idx) == 0:
            raise ValueError(f"Aucun noeud dans le tissu '{target_tissue_name}'")

        # Pré-calcul normalisé (une seule fois)
        print(f"Pre-calcul des {n_sources} champs sources (normalises)...")
        amplitudes_norm, prop_phases, positions = self._precompute_source_fields(
            n_sources, distance, frequency
        )

        cost_fn = _CostFunction(amplitudes_norm, prop_phases,
                                target_idx, other_idx, alpha, n_sources)

        bounds = ([(0, 2 * np.pi)] * n_sources +
                  [(0, I_max)] * n_sources)

        print(f"Optimisation en cours ({cpu_count()} workers)...")
        result = differential_evolution(cost_fn, bounds, seed=42, tol=1e-6,
                                        maxiter=500, polish=True,
                                        workers=cpu_count())

        opt_phases = result.x[:n_sources]
        opt_intensities = result.x[n_sources:]

        print(f"  Evaluations : {result.nfev}")
        print(f"  Cout final  : {result.fun:.4f}")
        print(f"  Convergence : {result.success}")

        return opt_phases, opt_intensities, result, positions

    def simulate_thermal(self, sources_file, t0, t1, T_body=37.0, n_frames=6):
        """Simulation thermique biologique à partir des sources optiques sauvegardées.

        Charge le fichier .npz, recalcule l'intensité optique, dépose
        Q = mu_a * I comme source de chaleur active pendant [0, t0], puis
        intègre l'équation de la chaleur sur [0, t1] (Euler explicite,
        Laplacien graphe creux, BC de Dirichlet à la frontière).

        Args:
            sources_file:         chemin vers le .npz sauvegardé
            t0:                   durée d'irradiation [s]
            t1:                   durée totale de simulation [s]  (t1 >= t0)
            T_body:               température initiale uniforme [°C]
            tissue_thermal_props: {nom: (rho [kg/m³], cp [J/kg/K], k [W/m/K])}
            n_frames:             nombre de snapshots pour l'affichage

        Returns:
            snapshots     : list de tableaux T [n_nodes] aux instants times_out
            times_out     : list des instants des snapshots [s]
            Q_field       : dépôt volumique absorbé [W/m³] par nœud
            src_pos       : positions des sources
            tissue_series : {tissue_name: (times_array, T_mean_array)}
        """
        # --- 1. Chargement des sources ---
        data = np.load(sources_file)
        opt_phases      = data['phases']
        opt_intensities = data['intensities']
        n_sources       = int(data['n_sources'])
        distance        = float(data['distance'])
        frequency       = float(data['frequency'])
        print(f"Sources chargees : {sources_file}")
        print(f"  n={n_sources}, d={distance:.3f} m, f={frequency:.2e} Hz")

        # --- 2. Champ d'intensité optique ---
        print("Propagation optique...")
        I_field, _, src_pos = self.propagate_multi_sources(
            n_sources, distance, frequency, opt_phases, opt_intensities)

        # --- 3. Source thermique Q = mu_a * I  [W/m³] ---
        mu_a    = np.array([t.absorption_coefficients for t in self.mesh_2d.node_tissue])
        Q_field = mu_a * I_field

        # --- 4. Propriétés thermiques lues depuis chaque objet Tissue ---
        # (density kg/m³, specific_heat J/kg/K, thermal_conductivity W/m/K)
        n_nodes = len(self.mesh_2d.nodes)
        rho  = np.array([t.density              for t in self.mesh_2d.node_tissue])
        cp   = np.array([t.specific_heat        for t in self.mesh_2d.node_tissue])
        k_th = np.array([t.thermal_conductivity for t in self.mesh_2d.node_tissue])

        # --- 5. Laplacien thermique creux L ---
        # G_ij = k_harmonique / d²  [W/m³/K] — approximation Laplacien graphe 2D
        print("Construction du Laplacien thermique...")
        L = lil_matrix((n_nodes, n_nodes))
        for link in self.mesh_2d.links:
            i, j = link.node_i, link.node_j
            d = link.length
            if d < 1e-12:
                continue
            k_ij = 2.0 * k_th[i] * k_th[j] / (k_th[i] + k_th[j])   # moy. harmonique
            g = k_ij / (d * d)
            L[i, i] -= g
            L[j, j] -= g
            L[i, j] += g
            L[j, i] += g
        L = csr_matrix(L)
        inv_rho_cp = 1.0 / (rho * cp)

        # --- 6. Pas de temps CFL (stabilité explicite) ---
        diag      = np.abs(L.diagonal())
        safe_diag = np.where(diag > 0, diag, diag[diag > 0].min())
        dt_cfl    = float(np.min(rho * cp / safe_diag))
        dt        = 0.4 * dt_cfl
        n_steps   = max(int(np.ceil(t1 / dt)), 1)
        dt        = t1 / n_steps
        print(f"  dt_CFL={dt_cfl:.4f} s | dt={dt:.4f} s | n_steps={n_steps}")

        # --- 7. Conditions aux limites (Dirichlet : frontière = T_body) ---
        boundary = np.array(self.mesh_2d.get_boundary_nodes())

        # Indices nœuds par tissu (séries temporelles)
        tissue_node_idx = {
            t.name: np.array([i for i, nt in enumerate(self.mesh_2d.node_tissue)
                               if nt.name == t.name])
            for t in self.tissues
        }

        # --- 8. Intégration temporelle (Euler explicite vectorisé) ---
        T            = np.full(n_nodes, float(T_body))
        snap_times   = np.linspace(t1 / n_frames, t1, n_frames)
        snapshots    = []
        times_out    = []
        series_t     = []
        series_T     = {name: [] for name in tissue_node_idx}
        next_snap    = 0

        print(f"\nSimulation thermique : t0={t0} s, t1={t1} s, n_steps={n_steps}")
        for step in range(n_steps):
            t_now = step * dt
            Q_now = Q_field if t_now < t0 else np.zeros(n_nodes)

            dT    = inv_rho_cp * (L.dot(T) + Q_now)
            T     = T + dt * dT
            T[boundary] = T_body       # BC Dirichlet

            t_after = (step + 1) * dt
            series_t.append(t_after)
            for name, idxs in tissue_node_idx.items():
                series_T[name].append(float(np.mean(T[idxs])))

            if next_snap < len(snap_times) and t_after >= snap_times[next_snap] - 1e-9:
                snapshots.append(T.copy())
                times_out.append(t_after)
                irrad = " [irrad.]" if t_after <= t0 + dt else " [refroid.]"
                means = " | ".join(f"{nm}={np.mean(T[ix]):.3f}°C" for nm, ix in tissue_node_idx.items())
                print(f"  t={t_after:7.1f} s{irrad} | {means}")
                next_snap += 1

        tissue_series = {
            name: (np.array(series_t), np.array(vals))
            for name, vals in series_T.items()
        }
        return snapshots, times_out, Q_field, src_pos, tissue_series

    def plot_thermal_results(self, snapshots, times_out, Q_field,
                             src_pos=None, t0=None, T_body=37.0,
                             tissue_series=None, save_path=None):
        """Affiche les résultats de la simulation thermique.

        Layout 2 x (n//2 + 1) :
          Ligne 0 : Q_field | T[0] | T[1] | ... | T[n//2-1]
          Ligne 1 : série temporelle | T[n//2] | ... | T[n-1]
        """
        n     = len(snapshots)
        n_col = n // 2 + 1
        mesh  = self.mesh_2d
        x, y  = mesh.nodes[:, 0], mesh.nodes[:, 1]

        T_min = T_body
        T_max = max(s.max() for s in snapshots)

        fig, axes = plt.subplots(2, n_col, figsize=(5 * n_col, 10))

        def _draw_circles(ax):
            for t in self.tissues:
                ax.add_patch(plt.Circle(t.Center[:2], t.Radius,
                                        fill=False, linestyle='--', color='cyan', lw=1))
            if src_pos is not None:
                for pos in src_pos:
                    ax.plot(pos[0], pos[1], 'g^', ms=6, zorder=10)

        # --- Q_field ---
        ax = axes[0, 0]
        ax.set_title("Source Q = μa·I  [W/m³]")
        sc = ax.scatter(x, y, c=Q_field, cmap='inferno', s=6)
        _draw_circles(ax)
        fig.colorbar(sc, ax=ax)
        ax.set_aspect('equal')

        # --- Snapshots première ligne ---
        for k in range(n // 2):
            ax = axes[0, k + 1]
            t_k = times_out[k]
            tag = " ⚡" if (t0 is not None and t_k <= t0 + 1e-9) else " ❄"
            ax.set_title(f"T  t={t_k:.0f} s{tag}  [°C]")
            sc = ax.scatter(x, y, c=snapshots[k], cmap='hot',
                            vmin=T_min, vmax=T_max, s=6)
            _draw_circles(ax)
            fig.colorbar(sc, ax=ax)
            ax.set_aspect('equal')

        # --- Série temporelle ---
        ax = axes[1, 0]
        ax.set_title("Évolution temporelle  T [°C]")
        if tissue_series is not None:
            colors_ts = plt.cm.tab10(np.linspace(0, 1, len(self.tissues)))
            for (name, (ts, Ts)), color in zip(tissue_series.items(), colors_ts):
                ax.plot(ts, Ts, label=name, color=color, lw=1.5)
        if t0 is not None:
            ax.axvline(t0, color='red', ls='--', lw=1.5, label=f't₀={t0} s')
        ax.axhline(T_body, color='gray', ls=':', lw=1, label=f'T₀={T_body}°C')
        ax.set_xlabel('Temps [s]')
        ax.set_ylabel('T moyenne [°C]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Snapshots deuxième ligne ---
        for k in range(n // 2, n):
            ax = axes[1, k - n // 2 + 1]
            t_k = times_out[k]
            tag = " ⚡" if (t0 is not None and t_k <= t0 + 1e-9) else " ❄"
            ax.set_title(f"T  t={t_k:.0f} s{tag}  [°C]")
            sc = ax.scatter(x, y, c=snapshots[k], cmap='hot',
                            vmin=T_min, vmax=T_max, s=6)
            _draw_circles(ax)
            fig.colorbar(sc, ax=ax)
            ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Figure sauvegardee : {save_path}")
        plt.show()

    def plot_thermal_animation(self, snapshots, times_out, Q_field,
                               src_pos=None, t0=None, T_body=37.0,
                               tissue_series=None, interval=300, save_gif=None):
        """Animation de la simulation thermique : carte T + curseur temporel.

        Layout 1×3 :
          [Q field statique] | [carte T animée] | [séries T(t) + curseur]

        Args:
            interval:  délai entre frames [ms]
            save_gif:  chemin .gif pour sauvegarder (optionnel, nécessite pillow)
        """
        from matplotlib.animation import FuncAnimation

        n   = len(snapshots)
        mesh = self.mesh_2d
        x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
        T_min = T_body
        T_max = max(s.max() for s in snapshots)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.subplots_adjust(wspace=0.35)

        def _circles(ax):
            for t in self.tissues:
                ax.add_patch(plt.Circle(t.Center[:2], t.Radius,
                                        fill=False, ls='--', color='cyan', lw=1))
            if src_pos is not None:
                for pos in src_pos:
                    ax.plot(pos[0], pos[1], 'g^', ms=6, zorder=10)

        # --- Panneau 0 : Q_field (statique) ---
        ax0 = axes[0]
        ax0.set_title("Source Q = μa·I  [W/m³]")
        sc0 = ax0.scatter(x, y, c=Q_field, cmap='inferno', s=6)
        _circles(ax0)
        fig.colorbar(sc0, ax=ax0)
        ax0.set_aspect('equal')

        # --- Panneau 1 : carte T (animée) ---
        ax1 = axes[1]
        sc1 = ax1.scatter(x, y, c=snapshots[0], cmap='hot',
                          vmin=T_min, vmax=T_max, s=6)
        _circles(ax1)
        fig.colorbar(sc1, ax=ax1, label='T [°C]')
        ax1.set_aspect('equal')
        title1 = ax1.set_title("")   # mis à jour dans update()

        # --- Panneau 2 : séries temporelles + curseur ---
        ax2 = axes[2]
        ax2.set_title("Évolution temporelle  T [°C]")
        if tissue_series is not None:
            colors_ts = plt.cm.tab10(np.linspace(0, 1, len(self.tissues)))
            for (name, (ts, Ts)), color in zip(tissue_series.items(), colors_ts):
                ax2.plot(ts, Ts, label=name, color=color, lw=1.5)
        if t0 is not None:
            ax2.axvline(t0, color='red', ls='--', lw=1.2, alpha=0.7,
                        label=f't₀ irrad. = {t0} s')
        ax2.axhline(T_body, color='gray', ls=':', lw=1,
                    label=f'T₀ = {T_body}°C')
        ax2.set_xlabel('Temps [s]')
        ax2.set_ylabel('T moyenne [°C]')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        # Curseur mobile (ligne verticale noire)
        cursor, = ax2.plot([times_out[0], times_out[0]],
                           ax2.get_ylim(), 'k-', lw=2, alpha=0.85)

        def update(frame):
            T_k = snapshots[frame]
            t_k = times_out[frame]
            tag = "⚡ irrad." if (t0 is not None and t_k <= t0) else "❄ refroid."

            # Carte de température
            sc1.set_array(T_k)
            ax1.set_title(f"T  —  t = {t_k:.1f} s   [{tag}]   [°C]")

            # Curseur temporel
            cursor.set_xdata([t_k, t_k])
            cursor.set_ydata(ax2.get_ylim())

            return sc1, cursor

        anim = FuncAnimation(fig, update, frames=n,
                             interval=interval, blit=False, repeat=True,
                             repeat_delay=1500)

        if save_gif:
            anim.save(save_gif, writer='pillow', fps=max(1, 1000 // interval))
            print(f"Animation sauvegardee : {save_gif}")

        plt.show()
        return anim

    def plot_results(self, intensity, phase, source_positions=None):
        """Affiche le maillage, les liens, et les résultats de simulation."""
        mesh = self.mesh_2d
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # --- 1) Maillage + sources ---
        ax = axes[0]
        ax.set_title("Maillage et sources")
        for link in mesh.links:
            p1 = mesh.nodes[link.node_i]
            p2 = mesh.nodes[link.node_j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=0.3, alpha=0.4)
        for tissue in self.tissues:
            ax.add_patch(plt.Circle(tissue.Center[:2], tissue.Radius,
                                    fill=False, linestyle='--', lw=1.5))
        for interface in mesh.interfaces:
            idxs = interface.node_indices
            if idxs:
                ax.scatter(mesh.nodes[idxs, 0], mesh.nodes[idxs, 1],
                           s=10, zorder=5,
                           label=f"{interface.tissue_outer.name}/{interface.tissue_inner.name}")
        ax.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1], s=2, c='blue', zorder=3)
        if source_positions is not None:
            for i, pos in enumerate(source_positions):
                ax.plot(pos[0], pos[1], 'r*', ms=12, zorder=10)
                ax.annotate(f"S{i}", pos, fontsize=7, color='red',
                            ha='center', va='bottom')
        ax.set_aspect('equal')
        ax.legend(fontsize=7)

        # --- 2) Intensité ---
        ax = axes[1]
        ax.set_title("Intensité (W/m²)")
        sc = ax.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1],
                        c=intensity, cmap='hot', s=8)
        for tissue in self.tissues:
            ax.add_patch(plt.Circle(tissue.Center[:2], tissue.Radius,
                                    fill=False, linestyle='--', color='cyan', lw=1))
        if source_positions is not None:
            for pos in source_positions:
                ax.plot(pos[0], pos[1], 'c*', ms=8, zorder=10)
        fig.colorbar(sc, ax=ax)
        ax.set_aspect('equal')

        # --- 3) Tissus ---
        ax = axes[2]
        ax.set_title("Répartition des tissus")

        tissue_colors = plt.cm.tab10(np.linspace(0, 1, len(self.tissues)))

        for i, tissue in enumerate(self.tissues):
            tissue_idx = np.array([j for j, t in enumerate(mesh.node_tissue)
                                if t.name == tissue.name])
            ax.scatter(mesh.nodes[tissue_idx, 0], mesh.nodes[tissue_idx, 1],
                    c=[tissue_colors[i]], s=8, label=tissue.name)

        for tissue in self.tissues:
            ax.add_patch(plt.Circle(tissue.Center[:2], tissue.Radius,
                                    fill=False, linestyle='--', color='black', lw=1))
        if source_positions is not None:
            for pos in source_positions:
                ax.plot(pos[0], pos[1], 'r*', ms=8, zorder=10)

        ax.legend(loc='upper right')
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig("results/simulation_2d.png", dpi=150)
        plt.show()

    def get_status(self):
        print(f"=== BioMeca Simulation Status ===")
        print(f"Tissus: {len(self.tissues)}")
        if self.mesh_2d:
            print(self.mesh_2d.summary())
        else:
            print("Maillage 2D: Non créé")


controller = BioMecaController()

# Tissus concentriques  (optique : mu_a [m⁻¹], n_r ; thermique : rho [kg/m³], cp [J/kg/K], k [W/m/K])
controller.add_tissue("skull", [0, 0, 0],        0.019, 0.0118, 1.4, density=1908.0, specific_heat=1313.0, thermal_conductivity=0.32)
controller.add_tissue("brain", [0, 0, 0],        0.012, 0.0041, 1.4, density=1040.0, specific_heat=3250.0, thermal_conductivity=0.51)
controller.add_tissue("tumor", [0.03, 0.03, 0],  0.002, 0.0200, 1.4, density=1040.0, specific_heat=3250.0, thermal_conductivity=0.51)

# Maillage 2D
mesh_2d = controller.create_mesh_2d(N=6000)
print(mesh_2d.summary())

# Parametres
n_sources = 40
distance = 0.20
frequency = 4e14
I_max = 100.0

# Indices tumeur / hors-tumeur
tumor_idx = np.array([i for i, t in enumerate(mesh_2d.node_tissue)
                    if t.name == "tumor"])
other_idx = np.array([i for i, t in enumerate(mesh_2d.node_tissue)
                    if t.name != "tumor"])

alpha = (0.5, 3.5, 1.0)



def simu_lumineuse():


    # --- 1) Avant optimisation (phases uniformes, intensités égales) ---
    phases_uniform = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    I_uniform = np.full(n_sources, I_max)
    I_before, ph_before, src_pos = controller.propagate_multi_sources(
        n_sources, distance, frequency, phases_uniform, I_uniform
    )
    print(f"\n=== AVANT optimisation ===")
    print(f"  I tumeur  (mean) : {np.mean(I_before[tumor_idx]):.2f} W/m2")
    print(f"  I dehors  (mean) : {np.mean(I_before[other_idx]):.2f} W/m2")
    print(f"  Ratio focus      : {np.mean(I_before[tumor_idx]) / max(np.mean(I_before[other_idx]), 1e-12):.2f}")

    # --- 2) Optimisation phases + intensités ---
    
    opt_phases, opt_intensities, result, src_pos = controller.optimize(
        "tumor", n_sources, distance, frequency,
        I_max=I_max, alpha = alpha
    )

    print(f"\nParametres optimaux :")
    for k in range(n_sources):
        print(f"  S{k}: phase={np.degrees(opt_phases[k]):6.1f} deg, "
              f"I={opt_intensities[k]:6.1f} W/m2")
    print(f"  I totale : {np.sum(opt_intensities):.1f} W/m2 "
          f"(uniforme: {n_sources * I_max:.1f})")

    # Sauvegarde des sources optimales
    save_path = f"results/sources_{n_sources}_{frequency}_{alpha}.npz"
    np.savez(save_path,
             phases=opt_phases,
             intensities=opt_intensities,
             source_positions=np.array(src_pos),
             n_sources=n_sources,
             distance=distance,
             frequency=frequency,
             I_max=I_max,
             alpha=alpha)
    print(f"\nSources sauvegardees : {save_path}")

    # --- 3) Propagation avec parametres optimaux ---
    I_after, ph_after, src_pos = controller.propagate_multi_sources(
        n_sources, distance, frequency, opt_phases, opt_intensities
    )

    boundary = mesh_2d.get_boundary_nodes()
    I_after[boundary] = 0.0

    print(f"\n=== APRES optimisation (phases + intensites) ===")
    print(f"  I tumeur  (mean) : {np.mean(I_after[tumor_idx]):.2f} W/m2")
    print(f"  I dehors  (mean) : {np.mean(I_after[other_idx]):.2f} W/m2")
    print(f"  Ratio focus      : {np.mean(I_after[tumor_idx]) / max(np.mean(I_after[other_idx]), 1e-12):.2f}")

    # --- Visualisation ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Avant
    ax = axes[0, 0]
    ax.set_title("Intensite AVANT")
    sc = ax.scatter(mesh_2d.nodes[:, 0], mesh_2d.nodes[:, 1],
                    c=I_before, cmap='hot_r', s=6)
    for t in controller.tissues:
        ax.add_patch(plt.Circle(t.Center[:2], t.Radius,
                                fill=False, linestyle='--', color='cyan', lw=1))
    fig.colorbar(sc, ax=ax)
    ax.set_aspect('equal')

    # Apres
    ax = axes[0, 1]
    ax.set_title("Intensite APRES")
    sc = ax.scatter(mesh_2d.nodes[:, 0], mesh_2d.nodes[:, 1],
                    c=I_after, cmap='hot_r', s=6)
    for t in controller.tissues:
        ax.add_patch(plt.Circle(t.Center[:2], t.Radius,
                                fill=False, linestyle='--', color='cyan', lw=1))
    fig.colorbar(sc, ax=ax)
    ax.set_aspect('equal')

    # Tissus
    ax = axes[0, 2]
    ax.set_title("Répartition des tissus")

    tissue_colors = plt.cm.tab10(np.linspace(0, 1, len(controller.tissues)))

    for i, tissue in enumerate(controller.tissues):
        tissue_idx = np.array([j for j, t in enumerate(mesh_2d.node_tissue)
                            if t.name == tissue.name])
        ax.scatter(mesh_2d.nodes[tissue_idx, 0], mesh_2d.nodes[tissue_idx, 1],
                c=[tissue_colors[i]], s=6, label=tissue.name)

    for t in controller.tissues:
        ax.add_patch(plt.Circle(t.Center[:2], t.Radius,
                                fill=False, linestyle='--', color='black', lw=1))

    ax.legend()
    ax.set_aspect('equal')

    # Bar chart intensité tumeur vs dehors
    ax = axes[1, 0]
    ax.set_title("Comparaison tumeur vs dehors")
    labels = ["Tumeur\n(avant)", "Dehors\n(avant)", "Tumeur\n(apres)", "Dehors\n(apres)"]
    values = [np.mean(I_before[tumor_idx]), np.mean(I_before[other_idx]),
              np.mean(I_after[tumor_idx]), np.mean(I_after[other_idx])]
    colors = ['#ff6666', '#6666ff', '#ff0000', '#0000ff']
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Intensite moyenne (W/m2)")

    # Intensités par source (polar)
    ax = axes[1, 1]
    ax.set_title("Intensite par source (optimale)")
    angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    ax.bar(range(n_sources), opt_intensities, color='orange')
    ax.axhline(I_max, color='red', ls='--', label=f'I_max={I_max}')
    ax.set_xlabel("Source")
    ax.set_ylabel("Intensite (W/m2)")
    ax.legend()

    # Phases par source
    ax = axes[1, 2]
    ax.set_title("Phase par source (optimale)")
    ax.bar(range(n_sources), np.degrees(opt_phases), color='steelblue')
    ax.set_xlabel("Source")
    ax.set_ylabel("Phase (deg)")
    ax.set_ylim(0, 360)

    plt.tight_layout()
    plt.savefig(f"results/optimization_2d_{n_sources}_{frequency}_{alpha}.png", dpi=150)
    plt.show()



def simu_thermique(data_file):
    t0_irrad = 1.0    # durée d'irradiation [s]  — sources actives sur [0, t0]
    t1_total = 3000.0   # durée totale de simulation [s] (inclut le refroidissement)

    snapshots, times_out, Q_field, src_pos_th, tissue_series = \
        controller.simulate_thermal(
            sources_file=data_file,
            t0=t0_irrad,
            t1=t1_total,
            T_body=37.0,
            n_frames=40,
        )
    
    eff = compute_efficiency_2d(controller.mesh_2d.nodes, controller.mesh_2d.node_tissue, snapshots, T_injury=46.0, T_lethal=60.0, tumor_name="tumor", healthy_name="brain")
    print("Efficiency :", eff)
    
    base_path = f"results/thermal_{n_sources}_{frequency}_{alpha}.gif"
    save_path = base_path

    counter = 1
    while os.path.exists(save_path):
        name, ext = os.path.splitext(base_path)
        save_path = f"{name}_{counter}{ext}"
        counter += 1

    controller.plot_thermal_animation(
        snapshots, times_out, Q_field,
        src_pos=src_pos_th,
        t0=t0_irrad,
        T_body=37.0,
        tissue_series=tissue_series,
        interval=50,
        save_gif=save_path
    )


# --- Menu principal ---
simu = input("Lancer une nouvelle simulation lumineuse ? (o/n) : ").lower() == 'o'
if simu:
    simu_lumineuse()
else:
    dossier = "results"
    fichiers_npz = [f for f in os.listdir(dossier) if f.endswith(".npz")]
    for i, fichier in enumerate(fichiers_npz):
        print(i + 1, " : ", fichier)
    data_file = os.path.join(dossier, fichiers_npz[int(input("Numero du .npz a charger : ")) - 1])
    simu_thermique(data_file)
