import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from Tissue import Tissue
from meshing_2d import Mesh2D
import Physique


class BioMecaController:

    def __init__(self):
        self.tissues = []
        self.mesh_2d = None

    def add_tissue(self, name, center, radius, absorption_coeff, refractive_index):
        tissue = Tissue(name, np.array(center, dtype=float), radius,
                        absorption_coeff, refractive_index)
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
        # Champs normalisés : propagation avec I_source = 1
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
        """Superposition rapide des champs pré-calculés.

        E_k = sqrt(I_k) * A_norm_k * exp(i * (phi_prop_k + phase_k))
        """
        field = np.zeros(amplitudes_norm.shape[1], dtype=complex)
        for k in range(amplitudes_norm.shape[0]):
            scale = np.sqrt(intensities[k])
            field += scale * amplitudes_norm[k] * np.exp(
                1j * (prop_phases[k] + phases[k]))
        intensity = np.abs(field) ** 2
        phase = np.angle(field) % (2 * np.pi)
        return intensity, phase

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

        n_eval = [0]



        def cost(x):
            phases = x[:n_sources]
            intensities = x[n_sources:]
            intensity, _ = self._superpose(
                amplitudes_norm, prop_phases, phases, intensities)
            n_eval[0] += 1

            return - alpha[0] * np.mean(intensity[target_idx]) + alpha[1] * np.mean(intensity[other_idx]) - alpha[2] * 0.01 * np.mean(intensities)



        bounds = ([(0, 2 * np.pi)] * n_sources +
                  [(0, I_max)] * n_sources)
        
        print("Optimisation en cours...")
        result = differential_evolution(cost, bounds, seed=42, tol=1e-6,
                                        maxiter=500, polish=True)

        opt_phases = result.x[:n_sources]
        opt_intensities = result.x[n_sources:]

        print(f"  Evaluations : {n_eval[0]}")
        print(f"  Cout final  : {result.fun:.4f}")
        print(f"  Convergence : {result.success}")

        return opt_phases, opt_intensities, result, positions

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


if __name__ == "__main__":
    controller = BioMecaController()

    # Tissus concentriques
    controller.add_tissue("skull", [0, 0, 0], 0.15, 0.3, 1.8)
    controller.add_tissue("brain", [0, 0, 0], 0.12, 0.1, 1.5)
    controller.add_tissue("tumor", [0.03, 0.03, 0], 0.02, 0.8, 1.6)

    # Maillage 2D
    mesh_2d = controller.create_mesh_2d(N=5000)
    print(mesh_2d.summary())

    # Parametres
    n_sources = 30
    distance = 0.20
    frequency = 4e14
    I_max = 100.0

    # Indices tumeur / hors-tumeur
    tumor_idx = np.array([i for i, t in enumerate(mesh_2d.node_tissue)
                          if t.name == "tumor"])
    other_idx = np.array([i for i, t in enumerate(mesh_2d.node_tissue)
                          if t.name != "tumor"])

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
    alpha = (0.5, 3.0, 1.0)
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