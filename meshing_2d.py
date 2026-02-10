import numpy as np
from typing import List, Tuple, Optional, Set
from Tissue import Tissue


class Link:
    """Lien entre deux noeuds du maillage pour les calculs de simulation."""

    def __init__(self, node_i: int, node_j: int, tissue: Tissue, length: float):
        self.node_i = node_i
        self.node_j = node_j
        self.tissue = tissue
        self.length = length

    def get_absorption(self, intensity: float) -> float:
        return intensity * np.exp(-self.tissue.absorption_coefficients * self.length)

    def get_phase_shift(self, frequency: float) -> float:
        c = 3e8 / self.tissue.refractive_index
        return (2 * np.pi * frequency * self.length / c) % (2 * np.pi)


class Interface:
    """Interface entre deux tissus pour les conditions limites.

    Sépare explicitement les noeuds côté extérieur et côté intérieur.
    """

    def __init__(self, tissue_inner: Tissue, tissue_outer: Tissue,
                 outer_node_indices: List[int], inner_node_indices: List[int],
                 edge_pairs: List[Tuple[int, int]],
                 radius: float, center: np.ndarray):
        self.tissue_inner = tissue_inner
        self.tissue_outer = tissue_outer
        self.outer_node_indices = outer_node_indices
        self.inner_node_indices = inner_node_indices
        self.node_indices = outer_node_indices + inner_node_indices
        self.node_indices_set: Set[int] = set(self.node_indices)
        self.edge_pairs = edge_pairs
        self.radius = radius
        self.center = center

    def get_normals(self, nodes: np.ndarray) -> np.ndarray:
        interface_nodes = nodes[self.node_indices]
        directions = interface_nodes - self.center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return directions / norms

    def apply_dirichlet(self, field: np.ndarray, value: float) -> np.ndarray:
        field[self.node_indices] = value
        return field

    def apply_neumann(self, field: np.ndarray, nodes: np.ndarray,
                      flux: float) -> np.ndarray:
        normals = self.get_normals(nodes)
        for idx, node_idx in enumerate(self.node_indices):
            field[node_idx] += flux * np.linalg.norm(normals[idx])
        return field

    def get_reflection_coefficient(self) -> float:
        n1 = np.sqrt(self.tissue_outer.refractive_index)
        n2 = np.sqrt(self.tissue_inner.refractive_index)
        return ((n1 - n2) / (n1 + n2)) ** 2

    def get_transmission_coefficient(self) -> float:
        return 1.0 - self.get_reflection_coefficient()


class Mesh2D:
    """Maillage 2D en coordonnées polaires avec liens et interfaces."""

    def __init__(self, N: int, List_Tissue: List[Tissue]):
        self._validate_tissues(List_Tissue)

        self.N = N
        self.List_Tissue = List_Tissue

        self.nodes: np.ndarray = np.empty((0, 2))
        self.node_tissue: List[Optional[Tissue]] = []
        self.links: List[Link] = []
        self.interfaces: List[Interface] = []

        self._build_mesh()

    # ---- Validation ----

    def _validate_tissues(self, tissues: List[Tissue]):
        """Vérifie que chaque tissu interne est contenu dans son parent direct."""
        for j in range(1, len(tissues)):
            parent_idx = self._find_parent_tissue_idx(tissues, j)
            if parent_idx is None:
                raise ValueError(
                    f"Tissu '{tissues[j].name}' n'est contenu dans aucun autre tissu."
                )

    @staticmethod
    def _find_parent_tissue_idx(tissues: List[Tissue], j: int) -> Optional[int]:
        """Trouve le plus petit tissu contenant tissues[j]."""
        inner = tissues[j]
        best_idx = None
        best_radius = np.inf
        for i in range(len(tissues)):
            if i == j:
                continue
            outer = tissues[i]
            dist = np.linalg.norm(inner.Center[:2] - outer.Center[:2])
            if dist + inner.Radius <= outer.Radius and outer.Radius < best_radius:
                best_idx = i
                best_radius = outer.Radius
        return best_idx

    # ---- Adjacence ----

    def _get_adjacent_pairs(self) -> List[Tuple[int, int]]:
        """Retourne les paires (idx_outer, idx_inner) de tissus directement adjacents."""
        pairs = []
        for j in range(1, len(self.List_Tissue)):
            parent_idx = self._find_parent_tissue_idx(self.List_Tissue, j)
            if parent_idx is not None:
                pairs.append((parent_idx, j))
        return pairs

    # ---- Tolérance et génération ----

    def _get_tolerance(self, radius: float) -> float:
        n_r = max(int(np.sqrt(self.N / len(self.List_Tissue))), 2)
        dr = radius / n_r
        return dr / 2

    def _cercle_2d(self, N: int, center: np.ndarray, R_min: float,
                   R_max: float) -> Tuple[np.ndarray, int, int]:
        n_radial = max(int(N ** 0.35), 2)
        n_angular = max(int(N ** 0.65), 4)

        points = []
        for i in range(n_radial):
            r = R_min + (i / max(n_radial - 1, 1)) * (R_max - R_min)
            for j in range(n_angular):
                theta = 2 * np.pi * j / n_angular
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                points.append([x, y])

        return np.array(points), n_radial, n_angular

    def _is_inside_tissue(self, point: np.ndarray, tissue: Tissue) -> bool:
        return np.linalg.norm(point - tissue.Center[:2]) <= tissue.Radius

    def _find_tissue(self, point: np.ndarray) -> Optional[Tissue]:
        """Trouve le tissu le plus interne contenant le point."""
        best = None
        best_radius = np.inf
        for tissue in self.List_Tissue:
            if self._is_inside_tissue(point, tissue) and tissue.Radius < best_radius:
                best = tissue
                best_radius = tissue.Radius
        return best

    def _compute_tissue_areas(self) -> List[float]:
        areas = []
        for i, tissue in enumerate(self.List_Tissue):
            area = np.pi * tissue.Radius ** 2
            for j in range(len(self.List_Tissue)):
                if j == i:
                    continue
                inner = self.List_Tissue[j]
                dist = np.linalg.norm(inner.Center[:2] - tissue.Center[:2])
                if dist + inner.Radius <= tissue.Radius and inner.Radius < tissue.Radius:
                    area -= np.pi * inner.Radius ** 2
            areas.append(max(area, 0.0))
        return areas

    # ---- Construction du maillage ----

    def _build_mesh(self):
        all_nodes = []
        all_tissues = []
        grid_info = {}
        node_offset = 0

        areas = self._compute_tissue_areas()
        total_area = sum(areas)
        adjacent_pairs = self._get_adjacent_pairs()

        for t_idx, tissue in enumerate(self.List_Tissue):
            center = tissue.Center[:2]
            R_max = tissue.Radius

            # R_min = rayon du plus grand tissu enfant direct (concentrique ou non)
            R_min = 0.0
            for idx_outer, idx_inner in adjacent_pairs:
                if idx_outer == t_idx:
                    child = self.List_Tissue[idx_inner]
                    child_dist = np.linalg.norm(child.Center[:2] - center)
                    # Seulement pour les enfants proches du centre (quasi-concentriques)
                    if child_dist < child.Radius * 0.5:
                        R_min = max(R_min, child.Radius)

            n_points = max(int(self.N * areas[t_idx] / total_area), 4)
            points, n_r, n_a = self._cercle_2d(n_points, center, R_min, R_max)

            kept_points = []
            kept_grid = []
            for p_idx, point in enumerate(points):
                actual_tissue = self._find_tissue(point)
                if actual_tissue == tissue:
                    i_r = p_idx // n_a
                    i_a = p_idx % n_a
                    kept_points.append(point)
                    kept_grid.append((i_r, i_a))
                    all_tissues.append(tissue)

            if kept_points:
                kept_arr = np.array(kept_points)
                grid_info[t_idx] = {
                    'offset': node_offset,
                    'points': kept_arr,
                    'grid': kept_grid,
                    'n_radial': n_r,
                    'n_angular': n_a,
                    'tissue': tissue,
                }
                node_offset += len(kept_points)
                all_nodes.append(kept_arr)

        if all_nodes:
            self.nodes = np.vstack(all_nodes)
        self.node_tissue = all_tissues

        self._build_intra_tissue_links(grid_info)
        self._build_interfaces_and_links()

    # ---- Liens intra-tissu ----

    def _build_intra_tissue_links(self, grid_info: dict):
        """Liens radiaux et angulaires à l'intérieur de chaque tissu."""
        for t_idx, info in grid_info.items():
            offset = info['offset']
            grid = info['grid']
            n_a = info['n_angular']
            tissue = info['tissue']
            points = info['points']

            grid_to_local = {}
            for local_idx, (i_r, i_a) in enumerate(grid):
                grid_to_local[(i_r, i_a)] = local_idx

            for local_idx, (i_r, i_a) in enumerate(grid):
                global_i = offset + local_idx
                pi = points[local_idx]

                # Lien radial
                neighbor_key = (i_r + 1, i_a)
                if neighbor_key in grid_to_local:
                    local_j = grid_to_local[neighbor_key]
                    global_j = offset + local_j
                    length = np.linalg.norm(pi - points[local_j])
                    self.links.append(Link(global_i, global_j, tissue, length))

                # Lien angulaire
                neighbor_key = (i_r, (i_a + 1) % n_a)
                if neighbor_key in grid_to_local:
                    local_j = grid_to_local[neighbor_key]
                    global_j = offset + local_j
                    if global_j != global_i:
                        length = np.linalg.norm(pi - points[local_j])
                        self.links.append(Link(global_i, global_j, tissue, length))

    # ---- Interfaces et liens inter-tissus (méthode unifiée) ----

    def _build_interfaces_and_links(self):
        """Construit les interfaces ET les liens inter-tissus en une passe.

        Pour chaque paire adjacente :
        1. Trouve les noeuds côté extérieur (appartenant au tissu parent)
        2. Trouve les noeuds côté intérieur (appartenant au tissu enfant)
        3. Crée les liens inter-tissus entre plus proches voisins
        4. Crée l'objet Interface avec les deux côtés séparés
        """
        for idx_outer, idx_inner in self._get_adjacent_pairs():
            tissue_outer = self.List_Tissue[idx_outer]
            tissue_inner = self.List_Tissue[idx_inner]
            boundary_radius = tissue_inner.Radius
            boundary_center = tissue_inner.Center[:2]
            tolerance = self._get_tolerance(boundary_radius)

            outer_nodes = []
            inner_nodes = []
            for idx, point in enumerate(self.nodes):
                dist = np.linalg.norm(point - boundary_center)
                if abs(dist - boundary_radius) < tolerance:
                    if self.node_tissue[idx] is tissue_outer:
                        outer_nodes.append(idx)
                    elif self.node_tissue[idx] is tissue_inner:
                        inner_nodes.append(idx)

            # Liens inter-tissus : chaque noeud extérieur -> noeud intérieur le plus proche
            for idx_o in outer_nodes:
                p_o = self.nodes[idx_o]
                min_dist = np.inf
                best_idx = -1
                for idx_i in inner_nodes:
                    d = np.linalg.norm(p_o - self.nodes[idx_i])
                    if d < min_dist:
                        min_dist = d
                        best_idx = idx_i
                if best_idx >= 0:
                    self.links.append(Link(idx_o, best_idx, tissue_outer, min_dist))

            # Arêtes de l'interface triées par angle
            all_interface = outer_nodes + inner_nodes
            edge_pairs = []
            if len(all_interface) > 1:
                angles = [np.arctan2(self.nodes[idx, 1] - boundary_center[1],
                                     self.nodes[idx, 0] - boundary_center[0])
                          for idx in all_interface]
                sorted_indices = [all_interface[k] for k in np.argsort(angles)]
                for k in range(len(sorted_indices)):
                    n1 = sorted_indices[k]
                    n2 = sorted_indices[(k + 1) % len(sorted_indices)]
                    edge_pairs.append((n1, n2))

            self.interfaces.append(Interface(
                tissue_inner=tissue_inner,
                tissue_outer=tissue_outer,
                outer_node_indices=outer_nodes,
                inner_node_indices=inner_nodes,
                edge_pairs=edge_pairs,
                radius=boundary_radius,
                center=boundary_center,
            ))

    # ---- Méthodes utilitaires ----

    def get_links_for_tissue(self, tissue: Tissue) -> List[Link]:
        return [link for link in self.links if link.tissue is tissue]

    def get_neighbors(self, node_idx: int) -> List[int]:
        neighbors = []
        for link in self.links:
            if link.node_i == node_idx:
                neighbors.append(link.node_j)
            elif link.node_j == node_idx:
                neighbors.append(link.node_i)
        return neighbors

    def get_link_between(self, node_i: int, node_j: int) -> Optional[Link]:
        for link in self.links:
            if ((link.node_i == node_i and link.node_j == node_j) or
                    (link.node_i == node_j and link.node_j == node_i)):
                return link
        return None

    def get_interface_between(self, tissue_a: Tissue,
                              tissue_b: Tissue) -> Optional[Interface]:
        for interface in self.interfaces:
            if ((interface.tissue_inner is tissue_a and
                 interface.tissue_outer is tissue_b) or
                    (interface.tissue_inner is tissue_b and
                     interface.tissue_outer is tissue_a)):
                return interface
        return None

    def get_boundary_nodes(self) -> List[int]:
        outer_tissue = self.List_Tissue[0]
        boundary_center = outer_tissue.Center[:2]
        boundary_radius = outer_tissue.Radius
        tolerance = self._get_tolerance(boundary_radius)

        boundary = []
        for idx, point in enumerate(self.nodes):
            dist = np.linalg.norm(point - boundary_center)
            if abs(dist - boundary_radius) < tolerance:
                boundary.append(idx)
        return boundary

    # ---- Simulation ----

    def compute_field_propagation(self, source_intensity: float,
                                  source_node: int,
                                  frequency: float) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.nodes)
        intensity = np.zeros(n)
        phase = np.zeros(n)
        visited = np.full(n, False)

        # Pré-calculer les sets pour O(1) lookup
        interface_sets = [(iface, iface.node_indices_set) for iface in self.interfaces]

        intensity[source_node] = source_intensity
        visited[source_node] = True
        queue = [source_node]

        while queue:
            current = queue.pop(0)
            for neighbor in self.get_neighbors(current):
                if visited[neighbor]:
                    continue
                link = self.get_link_between(current, neighbor)
                if link is None:
                    continue

                intensity[neighbor] = link.get_absorption(intensity[current])
                phase[neighbor] = (phase[current] +
                                   link.get_phase_shift(frequency)) % (2 * np.pi)

                # Transmission à l'interface : on traverse si le voisin est
                # sur l'interface mais pas le noeud courant
                for iface, iface_set in interface_sets:
                    if neighbor in iface_set and current not in iface_set:
                        intensity[neighbor] *= iface.get_transmission_coefficient()

                visited[neighbor] = True
                queue.append(neighbor)

        return intensity, phase

    def summary(self) -> str:
        lines = [
            "=== Maillage 2D ===",
            f"Noeuds    : {len(self.nodes)}",
            f"Liens     : {len(self.links)}",
            f"Interfaces: {len(self.interfaces)}",
            f"Tissus    : {len(self.List_Tissue)}",
        ]
        for tissue in self.List_Tissue:
            count = sum(1 for t in self.node_tissue if t is tissue)
            lines.append(f"  - {tissue.name}: {count} noeuds")
        for i, interface in enumerate(self.interfaces):
            lines.append(
                f"  Interface {i}: {interface.tissue_outer.name} <-> "
                f"{interface.tissue_inner.name} "
                f"(outer={len(interface.outer_node_indices)}, "
                f"inner={len(interface.inner_node_indices)}, "
                f"R={interface.get_reflection_coefficient():.4f}, "
                f"T={interface.get_transmission_coefficient():.4f})"
            )
        return "\n".join(lines)
