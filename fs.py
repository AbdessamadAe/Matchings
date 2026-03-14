import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_matching(G, n, M, filepath):
    pos = compute_positions(n + 1)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, edge_color="lightgray")
    nx.draw_networkx_edges(G, pos, edgelist=list(M), edge_color="blue", width=3)
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_size=200 * 6 / n)
    plt.title(filepath.split("/")[-1])
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(filepath, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def compute_positions(n):
    pos = {}
    for r in range(n):
        for c in range(r + 1):
            pos[("v", r, c)] = (c - r / 2, -r)
    for r in range(n - 1):
        for c in range(r + 1):
            x1, y1 = pos[("v", r, c)]
            x2, y2 = pos[("v", r + 1, c)]
            x3, y3 = pos[("v", r + 1, c + 1)]
            pos[("c", r, c)] = ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)
    return pos


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_G(n):
    height = n + 1
    G = nx.Graph()
    for r in range(height):
        for c in range(r + 1):
            G.add_node(("v", r, c))
    for r in range(height):
        for c in range(r + 1):
            if c < r:
                G.add_edge(("v", r, c), ("v", r, c + 1))
            if r + 1 < height:
                G.add_edge(("v", r, c), ("v", r + 1, c))
                G.add_edge(("v", r, c), ("v", r + 1, c + 1))
    for r in range(height - 1):
        for c in range(r + 1):
            center = ("c", r, c)
            G.add_node(center)
            G.add_edge(center, ("v", r, c))
            G.add_edge(center, ("v", r + 1, c))
            G.add_edge(center, ("v", r + 1, c + 1))
    return G


# ---------------------------------------------------------------------------
# Symmetry maps
#
# Coordinate system:
#   ("v", r, c)  for 0 <= c <= r <= n       (vertex nodes)
#   ("c", r, c)  for 0 <= c <= r <= n-1     (center nodes)
#
# Integer barycentric coords for ("v",r,c), summing to n:
#   alpha = n - r,   beta = r - c,   gamma = c
#
# Integer barycentric coords for ("c",r,c), summing to 3n:
#   alpha = 3(n-r)-2,  beta = 3(r-c)+1,  gamma = 3c+1
#
# Rotation rho: (alpha,beta,gamma) -> (gamma,alpha,beta)
# This gives different (r,c) formulas per node type (derived below).
#
# Reflection sigma1: (alpha,beta,gamma) -> (alpha,gamma,beta)
# => c -> r-c for both node types (same formula).
# ---------------------------------------------------------------------------

def rotate_vertex(node, n):
    """120-degree rotation rho."""
    kind, r, c = node
    if kind == "v":
        # alpha2=gamma=c => r2=n-c;  gamma2=beta=r-c => c2=r-c
        return (kind, n - c, r - c)
    else:
        # alpha2=gamma=3c+1 => 3(n-r2)-2=3c+1 => r2=n-c-1
        # gamma2=beta=3(r-c)+1 => 3c2+1=3(r-c)+1 => c2=r-c
        return (kind, n - c - 1, r - c)


def rotate_inverse_vertex(node, n):
    """240-degree rotation rho^{-1} = rho^2."""
    return rotate_vertex(rotate_vertex(node, n), n)


def reflect_vertex(node):
    """Reflection sigma1 (fixes the axis through corner ("v",0,0))."""
    kind, r, c = node
    return (kind, r, r - c)


def reflect2_vertex(node, n):
    """Reflection sigma2 = rho sigma1 rho^{-1}."""
    return rotate_vertex(reflect_vertex(rotate_inverse_vertex(node, n)), n)


def reflect3_vertex(node, n):
    """Reflection sigma3 = rho^{-1} sigma1 rho."""
    return rotate_inverse_vertex(reflect_vertex(rotate_vertex(node, n)), n)


def apply_sym(M, f):
    return frozenset(tuple(sorted((f(u), f(v)))) for u, v in M)


def is_fully_symmetric(M, n):
    """Check invariance under all 6 elements of D3."""
    return (
        M == apply_sym(M, reflect_vertex)
        and M == apply_sym(M, lambda v: rotate_vertex(v, n))
        and M == apply_sym(M, lambda v: rotate_inverse_vertex(v, n))
        and M == apply_sym(M, lambda v: reflect2_vertex(v, n))
        and M == apply_sym(M, lambda v: reflect3_vertex(v, n))
    )


# ---------------------------------------------------------------------------
# Axis-vertex identification
# ---------------------------------------------------------------------------

def get_axis_nodes(G, n):
    """
    Nodes fixed by each reflection lie on the corresponding axis.
    In a fully D3-symmetric matching, axis nodes must pair among themselves.
    """
    all_nodes = set(G.nodes())

    def fixed_by(sym_func):
        return {v for v in all_nodes if sym_func(v) == v}

    axis1 = fixed_by(reflect_vertex)
    axis2 = fixed_by(lambda v: reflect2_vertex(v, n))
    axis3 = fixed_by(lambda v: reflect3_vertex(v, n))
    return axis1, axis2, axis3


# ---------------------------------------------------------------------------
# D3 orbit decomposition
# ---------------------------------------------------------------------------

def get_orbits(nodes, n):
    """Partition `nodes` into D3-orbits."""
    all_syms = [
        lambda v: v,
        lambda v: reflect_vertex(v),
        lambda v: rotate_vertex(v, n),
        lambda v: rotate_inverse_vertex(v, n),
        lambda v: reflect2_vertex(v, n),
        lambda v: reflect3_vertex(v, n),
    ]
    remaining = set(nodes)
    orbits = []
    while remaining:
        seed = next(iter(remaining))
        orbit = frozenset(s(seed) for s in all_syms) & remaining
        orbits.append(orbit)
        remaining -= orbit
    return orbits


# ---------------------------------------------------------------------------
# Perfect matching enumerator (used for the axis subgraph)
# ---------------------------------------------------------------------------

def generate_perfect_matchings(G):
    if G.number_of_nodes() == 0:
        yield frozenset()
        return
    u = min(G.nodes(), key=lambda x: G.degree(x))
    for v in list(G.neighbors(u)):
        G1 = G.copy()
        G1.remove_nodes_from([u, v])
        for M in generate_perfect_matchings(G1):
            yield frozenset({tuple(sorted((u, v)))}) | M


# ---------------------------------------------------------------------------
# Enumerate axis pairings
# ---------------------------------------------------------------------------

def axis_pairings(axis_nodes, G):
    """Perfect matchings of the subgraph induced by axis_nodes."""
    if not axis_nodes:
        yield frozenset()
        return
    if len(axis_nodes) % 2 != 0:
        return
    H = G.subgraph(sorted(axis_nodes)).copy()
    yield from generate_perfect_matchings(H)


# ---------------------------------------------------------------------------
# Enumerate D3-symmetric off-axis pairings
# ---------------------------------------------------------------------------

def symmetric_matchings_off_axis(off_axis_nodes, G, n):
    """
    Enumerate matchings of off-axis nodes consistent with full D3 symmetry.

    Off-axis nodes come in D3-orbits of size 6.  Fixing the pairing of one
    orbit representative determines all 3 pairs in the orbit via symmetry.
    """
    all_syms = [
        lambda v: v,
        lambda v: reflect_vertex(v),
        lambda v: rotate_vertex(v, n),
        lambda v: rotate_inverse_vertex(v, n),
        lambda v: reflect2_vertex(v, n),
        lambda v: reflect3_vertex(v, n),
    ]

    def expand_pair(u, v):
        """Apply all 6 symmetries to (u,v) to get the full orbit of edges."""
        return frozenset(tuple(sorted((s(u), s(v)))) for s in all_syms)

    orbits = get_orbits(off_axis_nodes, n)

    def recurse(orbit_list, used, acc_edges):
        if not orbit_list:
            yield frozenset(acc_edges)
            return

        orbit = orbit_list[0]
        rest  = orbit_list[1:]

        rep = next((v for v in sorted(orbit) if v not in used), None)
        if rep is None:
            yield from recurse(rest, used, acc_edges)
            return

        for neighbor in G.neighbors(rep):
            if neighbor in used:
                continue
            orbit_edges = expand_pair(rep, neighbor)

            ok = True
            new_used = set()
            for eu, ev in orbit_edges:
                if not G.has_edge(eu, ev):
                    ok = False
                    break
                if eu in used or eu in new_used or ev in used or ev in new_used:
                    ok = False
                    break
                new_used.add(eu)
                new_used.add(ev)

            if ok:
                yield from recurse(rest, used | new_used, acc_edges | orbit_edges)

    yield from recurse(orbits, set(), set())


# ---------------------------------------------------------------------------
# Visualise G_n structure (unchanged from original)
# ---------------------------------------------------------------------------

def visualize_G(n):
    nz = 250 * 2 / n
    G  = build_G(n)
    pos = compute_positions(n + 1)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, edge_color="lightgray")
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_size=nz)

    height = n + 1
    top          = ("v", 0, 0)
    bottom_left  = ("v", height - 1, 0)
    bottom_right = ("v", height - 1, height - 1)

    x_top, y_top = pos[top]
    x_bl,  y_bl  = pos[bottom_left]
    x_br,  y_br  = pos[bottom_right]

    mid_bottom = ((x_bl + x_br) / 2, (y_bl + y_br) / 2)
    mid_right  = ((x_top + x_br) / 2, (y_top + y_br) / 2)
    mid_left   = ((x_top + x_bl) / 2, (y_top + y_bl) / 2)

    plt.plot([x_top, mid_bottom[0]], [y_top, mid_bottom[1]], color="red",    linewidth=2)
    plt.plot([x_bl,  mid_right[0]],  [y_bl,  mid_right[1]],  color="green",  linewidth=2)
    plt.plot([x_br,  mid_left[0]],   [y_br,  mid_left[1]],   color="orange", linewidth=2)

    def on_line(p, a, b, tol=1e-6):
        return abs((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) < tol

    shared = [v for v in G.nodes()
              if on_line(pos[v], (x_top, y_top), mid_bottom)
              and on_line(pos[v], (x_bl, y_bl),   mid_right)
              and on_line(pos[v], (x_br, y_br),   mid_left)]

    print("Centroid is a vertex?", "YES" if shared else "NO")
    if shared:
        nx.draw_networkx_nodes(G, pos, nodelist=shared, node_size=nz, node_color="purple")

    plt.title(f"G{n} Structure with 3 True Reflection Axes")
    plt.axis("equal")
    plt.axis("off")

    base_folder = f"matchings_output/G{n}"
    os.makedirs(base_folder, exist_ok=True)
    plt.savefig(os.path.join(base_folder, f"G{n}.png"), dpi=300)
    plt.close()
    print(f"Saved G{n}.png")


# ---------------------------------------------------------------------------
# Main: enumerate and save fully symmetric matchings
# ---------------------------------------------------------------------------

def count_fully_symmetric(G, n):
    base_folder = f"matchings_output/G{n}"
    full_folder  = os.path.join(base_folder, "full_D3")
    os.makedirs(full_folder, exist_ok=True)

    axis1, axis2, axis3 = get_axis_nodes(G, n)
    all_axis_nodes = axis1 | axis2 | axis3
    off_axis_nodes = set(G.nodes()) - all_axis_nodes

    print(f"Axis nodes: {len(all_axis_nodes)}  "
          f"(axis1={len(axis1)}, axis2={len(axis2)}, axis3={len(axis3)})")
    print(f"Off-axis nodes: {len(off_axis_nodes)}")

    full_index    = 0
    total_checked = 0

    for axis_M in axis_pairings(all_axis_nodes, G):
        for off_M in symmetric_matchings_off_axis(off_axis_nodes, G, n):
            M = axis_M | off_M
            total_checked += 1

            if len(M) * 2 != G.number_of_nodes():
                continue

            # Safety check: should always pass by construction
            if is_fully_symmetric(M, n):
                full_index += 1
                draw_matching(
                    G, n, M,
                    os.path.join(full_folder, f"full_{full_index}.png")
                )

    print(f"Combinations checked: {total_checked}")
    print(f"Fully D3-symmetric matchings: {full_index}")
    print(f"Fully symmetric even? {full_index % 2 == 0}")
    return full_index


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 app_optimized.py <n>")
        sys.exit(1)

    n = int(sys.argv[1])
    visualize_G(n)
    G = build_G(n)
    count_fully_symmetric(G, n)