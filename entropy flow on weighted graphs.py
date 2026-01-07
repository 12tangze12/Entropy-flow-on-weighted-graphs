import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics



class Entropyflow:
    def __init__(self, G, alpha, weight='weight', epsilon=1e-6):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.EPSILON = epsilon
        self._cache = {}

    def get_neighbors_distribution(self, start_node, n_x):
        key = (start_node, n_x)
        if key in self._cache:
            return self._cache[key].copy()
        distributions = {v: 0 for v in self.G.nodes()}
        current_layer = {start_node: 1}
        visited = {start_node}
        for step in range(n_x + 1):
            next_layer = {}
            for node, prob in current_layer.items():
                stay = self.alpha * prob
                move = (1 - self.alpha) * prob
                distributions[node] += stay
                nbrs = [nbr for nbr in self.G.neighbors(node) if nbr not in visited]
                if nbrs:
                    total_w = sum(self.G[node][nbr].get(self.weight, 1) for nbr in nbrs)
                    for nbr in nbrs:
                        w = self.G[node][nbr].get(self.weight, 1)
                        next_layer[nbr] = next_layer.get(nbr, 0) + move * w / total_w
                else:
                    distributions[node] += move
            visited.update(next_layer.keys())
            current_layer = next_layer
            if not current_layer:
                break
        self._cache[key] = distributions.copy()
        return distributions.copy()

    def _kl_divergence(self, source, target, n_x):
        dist_x = self.get_neighbors_distribution(source, n_x)
        dist_y = self.get_neighbors_distribution(target, n_x)
        nodes = list(self.G.nodes())
        px = np.array([dist_x[n] for n in nodes], dtype=float)
        py = np.array([dist_y[n] for n in nodes], dtype=float)
        px_safe = np.clip(px, self.EPSILON, None)
        py_safe = np.clip(py, self.EPSILON, None)
        kl = np.sum(px_safe * np.log(px_safe / py_safe))
        return float(kl)

    def _compute_entropy_single_edge(self, u, v, n_x):
        kl_uv = self._kl_divergence(u, v, n_x)
        kl_vu = self._kl_divergence(v, u, n_x)
        entropy = kl_uv + kl_vu
        return {(u, v): float(entropy)}

    def compute_entropy(self, n_x):
        self._cache.clear()
        for u, v in self.G.edges():
            entropy_dict = self._compute_entropy_single_edge(u, v, n_x)
            self.G[u][v]["entropy"] = list(entropy_dict.values())[0]

    def compute_entropy_flow(self, n_x, iterations=30, step=0.1):
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.compute_entropy(n_x=n_x)
        for i in range(iterations):
            for u, v in self.G.edges():
                self.G[u][v][self.weight] += step * self.G[u][v]["entropy"]
            self._cache.clear()
            self.compute_entropy(n_x=n_x)
        return self.G



def draw_graph(G, weight="weight", clustering_label="group", cutoff=1.0):
    G_copy = G.copy()
    edge_trim_list = [(n1, n2) for n1, n2 in G_copy.edges() if G_copy[n1][n2][weight] > cutoff]
    G_copy.remove_edges_from(edge_trim_list)
    complex_list = nx.get_node_attributes(G_copy, clustering_label)
    le = preprocessing.LabelEncoder()
    node_color = le.fit_transform(list(complex_list.values()))
    pos = nx.spring_layout(G_copy)
    nx.draw(G_copy, pos, node_color=node_color, cmap='rainbow', with_labels=True, alpha=0.1)
    plt.show()


def ARI(G, clustering, clustering_label="group"):
    complex_list = nx.get_node_attributes(G, clustering_label)
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))
    if isinstance(clustering, dict):
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        y_pred = clustering
    else:
        return -1
    return metrics.adjusted_rand_score(y_true, y_pred)


def NMI(G, clustering, clustering_label="group"):
    complex_list = nx.get_node_attributes(G, clustering_label)
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))
    if isinstance(clustering, dict):
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        y_pred = clustering
    else:
        return -1
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def check_accuracy(G_origin, weight="weight", clustering_label="group"):
    G = G_origin.copy()
    modularity, ari, nmi = [], [], []
    weights = list(nx.get_edge_attributes(G, weight).values())
    maxw = max(weights)
    minw = min(weights)
    cutoff_range = np.arange(maxw, minw, -0.01)
    for cutoff in cutoff_range:
        G_cut = G.copy()
        edge_trim_list = [(n1, n2) for n1, n2 in G_cut.edges() if G_cut[n1][n2][weight] > cutoff]
        G_cut.remove_edges_from(edge_trim_list)
        if G_cut.number_of_edges() == 0:
            break
        clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G_cut)) for c in comp}
        c_communities = list(nx.connected_components(G_cut))
        modularity.append(nx.community.modularity(G_cut, c_communities))
        ari.append(ARI(G_cut, clustering, clustering_label=clustering_label))
        nmi.append(NMI(G_cut, clustering, clustering_label=clustering_label))
    plt.plot(cutoff_range, modularity, alpha=0.8)
    plt.plot(cutoff_range, ari, alpha=0.8)
    plt.plot(cutoff_range, nmi, alpha=0.8)
    plt.xlabel("Edge weight cutoff")
    plt.legend(['Modularity', 'ARI', 'NMI'])
    plt.show()
    return max(ari), max(nmi), max(modularity)



def main():
    #G = nx.read_gexf(r"D:\PythonRNNGCN\karate.gexf")
    G = nx.read_gexf(r"D:\PythonRNNGCN\football.gexf")
    #G=nx.read_gexf(r"D:\PythonRNNGCN\facebook.gexf")
    print(f"Number of nodes in G: {G.number_of_nodes()}")
    print(f"Number of edges in G: {G.number_of_edges()}")
    diameters = [nx.diameter(G.subgraph(comp)) for comp in nx.connected_components(G)]
    n_x = max(diameters)
    print(f"Graph diameter={n_x}, using n_steps={n_x} for entropy flow")

    entropy_flow = Entropyflow(G, alpha=0.5)
    print("Computing entropy...")
    entropy_flow.compute_entropy(n_x=n_x)
    print("Performing entropy flow...")
    G_new = entropy_flow.compute_entropy_flow(n_x=n_x, iterations=30, step=0.01)
    print("Visualizing results...")
    draw_graph(G_new, weight="weight", clustering_label="value", cutoff=2)
    print("Checking clustering quality...")
    max_ari, max_nmi, max_mod = check_accuracy(G_new, weight="weight", clustering_label="value")
    print(f"Max ARI: {max_ari:.4f}, Max NMI: {max_nmi:.4f}, Max Modularity: {max_mod:.4f}")


if __name__ == "__main__":
    main()