import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics


class Entropyflow:
    def __init__(self, G, alpha, weight='weight'):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self._cache = {}

    def get_neighbors_distribution(self, start_node):
        if start_node in self._cache:
            return self._cache[start_node].copy()
        node_distributions = {}
        current_layer = {start_node: 1}
        visited_nodes = set([start_node])

        while current_layer:
            next_layer = {}
            for node, probability in current_layer.items():
                stay = self.alpha * probability
                move = (1 - self.alpha) * probability
                node_distributions[node] = node_distributions.get(node, 0) + stay
                nbrs = [nbr for nbr in self.G.neighbors(node) if nbr not in visited_nodes]
                if nbrs:
                    total_weights = sum(self.G[node][nbr].get(self.weight, 1.0) for nbr in nbrs)
                    for nbr in nbrs:
                        edge_weight = self.G[node][nbr].get(self.weight, 1.0)
                        next_prob = move * edge_weight / total_weights
                        next_layer[nbr] = next_layer.get(nbr, 0) + next_prob
                else:
                    node_distributions[node] = node_distributions.get(node, 0) + move

            if not next_layer:
                break
            visited_nodes.update(next_layer.keys())
            current_layer = next_layer
        self._cache[start_node] = node_distributions.copy()
        return node_distributions.copy()

    def _kl_divergence(self, source, target):
        dist_x = self.get_neighbors_distribution(source)
        dist_y = self.get_neighbors_distribution(target)
        all_nodes = set(dist_x.keys()) | set(dist_y.keys())
        px = np.array([dist_x[node] for node in all_nodes])
        py = np.array([dist_y[node] for node in all_nodes])
        kl = np.sum(px * np.log(px / py))
        return float(kl)

    def _compute_entropy_single_edge(self, u, v):
        kl_uv = self._kl_divergence(u, v)
        kl_vu = self._kl_divergence(v, u)
        entropy = kl_uv + kl_vu
        return {(u, v): float(entropy)}

    def compute_entropy(self):
        self._cache.clear()
        for u, v in self.G.edges():
            entropy_dict = self._compute_entropy_single_edge(u, v)
            self.G[u][v]["entropy"] = list(entropy_dict.values())[0]

    def compute_entropy_flow(self, iterations=5, step=0.01):
        self.compute_entropy()
        for i in range(iterations):
            for u, v in self.G.edges():
                self.G[u][v][self.weight] += step * self.G[u][v]["entropy"]
            self._cache.clear()
            self.compute_entropy()
        return self.G

def compare_entropy_before_after(G_original, G_flowed):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    entropy_flow = Entropyflow(G_original.copy(), alpha=0.5)
    entropy_flow.compute_entropy()
    initial_entropy = list(nx.get_edge_attributes(entropy_flow.G, "entropy").values())
    initial_weights = list(nx.get_edge_attributes(entropy_flow.G, "weight").values())
    final_entropy = list(nx.get_edge_attributes(G_flowed, "entropy").values())
    final_weights = list(nx.get_edge_attributes(G_flowed, "weight").values())
    ax1 = axes[0]
    bins1 = np.linspace(min(initial_entropy + final_entropy),
                        max(initial_entropy + final_entropy), 35)
    ax1.hist(initial_entropy, bins=bins1, alpha=0.7, color='#3498DB', label='Initial', density=True, edgecolor='white', linewidth=1.2)
    ax1.hist(final_entropy, bins=bins1, alpha=0.7, color='#E74C3C', label='After Flow', density=True, edgecolor='white', linewidth=1.2)
    from scipy.stats import gaussian_kde
    if len(initial_entropy) > 1:
        kde_initial = gaussian_kde(initial_entropy)
        x_range1 = np.linspace(min(initial_entropy), max(initial_entropy), 200)
        ax1.plot(x_range1, kde_initial(x_range1), color='#2980B9',
                 linewidth=2.5, linestyle='--', alpha=0.9)

    if len(final_entropy) > 1:
        kde_final = gaussian_kde(final_entropy)
        x_range2 = np.linspace(min(final_entropy), max(final_entropy), 200)
        ax1.plot(x_range2, kde_final(x_range2), color='#C0392B',
                 linewidth=2.5, linestyle='--', alpha=0.9)
    ax1.set_xlabel('Edge Entropy', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Density', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_title('Entropy distribution before and after flow',
                  fontsize=16, fontweight='bold', pad=15)
    legend1 = ax1.legend(loc='upper right', fontsize=12, frameon=True,
                         shadow=True, fancybox=True, borderpad=1,
                         facecolor='white', edgecolor='#34495E')
    legend1.get_frame().set_alpha(0.9)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.7)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#2C3E50')
    ax2 = axes[1]
    bins2 = np.linspace(min(initial_weights + final_weights),
                        max(initial_weights + final_weights), 35)
    ax2.hist(initial_weights, bins=bins2, alpha=0.7, color='#2ECC71', label='Initial', density=True, edgecolor='white', linewidth=1.2)
    ax2.hist(final_weights, bins=bins2, alpha=0.7, color='#F39C12',label='After Flow', density=True, edgecolor='white', linewidth=1.2)
    if len(initial_weights) > 1:
        kde_initial_w = gaussian_kde(initial_weights)
        x_range3 = np.linspace(min(initial_weights), max(initial_weights), 200)
        ax2.plot(x_range3, kde_initial_w(x_range3), color='#27AE60',
                 linewidth=2.5, linestyle='--', alpha=0.9)
    if len(final_weights) > 1:
        kde_final_w = gaussian_kde(final_weights)
        x_range4 = np.linspace(min(final_weights), max(final_weights), 200)
        ax2.plot(x_range4, kde_final_w(x_range4), color='#D35400',
                 linewidth=2.5, linestyle='--', alpha=0.9)
    ax2.set_xlabel('Edge Weight', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Density', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_title('Weight distribution before and after flow',
                  fontsize=16, fontweight='bold', pad=15)
    legend2 = ax2.legend(loc='upper right', fontsize=12, frameon=True,
                         shadow=True, fancybox=True, borderpad=1,
                         facecolor='white', edgecolor='#34495E')
    legend2.get_frame().set_alpha(0.9)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.7)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#2C3E50')
    plt.tight_layout(pad=3.0)
    plt.show()


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
    plt.figure(figsize=(13, 8))
    colors = ['#FF4E50', '#00C9FF', '#92D050']
    plt.plot(cutoff_range[:len(modularity)], modularity,
             label='Modularity', color=colors[0],
             linewidth=3.5, alpha=0.9, marker='',
             linestyle='-', markersize=0)
    plt.plot(cutoff_range[:len(ari)], ari,
             label='ARI', color=colors[1],
             linewidth=3.5, alpha=0.9, marker='',
             linestyle='-', markersize=0)
    plt.plot(cutoff_range[:len(nmi)], nmi,
             label='NMI', color=colors[2],
             linewidth=3.5, alpha=0.9, marker='',
             linestyle='-', markersize=0)
    plt.fill_between(cutoff_range[:len(modularity)], 0, modularity,
                     color=colors[0], alpha=0.15, label='_nolegend_')
    plt.fill_between(cutoff_range[:len(ari)], 0, ari,
                     color=colors[1], alpha=0.15, label='_nolegend_')
    plt.fill_between(cutoff_range[:len(nmi)], 0, nmi,
                     color=colors[2], alpha=0.15, label='_nolegend_')
    if modularity:
        max_mod_idx = np.argmax(modularity)
        plt.scatter(cutoff_range[max_mod_idx], modularity[max_mod_idx],
                    color=colors[0], s=120, zorder=5,
                    edgecolors='white', linewidth=2,
                    marker='o', label=f'Max Modularity: {max(modularity):.3f}')

    if ari:
        max_ari_idx = np.argmax(ari)
        plt.scatter(cutoff_range[max_ari_idx], ari[max_ari_idx],
                    color=colors[1], s=120, zorder=5,
                    edgecolors='white', linewidth=2,
                    marker='s', label=f'Max ARI: {max(ari):.3f}')

    if nmi:
        max_nmi_idx = np.argmax(nmi)
        plt.scatter(cutoff_range[max_nmi_idx], nmi[max_nmi_idx],
                    color=colors[2], s=120, zorder=5,
                    edgecolors='white', linewidth=2,
                    marker='^', label=f'Max NMI: {max(nmi):.3f}')
    plt.xlabel("Edge Weight Cutoff", fontsize=15, fontweight='bold', labelpad=12)
    plt.ylabel("Metric Value", fontsize=15, fontweight='bold', labelpad=12)
    plt.title("Community Detection Analysis: Metrics vs Edge Weight Cutoff",
              fontsize=17, fontweight='bold', pad=20,
              color='#2C3E50')
    legend = plt.legend(fontsize=12, frameon=True,
                        shadow=True, fancybox=True,
                        borderpad=1, loc='upper right',
                        facecolor='white', edgecolor='#BDC3C7')
    legend.get_frame().set_alpha(0.95)
    plt.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, color='#7F8C8D')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#34495E')
    ax.spines['bottom'].set_color('#34495E')
    ax.tick_params(axis='both', which='major', labelsize=12,
                   width=1.5, length=6, color='#34495E')
    all_metrics = modularity + ari + nmi
    if all_metrics:
        y_min = min(all_metrics)
        y_max = max(all_metrics)
        plt.ylim(max(y_min - 0.08, -0.05), min(y_max + 0.08, 1.05))
    ax.set_facecolor('#F8F9FA')
    ax.set_axisbelow(True)
    plt.figtext(0.98, 0.02, 'Community Detection Analysis',
                fontsize=10, color='gray', ha='right', alpha=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    return max(ari), max(nmi), max(modularity)


def main():
    #G = nx.read_gexf(r"D:\PythonRNNGCN\karate.gexf")
    G = nx.read_gexf(r"D:\PythonRNNGCN\football.gexf")
    #G = nx.read_gexf(r"D:\PythonRNNGCN\facebook.gexf")
    entropy_flow = Entropyflow(G, alpha=0.5)
    print("\nComputing entropy...")
    entropy_flow.compute_entropy()
    print("Performing entropy flow...")
    G_new = entropy_flow.compute_entropy_flow(iterations=30, step=0.1)
    compare_entropy_before_after(G, G_new)
    draw_graph(G_new, weight="weight", clustering_label="value", cutoff=2)
    max_ari, max_nmi, max_mod = check_accuracy(G_new, weight="weight", clustering_label="value")
    print(f"Max ARI: {max_ari:.4f}, Max NMI: {max_nmi:.4f}, Max Modularity: {max_mod:.4f}")


if __name__ == "__main__":
    main()
