import numpy as np
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import decomposition

class GrowingNeuralGas:

    def __init__(self, input_data,e_b, e_n, a_max, l, a, d, passes, plot_evolution):
        self.network = None
        self.e_b = e_b
        self.e_n = e_n
        self.a_max  = a_max
        self.a  = a
        self.l = l
        self.d  = d 
        self.passes = passes
        self.plot_evolution = plot_evolution
        self.data = input_data
        self.units_created = 0
        plt.style.use('ggplot')

    def find_nearest_units(self, observation):
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes['vector']
            dist = spatial.distance.euclidean(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def prune_connections(self, a_max):
        nodes_to_remove = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > a_max:
                nodes_to_remove.append((u, v))
        for u, v in nodes_to_remove:
            self.network.remove_edge(u, v)

        nodes_to_remove = []
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                nodes_to_remove.append(u)
        for u in nodes_to_remove:
            self.network.remove_node(u)

    def fit_network(self):
        # logging variables
        accumulated_local_error = []
        global_error = []
        network_order = []
        network_size = []
        total_units = []
        self.units_created = 0
        # 0. start with two units a and b at random position w_a and w_b
        w_a = [np.random.uniform(-2, 2) for _ in range(np.shape(self.data)[1])]
        w_b = [np.random.uniform(-2, 2) for _ in range(np.shape(self.data)[1])]
        self.network = nx.Graph()
        self.network.add_node(self.units_created, vector=w_a, error=0)
        self.units_created += 1
        self.network.add_node(self.units_created, vector=w_b, error=0)
        self.units_created += 1
        # 1. Стартоовые значения и итерации
        sequence = 0
        for p in range(self.passes):
            print('   Pass #%d' % (p + 1))
            np.random.shuffle(self.data)
            steps = 0
            for observation in self.data:
                # 2. Поиск первого и второго ближайшего
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]
                # 3. Определение следующего поколения для всех итераций 1 ближайшего
                for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                    self.network.add_edge(u, v, age=attributes['age']+1)
                # 4. Метрика расстояния (евклидова)
                self.network.node[s_1]['error'] += spatial.distance.euclidean(observation, self.network.node[s_1]['vector'])**2
                # 5 . Перемещение ближайшего по сетке в сторону кратчайшего расстояния 
                update_w_s_1 = self.e_b * \
                    (np.subtract(observation,
                                 self.network.node[s_1]['vector']))
                self.network.node[s_1]['vector'] = np.add(
                    self.network.node[s_1]['vector'], update_w_s_1)

                for neighbor in self.network.neighbors(s_1):
                    update_w_s_n = self.e_n * \
                        (np.subtract(observation,
                                     self.network.node[neighbor]['vector']))
                    self.network.node[neighbor]['vector'] = np.add(
                        self.network.node[neighbor]['vector'], update_w_s_n)
                # 6. если 1 и 2 ближайший лежат в одном подпростратсве - ставим 0
                #    если такого подпространства нет - создаеи
                self.network.add_edge(s_1, s_2, age=0)
                # 7. убираем ребра на ограничение поколений a_max
                # если у результата нет связующих ребер - убираем 
                self.prune_connections(self.a_max)
                # 8. пока число шагов не выходит за рамки l создаем новый рассчет 
                steps += 1
                if steps % self.l == 0:
                    if self.plot_evolution:
                        self.plot_network('visualization/sequence/' + str(sequence) + '.png')
                    sequence += 1
                    # 8. переменная q накопленная ошибка
                    q = 0
                    error_max = 0
                    for u in self.network.nodes():
                        if self.network.node[u]['error'] > error_max:
                            error_max = self.network.node[u]['error']
                            q = u
                    # 8.b создадим новую переменную r, лежащую между q и ее соседом(f) с максимальной накопленной ошибкой
                    f = -1
                    largest_error = -1
                    for u in self.network.neighbors(q):
                        if self.network.node[u]['error'] > largest_error:
                            largest_error = self.network.node[u]['error']
                            f = u
                    w_r = 0.5 * (np.add(self.network.node[q]['vector'], self.network.node[f]['vector']))
                    r = self.units_created
                    self.units_created += 1
                    # 8.c создаем ребра на q r f
                    #  обнулим первоначальную реализацию
                    self.network.add_node(r, vector=w_r, error=0)
                    self.network.add_edge(r, q, age=0)
                    self.network.add_edge(r, f, age=0)
                    self.network.remove_edge(q, f)
                    # 8.d используем параметр а для уменьшения разброса ошибок на q f 
                    #     инициализируем параметр ошибок r для нового параметра q
                    self.network.node[q]['error'] *= self.a
                    self.network.node[f]['error'] *= self.a
                    self.network.node[r]['error'] = self.network.node[q]['error']
                # 9. используем параметр d для уменьшения вектора ошибок
                error = 0
                for u in self.network.nodes():
                    error += self.network.node[u]['error']
                accumulated_local_error.append(error)
                network_order.append(self.network.order())
                network_size.append(self.network.size())
                total_units.append(self.units_created)
                for u in self.network.nodes():
                    self.network.node[u]['error'] *= self.d
                    if self.network.degree(nbunch=[u]) == 0:
                        print(u)
            global_error.append(self.compute_global_error())
        plt.clf()
        plt.title('Accumulated local error')
        plt.xlabel('iterations')
        plt.plot(range(len(accumulated_local_error)), accumulated_local_error)
        plt.savefig('visualization/accumulated_local_error.png')
        plt.clf()
        plt.title('Global error')
        plt.xlabel('passes')
        plt.plot(range(len(global_error)), global_error)
        plt.savefig('visualization/global_error.png')
        plt.clf()
        plt.title('Neural network properties')
        plt.plot(range(len(network_order)), network_order, label='Network order')
        plt.plot(range(len(network_size)), network_size, label='Network size')
        plt.legend()
        plt.savefig('visualization/network_properties.png')

    def plot_network(self, file_path):
        plt.clf()
        plt.scatter(self.data[:, 0], self.data[:, 1])
        node_pos = {}
        for u in self.network.nodes():
            vector = self.network.node[u]['vector']
            node_pos[u] = (vector[0], vector[1])
        nx.draw(self.network, pos=node_pos)
        plt.draw()
        plt.savefig(file_path)

    def number_of_clusters(self):
        return nx.number_connected_components(self.network)

    def cluster_data(self):
        unit_to_cluster = np.zeros(self.units_created)
        cluster = 0
        for c in nx.connected_components(self.network):
            for unit in c:
                unit_to_cluster[unit] = cluster
            cluster += 1
        clustered_data = []
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s = nearest_units[0]
            clustered_data.append((observation, unit_to_cluster[s]))
        return clustered_data

    def reduce_dimension(self, clustered_data):
        transformed_clustered_data = []
        svd = decomposition.PCA(n_components=2)
        transformed_observations = svd.fit_transform(self.data)
        for i in range(len(clustered_data)):
            transformed_clustered_data.append((transformed_observations[i], clustered_data[i][1]))
        return transformed_clustered_data

    def plot_clusters(self, clustered_data):
        number_of_clusters = nx.number_connected_components(self.network)
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(number_of_clusters):
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.array(observations)
                plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
        plt.legend()
        plt.savefig('visualization/clusters.png')

    def compute_global_error(self):
        global_error = 0
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            global_error += spatial.distance.euclidean(observation, self.network.node[s_1]['vector'])**2
        return global_error