import numpy as np
import copy
from collections import defaultdict
import random
import time
random.seed(time.time())

o_rate = 1024 * 1024 / 8
origin_size = 3*224*224/o_rate
# print(origin_size)

class Graph:

    def __init__(self, v_number, flops, power_e, power_c, data_sizes, band):

        self.dict_graph=defaultdict(list)
        self.dict_parent=defaultdict(list)
        self.dict_graph_sorted = defaultdict(list)
        self.dict_parent_sorted = defaultdict(list)
        self.V = v_number

        for i in range(v_number):
            self.dict_graph[i]=[]
            self.dict_parent[i] =[]
            self.dict_graph_sorted[i] = []
            self.dict_parent_sorted[i] = []

        self.flops = np.array(flops)
        self.power_e=power_e
        self.power_c=power_c
        self.data_sizes=np.array(data_sizes)
        self.band=band
        self.cal_time_e=self.flops/power_e
        self.cal_time_c=self.flops/power_c
        self.trans_time=self.data_sizes/band

    def addEdge(self, u, v):
        self.dict_graph[u].append([v, self.trans_time[u]])
        self.dict_parent[v].append(u)

    def generate_graph(self):
        sort_result=self.topologicalSort()
        self.sort_result=sort_result
        len_sort=len(sort_result)
        self.len_sort=len_sort
        self.dict_sort_idx={}
        self.flops_sort=[]
        self.data_sizes_sort=[]
        for i in range(len_sort):
            self.dict_sort_idx[sort_result[i]]=i
            self.flops_sort.append(self.flops[sort_result[i]])
            self.data_sizes_sort.append(self.data_sizes[sort_result[i]])

        for i in range(len_sort):
            self.dict_graph_sorted[i]=copy.deepcopy(self.dict_graph[sort_result[i]])
            for j in range(len(self.dict_graph_sorted[i])):
                self.dict_graph_sorted[i][j][0]=list(sort_result).index(self.dict_graph_sorted[i][j][0])
            self.dict_parent_sorted[i]=copy.deepcopy(self.dict_parent[self.sort_result[i]])
            for j in range(len(self.dict_parent_sorted[i])):
                self.dict_parent_sorted[i][j]=list(sort_result).index(self.dict_parent_sorted[i][j])

        graph=defaultdict(list)
        first_line = [0]
        for i in range(len_sort):
            first_line.append(self.cal_time_c[i])
        first_line.append(0)
        graph[0]=first_line
        len_first_line=len(first_line)

        for i in range(len_sort):
            line=[0 for j in range(len_first_line)]
            line[len_first_line-1]=self.cal_time_e[i]
            idx=sort_result[i]
            for child in self.dict_graph[idx]:
                line[child[0]+1]=child[1]
            graph[idx+1]=line

        graph[len_sort+1]=[0 for j in range(len_first_line)]
        self.graph=[]
        for i in range(len_first_line):
            self.graph.append(copy.deepcopy(graph[i]))
        # self.graph = graph  # residual graph
        self.org_graph = [i[:] for i in self.graph]
        self.ROW = len(self.graph)
        self.COL = len(self.graph[0])


    def topologicalSortUtil(self, v, visited, stack):

        visited[v] = True

        for i in self.dict_graph[v]:
            if visited[i[0]] == False:
                self.topologicalSortUtil(i[0], visited, stack)

        stack.insert(0, v)

    def topologicalSort(self):
        import numpy as np
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)
        stack=np.array(stack)

        return stack

    '''Returns true if there is a path from 
    source 's' to sink 't' in 
    residual graph. Also fills 
    parent[] to store the path '''

    def BFS(self, s, t, parent):

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            # Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of
            # the dequeued vertex u
            # If a adjacent has not been
            # visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

                    # If we reached sink in BFS starting
        # from source, then return
        # true, else false
        return True if visited[t] else False

    # Function for Depth first search
    # Traversal of the graph
    def dfs(self, graph, s, visited):
        visited[s] = True
        for i in range(len(graph)):
            if graph[s][i] > 0 and not visited[i]:
                self.dfs(graph, i, visited)

    def DFS(self, s, t, now, parent):
        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)

        # Create a queue for BFS
        # queue = []

        # Mark the source node as visited and enqueue it
        # queue.append(s)
        visited[s] = True
        for i in range(now + 1, t + 1):
            if visited[i] == False and self.graph[now][i] > 0:
                visited[i] = True
                parent[i] = now
                if (visited[t]):
                    return True
                self.DFS(s, t, i, parent)
                if (visited[t]):
                    return True
        return visited[t]

    # Returns the min-cut of the given graph
    def minCut(self, source, sink):

        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially

        # print(self.DFS(source, sink, source, parent))
        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):
            # while self.DFS(source, sink, source, parent):

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            max_flow += path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        visited = len(self.graph) * [False]
        self.dfs(self.graph, 0, visited)

        c_graph = [[0 for i in range(self.COL)] for j in range(self.ROW)]
        for i in range(self.ROW):
            for j in range(self.COL):
                if self.graph[i][j] == 0 and \
                        self.org_graph[i][j] > 0 and visited[i]:
                    c_graph[i][j] = 1

        # print(c_graph)

        need_fix = [0 for i in range(self.COL)]
        for i in range(self.COL):
            if c_graph[0][i]==1 and c_graph[i][self.COL-1]==1:
                need_fix[i]=1
        min_time = float("inf")
        edge_time = 0
        cloud_time = 0
        trans_time = -1
        fix_idx=-1
        if np.sum(need_fix)>0:
            return "ERROR CODE"
        else:
            tmp_time_c = 0
            tmp_time_e = 0
            tmp_trans = 0
            for i in range(self.ROW):
                for j in range(self.COL):
                    if c_graph[i][j]==1:
                        if i==0:
                            tmp_time_c+=self.org_graph[i][j]
                        elif j==self.COL-1:
                            tmp_time_e+=self.org_graph[i][j]
                        else:
                            tmp_trans+=self.org_graph[i][j]
            tmp_min_time = max(max(tmp_time_e, tmp_time_c), tmp_trans)
            if tmp_min_time < min_time:
                min_time = tmp_min_time
                trans_time = tmp_trans
                edge_time = tmp_time_e
                cloud_time = tmp_time_c

        if fix_idx!=-1:
            for i in range(fix_idx, self.ROW-1):
                c_graph[i][self.COL-1]=0
                c_graph[0][i]=1

        cuted_graph=copy.deepcopy(self.org_graph)
        for i in range(self.ROW):
            for j in range(self.COL):
                if c_graph[i][j]==1:
                    cuted_graph[i][j]=0
        # edge_visited=self.ROW * [False]
        # self.dfs(cuted_graph,0,edge_visited)

        layer_edge = []
        layer_cloud = []

        for i in range(1,self.ROW-1):
            if c_graph[0][i]==1:
                layer_cloud.append(i-1)
            # elif c_graph[i][self.COL-1]:
            else:
                layer_edge.append(i-1)

        edge_time=0
        cloud_time=0
        trans_time=0
        for i in layer_cloud:
            cloud_time+=self.org_graph[0][i+1]
            children=self.dict_graph_sorted[i]
            for child in children:
                if child[0] in layer_edge:
                    trans_time+=child[1]
        for i in layer_edge:
            edge_time+=self.org_graph[i+1][self.COL-1]
            children = self.dict_graph_sorted[i]
            for child in children:
                if child[0] in layer_cloud:
                    trans_time += child[1]
        min_time = np.max([edge_time,cloud_time,trans_time])

        layer_edge_sorted = []
        layer_cloud_sorted = []
        for i in range(len(layer_edge)):
            # edge_layer = [[i, self.org_graph[layer_edge[i]][self.COL-1], self.data_sizes[layer_edge[i]-1]]]
            edge_layer = [[i, self.flops_sort[layer_edge[i]], self.data_sizes_sort[layer_edge[i]]]]
            # layer_edge_sorted.append(edge_layer)
            children=[]
            for j in range(len(self.dict_graph_sorted[layer_edge[i]])):
                if self.dict_graph_sorted[layer_edge[i]][j][0] in layer_edge:
                    tmp = self.dict_graph_sorted[layer_edge[i]][j]
                    tmp_child = layer_edge.index(tmp[0])
                    children.append(tmp_child)
            edge_layer.append(children)
            layer_edge_sorted.append(edge_layer)

        for i in range(len(layer_cloud)):
            # cloud_layer = [[i, self.org_graph[0][layer_cloud[i]], self.data_sizes[layer_cloud[i]-1]]]
            cloud_layer = [[i, self.flops_sort[layer_cloud[i]], self.data_sizes_sort[layer_cloud[i]]]]
            children=[]
            for j in range(len(self.dict_graph_sorted[layer_cloud[i]])):
                if self.dict_graph_sorted[layer_cloud[i]][j][0] in layer_cloud:
                    tmp=self.dict_graph_sorted[layer_cloud[i]][j]
                    tmp_child=layer_cloud.index(tmp[0])
                    children.append(tmp_child)
            cloud_layer.append(children)
            layer_cloud_sorted.append(cloud_layer)

        return layer_edge_sorted, layer_cloud_sorted, min_time, trans_time, edge_time, cloud_time

def others(layers, units, band=1024/8):
    cm=0
    for i in range(len(layers)):
        cm+=layers[i][0][1]
    return max(np.sum(cm)/np.max(units), origin_size/band)

def middle_cut(layers, units, band=1024/8):
    len_layers = len(layers)
    cal_left = []
    cal_right = []
    trans = []
    for i in range(len_layers):
        cal_left.append(layers[i][0][1]/units[0])
        cal_right.append(layers[i][0][1]/units[1])
        trans.append(layers[i][0][2]/band)

    sum_left=copy.deepcopy(cal_left)
    for i in range(1,len_layers):
        sum_left[i]+=sum_left[i-1]

    sum_right=copy.deepcopy(cal_right)
    for i in range(len_layers-2,-1,-1):
        sum_right[i]+=sum_right[i+1]
    for i in range(len_layers):
        sum_right[i]-=cal_right[i]

    min_time = float('inf')
    cut_point = -1
    for i in range(len_layers):
        tmp_time = np.max([sum_left[i],sum_right[i],trans[i]])
        if tmp_time<min_time:
            min_time=tmp_time
            cut_point=i+1

    left_layers=layers[0:cut_point]
    right_layers=layers[cut_point:]
    trans_time=trans[cut_point-1]

    return [min_time, trans_time, left_layers, right_layers]


# layers: [[[index, flops, out_size], [child_idx_1, child_idx_2, ...]],
# [[index, flops, out_size], [child_idx_1, child_idx_2, ...]],
# ...]
#
# units: [power_1, power_2, power_3, ...]
def recursion(layers, units, band=1024/8):
    len_units = len(units)
    len_layers = len(layers)
    if len_layers==0 or len_units==0:
        return 0
    if len_units==1:
        min_time = 0
        for i in range(len(layers)):
            min_time+=layers[i][0][1]/units[0]
        return min_time
    if len_layers==1:
        return layers[0][0][1]/np.max(units)
    mid = int(len_units/2)
    left_units = units[0:mid]
    right_units = units[mid:]
    left_sum = np.sum(left_units)
    right_sum = np.sum(right_units)
    left_first = [left_sum, right_sum]
    right_first = [right_sum, left_sum]

    left_result=middle_cut(layers, left_first, band)
    right_result = middle_cut(layers, right_first, band)
    # print(left_result)
    # print(right_result)
    time_1=left_result[0]
    time_2=right_result[0]
    if left_result[0]<right_result[0]:
        if left_sum>0:
            time_1 = recursion(left_result[2], left_units, band)
        if right_sum>0:
            time_2 = recursion(left_result[3], right_units, band)
        trans_time = left_result[1]
    else:
        if right_sum>0:
            time_1 = recursion(right_result[2], right_units, band)
        if left_sum>0:
            time_2 = recursion(right_result[3], left_units, band)
        trans_time=right_result[1]
    return np.max([time_1, time_2, trans_time])

def middle_cut_call(layers, units, band=1024/8):
    result = middle_cut(layers, units, band)
    return max(result[0], origin_size/band)

def recursion_call(layers, units, band=1024/8):
    result = recursion(layers, units, band)
    return max(result, origin_size/band)

def min_cut(layers, units, band=1024/8):
    len_units=len(units)
    times=[]
    for i in range(len_units-1):
        for j in range(i+1, len_units):
            tmp_units=[units[i], units[j]]
            tmp_units.sort()
            times.append(min_cut_util(layers, tmp_units, band))
    return np.min(times)

def min_cut_util(layers, units, band=1024/8, status=1):
    len_units = len(units)
    if len_units < 2:
        return 0

    units = copy.deepcopy(units)
    units.sort()
    if status==1:
        left_units = [units[0]]
        right_units = units[1:]
    else:
        mid = int(len_units / 2)
        left_units=units[0:mid]
        right_units=units[mid:]

    # def __init__(self, v_number, flops, power_e, power_c, data_sizes, band):
    v_number = len(layers)
    flops = []
    data_sizes=[]
    for i in range(v_number):
        flops.append(layers[i][0][1])
        data_sizes.append(layers[i][0][2])
    power_e=np.sum(left_units)
    power_c=np.sum(right_units)

    g=Graph(v_number, flops, power_e, power_c, data_sizes, band)
    for i in range(v_number):
        children = layers[i][1]
        for child in children:
            g.addEdge(i, child)
    g.generate_graph()

    # def minCut(self, source, sink):
    tmp_result = g.minCut(0, g.ROW-1)
    if tmp_result == "ERROR CODE":
        avg_time = middle_cut_call(layers, units, band)
        return max(avg_time, origin_size/band)
    layer_edge_sorted, layer_cloud_sorted, min_time, trans_time, edge_time, cloud_time = tmp_result

    time_edge = min_cut_util(layer_edge_sorted, left_units, band)
    time_cloud = min_cut_util(layer_cloud_sorted, right_units, band)

    avg_time = 0
    if time_edge == 0 and time_cloud == 0:
        avg_time = min_time
    elif time_edge == 0:
        avg_time = np.max([edge_time, trans_time, time_cloud])
    elif time_cloud == 0:
        avg_time = np.max([time_edge, trans_time, cloud_time])
    else:
        avg_time = np.max([time_edge, trans_time, time_cloud])
    return max(avg_time, origin_size/band)

# layers: [[[index, flops, out_size], [child_idx_1, child_idx_2, ...]],
# [[index, flops, out_size], [child_idx_1, child_idx_2, ...]],
# ...]
class GS():
    def __init__(self):
        f_rate=1e9
        self.ale_f = np.array([70470400 + 193600 + 193600, 224088768 + 139968 + 139968, 112205184 + 64896,
                               149563648 + 43264, 99723520 + 43264 + 43264, 37748736 + 4096, 16777216 + 4096, 4096000]) / f_rate
        self.ale_o = np.array([64 * 27 * 27, 192 * 13 * 13, 384 * 13 * 13, 256 * 13 * 13,
                               256 * 6 * 6, 4096, 4096, 0]) / 1024 / 1024 * 8
        self.ale_layers = []
        for i in range(7):
            layer_param = [i, self.ale_f[i]/2, self.ale_o[i]/2]
            children = [i+1]
            if i in [1,4,5]:
                children.append(i+9)
            self.ale_layers.append([layer_param, children])
        self.ale_layers.append([[7, self.ale_f[7], self.ale_o[7]], []])
        for i in range(7):
            layer_param = [i+8, self.ale_f[i]/2, self.ale_o[i]/2]
            if i == 6:
                children = [7]
            else:
                children = [i + 8 + 1]
            if i in [1, 4, 5]:
                children.append(i+1)
            self.ale_layers.append([layer_param, children])
        self.ale_origin=[]
        for i in range(8):
            layer_param=[i, self.ale_f[i], self.ale_o[i]]
            if i==7:
                children = []
            else:
                children = [i + 1]
            self.ale_origin.append([layer_param, children])


        self.res_f = np.array([np.sum([118013952, 1605632, 802816, 802816]), np.sum([115605504, 401408, 200704]),
                               np.sum([115605504, 401408]), np.sum([115605504, 401408, 200704]),
                               115605504 + 401408, 57802752 + 200704 + 100352, 115605504 + 200704,
                               115605504 + 200704 + 100352, 115605504 + 200704, 57802752 + 100352 + 50176,
                               115605504 + 100352, 115605504 + 100352 + 50176, 115605504 + 100352,
                               57802752 + 50176 + 25088, 115605504 + 50176, 115605504 + 50176 + 25088,
                               115605504 + 50176, 512000,
                               6422528 + 200704, 6422528 + 100352, 6422528 + 50176]) / f_rate
        self.res_o = np.array([1.53125, 1.53125, 1.53125, 1.53125, 1.53125,
                               0.765625, 1.53125, 0.765625, 0.765625, 0.3828125, 0.3828125,
                               0.3828125, 0.3828125, 0.19140625, 0.19140625, 0.19140625,
                               0.19140625, 0,
                               0.765625, 0.3828125, 0.19140625])
        # layers: [[[index, flops, out_size], [child_idx_1, child_idx_2, ...]],
        # [[index, flops, out_size], [child_idx_1, child_idx_2, ...]],
        # ...]
        self.res_layers = []
        for i in range(21):
            layer_param = [i, self.res_f[i], self.res_o[i]]
            if i in [18, 19, 20]:
                children=[i-11+(i-18)*3]
            elif not i%2:
                if i in [4, 8, 12]:
                    children=[i+1, i+14-int((i-4)/4)*3]
                elif i == 16:
                    children=[17]
                else:
                    children = [i + 1, i + 3]
            elif i == 17:
                children=[]
            else:
                children=[i+1]
            self.res_layers.append([layer_param, children])

        self.vgg_f = np.array([89915392 + 3211264, 1852899328 + 3211264 + 3211264, 926449664 + 1605632,
                               1851293696 + 1605632 + 1605632, 925646848 + 802816, 1850490880 + 802816,
                               1850490880 + 802816 + 802816, 925245440 + 401408, 1850089472 + 401408,
                               1850089472 + 401408 + 401408, 462522368 + 100352, 462522368 + 100352,
                               462522368 + 100352 + 100352, 102760448.0 + 4096, 16777216 + 4096,
                               4096000]) / f_rate
        self.vgg_o = np.array([64 * 224 * 224, 64 * 112 * 112, 128 * 112 * 112, 128 * 56 * 56, 256 * 56 * 56,
                               256 * 56 * 56, 256 * 28 * 28, 512 * 28 * 28, 512 * 28 * 28, 512 * 14 * 14,
                               512 * 14 * 14, 512 * 14 * 14, 512 * 7 * 7, 4096, 4096, 0]) / 1024 / 1024 * 8
        self.vgg_layers = []
        for i in range(16-1):
            layer_param=[i, self.vgg_f[i], self.vgg_o[i]]
            children=[i+1]
            self.vgg_layers.append([layer_param, children])
        self.vgg_layers.append([[15, self.vgg_f[15], self.vgg_o[15]], []])



        self.nin_f = np.array([np.sum([105705600, 290400, 28168800, 290400, 28168800, 290400]),
                               290400,
                               np.sum([448084224, 186624, 47962368, 186624, 47962368, 186624]),
                               186624,
                               np.sum([149585280, 64896, 24984960, 64896, 24984960, 64896]),
                               64896,
                               np.sum([248832000, 36000, 72000000, 36000, 72000000, 36000]),
                               0]) / f_rate
        self.nin_o = np.array([0.27694702, 0.06674194, 0.17797852, 0.04125977, 0.06188965, 0.01318359, 0.03433228, 0]) * 8
        self.nin_layer=[]
        for i in range(8):
            layer_param=[i, self.nin_f[i], self.nin_o[i]]
            if i<7:
                children = [i+1]
            else:
                children=[]
            self.nin_layer.append([layer_param, children])


