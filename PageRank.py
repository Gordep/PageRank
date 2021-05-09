import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys


def animate(current_iteration):
    nodes, values, iteration = current_iteration
    values = np.asarray(values).ravel()
    plt_nodes = nx.draw_networkx_nodes(G, pos,ax=ax,node_color=values,alpha=1,node_size=700,cmap=plt.cm.Oranges,vmin=0,vmax=0.2)
    ax.set_title(f"Iteration {iteration}")
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=12)
    return [plt_nodes, ]

#G our graph
#D Damping factor
#T tolerance for convergence
#max_iter maximum number of iteration allowed
#doing iterative version of page range
def page_rank(G, d=0.85, t=1e-6, max_iter=100):
    nodes = G.nodes()
    matrix = nx.adjacency_matrix(G, nodelist=nodes)
    out_degree = matrix.sum(axis=0) 
    weight = matrix / out_degree
    #Get number of pages 
    N = G.number_of_nodes()
    #initialize pages with value of 1
    pr = np.ones(N).reshape(N, 1) * 1./N

    #Initilize the first frame of graph
    yield nodes, pr, "Graph"

    for i in range(max_iter):
        old_pr = pr[:]
        pr = d * weight.dot(pr) + (1-d)/N
        #Initilize node i of the graph
        yield nodes, pr, i
        err = np.absolute(pr - old_pr).sum()
        if err < t:
            return pr
        



  
#Generate random 
# Has chance to Generate a graph that contains a node with zero edges this will break the code
G = nx.gnp_random_graph(10,.4)

#Get the page Rank of graph G 
PageRank_of_G = page_rank(G)

# Generate positons for each node 
pos = nx.kamada_kawai_layout(G)

f, ax = plt.subplots()
#Create the gif animation and save it
ani = FuncAnimation(f,animate,frames=PageRank_of_G,interval=1000,blit=True)
ani.save("PageRank_Graph.gif", writer="pillow")

