import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import display, Markdown
from node2vec import Node2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.cm as cm
from sklearn.metrics import  silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

def myprint():
    print (me)
def print_dict(mydict, n_per_line):
    n_lines = int(len(mydict)/n_per_line) + 1
    for i in range(n_lines):
        text1 = ''
        for item in list(node_edge_map.items())[i*n_per_line:(i+1)*n_per_line]:
            #print ('    ',item[0],': ', item[1], end =" ")
            text1 +=  item[0] + ': &nbsp;' + item[1] + '&emsp; &emsp;&emsp;&emsp;'
        display(Markdown('<b>' + text1 + '</b>'))

def create_graph_from_df(DF):
    G = nx.Graph()
    G.add_nodes_from(list(zip(DF['source_id'], [{'node_type':v} for v in DF['source_type']])))
    G.add_nodes_from(list(zip(DF['destination_id'], [{'node_type':v} for v in DF['destination_type']])))
    for edge_type in DF.edge_type.unique():
        dff = DF[DF.edge_type == edge_type]
        G.add_edges_from(list(zip(dff['source_id'], dff['destination_id'])), edge_type=edge_type)
    return G

def create_multigraph_from_df(DF):
    G = nx.MultiGraph()
    G.add_nodes_from(list(zip(DF['source_id'], [{'node_type':v} for v in DF['source_type']])))
    G.add_nodes_from(list(zip(DF['destination_id'], [{'node_type':v} for v in DF['destination_type']])))
    for edge_type in DF.edge_type.unique():
        dff = DF[DF.edge_type == edge_type]
        G.add_edges_from(list(zip(dff['source_id'], dff['destination_id'])), edge_type=edge_type)
    return G

def visualize_subgraphs(graph_id, sub_graph_size, shift, max_number_subgraphs, label, ncols = 4):
    graph_df = pd.read_parquet('./data/part.0.parquet', filters = [('graph_id','==', graph_id)])
    graph_df.reset_index(drop=True, inplace=True)
    graph_df.dropna(inplace=True)
    graph_size = len(graph_df)

    graphs = []
    node_colors = []
    number_subgraphs = int(graph_size/sub_graph_size)
    number_subgraphs = min(max_number_subgraphs, number_subgraphs)
    for start in range(1,number_subgraphs+1):
        sub_chunk = graph_df[start * shift : start * shift + sub_graph_size]

        subG = create_multigraph_from_df(sub_chunk)    
        graphs.append(subG)
        colors = [color_map[x] for x in list(nx.get_node_attributes(subG,'node_type').values())]
        node_colors.append(colors)

    image_size = 24/ncols
    nrows = int(number_subgraphs/ncols)
    xsize = 24
    ysize = image_size * nrows
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(xsize, ysize)
    ax = axes.flatten()

    for i in range(number_subgraphs):
        pos = nx.kamada_kawai_layout(graphs[i])
        nx.draw(graphs[i], pos, node_size = 50, width = 0.2, node_color = node_colors[i], ax=ax[i])
        ax[i].set_axis_off()
        offset = str((i+1) * shift) 
        ax[i].set_title(label + " offset: " + offset)

    plt.show()    

graph_type_map = {}
graph_types = ['YouTube', 'GMail', 'VGame', 'Attack', 'Download', 'CNN']
for i, gtype in enumerate(graph_types):
    graph_type_map.update(dict(zip(np.linspace(i*100,i*100+99,100), 100*[gtype])))

def display_silhouette(silhouette_values, kmeans_labels, n_clusters, data_size):
    plt.figure(figsize = (10,3))
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    plt.xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, data_size + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_values[kmeans_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
            # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    plt.show()
def visualize_graphs(allgraphs, allcolors, indices, layout):
    node_colors = [allcolors[ind] for ind in indices]
    graphs = [allgraphs[ind] for ind in indices]

    image_size = 24/layout[0]
    fig, axes = plt.subplots(nrows = layout[1], ncols = layout[0])
    xsize = image_size * layout[0]
    ysize = image_size * layout[1]
    fig.set_size_inches(xsize, ysize)
    ax = axes.flatten()
    for a in ax:
        a.set_axis_off()
    for i, subG in enumerate(graphs):
        pos = nx.kamada_kawai_layout(subG)
        nx.draw(subG, pos, node_size = 50, width = 0.2, node_color = node_colors[i], ax=ax[i])
        #ax[i].set_axis_off()
        
    plt.show()

def dataframe2graphs(data_df, sentence_size):

    all_graphs = []
    node_colors = []
    number_sent = int(data_df.shape[0]/sentence_size)
    for start in range(number_sent):
        sub_chunk = data_df[start*sentence_size:start*sentence_size+sentence_size]
        subG = create_multigraph_from_df(sub_chunk)
        colors = [color_map[x] for x in list(nx.get_node_attributes(subG, 'node_type').values())]
        all_graphs.append(subG)
        node_colors.append(colors)
        
    return all_graphs, node_colors

def dataframe2sentences(data_df, sentence_size):

    all_sentences = []
    number_sent = int(data_df.shape[0]/sentence_size)
    for start in range(number_sent):
        sub_chunk = data_df[start*sentence_size:start*sentence_size+sentence_size]
        sent = []
        for _, row in sub_chunk[:sentence_size].iterrows():
            word = list(row[['source_type','edge_type', 'destination_type']])
            word = '_'.join(word)
            sent.append(word)
        all_sentences.append( sent)
        
    return all_sentences

def sentence2doc2vec(sentences, epochs):
    textLabeled = []
    for textID, sent in enumerate(sentences):
        textL = TaggedDocument(sent, tags = ['text_%s' %textID])
        textLabeled.append(textL)

    doc2vec_model = Doc2Vec(vector_size=1024, window=10, dm = 0, dbow_words = 0,
                            min_count=2,  alpha=0.025,  min_alpha=0.025, workers = 8)
    doc2vec_model.build_vocab(textLabeled)


    for epoch in range(epochs):
        doc2vec_model.train(textLabeled, total_examples=doc2vec_model.corpus_count, epochs=1)
        doc2vec_model.alpha -= 0.0001  # decrease the learning rate
        doc2vec_model.min_alpha = doc2vec_model.alpha  # fix the learning rate, no decay    
        
    return doc2vec_model.dv.vectors

def applyKMeans(Vect, n_components, verbose):
    km = KMeans(n_clusters=n_components, max_iter=10000)
    km.fit(Vect)
    sample_silhouette_values = silhouette_samples(Vect, km.labels_)
    if verbose:
        print (silhouette_score(Vect, km.labels_, sample_size=10000), np.mean(sample_silhouette_values))
    for i in range(n_components):
        ith_cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
        if verbose:
            print (i, np.mean(ith_cluster_silhouette_values), np.min(ith_cluster_silhouette_values),
                   sum(ith_cluster_silhouette_values < 0)/len(ith_cluster_silhouette_values))    
    display_silhouette(sample_silhouette_values, km.labels_, n_components, Vect.shape[0])
    return sample_silhouette_values, km

def process_data(graph_id, n_components, sentence_size = 100, epochs =1, verbose = False, datafile = './data/part.0.parquet'):
    
    panda_df = pd.read_parquet(datafile, filters = [('graph_id','==', graph_id)])
    panda_df.reset_index(drop=True, inplace=True)
    panda_df.dropna(inplace=True)
    panda_df['graph_type'] = panda_df.graph_id.map(lambda x: graph_type_map[x])
    graph_size = len(panda_df)

    all_graphs, node_colors = dataframe2graphs(panda_df, sentence_size)
    all_sentences= dataframe2sentences(panda_df, sentence_size)

    textVect = sentence2doc2vec(all_sentences, epochs)
    sample_silhouette_values, km = applyKMeans(textVect, n_components, verbose)
    
    return textVect, sample_silhouette_values, km, all_graphs, node_colors

def dataframe2node2vec(data_df, node2vec_size, sub_graph_size):

    graph_size = len(data_df)
    all_node2vec = []
    real_graph_size = 0
    number_subgraphs = int(graph_size/sub_graph_size)
    for start in range(number_subgraphs):
        sub_chunk = data_df[start*sub_graph_size:start*sub_graph_size+sub_graph_size]
  
        sub_G = create_graph_from_df(sub_chunk)
        real_graph_size += sub_G.size()

        n2v = Node2Vec(sub_G, dimensions=node2vec_size, walk_length=16, num_walks=20, workers=4, quiet=True)
        mod = n2v.fit(window=4, min_count=0)      
        edge_vec = graph2vec_edge_arithmetic(sub_G, mod)
        all_node2vec.append(edge_vec)

    return np.array(all_node2vec)

def process_data_node2vec(graph_id, n_components, node2vec_size = 20, sub_graph_size = 100, verbose = False, datafile = './data/part.0.parquet'):
    
    panda_df = pd.read_parquet(datafile, filters = [('graph_id','==', graph_id)])
    panda_df.reset_index(drop=True, inplace=True)
    panda_df.dropna(inplace=True)
    panda_df['graph_type'] = panda_df.graph_id.map(lambda x: graph_type_map[x])
    graph_size = len(panda_df)

    all_graphs, node_colors = dataframe2graphs(panda_df, sub_graph_size)
    textVect = dataframe2node2vec(panda_df, node2vec_size, sub_graph_size)

    sample_silhouette_values, km = applyKMeans(textVect, n_components, verbose)
    
    return textVect, sample_silhouette_values, km, all_graphs, node_colors

node_edge_map = {
     'a': 'process',
     'b': 'thread',
     'c': 'file',
     'd': 'MAP_ANONYMOUS',
     'e': 'NA',
     'f': 'stdin',
     'g': 'stdout',
     'h': 'stderr',
     'i': 'accept',
     'j': 'access',
     'k': 'bind',
     'l': 'chmod',
     'm': 'clone',
     'n': 'close',
     'o': 'connect',
     'p': 'execve',
     'q': 'fstat',
     'r': 'ftruncate',
     's': 'listen',
     't': 'mmap2',
     'u': 'open',
     'v': 'read',
     'w': 'recv',
     'x': 'recvfrom',
     'y': 'recvmsg',
     'z': 'send',
     'A': 'sendmsg',
     'B': 'sendto',
     'C': 'stat',
     'D': 'truncate',
     'E': 'unlink',
     'F': 'waitpid',
     'G': 'write',
     'H': 'writev'}

color_map = {'a': 'red', 'b':'orange', 'c':'green', 'f':'yellow', 'g':'blue', 'h':'brown', 'd':'black', 'e':'black'}

graph_type_color_map = {'Attack': 'red', 'CNN':'orange', 'GMail':'green', 'YouTube':'yellow', 'VGame':'blue', 'Download':'brown'}

heading_properties = [('font-size', '18px')]
cell_properties = [('font-size', '16px'), ('width', '300px')]
caption_properties = [("text-align", "center"), ("font-size", "20px"), ( "font-weight", "bold"), ("color", 'blue')]

dfstyle = [dict(selector="th", props=heading_properties),
     dict(selector="td", props=cell_properties),
     dict(selector="caption", props=caption_properties)]

def graph2vec_edge_arithmetic(graph: nx.Graph, n2vmodel):
    ## CREDITS: https://www.kaggle.com/code/tangodelta/api-call-graph-analytics/notebook
    # Graphs can have multiple connected components. Sometimes there can be trivial connected components, 
    # meaning they are just nodes eithout any edges.
    # In cases where we have trivial connected components we perform only node addition.
    # For non-trivial connected components each edge is taken. embeddings for the nodes in the edge are multiplied to get an edge vector.
    # These edge vectors are summed to get the vector for the connected component
    
    sum_vec = None
    count = 0
    
    # for non trivial connected components
    for e in list(graph.edges):
        edgevec = np.multiply(n2vmodel.wv.get_vector(str(e[0])),n2vmodel.wv.get_vector(str(e[1])))
        norm = np.linalg.norm(edgevec)
        edgevec = edgevec if norm == 0 else edgevec/norm
        sum_vec = edgevec if sum_vec is None else np.add(sum_vec, edgevec)
        count += 1
    # getting all nodes that have a zero degree. These nodes a part of the trivial connected components
    node_degrees = list(map(lambda x: (x, graph.degree(x)), list(graph.nodes)))
    node_zero_degrees = list(filter(lambda x: x[1] == 0 , node_degrees))
    for (n, d) in node_zero_degrees:
        nodevec = n2vmodel.wv.get_vector(str(n))
        norm = np.linalg.norm(nodevec)
        nodevec = nodevec if norm == 0 else nodevec/norm
        sum_vec = nodevec if sum_vec is None else np.add(sum_vec, nodevec)
        count += 1
    try:
        graph_vector = sum_vec/count
        return graph_vector 
    except:
        print(sum_vec, count, len(graph.nodes), len(graph.edges))
        return None