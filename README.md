# GraphEmbeding
<i> Illustration of graph embedding techniques as applied to a specific data set </i>

Graphs are playing an ever important role in various areas of data science. The building blocks of graphs are nodes and edges (or links). There exists a number of different formats to store graph data. They all fall in the category of semi-structured data. 

Graphs can be used for visualization of complex data, for logical reasoning, for machine learning (ML) and deep learning (DL). In order to be used for machine and deep learning the graph data have to be converted to the format the ML and DL algorithms 'understand'. This format consists of rows and columns of numbers, commonly known as numerical matrices. The conversion from the node-edge to a matrix representation is not trivial and not unique. I.e. there are multiple ways to convert the way graph to a numerical matrix. The benefits and downsides of each method depend on the application area. 

The number of different ways to encode, or embed, graphs is constantly growning. It is not a goal of this repository  to present a complete, or even imcomplete list of graph embeddings developed up-to-date. Rather, the goal is to illustrate application of several graph embedding algorithms to one graph dataset. 

## Graph Dataset
We use the  [StreamSpot] data set

[StreamSpot]: https://github.com/sbustreamspot/sbustreamspot-data
The StreamSpot dataset is composed of 600 provenance graphs derived from 5 benign and 1 attack scenarios. 
See the [StreamSpot Explore Data.ipynb notebook](https://github.com/CyberActive/GraphEmbeding/blob/main/StreamSpot%20Explore%20Data.ipynb) for a detailed analysis of the dataset

It is important to visualize data before doing any ML. This falls in the category if descriptive analytics. 

The size of each graph is in tens and hundreds of thousands of edges; it is too big to visualize. Moreover, each graph is really a time progression, the processes captured there didn't happen at the same time. Therefore it is only reasonable to split each graph in a set of subgraphs for the purpose of visualization and for any further analytics, including clustering. We can see the activities in each of the 600 graphs can be split roughlty into several distinct types:  
- <img src="images/type1.png" height="40" /> a process is performing single I/O operations with a large number of files 
- <img src="images/type2.png" height="40" /> a process is performing multiple I/O operations with a small number of files, stin, and stdout
- <img src="images/type3.png" height="40" /> several processes and threads are performing I/O operations with several potentially overlapping files, stin, and stdout
- <img src="images/type4.png" height="40" /> a process performin single I/O operations witha large number of NA objects

Not all subgraphs fall cleanly into one the above categories, but we feel that they can used as a basis for assessing how well clusterin works. 

## Graph Embedding

Graph embedding aims to create a numerical representation of graphs in order to achieve some specific goals. Some practical applications of graph embeddin are
- Link prediction
- Anomaly detection
- Graph clustering

There are a number of different graph embedding algorithms 
- [node2vec](https://github.com/aditya-grover/node2vec)

Here we compare node2vec embedding with CyberActive open-sourced embedding. Our embedding is designed for dynamic graphs, such as StreamSpot. It applies batch processing by splitting a large and potentially 'infinite' streaming graph data into small subgraphs with the edges, which represent timestamped events, being in a close time proximity. Using this approach we account for both the structural and temporal proximity of different nodes in the graph.

What is presented here in no way represent a rigorous analysis and comparison of two graph embedding technique. But it is a usefull and practial introduction to such an analysis. Another benefit of this analysis is visualization of subgraphs in the StreamSpot dataset, which to the best of our knowledge has not been done anywhere else. 
