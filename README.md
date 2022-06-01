# GraphEmbeding
Illustration of graph embedding technique as applied to a specific data set

Graphs are playing an ever important role in various areas of data science. The building blocks of graphs are nodes and edges (or links). There exists a number of different formats to store graph data. They all fall in the category of semi-structured data. 

Graphs can be used for visualization of complex data, for logical reasoning, for machine learning (ML) and deep learning (DL). In order to be used for machine and deep learning the graph data have to be converted to the format the ML and DL algorithms 'understand'. This format consists of rows and columns of numbers, commonly known as numerical matrices. The conversion from the node-edge to a matrix representation is not trivial and not unique. I.e. there are multiple ways to convert the way graph to a numerical matrix. The benefits and downsides of each method depend on the application area. 

The number of different ways to encode, or embed, graphs is constantly growning. It is not a goal of this repository  to present a complete, or even imcomplete list of graph embeddings developed up-to-date. Rather, the goal is to illustrate application of several graph embedding algorithms to one graph data set. 

## Graph Data Set
We use the  [StreamSpot] data set

[StreamSpot]: https://github.com/sbustreamspot/sbustreamspot-data
This repository contains the `ALL` dataset, which includes edges from all the
600 benign and attack scenario graphs. 
   
   1. YouTube (graph ID's 0 - 99)
   2. GMail (graph ID's 100 - 199)
   3. VGame (graph ID's 200 - 299)
   4. Drive-by-download attack (graph ID's 300 - 399)
   5. Download (graph ID's 400 - 499)
   6. CNN (graph ID's 500 - 599)
