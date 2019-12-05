---
layout: post
title: Graph Neural Networks
---

In the previous section, we have learned how to represent a graph using "shallow" encoders. Although those techniques give us a powerful expression of the graph in a vector space there are some limitations. In this section, we will present several different approaches using deep encoder to represent the graph. In this section, we will explore how to embed a graph through multiple layers of non-linear transformations of a graph structure.


## Limitations of "Shallow Encoder"
* $$O(\rvert V \rvert)$$ parameters are needed
 
 The number of parameters we estimate equals the number of nodes. There are is no sharing of parameters between nodes and every node has its own unique embedding.   

* Inherently "transductive"

 It cannot generate embeddings for nodes that are not seen
during training, so for a new node arrives we need to embed the entire graph again.

* Do not incorporate node features

 It cannot incorporate node features and many graphs have features that we can and should
leverage.

* Embedding is not task-specific

 For example, in the case of classification, the embedding should obey link classification.

## Deep Graph Encoders using Graph Convolutional Networks

### Why is this hard?
In the traditional neural network are designed either for fixed-sized graphs. For example, we can consider images as fixed-size graphs, or text as a line graph and arbitrary size and complex topological structures. There is no fixed node ordering or reference point.

### Setup
Assume that we have a graph $$G$$ with the following properties:
* $$V$$ is the vertex set
* $$A$$ is the adjacency matrix
* $$X\in R^{m\times\rvert V \rvert}$$ is a matrix of node features
* Node features can be for example user profile, gene expressions, indicator vectors and so on

### Computation Graph and Generalize Convolution
We want to not only keep the structure of the network but also borrow the neighboring features. In order to achieve this, we can generate node embeddings based on local network neighborhoods. For example, if we want to create an embedding for node $$A$$, we can collect information from its neighbour: $$B, C, D$$ and so on. The following figure demonstrates a two-layer graph network.
![aggregate_neighbors](../assets/img/aggregate_neighbors.png?style=centerme)
The little boxes are some neural networks that aggregate neighboring features. The aggregation function needs to be **order invariant** (max, average and so on).
The computation graph for all the nodes in the input graph with two layers deep will look like the following:
![computation_graph](../assets/img/computation_graph.png?style=centerme)
We can see that every node defines a computation graph based on its neighbors. In particular, the computation graph for node $$A$$ can be viewed as the following:
![computation_graph_for_a](../assets/img/computation_graph_for_a.png?style=centerme)
Layer-0 is the input layer with node feature $$X$$, and in each layer, we will do an aggregation to combine the node features and create some hidden representation of nodes.
Also, notice that the concept of the number of layers here means the number of hops from the targeting node. In this example, we are looking at a graph neural network with two layers.

### Deep Encoder
In each layer we need to first aggregate the neighbour messages and apply a neural network, here we will see a simple example using the average aggregation function:
* 0th layer: $$h^0_v = x_v$$, this is just the node feature
* kth layer: $$ h_v^{K} = \sigma(W_k\sum_{u\in N(v)}\frac{h_u^{k-1}}{\rvert N(v)\rvert} + B_kh_v^{k-1}), \forall k \in \{1, .., K\}$$,

 $$h_v^{k-1}$$ is the embedding from the previous layer, $$\sigma$$ is the activation function (e.g. ReLU), $$W_k, B_k$$ are the trainable parameters, and $$\rvert N(v) \rvert$$ are the neighbours of node $$v$$

* $$z_v = h_v^{k}$$, this is the final embedding for after $$K$$ layers

Equivalently, these computation for be written in the vector so that its easier to implement in practice:

$$ H^{l+1} = \sigma(H^{l}W_0^{l} + \tilde{A}H^{l}W_1^{l}) $$ with $$\tilde{A}=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$$

### Training the Model
We can feed these embeddings into any loss functions and run stochastic gradient descent to train the weight parameters.
For example, in a binary classification we can define the loss function as:

$$L = \sum_{v\in V} y_v \log(\sigma(z_v^T\theta)) + (1-y_v)\log(1-\sigma(z_v^T\theta))$$

Where $$y_v$$ are the node class label, $$z_v$$ are the encoder outputs, $$\theta$$ is the classification weights, and $$\sigma$$ is sigmoid function.
In addition, we can also train the model in an unsupervised manner by using: random walk, graph factorization, node proximity and so on.

### Inductive Capability
Suppose we only trained the model using the nodes $$A, B, C$$, the newly added nodes $$D,E,F$$ can also be evaluated since the parameters are share crossed all nodes in the model.
![apply_to_new_nodes](../assets/img/apply_to_new_nodes.png?style=centerme)


## GraphSage
So far we have explored a simple neighbourhood aggregation methods, but we can also generalize the aggregations in the following form:

$$ h_v^{K} = \sigma([W_k AGG(\{h_u^{k-1}, \forall u \in N(v)\}) ,B_kh_v^{k-1}])$$

Where for a node $$v$$ we can apply different aggregation methods to the neighbors using $$AGG$$, then concatenate with itself.
For example, the aggregation function can be:
* Mean: Take a weighted average of neighbors

 $$AGG = \sum_{u\in N(v)} \frac{h_u^{k-1}}{\rvert N(v) \rvert}$$

* Pool: Transform neighbor vectors and apply symmetric vector function ($$\gamma$$ can be element-wise mean or max)

 $$AGG = \gamma(\{ Qh_u^{k-1}, \forall u\in N(v)\})$$

* LSTM: Apply LSTM to reshuffled of neighbors

 $$AGG = LSTM(\{ h_u^{k-1}, \forall u\in \pi(N(v)\}))$$

## Attention
What happened if we want to incorporate different weights for different nodes? For example, some nodes are more important than other nodes.

Let $$\alpha_{vu}$$ be the weighting factor (importance) of node $$u$$'s message to node $$v$$. Previously, in the average aggregation, we defined $$\alpha=\frac{1}{\rvert N(v) \rvert}$$, but we can also explicitly define $$\alpha$$ based on the structural property of the graph.

### Attention Mechanism
Let $$\alpha_{uv}$$ be computed as a byproduct of an attention mechanism $$a$$:
Let $$a$$ compute attention coefficients $$e_{vu}$$ across pairs of nodes $$u, v$$ based on their messages:

$$e_{vu} = a(W_kh_u^{k-1}, W_kh)v^{k-1}$$

where $$e_{vu}$$ indicates the importance of node $$u$$'s message to node $$v$$. Then normalize the coefficients using the softmax function in
order to be comparable across different neighborhoods:

$$\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k\in N(v)}\exp(e_{vk})}$$

Therefore we have:

$$h_{v}^k = \sigma(\sum_{u\in N(v)}\alpha_{vu}W_kh^{k-1}_u)$$

Notice that this approach is agnostic to the choice of $$a$$ and parameters of $$a$$ are trained jointly.

## Reference
Here is a list of useful references:

**Tutorials and overviews:**
* [Relational inductive biases and graph networks (Battaglia et al., 2018)](https://arxiv.org/pdf/1806.01261.pdf)
* [Representation learning on graphs: Methods and applications (Hamilton et al., 2017)](https://arxiv.org/pdf/1709.05584.pdf)

**Attention-based neighborhood aggregation:**
* [Graph attention networks (Hoshen, 2017; Velickovic et al., 2018; Liu et al., 2018)](https://arxiv.org/pdf/1710.10903.pdf)

**Embedding entire graphs:**
* Graph neural nets with edge embeddings ([Battaglia et al., 2016](https://arxiv.org/pdf/1806.01261.pdf); [Gilmer et. al., 2017](https://arxiv.org/pdf/1704.01212.pdf))
* Embedding entire graphs ([Duvenaud et al., 2015](https://dl.acm.org/citation.cfm?id=2969488); [Dai et al., 2016](https://arxiv.org/pdf/1603.05629.pdf); [Li et al., 2018](https://arxiv.org/abs/1803.03324)) and graph pooling
([Ying et al., 2018](https://arxiv.org/pdf/1806.08804.pdf), [Zhang et al., 2018](https://arxiv.org/pdf/1911.05954.pdf))
* [Graph generation](https://arxiv.org/pdf/1802.08773.pdf) and [relational inference](https://arxiv.org/pdf/1802.04687.pdf) (You et al., 2018; Kipf et al., 2018)
* [How powerful are graph neural networks(Xu et al., 2017)](https://arxiv.org/pdf/1810.00826.pdf)

**Embedding nodes:**
* Varying neighborhood: [Jumping knowledge networks Xu et al., 2018)](https://arxiv.org/pdf/1806.03536.pdf), [GeniePath (Liu et al., 2018](https://arxiv.org/pdf/1802.00910.pdf)
* [Position-aware GNN (You et al. 2019)](https://arxiv.org/pdf/1906.04817.pdf)

**Spectral approaches to graph neural networks:**
* [Spectral graph CNN](https://arxiv.org/pdf/1606.09375.pdf) & [ChebNet](https://arxiv.org/pdf/1609.02907.pdf) [Bruna et al., 2015; Defferrard et al., 2016)
* [Geometric deep learning (Bronstein et al., 2017; Monti et al., 2017)](https://arxiv.org/pdf/1611.08097.pdf)

**Other GNN techniques:**
* [Pre-training Graph Neural Networks (Hu et al., 2019)](https://arxiv.org/pdf/1905.12265.pdf)
* [GNNExplainer: Generating Explanations for Graph Neural Networks (Ying et al., 2019)](https://arxiv.org/pdf/1903.03894.pdf)

