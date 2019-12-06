---
layout: post
title: Graph Neural Networks
---

In the previous section, we have learned how to represent a graph using "shallow" encoders. Those techniques give us a powerful expression over a graph in a vector space, but there are limitations. In this section, we will explore several different approaches using graph neural networks.


## Limitations of "Shallow Encoders"
* Scalability: every node has its own embeddings
* Inherently Transductive: it cannot generate embeddings for unseen nodes.
* Node Feature Excluded: it cannot leverage node features.
* Not Task Specific: it cannot be generalized to train with different loss function.

Fortunately, the above limitations can be solved by using graph neural networks.

## Graph Convolutional Networks (GCN)

Traditionally, neural network are designed for fixed-sized graphs. For example, we could consider a image as fixed-size graph or text as a line graph. However, most of the graphs in the real world has arbitrary size and complex topological structure. Therefore, we need to define the computation graph of GCN differently.

### Setup
Given $$G = (V, A, X)$$ be a graph such that:
* $$V$$ is the vertex set
* $$A$$ is the adjacency matrix
* $$X\in \mathbb{R}^{m\times\rvert V \rvert}$$ is the node feature matrix 

### Computation Graph and Generalized Convolution
![aggregate_neighbors](../assets/img/aggregate_neighbors.png?style=centerme)
Suppose $$G$$ is the graph in the above figure on the left, our goal is to define a computation graph of GCN with convolution. The GCN should keep the structure of the graph and incorporate the neighboring features. For example, if we want to create an embedding for node $$A$$, we can aggregate the information from its neighbour: $$B, C, D$$.
The aggregation (little boxes) needs to be **order invariant** (max, average, etc.). 
The computation graph for all the nodes in the graph with two layers deep will look like the following:
![computation_graph](../assets/img/computation_graph.png?style=centerme)
Notice that every node defines a computation graph based on its neighbors. In particular, the computation graph for node $$A$$ can be viewed as the following:
![computation_graph_for_a](../assets/img/computation_graph_for_a.png?style=centerme)
Layer-0 is the input layer with node feature $$X$$. In each layer, GCN combines the node features and transform them into some hidden representations.

### Deep Encoders
With the above idea, here is the mathematical expression at each layer using the average aggregation function:
* at 0th layer: $$h^0_v = x_v$$, this is the node feature
* at kth layer: $$ h_v^{K} = \sigma(W_k\sum_{u\in N(v)}\frac{h_u^{k-1}}{\rvert N(v)\rvert} + B_kh_v^{k-1}), \forall k \in \{1, .., K\}$$,

 $$h_v^{k-1}$$ is the embedding from the previous layer, $$\sigma$$ is the activation function (e.g. ReLU), $$W_k, B_k$$ are the trainable parameters, and $$\rvert N(v) \rvert$$ are the neighbours of node $$v$$.

* output layer: $$z_v = h_v^{k}$$, this is the final embedding for after $$k$$ layers

Equivalently, these computation can be written in a vector form: $$ H^{l+1} = \sigma(H^{l}W_0^{l} + \tilde{A}H^{l}W_1^{l}) $$ such that $$\tilde{A}=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$$


### Training the Model
We can feed these embeddings into any loss functions and run stochastic gradient descent to train the parameters.
For example, for a binary classification task, we can define the loss function as:

$$L = \sum_{v\in V} y_v \log(\sigma(z_v^T\theta)) + (1-y_v)\log(1-\sigma(z_v^T\theta))$$

Here, $$y_v$$ is the node class label, $$z_v$$ is the encoder output, $$\theta$$ is the classification weight, and $$\sigma$$ can be the sigmoid function.
In addition, we can also train the model in an unsupervised manner by using: random walk, graph factorization, node proximity, etc.

### Inductive Capability
GCN can also be generalized to unseen nodes in a graph. For example, if the model is trained using nodes $$A, B, C$$, the newly added nodes $$D,E,F$$ can also be evaluated since all the parameters are share crossed all nodes.
![apply_to_new_nodes](../assets/img/apply_to_new_nodes.png?style=centerme)


## GraphSage
So far we have explored a simple neighbourhood aggregation methods, but we can also generalize the aggregations in the following form:

$$ h_v^{K} = \sigma([W_k AGG(\{h_u^{k-1}, \forall u \in N(v)\}), B_kh_v^{k-1}])$$

For a node $$v$$, we can apply different aggregation methods to the neighbors using other aggregation functions ($$AGG$$), then concatenating the features with the target node itself.
Here are some commonly used aggregation functions:
* Mean: Take a weighted average of neighbors

 $$AGG = \sum_{u\in N(v)} \frac{h_u^{k-1}}{\rvert N(v) \rvert}$$

* Pooling: Transform neighbor vectors and apply symmetric vector function ($$\gamma$$ can be element-wise mean or max)

 $$AGG = \gamma(\{ Qh_u^{k-1}, \forall u\in N(v)\})$$

* LSTM: Apply LSTM to reshuffled neighbors

 $$AGG = LSTM(\{ h_u^{k-1}, \forall u\in \pi(N(v)\}))$$

## Graph Attention Network
What happened if we want to incorporate different weights for different nodes? Maybe some nodes can express more important information than others.

Let $$\alpha_{vu}$$ be the weighting factor (importance) of node $$u$$'s message to node $$v$$. Previously, in the average aggregation, we defined $$\alpha=\frac{1}{\rvert N(v) \rvert}$$, but we can also explicitly define $$\alpha$$ based on the structural property of a graph.

### Attention Mechanism
Let $$\alpha_{uv}$$ be computed as a byproduct of an attention mechanism $$a$$ where
$$a$$ computes the attention coefficients $$e_{vu}$$ across pairs of nodes $$u, v$$ based on their messages:

$$e_{vu} = a(W_kh_u^{k-1}, W_kh)v^{k-1}$$

$$e_{vu}$$ indicates the importance of node $$u$$'s message to node $$v$$. Then, we can normalize the coefficients using the softmax in
order to compare importance across different neighborhoods:

$$\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k\in N(v)}\exp(e_{vk})}$$

Therefore, we have:

$$h_{v}^k = \sigma(\sum_{u\in N(v)}\alpha_{vu}W_kh^{k-1}_u)$$

Notice that this approach is agnostic to the choice of $$a$$ and parameters of $$a$$ can be trained jointly.

## Reference
Here is a list of useful references:

**Tutorials and Overview:**
* [Relational inductive biases and graph networks (Battaglia et al., 2018)](https://arxiv.org/pdf/1806.01261.pdf)
* [Representation learning on graphs: Methods and applications (Hamilton et al., 2017)](https://arxiv.org/pdf/1709.05584.pdf)

**Attention-based Neighborhood Aggregation:**
* [Graph attention networks (Hoshen, 2017; Velickovic et al., 2018; Liu et al., 2018)](https://arxiv.org/pdf/1710.10903.pdf)

**Embedding the Entire Graphs:**
* Graph neural nets with edge embeddings ([Battaglia et al., 2016](https://arxiv.org/pdf/1806.01261.pdf); [Gilmer et. al., 2017](https://arxiv.org/pdf/1704.01212.pdf))
* Embedding entire graphs ([Duvenaud et al., 2015](https://dl.acm.org/citation.cfm?id=2969488); [Dai et al., 2016](https://arxiv.org/pdf/1603.05629.pdf); [Li et al., 2018](https://arxiv.org/abs/1803.03324)) and graph pooling
([Ying et al., 2018](https://arxiv.org/pdf/1806.08804.pdf), [Zhang et al., 2018](https://arxiv.org/pdf/1911.05954.pdf))
* [Graph generation](https://arxiv.org/pdf/1802.08773.pdf) and [relational inference](https://arxiv.org/pdf/1802.04687.pdf) (You et al., 2018; Kipf et al., 2018)
* [How powerful are graph neural networks(Xu et al., 2017)](https://arxiv.org/pdf/1810.00826.pdf)

**Embedding Nodes:**
* Varying neighborhood: [Jumping knowledge networks Xu et al., 2018)](https://arxiv.org/pdf/1806.03536.pdf), [GeniePath (Liu et al., 2018](https://arxiv.org/pdf/1802.00910.pdf)
* [Position-aware GNN (You et al. 2019)](https://arxiv.org/pdf/1906.04817.pdf)

**Spectral Approaches to Graph Neural Networks:**
* [Spectral graph CNN](https://arxiv.org/pdf/1606.09375.pdf) & [ChebNet](https://arxiv.org/pdf/1609.02907.pdf) [Bruna et al., 2015; Defferrard et al., 2016)
* [Geometric deep learning (Bronstein et al., 2017; Monti et al., 2017)](https://arxiv.org/pdf/1611.08097.pdf)

**Other GNN Techniques:**
* [Pre-training Graph Neural Networks (Hu et al., 2019)](https://arxiv.org/pdf/1905.12265.pdf)
* [GNNExplainer: Generating Explanations for Graph Neural Networks (Ying et al., 2019)](https://arxiv.org/pdf/1903.03894.pdf)

