# meta_learning
Implementation of Meta Learning Algorithm


1. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks(![Paper](https://arxiv.org/abs/1703.03400))
   
   This paper proposed simple, but effective gradient-based meta-learning algorithms called MAML. MAML tries to train model's initial parameters such that the model has a maximal performance on a new task after the parameters have been updated through one or more gradient steps computed with a small amount of data from that new task.

   Formally, the meta-objective of MAML algorithm can be written as below:
   $$
   \theta^{*} = \argmin_{\theta} \sum_{\mathcal{T}_{i} \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_{i}}(\theta-\alpha\nabla_{\theta}\mathcal{L}(\theta;\;\mathcal{D}_{i}^{tr});\;\mathcal{D}_{i}^{ts})
   $$
   We optimize the meta-objective via gradient descent. Although this requires an additional backward pass, it is supported by Pytorch or Tensorflow

   Below figures show the results of MAML compared to pure pretrained model in regression and reinforcement learning, which is almost same dataset and environment proposed in the paper. As shown in figures, MAML can learn quickly to adapt new task with a small number of gradient update than pure pretrained model.

   - Regression 
   ![](MAML/regression/results/MAML.png)    ![](MAML/regression/results/pretrained.png)
   
   - Reinforcement Learning
   ![](MAML/reinforcement_learning/results/MAML.png)    ![](MAML/reinforcement_learning/results/pretrained.png)