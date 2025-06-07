<h1> Neural Networks from Scratch - My Learning Attempt</h1>

<p>
  This project is a hands-on implementation of a basic neural network, built entirely with <code>NumPy</code>, and trained on the classic spiral dataset. My goal is to better understand how neural networks work under the hood by writing all key components myself — no high-level frameworks like TensorFlow or PyTorch.
</p>

<h2> What's Inside</h2>
<ul>
  <li>Fully connected (dense) layers</li>
  <li>Activation functions (ReLU and Softmax)</li>
  <li>Categorical cross-entropy loss</li>
  <li>Backpropagation and gradient calculation</li>
  <li>Stochastic gradient descent (SGD) with mini-batch training</li>
</ul>

<h2> Dataset</h2>
<p>
  The network is trained on a non-linearly separable <code>spiral_data</code> set (from the <code>nnfs</code> library), which is great for testing nonlinear decision boundaries.
</p>

<h2> Learning Goals</h2>
<ul>
  <li>Understand the structure of a feedforward neural network</li>
  <li>Implement each part manually (no magic boxes!)</li>
  <li>Gain intuition about gradients, loss functions, and weight updates</li>
  <li>Experiment with batch sizes, learning rates, and accuracy evaluation</li>
</ul>

<h2> Results</h2>
<p>
  After training, the model achieves over <strong>96% accuracy</strong> on the spiral dataset, demonstrating that even simple neural networks can learn complex patterns with the right architecture and training process.
</p>


<pre><code>pip install nnfs
python neural.py
</code></pre>

<hr>
