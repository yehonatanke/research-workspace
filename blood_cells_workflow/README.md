[VERSION 2.3]<br>

Machine learning techniques for blood cells analysis research. It includes various models implemented using PyTorch and PyTorch Lightning, and provides utilities for training, evaluating, and visualizing the performance of these models.

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-title">Project Title</a>
    </li>
    <li>
      <a href="#table-of-contents">Table of Contents</a>
    </li>
    <li>
      <a href="#description">Description</a>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
    </li>
    <li>
      <a href="#models">Models</a>
      <ul>
        <li><a href="#modelv0">ModelV0</a></li>
        <li><a href="#modelv1">ModelV1</a></li>
        <li><a href="#modelresnet18">ModelResNet18</a></li>
        <li><a href="#modeleffnetv2">ModelEffNetV2</a></li>
        <li><a href="#convolutionalnetwork">ConvolutionalNetwork</a></li>
        <li><a href="#trainermodel8">TrainerModel8</a></li>
        <li><a href="#modelcfw">ModelCFW</a></li>
      </ul>
    </li>
    <li>
      <a href="#training-and-evaluation">Training and Evaluation</a>
    </li>
    <li>
      <a href="#visualization">Visualization</a>
    </li>
  </ol>
</details>

<details>
  <summary>Models</summary>
  <ul>
    <li>
      <a href="#modelv0">ModelV0</a>: Fedforward neural network.
    </li>
    <li>
      <a href="#modelv1">ModelV1</a>: A feedforward neural network with multiple hidden layers.
    </li>
    <li>
      <a href="#modelresnet18">ModelResNet18</a>: A ResNet-18 model.
    </li>
    <li>
      <a href="#modeleffnetv2">ModelEffNetV2</a>: An EfficientNet V2 model.
    </li>
    <li>
      <a href="#convolutionalnetwork">ConvolutionalNetwork</a>: A custom convolutional neural network.
    </li>
    <li>
      <a href="#trainermodel8">TrainerModel8</a>: A trainer class for EfficientNet V2.
    </li>
    <li>
      <a href="#modelcfw">ModelCFW</a>: A custom convolutional neural network with different transformations.
    </li>
  </ul>
</details>

## Structure

```
blood-cells-analysis/
├── archive/
│   └── run_func.py
├── src/
│   ├── models/
│   │   ├── model0.py
│   │   ├── model1.py
│   │   ├── model5.py
│   │   ├── model6.py
│   │   ├── model_v8.py
│   │   └── modelCFW.py
│   ├── train_util/
│   │   └── train.py
│   ├── utilities/
│   │   └── plot_util.py
├── func_testing.py
├── main.py
└── requirements.txt
```


### ModelV0

**Architecture:**
- Input Layer: $\mathbf{x} \in \mathbb{R}^{n}$
- Hidden Layer: Fully connected layer with $h$ units
- Output Layer: Fully connected layer with $c$ units (number of classes)

1. **Input Layer:**
   $$\mathbf{x} \in \mathbb{R}^{n}$$
   where $ n $ is the number of input features.

2. **Hidden Layer:**
   $$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
   where:
   - $\mathbf{W}_1 \in \mathbb{R}^{h \times n}$ is the weight matrix.
   - $\mathbf{b}_1 \in \mathbb{R}^{h}$ is the bias vector.
   - $\sigma$ is the activation function (e.g., ReLU).

3. **Output Layer:**
   $$\mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$
   where:
   - $\mathbf{W}_2 \in \mathbb{R}^{c \times h}$ is the weight matrix.
   - $\mathbf{b}_2 \in \mathbb{R}^{c}$ is the bias vector.
### ModelEffNetV2

**Architecture:**
- Base Model: EfficientNetV2-S
- Custom Block: Fully connected layers

1. **Base Model:**
   $$\mathbf{f} = \text{EfficientNetV2-S}(\mathbf{x})$$
   where $\mathbf{f} \in \mathbb{R}^{1280}$ is the feature vector from the base model.

2. **Custom Block:**
   $$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{f} + \mathbf{b}_1)$$
   $$\mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$
   where:
   - $\mathbf{W}_1 \in \mathbb{R}^{128 \times 1280}$
   - $\mathbf{b}_1 \in \mathbb{R}^{128}$
   - $\mathbf{W}_2 \in \mathbb{R}^{c \times 128}$
   - $\mathbf{b}_2 \in \mathbb{R}^{c}$

### ConvolutionalNetwork

**Architecture:**
- Convolutional Layers: Two convolutional layers followed by max-pooling
- Fully Connected Layers: Three fully connected layers

1. **Convolutional Layers:**
   $$\mathbf{h}_1 = \sigma(\text{Conv2D}_1(\mathbf{x}))$$
   $$\mathbf{h}_2 = \sigma(\text{Conv2D}_2(\mathbf{h}_1))$$
   where $\text{Conv2D}_i$ represents the $i$-th convolutional layer.

2. **Fully Connected Layers:**
   $$\mathbf{h}_3 = \sigma(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3)$$
   $$\mathbf{h}_4 = \sigma(\mathbf{W}_4 \mathbf{h}_3 + \mathbf{b}_4)$$ 
   $$\mathbf{y} = \mathbf{W}_5 \mathbf{h}_4 + \mathbf{b}_5$$
   where:
   - $\mathbf{W}_i$ and $\mathbf{b}_i$ are the weights and biases of the fully connected layers.

