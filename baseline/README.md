## **Perceptron Learning Algorithm**

### Model Overview

**Multi-class Perceptron and One-vs-All Strategy**<br>
The perceptron algorithm, initially designed for binary classification, is extended to multi-class problems using a **one-vs-all strategy**:  
- For each class $k$, the perceptron distinguishes between samples of that class (positive) and all others (negative).  
- This creates $C$ binary classifiers (where $C$ is the number of classes).  
- During prediction, the classifier assigns a sample to the class with the highest score (dot product of input and weights).  

**Update Rule**<br>
The perceptron updates its weights when a sample is misclassified:  
$$w_{k} = w_{k} + y \cdot x$$

Where:  
- $w_{k}$: Weights for class $k$.  
- $y$: Label $+1$ for positive class, $-1$ otherwise.  
- $x$: Feature vector of the misclassified sample.

**Pocket Algorithm**<br>
To handle non-linearly separable cases:  
- The **Pocket Algorithm** retains the best-performing weights (based on error) during training, ensuring stability even when convergence isn't guaranteed.


### **Model Implementation**

#### Model Structure and Mechanisms
The `MulticlassPerceptron_V1` class is designed for flexibility, modularity, and performance. Here's an explanation of its key elements:

1. **Attributes**:  
   - `weights` and `pocket_weights` track current and best weights for each class.  
   - `training_errors` and `test_errors` log error rates during training for analysis.  

2. **Initialization Choices**:  
   - Weight initialization starts at zero for simplicity and reproducibility.  
   - The `max_iter` parameter controls the maximum training epochs, balancing training time and performance.  

3. **Vectorization**:  
   - Operations like dot products and error calculations are vectorized for efficiency, especially given the high dimensionality of MNIST data.  

4. **Binary Label Conversion**:  
   - The `_get_binary_labels` method ensures each class's binary classification is seamlessly handled.  

5. **Error Calculation**:  
   - Error rates are computed as the proportion of misclassified samples, allowing easy comparison and pocket weight updates.  


#### **Code Summary**

The `MulticlassPerceptron_V1` implementation follows these key steps:  

1. **Weight Initialization**:  
   - Both current and pocket weights are initialized, ensuring a baseline for comparison.  

2. **Training Loop**:  
   - Each epoch processes all classes, updating weights and errors as necessary.  
   - Pocket weights are updated only when a better solution is found.  

3. **Prediction**:  
   - During inference, the class with the highest score is selected, ensuring consistency with the multi-class approach.  

4. **Sensitivity Calculation**:  
   - Evaluates performance per class, aiding in understanding the model's strengths and weaknesses.


#### **Reasoning Behind Design Choices**

- **Pocket Algorithm**:
- **Error Logging**: Provides insights into model behavior during training and testing phases.  
- **Progress Indicators**: Nested progress bars offer detailed feedback.
- **Vectorized Operations**:
---

#### **Approach Summary**

The implementation reflects a balance between simplicity (e.g., zero initialization) and advanced features (e.g., Pocket Algorithm). The model prioritizes interpretability, efficiency, and adaptability to real-world scenarios, aligning with modern best practices in machine learning.
"""


## SoftMax
Softmax regression calculates the probability of each class by using the softmax function. The softmax function converts a vector of values into a probability distribution, where each value is between 0 and 1, and the sum of all values is 1.


The goal is to estimate the probability $P(y = k | x)$ for each class $k \in \{1, \dots, K\}$, where $K=10$ is the total number of classes. The Softmax function ensures that the predicted probabilities for each class sum to 1. The hypothesis of the model can be expressed as:

$$h(x) = \left( P(y = 1 | x), P(y = 2 | x), \dots, P(y = K | x) \right)$$

Where the probabilities are computed using the softmax function applied to the weighted input $x$.


 #### **Define the Softmax Regression Model**
In this model, we aim to minimize the softmax cost function, which is a cross-entropy loss function. The model calculates probabilities for each of the possible classes and selects the class with the highest probability. We will implement the gradient descent method to minimize the loss function.


### **Softmax Function and Loss Computation**
1. **Softmax Function**:
   - The softmax function transforms logits into probabilities:  
     $$h(x) = \frac{e^{w_k^T x}}{\sum_{j=1}^K e^{w_j^T x}}$$  
     where $w_k$ is the weight vector for class $k$, and $x$ is the input.

2. **Loss Calculation**:
   - The model minimizes the cross-entropy loss, defined as:
 $$E_{\text{in}}(w) = - \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K 1\{y_n = k\} \log P(y = k | x_n)$$
     where $P(y = k | x_n)$ is the predicted probability for class $k$, and $y_n$ is the true label.

3. **Gradient Calculation**:
   - The gradient of the loss with respect to the weights is computed as:  
     $$\nabla_w E_{\text{in}} = \frac{1}{N} \sum_{n=1}^N (P(y = k | x_n) - 1\{y_n = k\}) x_n^T$$

4. **Prediction**:
   - For a given input $x$, the predicted class is:  
     $$\hat{y} = \arg\max_k P(y = k | x)$$

The model is trained using mini-batch gradient descent, updating weights iteratively to minimize the loss function.

### **SoftmaxRegression Model Overview**

This class implements a softmax regression model using the MNIST dataset for digit classification. This model is based on the softmax function, which is commonly used for multi-class classification problems, such as digit recognition in this case. The model is trained using stochastic gradient descent (SGD) with mini-batch updates. It includes mechanisms for training, evaluation, and visualization of results, including accuracy, loss, and confusion matrix plots.

### **Model Implementation**

#### Model Structure and Mechanisms

The class defines the following key components:

1. **Initialization**:
   - `learning_rate`: Sets the learning rate for gradient descent.
   - `num_epochs`: Defines how many times the model will iterate over the entire dataset during training.
   - `batch_size`: Specifies the number of samples per batch during training.

2. **Data Loading and Preprocessing**:
   - The `load_mnist` function fetches and preprocesses the MNIST dataset, normalizing the pixel values to the range [0, 1] and adding a bias term. The dataset is split into training and testing sets.

3. **Model Mechanics**:
   - **One-Hot Encoding**: Converts the target labels into a one-hot encoded format.
   - **Softmax Function**: Converts raw model outputs (logits) into probability distributions for each class.
   - **Loss Calculation**: Uses the cross-entropy loss function to quantify the model's performance.
   - **Gradient Calculation**: Computes gradients for weight updates using backpropagation.
   - **Prediction**: The model predicts the class with the highest probability for each input.

4. **Training Process**:
   - The `train` method updates model weights using mini-batch gradient descent.
   - The loss for each epoch is computed, and the model's performance on the test set is evaluated after each epoch.

5. **Evaluation**:
   - **Accuracy**: The model's prediction accuracy is calculated.
   - **Confusion Matrix**: Generates a confusion matrix to assess performance for each class (digit).
   - **Sensitivity**: Calculates the sensitivity (True Positive Rate) for each class.

6. **Visualization**:
   - **Loss Curves**: The `plot_loss` method visualizes the training and test loss over epochs.
   - **Confusion Matrix**: The `plot_confusion_matrix` method displays a heatmap of the confusion matrix.

#### **Code Summary**

The `SoftmaxRegression` class includes:
- `__init__`: Initializes hyperparameters and model parameters.
- `load_mnist`: Loads and preprocesses the MNIST dataset.
- `one_hot_encode`: Converts labels to one-hot encoding.
- `softmax`: Applies the softmax function to logits.
- `compute_loss`: Computes the cross-entropy loss.
- `compute_gradient`: Computes the gradient for backpropagation.
- `predict`: Predicts the class for input data.
- `train`: Trains the model using mini-batch gradient descent.
- `calculate_accuracy`: Computes the accuracy of the model.
- `calculate_confusion_and_sensitivity`: Computes the confusion matrix and sensitivity for each class.
- `plot_confusion_matrix`: Plots the confusion matrix as a heatmap.
- `plot_loss`: Plots the training and test loss curves.
- `run_model`: Executes the full workflow, including training, prediction, evaluation, and visualization.

#### **Reasoning Behind Design Choices**

1. **Softmax for Multi-class Classification**:
   Softmax regression is chosen because it is a natural fit for multi-class classification problems, like digit recognition in MNIST, where each input belongs to one of ten possible classes.

2. **Stochastic Gradient Descent (SGD)**:
   The model uses SGD with mini-batches to balance computational efficiency and convergence speed. This allows the model to process large datasets like MNIST effectively.

3. **Cross-Entropy Loss**:
   Cross-entropy loss is used because it is the standard loss function for classification problems involving probabilities and softmax outputs.

4. **Evaluation with Confusion Matrix and Sensitivity**:
   The confusion matrix and sensitivity are used to evaluate how well the model performs for each class, providing more detailed insights into its strengths and weaknesses.

#### **Approach Summary**

This model follows a straightforward approach to multi-class classification using softmax regression. The training is done through mini-batch gradient descent, and the model is evaluated using a combination of accuracy, confusion matrix, and sensitivity metrics. The use of visualization tools for the loss curves and confusion matrix enhances the interpretability and understanding of the model’s performance.

**Note:**<br>
The `load_mnist` function is provided for convenience, but the data remains identical to that mentioned previously.

## Linear Regression

### **Thought Process:**

**Formulating the Problem as a Linear Regression Task:**<br>
The task at hand is the classification of handwritten digits, which traditionally represents a classification problem. However, linear regression is generally used for predicting continuous variables, not categorical ones. To adapt linear regression to this context, we frame it as a multi-output regression problem, where the objective is to predict the digit labels as numerical values.

A significant challenge arises because the output values are discrete integers (ranging from 0 to 9), differing from the continuous outputs in typical regression settings.

To address this, we:
  - Apply one-hot encoding to the target labels, transforming each digit into a binary vector where the position corresponding to the correct digit is set to 1 (e.g., the digit 1 becomes [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).
  - Conceptualize the problem as "multi-class regression," where the model predicts the one-hot encoded vector for each digit.
  - Represent each image as a feature vector consisting of 784 pixel intensities (one for each pixel).
  
**Evaluating Linear Regression on the Test Set:**<br>
  - Performance is assessed by how effectively the model predicts digit labels on the test set.
  - Since linear regression outputs continuous values, a thresholding approach is used to map the predictions back to discrete classes.
  - Evaluation metrics may include accuracy (as a classification metric) along with potential metrics such as mean squared error or cross-entropy loss (more common in regression tasks).
  - Given that linear regression is not ideal for categorical tasks, especially with the complexity and high dimensionality of handwritten digits, we expect its performance to be subpar.

### **Plan for Implementation:**

**Data Preparation:**
  - Load the MNIST dataset.
  - Flatten the 28x28 pixel images into 784-dimensional vectors.
  - One-hot encode the digit labels (0-9).

**Training the Linear Regression Model:**
  - Utilize the least squares method to train the linear regression model on the flattened image data.
  - The weight matrix is computed using the formula:
  $$W = (X^T X)^{-1} X^T y$$
  where $X$ represents the matrix of input images size  $N \times 784$, and $y$ is the matrix of one-hot encoded labels.

**Performance Evaluation:**<br>
  - Use the trained model to predict digit labels on the test set.
  - Convert the continuous predicted values to discrete labels by selecting the class with the highest predicted value.
  - Compute classification accuracy by comparing the predicted labels with the actual labels.

### **Linear Regression Approach:**

- **Advantages:**
   - Simplicity in both implementation and concept.
   - Fast training on small datasets.
  
- **Disadvantages:**
   - Linear regression is fundamentally unsuitable for classification tasks.
   - It presupposes a linear relationship between the features and the target, which is unrealistic for image data such as handwritten digits.
   - The model may fail to generalize effectively, leading to suboptimal accuracy when compared to more appropriate classification algorithms like logistic regression, support vector machines, or neural networks.

### **MulticlassLinearRegression - Model Overview**

- `self.weights`: Stores the learned weights after training.
- `self.training_errors`: Tracks errors during training (not actively used in the code).
- `self.test_errors`: Tracks errors during testing (not actively used in the code).

**Fit Method (`fit`)**:
- **Purpose**: This method trains the model using the normal equation for linear regression.
- **Formula**: The weights are computed using the formula:
  $$w = (X^T X)^{-1} X^T y$$
  Where $X$ is the feature matrix (size: $n_{\text{samples}} \times n_{\text{features}}$) and $y$ is the one-hot encoded label matrix (size: $n_{\text{samples}} \times n_{\text{classes}}$).
- **Procedure**:
  - It calculates the pseudo-inverse of $X^T X$, multiplies it by $X^T$, and then multiplies by the one-hot encoded labels $y$ to obtain the weight matrix `self.weights`.
   
3. **Predict Method (`predict`)**:
   - **Purpose**: This method makes predictions for the input data based on the learned weights.
   - **Procedure**:
     - Computes the scores by multiplying the feature matrix $X$ by the weight matrix $w$.
     - Uses `np.argmax` to select the class with the highest score for each sample, thus predicting the class label.

4. **Predict Proba Method (`predict_proba`)**:
   - **Purpose**: This method returns the raw regression scores (before applying `argmax`), which can be useful for evaluating errors or understanding the confidence of predictions.
   - **Procedure**:
     - Computes the scores by multiplying the feature matrix $X$ by the weight matrix $w$, without applying any thresholding or argmax.
   
### **Evaluation Function (`evaluate_model`)**:

This function calculates and prints the classification accuracy of the model. It takes the true labels (`y_true`) and the predicted labels (`y_pred`) as inputs, calculates the proportion of correct predictions, and prints the accuracy.

### **Accuracy Score Function (`accuracy_score`)**:

This function calculates the classification accuracy as the ratio of correct predictions to the total number of samples. It returns the accuracy as a float.


### **Potential Improvements or Notes**:
- **Error Tracking**: The `training_errors` and `test_errors` attributes are initialized but not actively used. Implementing error tracking could provide insight into how well the model generalizes.
- **Regularization**: The model uses the normal equation for fitting, which may lead to overfitting in cases with high-dimensional data (like MNIST). Regularization techniques (e.g., L2 regularization) might improve performance.
- **Model Comparison**: While the `plot_comparison` function is defined, it's not invoked in the provided code. Adding functionality to compare multiple models might be useful for evaluating different algorithms.


