# DECISION-TREE-IMPLEMENTATION

"COMPANY" : CODTECH

"NAME"    : VAISHNAVI NARAYANDAS

INTERN ID : CT04WT240

"DOMAIN"  : MACHINE LEARNING

"DURATION": 4WEEKS

"MENTOR"  : NEELA SANTOSH
---
#DESCRIPTION

This project demonstrates the implementation, training, evaluation, and visualization of a Decision Tree Classification model using the Scikit-learn library in Python. Decision Trees are a fundamental supervised learning algorithm used for classification and regression tasks in machine learning. They mimic human decision-making by learning rules inferred from the data features. This project specifically focuses on classifying data and visualizing the tree structure for better interpretability and analysis.

I used the **Iris dataset**, a classic dataset often used to benchmark classification algorithms. It contains 150 records under 3 different species of Iris flowers: Setosa, Versicolour, and Virginica. Each record consists of four features: sepal length, sepal width, petal length, and petal width. The target variable is the species label. This dataset is available directly from Scikit-learn's built-in datasets module, making it easy to load and explore.
---
### Tools and Technologies Used

* **Python**: The programming language used for this project due to its extensive machine learning libraries and ease of use.
* **Scikit-learn**: A powerful open-source machine learning library in Python used for data preprocessing, model training, and visualization. The `DecisionTreeClassifier` from `sklearn.tree` was used to build the classification model.
* **Matplotlib & Seaborn**: For plotting and visualizing the decision boundaries and feature relationships in the dataset.
* Visual Studio Code (VS Code): The IDE used for writing and running the code. VS Code’s support for Python extensions and interactive notebooks made it a productive environment for this machine learning task.
**Jupyter Notebook**: Although the primary development was done in VS Code, the notebook format (with .ipynb extension) was used within VS Code for a step-by-step and interactive execution experience.
* **Pandas & NumPy**: Used for data manipulation and numerical computations.
---
### Implementation and Workflow

1. **Dataset Loading and Exploration**: The Iris dataset was loaded using Scikit-learn’s `load_iris()` function. Basic exploration was performed using Pandas to understand the shape, feature names, and distribution of classes.
2. **Data Preparation**: The features and target labels were separated and the dataset was split into training and testing sets using `train_test_split` to evaluate the model fairly.
3. **Model Training**: A `DecisionTreeClassifier` was instantiated with parameters such as `max_depth` to prevent overfitting and then trained using the training set.
4. **Evaluation**: The trained model was evaluated on the test data using accuracy score and confusion matrix. These metrics helped determine how well the model generalizes to new data.
5. **Visualization**: The decision tree was visualized using Scikit-learn’s `plot_tree()` function from `sklearn.tree`, which provides a clear, interpretable diagram of the decision rules at each node.
---
### Model Analysis

The decision tree successfully classified the Iris flowers into their respective categories with high accuracy. Visualizing the tree made it possible to see which features were most influential in making decisions (e.g., petal length and width are generally more informative than sepal length). The confusion matrix and classification report helped to understand the performance of the model on each class.
---
### Applications of Decision Trees

Decision Trees are used across various domains for their simplicity, interpretability, and speed:

* **Healthcare**: Diagnosing diseases based on symptoms and test results.
* **Finance**: Credit risk analysis and fraud detection.
* **Marketing**: Predicting customer churn or response to promotions.
* **Education**: Classifying students based on performance or predicting dropouts.
* **Retail**: Inventory classification and customer segmentation.

They are also widely used in ensemble methods like Random Forests and Gradient Boosting for more robust and accurate models.
---
### Conclusion

This project showcases a complete pipeline for building and visualizing a Decision Tree model using Scikit-learn. The Iris dataset was ideal for demonstrating the effectiveness of the model due to its simplicity and balanced classes. The use of Jupyter Notebook enabled step-by-step explanations and visualization, making it an educational and insightful machine learning exercise. This implementation serves as a strong foundation for exploring more advanced models or applying decision trees to real-world datasets.

#OUTPUT

![Image](https://github.com/user-attachments/assets/0e93f0d8-93b0-417c-b665-a8d1205315d7)
