Predicting Irrigation Need is a machine learning project that predicts the required irrigation level for agricultural fields based on environmental and soil-related factors. The project uses data such as temperature, humidity, rainfall, soil moisture, wind speed, sunlight hours, soil pH, organic carbon, and previous irrigation levels to classify each field into one of three categories:

Low irrigation need
Medium irrigation need
High irrigation need
The main workflow is implemented in a Jupyter Notebook using Python. It begins by loading training and testing datasets, removing irrelevant features and identifiers, and converting the categorical target labels into numerical values.

Several agricultural and weather features are combined to create more informative variables, including temperature-humidity interaction, rainfall relative to soil moisture, evaporation index, and soil-thirst index. Categorical fields are converted into numerical values using ordinal encoding, while numerical features are standardized using StandardScaler.

Because the irrigation classes may be unevenly distributed, the project applies SMOTE to balance the training data and class weights to improve the model's ability to learn from minority classes. A deep neural network built with TensorFlow and Keras is then used for classification. The network contains several dense layers, batch normalization, dropout regularization, and a softmax output layer that produces probabilities for the three irrigation levels.

During training, early stopping and learning-rate reduction callbacks prevent overfitting and adjust the learning process when validation performance stops improving. Finally, the model predicts the irrigation level for each test record, converts the numerical predictions back into labels, and exports them to a CSV submission file.

Technologies used: Python, Jupyter Notebook, Pandas, NumPy, TensorFlow/Keras, Scikit-learn, and imbalanced-learn.
