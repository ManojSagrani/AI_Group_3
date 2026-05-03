import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# 1. Create a dummy dataset structure matching common crop datasets
# Typically: N, P, K, temperature, humidity, ph, rainfall -> label
data = {
    'N': np.random.randint(0, 140, 100),
    'P': np.random.randint(5, 145, 100),
    'K': np.random.randint(5, 205, 100),
    'temperature': np.random.uniform(10, 45, 100),
    'humidity': np.random.uniform(15, 100, 100),
    'ph': np.random.uniform(3.5, 10, 100),
    'rainfall': np.random.uniform(20, 300, 100),
    'label': np.random.choice(['rice', 'maize', 'coffee', 'cotton'], 100)
}
df = pd.DataFrame(data)

# 2. Preprocessing
X = df.drop('label', axis=1)
y = df['label']

# Encode the crop names into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features (Critical for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Create the MLP (Multi-Layer Perceptron)
# hidden_layer_sizes=(64, 32) means 2 hidden layers with 64 and 32 neurons respectively
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# 4. Train
mlp.fit(X_train_scaled, y_train)

# 5. Output structure
print("MLP Layers:", mlp.n_layers_)
print("Classes identified:", label_encoder.classes_)

# 6. Evaluate
y_pred = mlp.predict(X_test_scaled) 
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
