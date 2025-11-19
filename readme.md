# 📘 **Iris Prediction API**

This project provides a **machine learning classifier** for predicting the species of an Iris flower using a **Decision Tree** model trained with the **scikit-learn** library.
The trained model is saved using **joblib** and served via a **FastAPI** application running inside a **Docker container**.

---

# 📂 **Project Structure**

```
iris-prediction/
│
├── main.py                             # FastAPI application
├── iris_classifier.ipynb               # Jupyter notebook to train & export the model
├── iris_decision_tree_pipeline.joblib  # Saved classifier (generated after training)
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container definition
└── README.md
```

---

# 🌱 **1. Training the Model**

You can train the model inside the notebook **iris_classifier.ipynb**.

The training process includes:

1. Loading the Iris dataset
2. Creating a DecisionTreeClassifier
3. Fitting the model
4. Saving it using **joblib**

Example code used in the notebook:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, stratify = y, random_state = 42
)

model = Pipeline([
    ("dectree", DecisionTreeClassifier(random_state = 42))
])

model.fit(X_train, y_train)

joblib.dump(model, "iris_decision_tree_pipeline.joblib")
print("\nModel saved to iris_decision_tree_pipeline.joblib")
```

After running this, the file **iris_decision_tree_pipeline.joblib** will be created in the project root.

---

# 💾 **2. Saving the Model**

The line responsible for saving the model is:

```python
joblib.dump(model, "iris_decision_tree_pipeline.joblib")
```

This file will be loaded by the API at runtime.

---

# 🐳 **3. Building the Docker Image**

Make sure your `Dockerfile` is in the project root.

Build the Docker image:

```
docker build -t iris-api .
```

---

# ▶️ **4. Running the API in Docker**

Start the container:

```
docker run -p 8000:8000 iris-api
```

The API is now available at:

👉 [http://localhost:8000](http://localhost:8000)
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

# 📡 **5. Making a Prediction Request**

Use any HTTP client (Insomnia, Postman, curl).

### **Endpoint**

```
POST /predict
URL: http://localhost:8000/predict
```

### **Request Body example (JSON)**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### **Example with curl**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      }'
```

### **Example Response**

```json
{
  "predicted_class": 0
}
```

---

# 🧪 **6. Testing the API**

Visit:

👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

FastAPI automatically generates an interactive Swagger UI where you can test the endpoint directly.

---

# 🎉 **7. Conclusion**

This project demonstrates:

* Training a Decision Tree classifier
* Saving the model with joblib
* Serving predictions using FastAPI
* Containerizing the entire service with Docker

Feel free to extend the project with:

* Model versioning
* Improved preprocessing
* Batch prediction endpoint
* Logging and monitoring
* CI/CD pipeline

---

# 🙋‍♂️ **Author**

**Romulo Leite**

Statistician & Data Scientist

📫 [GitHub](https://github.com/romulool)

📧 [romulool@yahoo.com.br](mailto:romulool@yahoo.com.br)

---

# 📜 **License**

This project is open-source and licensed under the **MIT License**.

```
