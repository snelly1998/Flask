import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score

app = Flask(__name__)

data = pd.read_csv('./content/Crop_recommendation.csv')

X = data.drop('label', axis=1)
y = data['label'] 

unique_labels = sorted(set(y))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_int = [label_map[label] for label in y]

X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.1, random_state=42)

params = {
    'objective': 'multiclass',
    'num_class': len(unique_labels),
    'metric': 'multi_logloss',
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

num_round = 100
model = lgb.train(params, train_data, num_round, valid_sets=[test_data])

y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_class = [list(x).index(max(x)) for x in y_pred]

accuracy = accuracy_score(y_test, y_pred_class)

model.save_model('crop_recommendation_model.txt')

model = lgb.Booster(model_file='./crop_recommendation_model.txt')  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = request.json['input_features']
    input_features = [input_features]
    predicted_labels = predict_crop(input_features)
    return jsonify({'predicted_label': unique_labels[predicted_labels[0]], 'accuracy': accuracy})

def predict_crop(input_features):
    predictions = model.predict(input_features)
    predicted_labels = [list(x).index(max(x)) for x in predictions]
    return predicted_labels

def return_accuracy():
    return accuracy 

if __name__ == '__main__':
    app.run()
    
