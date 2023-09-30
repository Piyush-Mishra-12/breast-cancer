from flask import Flask, render_template, request, send_file
from src.exception import CustomException
from src.log import logging
import sys
from src.pipelines.train_pipeline import TrainingPipeline
from src.pipelines.predict_pipeline import PredictingPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home_page():
    return "Welcome to my App"

@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.start_pipeline()
        return "Training Completed."
    except Exception as e:
        raise CustomException(e,sys) # type: ignore

@app.route('/predict', methods=['GET','Post'])
def predict():
    try:
        if request.method=='GET':
            return render_template('form.html')
        else:
            data = CustomData(worst_concave_points = float(request.form.get('worst_concave_points')),
                              worst_area = float(request.form.get('worst_area')),
                              worst_perimeter = float(request.form.get('worst_perimeter')),
                              worst_radius = float(request.form.get('worst_radius')),
                              mean_concave_points = float(request.form.get('mean_concave_points')))
            logging.info(f'the data is {data}')
            final_data = data.get_data()
            logging.info(f'the final data is\n{final_data}')
            predict_pipeline = PredictingPipeline()
            pred = predict_pipeline.predict(final_data)
            results = {'Prediction': 'Malignant' if pred[0] == 1 else 'Benign'}
            return render_template('form.html', final_result = results)
    except Exception as e:
        raise CustomException(e,sys) # type: ignore
if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)