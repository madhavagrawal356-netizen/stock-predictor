
import os 
import joblib
from datetime import datetime, timedelta
from main_files.predictor import deep_predict
Model_dir  ='models'
from apscheduler.schedulers.background import BackgroundScheduler

def retrain_models():
    for files in os.listdir(Model_dir):
        if not files.endswith('.pkl'):
            continue
        else:
            model_path = os.path.join(Model_dir, files)
            model = joblib.load(model_path)
            timestamp = model['timestamp']
            if datetime.now() - timestamp > timedelta(days=60):
                ticker = files.replace('.pkl', '')
                deep_predict(ticker)

# %%
scheduler = BackgroundScheduler()
scheduler.add_job(retrain_models, trigger = 'interval', days=1)
scheduler.start()


