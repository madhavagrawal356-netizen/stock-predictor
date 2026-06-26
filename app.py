
import os
from fastapi import FastAPI
from main_files.predictor import quick_predict , deep_predict
import uuid
import json
import threading
from main_files.job import update_job
from scheduler import scheduler
Job_dir = 'jobs'
os.makedirs(Job_dir, exist_ok=True)
# %%


app = FastAPI()



def run_analysis(job_id, ticker):
    try:
        update_job(job_id, "running", "Downloading data")
        result = deep_predict(ticker, job_id)
        update_job(job_id, "completed", "Completed", result=result)
        
    except Exception as e:
        update_job(job_id,"error","Failed",error=str(e))


@app.get("/")
def home():
    return {"status": "running"}

# %%
@app.get("/quick_predict/{ticker}")
def quickk_predict(ticker:str):
    return quick_predict(ticker)

@app.get("/start_analysis/{ticker}")
def start_analysis(ticker:str):
    job_id = str(uuid.uuid4())
    threading.Thread(target=run_analysis, args=(job_id, ticker)).start()
    return {"job_id": job_id , "status": "running"}

# %%
@app.get("/job_status/{job_id}")
def job_status(job_id):
    path = f"{Job_dir}/{job_id}.json"
    if not os.path.exists(path):
        return {"status": "running", "step": "Starting"}
    with open(path, "r") as f:
        result = json.load(f)

# %%



