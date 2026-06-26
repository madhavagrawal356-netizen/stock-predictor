
import json
import os

Job_dir = 'jobs'
os.makedirs(Job_dir, exist_ok=True)
def update_job(job_id, status, step,result=None, error=None):
    data = {
        "status": status,
        "step": step,}
    if result is not None:
        data["result"] = result
    if error is not None:
        data["message"] = error
    with open(f"{Job_dir}/{job_id}.json", "w") as f:
        json.dump(data, f)


