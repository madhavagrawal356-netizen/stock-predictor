

from fastapi import FastAPI
from main_files.predictor import quick_predict , deep_predict

# %%
app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

# %%
@app.get("/quick_predict/{ticker}")
def quickk_predict(ticker:str):
    return quick_predict(ticker)

@app.get("/deep_predict/{ticker}")
def deeep_predict(ticker:str):
    return deep_predict(ticker)

# %%


# %%



