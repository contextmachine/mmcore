import os
from boxes import *
import uvicorn
from fastapi import FastAPI

app = FastAPI()

app.mount(os.getenv('MMCORE_API_PREFIX'), serve.app)
if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=7711)
