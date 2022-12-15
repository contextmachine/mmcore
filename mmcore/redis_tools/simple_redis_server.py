import os

import pydantic
from fastapi import FastAPI

app = FastAPI()
from fastapi.responses import JSONResponse

from .connect import bootstrap_local
REDIS_URL = os.getenv('REDIS_URL')
PRIMARY_KEY = ""
REDIS_CONN = bootstrap_local(REDIS_URL)


@app.get("/take/task/{uid}")
async def take(uid: str):
    if 'REDIS_URL' in os.environ.keys() and PRIMARY_KEY + uid in REDIS_CONN.keys(PRIMKEY + "*"):
        model = done_pool[uid]
        return JSONResponse({"msg": {"x": model.x.tolist(), "fun": model.final()}}, status_code=200)


import plotly.graph_objects as go


class ExecModelParam(pydantic.BaseModel):
    x: list[float]


@app.post("/execute/{uid}")
async def take(uid: str, x: ExecModelParam):
    if 'REDIS_URL' in os.environ.keys():
        if PRIMARY_KEY + uid in REDIS_CONN.keys(PRIMARY_KEY + "*"):
            model = done_pool[uid]
            fun = model(x.x)
            fig = go.Figure(model.metrics[0].masked_plot)
            fig.to_plotly_json()
            return JSONResponse({"msg": {"fun": fun, "plot": fig.to_plotly_json()}}, status_code=200)
        else:
            return JSONResponse({"msg": f"Connection successes, but key {uid} is not found."}, status_code=400)
    else:
        return JSONResponse({"msg": "Redis is not provide."}, status_code=400)


@app.get("/take/keys")
async def take():
    if 'REDIS_URL' in os.environ.keys():
        fullkeys = REDIS_CONN.keys(PRIMARY_KEY + "*")
        s = []
        for k in fullkeys:
            s.append(k.replace(PRIMARY_KEY, ""))
        return JSONResponse({"msg": s}, status_code=200)
    else:
        return JSONResponse({"msg": "Redis is not provide"}, status_code=400)
