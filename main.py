from fastapi import FastAPI
from majorproject import calculate_S11
from majorproject import predictingValues
from majorproject import lrperformance
from majorproject import rfperformance
from majorproject import enperformance
from majorproject import dtperformance
from majorproject import lassoperformance
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "The Server is up and runing"}


class Data(BaseModel):
    Freq: float
    length_of_patch: float
    width_of_patch: float
    Slot_length: float
    slot_width: float


@app.post("/predict")
async def predicting(data: Data):
    return {
        "s11": predictingValues(
            {
                "Freq(GHz)": data.Freq,
                "length of patch in mm": data.length_of_patch,
                "width of patch in mm": data.width_of_patch,
                "Slot length in mm": data.Slot_length,
                "slot width in mm": data.slot_width,
            }
        )
    }


@app.get("/performance")
async def performance():
    return {
        "Linear": lrperformance(),
        "Random Forest ": rfperformance(),
        "ElasticNet": enperformance(),
        "Lasso": lassoperformance(),
        "Decision Tree": dtperformance(),
    }
