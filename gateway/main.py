from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel
from typing import List

app = FastAPI(title="SignSight Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expected input: a window of frames, each frame a list of keypoints [x, y, z, visibility]
class KeypointRequest(BaseModel):
    keypoints: List[List[List[float]]]

PREPROCESSING_URL = "http://127.0.0.1:8001"
VISION_URL = "http://127.0.0.1:8002"
NLP_URL = "http://127.0.0.1:8003"

@app.post("/translate")
async def translate_sign(data: KeypointRequest):
    async with httpx.AsyncClient() as client:
        try:
            # Step 1: Preprocessing
            prep_resp = await client.post(f"{PREPROCESSING_URL}/filter", json={"keypoints": data.keypoints}, timeout=5.0)
            if prep_resp.status_code != 200: raise HTTPException(status_code=500, detail="Preprocessing failed")
            clean_kp = prep_resp.json()["clean_keypoints"]

            if not clean_kp:
                return {"sentence": "", "confidence": 0.0}

            # Step 2: Vision Inference
            vis_resp = await client.post(f"{VISION_URL}/predict", json={"keypoints": clean_kp}, timeout=5.0)
            if vis_resp.status_code != 200: raise HTTPException(status_code=500, detail="Vision inference failed")
            vis_result = vis_resp.json()

            # Step 3: NLP Translation (confidence passes through from the vision model)
            nlp_resp = await client.post(f"{NLP_URL}/correct", json=vis_result, timeout=5.0)
            if nlp_resp.status_code != 200: raise HTTPException(status_code=500, detail="NLP failed")
            
            return nlp_resp.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")

@app.post("/fingerspell")
async def fingerspell(data: KeypointRequest):
    """Letter recognition: same preprocessing gate + smoothing, then the
    static-handshape classifier instead of the word model."""
    async with httpx.AsyncClient() as client:
        try:
            # smooth=False: the moving average mangles the edges of a short
            # fingerspell window and the letter model trains on unsmoothed
            # landmarks.
            # use_validator=False: the SVM validator's negative class includes
            # "signer holding still", which is precisely what a fingerspelled
            # letter looks like — it rejected 23/23 real letters. The validator
            # gates the word path, where stillness genuinely means idle; the
            # cheap threshold rule gates this one.
            prep_resp = await client.post(f"{PREPROCESSING_URL}/filter",
                                          json={"keypoints": data.keypoints, "smooth": False,
                                                "use_validator": False}, timeout=5.0)
            if prep_resp.status_code != 200: raise HTTPException(status_code=500, detail="Preprocessing failed")
            clean_kp = prep_resp.json()["clean_keypoints"]
            if not clean_kp:
                return {"letter": "", "confidence": 0.0}

            vis_resp = await client.post(f"{VISION_URL}/predict_letter", json={"keypoints": clean_kp}, timeout=5.0)
            if vis_resp.status_code != 200: raise HTTPException(status_code=500, detail="Letter inference failed")
            return vis_resp.json()  # includes `candidates` for the live debug panel
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")

class CalibrateRequest(BaseModel):
    letter: str
    keypoints: List

@app.post("/calibrate")
async def calibrate(data: CalibrateRequest):
    """Save a labelled hand from the live camera. No preprocessing gate and no
    smoothing: calibration must record exactly what /fingerspell will see."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{VISION_URL}/calibrate/save",
                                     json={"letter": data.letter, "keypoints": data.keypoints}, timeout=10.0)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.json().get("detail", "Calibration failed"))
            return resp.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")

class ComposeRequest(BaseModel):
    words: List[str]

@app.post("/compose")
async def compose(data: ComposeRequest):
    async with httpx.AsyncClient() as client:
        try:
            nlp_resp = await client.post(f"{NLP_URL}/compose", json={"words": data.words}, timeout=5.0)
            if nlp_resp.status_code != 200: raise HTTPException(status_code=500, detail="NLP failed")
            return nlp_resp.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")

@app.get("/health")
def health():
    return {"status": "Gateway Healthy"}