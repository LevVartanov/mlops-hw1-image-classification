from pydantic import BaseModel
from typing import List, Union, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from PIL import Image
import onnxruntime as ort
import json
import numpy as np
import base64
import io

class Tensor(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    data: Union[List[float], List[int], List[str]]

class InferRequest(BaseModel):
    id: Optional[str] = None
    inputs: List[Tensor]

class InferResponse(BaseModel):
    model_name: str
    id: str
    outputs: List[Tensor]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session = ort.InferenceSession(
        "artifacts/model.onnx",
        providers=["CPUExecutionProvider"]
    )

    with open("artifacts/data_config.json") as f:
        app.state.data_config = json.load(f)

    app.state.input_name = app.state.session.get_inputs()[0].name
    app.state.output_name = app.state.session.get_outputs()[0].name

    yield

    app.state.session = None
    app.state.data_config = None


def preprocess_image(image_bytes: bytes, data_config: dict):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}") from e
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(h * 256 / w)
    else:
        new_h, new_w = 256, int(w * 256 / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(data_config["mean"], dtype=np.float32)
    std = np.array(data_config["std"], dtype=np.float32)
    arr = (arr - mean) / std

    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


def predict(input_array: np.ndarray, session, input_name: str):
    return session.run(None, {input_name: input_array})[0]

def onnx_type_to_oip(onnx_type: str):
    mapping = {
    "tensor(float)": "FP32",
    "tensor(double)": "FP64",
    "tensor(float16)": "FP16",
    "tensor(int8)": "INT8",
    "tensor(int16)": "INT16",
    "tensor(int32)": "INT32",
    "tensor(int64)": "INT64",
    "tensor(uint8)": "UINT8",
    "tensor(uint16)": "UINT16",
    "tensor(uint32)": "UINT32",
    "tensor(uint64)": "UINT64",
    "tensor(bool)": "BOOL",
    "tensor(string)": "BYTES",
    }
    if onnx_type not in mapping:
        raise ValueError(f"Unknown ONNX type: {onnx_type}")
    return mapping[onnx_type]

app = FastAPI(lifespan=lifespan)

@app.get("/v2/health/live")
def live():
    return {"live": True}

@app.get("/v2/health/ready")
def ready():
    if app.state.session is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"ready": True}

@app.get("/v2/models/{model_name}")
def model_metadata(model_name: str):
    if app.state.session is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    input_ = app.state.session.get_inputs()[0]
    output_ = app.state.session.get_outputs()[0]

    return {
        "name": model_name,
        "versions": ["1"],
        "platform": "onnxruntime",
        "inputs": [
            {
                "name": input_.name,
                "shape": [s if isinstance(s, int) else -1 for s in input_.shape],
                "datatype": onnx_type_to_oip(input_.type),
            }
        ],
        "outputs": [
            {
                "name": output_.name,
                "shape": [s if isinstance(s, int) else -1 for s in output_.shape],
                "datatype": onnx_type_to_oip(output_.type),
            }
        ],
    }

@app.get("/v2/models/{model_name}/ready")
def model_ready(model_name: str):
    if app.state.session is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"ready": True}


@app.post("/v2/models/{model_name}/infer", response_model=InferResponse)
def infer(model_name: str, request: InferRequest):
    if app.state.session is None:
        raise HTTPException(503, "Model not loaded")

    if not request.inputs:
        raise HTTPException(422, "No inputs")
    tensor = request.inputs[0]
    if tensor.datatype != "BYTES":
        raise HTTPException(422, "Only BYTES supported")
    try:
        image_bytes = base64.b64decode(tensor.data[0])
    except Exception:
        raise HTTPException(422, "Invalid base64 in data")
    
    try:
        arr = preprocess_image(image_bytes, app.state.data_config)
    except ValueError as e:
        raise HTTPException(422, str(e))
    logits = predict(arr, app.state.session, app.state.input_name)

    return InferResponse(
        model_name=model_name,
        id=request.id or "",
        outputs=[Tensor(
            name="output",
            shape=list(logits.shape),
            datatype="FP32",
            data=logits.flatten().tolist(),
        )],
    )




    