from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import uvicorn

from mlp import SimpleMLP

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 모델 로드
model = SimpleMLP()
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
model.eval()

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    pixels = data["pixels"]  # shape (64,)
    x_tensor = torch.tensor([pixels], dtype=torch.float)

    with torch.no_grad():
        out = model(x_tensor)           # shape (1,2)
        pred_idx = torch.argmax(out, dim=1).item()  # 0 or 1

    # hook 갱신
    with torch.no_grad():
        _ = model(x_tensor)

    # 0->1, 1->2
    label_map = {0:1, 1:2}
    pred_label = label_map[pred_idx]

    # 시각화
    img_data = model.visualize_activations()

    return JSONResponse({
        "prediction": str(pred_label),
        "img_data": img_data
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
