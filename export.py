# export_onnx.py
import torch
import timm
from timm.data import resolve_data_config
import json
import os

# 1. Создаем папку для артефактов, если ее нет
os.makedirs("artifacts", exist_ok=True)

# 2. Загружаем предобученную модель и переводим в режим инференса
model_name = 'tf_efficientnet_lite0'
model = timm.create_model(model_name, pretrained=True)
model.eval()

# 3. Получаем и сохраняем конфигурацию предобработки
#    (размер входа, mean, std) — они понадобятся в сервисе
data_config = resolve_data_config({}, model=model)
with open("artifacts/data_config.json", "w") as f:
    json.dump(data_config, f, indent=2)
print(f"Data config saved: {data_config}")

# 4. Создаем "фиктивный" входной тензор для трассировки графа
dummy_input = torch.randn(1, 3, 224, 224)

# 5. Экспортируем модель в ONNX
torch.onnx.export(
    model,
    dummy_input,
    "artifacts/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)
print("Model exported to artifacts/model.onnx")