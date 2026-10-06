import base64
import json
import requests


URL = "http://127.0.0.1:8000/v2/models/efficientnet/infer"
IMAGE_PATH = "test_image.jpg"


def main():
    # 1. Кодируем изображение в base64
    with open(IMAGE_PATH, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 2. Формируем KServe-запрос
    payload = {
        "id": "test-1",
        "inputs": [
            {
                "name": "image",
                "shape": [1],
                "datatype": "BYTES",
                "data": [image_b64],
            }
        ],
    }

    # 3. Отправляем
    r = requests.post(URL, json=payload)

    # 4. Печатаем ответ
    print("Status:", r.status_code)
    if r.status_code != 200:
        print("Error:", r.text)
        return

    data = r.json()
    print("model_name:", data.get("model_name"))
    print("id:", data.get("id"))

    out = data["outputs"][0]
    print("shape:", out["shape"])
    print("datatype:", out["datatype"])

    logits = out["data"]
    top5_idx = sorted(range(len(logits)), key=lambda i: -logits[i])[:5]
    print("Top-5 indices:", top5_idx)
    print("Top-5 logits:", [round(logits[i], 3) for i in top5_idx])


if __name__ == "__main__":
    main()