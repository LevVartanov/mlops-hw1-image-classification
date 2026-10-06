import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "artifacts/model.onnx",
    providers=["CPUExecutionProvider"],
)

print("Inputs:")
for i in session.get_inputs():
    print(f"  name={i.name}, shape={i.shape}, type={i.type}")

print("Outputs:")
for o in session.get_outputs():
    print(f"  name={o.name}, shape={o.shape}, type={o.type}")

dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
out = session.run(None, {session.get_inputs()[0].name: dummy})
print("Output shape:", out[0].shape)