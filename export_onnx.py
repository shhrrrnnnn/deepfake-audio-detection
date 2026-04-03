# Export trained LCNN to ONNX format for backend deployment
# Run with: python export_onnx.py

import os
import sys
import torch
import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from models.lcnn import LCNN
from utils.features import IMG_SIZE

OUTPUT_DIR  = r"C:\Users\shara\deepfake_audio\output"
MODEL_PATH  = os.path.join(OUTPUT_DIR, "best_lcnn.pt")
ONNX_PATH   = os.path.join(OUTPUT_DIR, "audio_deepfake_lcnn.onnx")

device = torch.device("cpu")   # always export from CPU

if __name__ == '__main__':

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model      = LCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model loaded — EER: {checkpoint.get('eer', 0)*100:.2f}%")

    # Export
    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["mel_input"],
        output_names=["output"],
        dynamic_axes={
            "mel_input": {0: "batch_size"},
            "output":    {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )

    size_mb = os.path.getsize(ONNX_PATH) / 1e6
    print(f"ONNX exported ✅  {size_mb:.1f} MB → {ONNX_PATH}")

    # Verify
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print("[1] Structure check ✅")

    session     = ort.InferenceSession(ONNX_PATH,
                      providers=["CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    test_inp = np.random.randn(1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
    out      = session.run([output_name], {input_name: test_inp})
    probs    = torch.softmax(torch.tensor(out[0]), dim=1).numpy()
    print(f"[2] Single inference ✅  real={probs[0][0]:.3f} fake={probs[0][1]:.3f}")

    import time
    t0 = time.time()
    for _ in range(100):
        session.run([output_name], {input_name: test_inp})
    avg_ms = (time.time() - t0) / 100 * 1000
    print(f"[3] Latency          ✅  {avg_ms:.1f} ms")

    batch_inp = np.random.randn(4, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
    batch_out = session.run([output_name], {input_name: batch_inp})
    assert batch_out[0].shape == (4, 2)
    print(f"[4] Batch size=4     ✅  {batch_out[0].shape}")
    print(f"[5] File size        ✅  {size_mb:.1f} MB")

    print("\n" + "=" * 50)
    print("  ONNX export verified ✅")
    print(f"  Deploy: {ONNX_PATH}")
    print("=" * 50)

    # Save threshold
    np.save(os.path.join(OUTPUT_DIR, "eer_threshold.npy"),
            np.array([checkpoint.get("threshold", 0.5)]))
    print("EER threshold saved ✅")
```

---

Now run this in terminal to set everything up and start training:
```
mkdir C:\Users\shara\deepfake_audio
mkdir C:\Users\shara\deepfake_audio\models
mkdir C:\Users\shara\deepfake_audio\utils
mkdir C:\Users\shara\deepfake_audio\output
mkdir C:\Users\shara\deepfake_audio\cache

echo. > C:\Users\shara\deepfake_audio\models\__init__.py
echo. > C:\Users\shara\deepfake_audio\utils\__init__.py

cd C:\Users\shara\deepfake_audio
python trainer.py
```

Once training finishes:
```
python export_onnx.py
python app_inference.py path\to\any\audio.mp3