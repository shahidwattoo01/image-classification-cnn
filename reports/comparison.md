# Model comparison (test split, 551 images)

| Model | Params | Size (MB) | Accuracy | Top-3 | Macro F1 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| mobilenet_v2 | 2.26M | 23.0 | 0.9201 | 0.9946 | 0.9191 | 303.5 |
| mobilenet_v2 (TFLite) | — | 2.5 | 0.9165 | — | 0.9155 | 132.8 |
| custom_cnn | 1.18M | 14.2 | 0.6606 | 0.9256 | 0.6522 | 89.8 |

Latency = median single-image CPU inference (OpenCV preprocessing + forward pass).
