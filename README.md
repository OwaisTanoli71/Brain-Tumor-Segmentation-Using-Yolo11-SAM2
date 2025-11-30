🧠 Brain Tumor Segmentation & Detection (YOLOv11 + SAM2)

📘 Project Overview
This project focuses on the automated analysis of medical imaging. The goal is to perform **Brain Tumor Detection and Segmentation** by combining two powerful AI architectures:
1.  **YOLOv11 (You Only Look Once):** For accurate object detection (bounding boxes).
2.  **SAM2 (Segment Anything Model 2):** For precise instance segmentation (pixel-perfect shapes).

Additionally, the project features an interactive **Streamlit Dashboard**, which allows users to upload MRI scans and visualize detection results in real-time without requiring any code.

## 🧪 Inference Pipeline
The system follows a two-step logic to achieve high precision:
1.  **Load YOLOv11:** The custom-trained model detects the tumor and generates a bounding box.
2.  **Bounding Box Prompt:** The coordinates of this box are passed as a "prompt" to SAM2.
3.  **SAM2 Segmentation:** SAM2 uses the box to generate a precise mask (segmentation) of the tumor region.
4.  **Visualization:** The final result is displayed on the dashboard, showing the tumor type (Glioma, Meningioma, Pituitary) and its location.

## 📂 Repository Contents

| File/Folder | Description |
| :--- | :--- |
| **`app.py`** | **The Streamlit Dashboard application code.** |
| `brain_tumor_detection.ipynb` | Full Jupyter notebook (Training + Inference logic). |
| `requirements.txt` | List of required libraries (`streamlit`, `ultralytics`, `sam2`, etc.). |
| `my_tumor_model.pt` | Custom trained YOLOv11 weights (Best Model). |
| `sam2_b.pt` | Pre-trained weights for the SAM2 model. |
| `test_images/` | Sample MRI scans for testing the system. |
| `README.md` | Project documentation. |

## ⚙️ Setup & Installation

**1. Clone the Repository**
```bash
git clone [https://github.com/YourUsername/Brain-Tumor-Segmentation.git](https://github.com/YourUsername/Brain-Tumor-Segmentation.git)
cd Brain-Tumor-Segmentation
