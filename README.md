# SwiftGate: Fairness-Aware Biometric Authentication 

SwiftGate is an algorithmic security pipeline designed for high-throughput operational environments (such as airport security checkpoints). It utilizes deep learning face recognition models to authenticate identities while actively auditing and mitigating demographic bias to ensure fairness across all user groups.

---

##  Key Features & Engineering Highlights

###  Bias Mitigation & Demographic Auditing
* **Fairness-Aware Core:** Built a specialized pipeline using **FaceNet** and **MTCNN** architectures explicitly designed to identify, audit, and systematically reduce demographic bias in facial verification.
* **Balanced Disparity:** Benchmarked multi-model architectures—including **AdaFace, ArcFace, and GhostFaceNet**—to evaluate deep learning performance tradeoffs.
* **Rigorous Security Constraints:** Successfully lowered subgroup error disparities while strictly adhering to a mission-critical **0.02% False Acceptance Rate (FAR)** security mandate.

###  High-Performance Backend & Preprocessing
* **Modular Infrastructure:** Engineered a fast, production-ready backend framework powered by **FastAPI**.
* **Data Pipelines:** Designed data preprocessing sequences that apply strict data normalization techniques to maintain consistency under raw camera inputs.
* **Threshold Optimization:** Implemented custom **grid-search thresholding** algorithms to fine-tune decision boundaries, optimizing accuracy against intense, high-throughput airport operational constraints.

---

##  Tech Stack

* **Language:** Python
* **Deep Learning & CV:** FaceNet, MTCNN, AdaFace, ArcFace, GhostFaceNet
* **Backend API:** FastAPI
* **Data Processing:** NumPy, Pandas, Scikit-learn (Grid-Search)

---

##  Local Installation & Setup

To replicate this environment and test the API endpoints locally:
