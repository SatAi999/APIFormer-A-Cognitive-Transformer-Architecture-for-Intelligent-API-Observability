# APIFormer+ 
> **A Cognitive Transformer Architecture for Intelligent API Traffic Understanding, Prediction, Security, Reasoning, and Generation**

APIFormer+ is a  AI observability platform designed to learn the language, semantic behaviors, service dependency topologies, and cybersecurity profiles of cloud-native microservice API traffic. Unlike traditional signature-based security gateways or static threshold monitors, APIFormer+ models enterprise API traffic as a multi-dimensional graph-temporal sequence using a custom-engineered **Transformer Architecture** built completely from scratch in PyTorch.

---

## 🖥️ Unified Observability Interface

![APIFormer+ Observability Dashboard](docs/images/dashboard.png)

---

## 📐 Project Vision & Architectural Philosophy
Modern cloud architectures run hundreds of microservices processing billions of requests. When failures or cyber attacks occur, they propagate rapidly across call paths, making root-cause analysis extremely difficult. 

APIFormer+ resolves this by treating microservice traces not as text logs, but as a **dynamic, time-decaying microservice graph**. By combining GCN-propagated topological structures, Hawkes-process time-delays, and historical session memories, the Transformer has a cognitive understanding of microservice context:
1. **Structural Context**: Where does this endpoint sit in the microservice topology? (GCN + PageRank)
2. **Temporal Context**: How long has it been since the last call, and is this request taking abnormally long? (Time2Vec + Hawkes Decay)
3. **Historical Context**: Have we seen a similar transaction pattern in the past? (Session Memory Bank + Cross-Attention)
4. **Behavioral Semantics**: What endpoint is being called, with what auth headers, env, location, and payload size? (Multi-Feature Embedding)

---

## 🧬 Core Neural Architecture (Built from Scratch)

Every component of the deep learning stack is implemented using raw PyTorch tensors, avoiding black-box prebuilt APIs to maintain absolute numerical explainability:

```
               [ Unified API Log Stream ]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      [ GNN Adjacency ]          [ Tokenized Sessions ]
      [ PageRank Scores]         [ Time & Latency Vectors ]
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  [ Time2Vec + RoPE ]
                           │
            [ stacked Transformer Encoder ]
                           │
             [ Hawkes Decay attention ]
                           │
         [ Memory Cross-Attention Fusion ] <─── [ Cosine Retrieval Bank ]
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
[ SSL Pretraining Heads ]              [ Downstream Predictors ]
 - Masked API Modeling (MAM)            - Sequence Anomaly Head
 - Next API Causal Shift                - Workflow Intent Class
 - Latency continuous MSE               - Bot Agent Detection
 - Status Code Classifier
 - InfoNCE Contrastive alignment
```

### 1. Multi-Feature Fusion Embeddings (`embeddings.py`)
Incoming telemetry events contain mixed categorical and continuous numerical features. The embedding layer converts these into a unified vector space:
*   **Categorical Embeddings**: High-dimensional tokens (Endpoint, HTTP Method, HTTP Status Code, Calling Service, Device, Location, Environment, and Auth headers) are projected to embedding dimensions and concatenated.
*   **Time2Vec Continuous Embeddings**: Project continuous numerical dimensions (Latency, Payload Size, and Inter-Request Time Gap) into periodic and linear representations to ease gradient calculations:
    $$\text{T2V}(x)[i] = \begin{cases} \omega_0 x + \varphi_0 & \text{if } i = 0 \\ \sin(\omega_i x + \varphi_i) & \text{if } 1 \le i \le d \end{cases}$$
*   **Rotary Positional Embeddings (RoPE)**: Orthogonally rotates key-value pairs in the self-attention mechanism, preserving relative distances:
    $$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$
*   **Absolute Position Encodings**: Traditional sinusoidal absolute positional encodings are added to provide sequential coordinate grids.

### 2. Custom Layer Normalization (`attention.py`)
Layer Normalization is implemented from mathematical definitions to enforce Pre-LN stability during deep Transformer runs:
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

### 3. Scaled Dot-Product Attention with T5 Relative Position Bias (`attention.py`)
Augments standard scaled dot-product attention queries and keys with relative distance offsets:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + B\right) V$$
where $B$ is a learnable relative position bias matrix.

### 4. Graph Convolutional Network (GCN) service topologizer (`cognitive.py`)
Refines node representations by propagating topological properties across microservice call connections:
$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$
where $\tilde{A}$ is the microservice adjacency matrix and $\tilde{D}$ is the diagonal degree matrix.

### 5. Hawkes Temporal Decay Attention (`cognitive.py`)
Applies exponential decay penalties to attention weights based on the time delay between events. This models the assumption that events occurring closer in time have higher causal correlation:
$$\text{Score}_{ij} = \frac{q_i k_j^T}{\sqrt{d_k}} - \gamma \odot \Delta t_{ij}$$
where $\gamma = \exp(\log \gamma)$ is parameterized in log-space to constrain the decay rate to positive values ($\gamma > 0$).

### 6. Cosine Retrieval Session Memory Bank (`cognitive.py`)
*   **Storage**: Pre-allocates CPU-side tensor arrays to store historical normal/anomalous session embeddings.
*   **Retrieval**: Computes the cosine similarity between the current session vector $z_{\text{query}}$ and stored keys:
    $$\text{sim}(z_{\text{query}}, z_{\text{bank}}) = \frac{z_{\text{query}} \cdot z_{\text{bank}}}{\|z_{\text{query}}\| \|z_{\text{bank}}\|}$$
*   **Fusion**: Retrieves the top $K=3$ closest sessions and fuses them using memory cross-attention layers.

---

## 📊 Telemetry Ingestion & Dataset Generation (`src/data/`)

### 1. Ingestion Generator (`synthetic_gen.py`)
A state-machine simulator modeling trace structures based on OpenTelemetry and DeathStarBench specifications. Injects:
*   **Normal Traffic**: Standard API calling sequences.
*   **503 Cascades**: Simulates upstream microservice failures that propagate timeouts and service outages.
*   **Credential Stuffing**: Repetitive failed login endpoints.
*   **Bot Scraping**: Fast sequential product requests using custom bot user-agents.
*   **Sequence Abuse**: Accessing protected checkout endpoints without authentication.

### 2. Dynamic Schema Inspector & Normalizer (`inspector.py`, `normalizer.py`)
Uses Jaro-Winkler string similarity metrics to scan raw databases, automatically map varying column names to standard fields (`timestamp`, `sessionID`, `url_path`, `duration`, `bytes_size`), and normalize inputs.

### 3. Topological Graph Builder (`graph_builder.py`)
Builds dependency adjacency matrices of endpoints, calculating PageRank centrality and degree statistics to quantify topological importance.

---

## 📈 Dual-Phase Optimization Loops (`scripts/`)

```
                          ┌───────────────────────────┐
                          │   Step 1: Ingestion       │
                          │   Raw OTEL Trace Logs     │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │   Step 2: Processing      │
                          │   GNN Graph & Sessions    │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │   Step 3: Pretraining     │
                          │   MAM, Next, MSE, InfoNCE │
                          └─────────────┬─────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │   Step 4: Fine-Tuning     │
                          │   Anomaly, Intent, Bot    │
                          └───────────────────────────┘
```

### Phase 1: Self-Supervised Pretraining (`train.py`)
Pretrains the model on unlabeled trace streams using a joint self-supervised loss:
$$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{mam}} + \mathcal{L}_{\text{next}} + 0.1 \cdot \mathcal{L}_{\text{latency}} + \mathcal{L}_{\text{status}} + 0.5 \cdot \mathcal{L}_{\text{contrastive}}$$
*   **Masked API Modeling (MAM)**: Predicts 15% masked endpoint tokens.
*   **Causal Next-API Prediction**: Autocompletes API calling paths.
*   **Latency MSE Regression**: Forecasts continuous log-transformed latency.
*   **Status Code Prediction**: Forecasts HTTP response status codes.
*   **InfoNCE Contrastive Session Alignment**: Aligns session embeddings under augmented timing jitter and token dropout views:
    $$\mathcal{L}_{\text{InfoNCE}} = - \log \frac{\exp(\text{sim}(z_i^{(1)}, z_i^{(2)}) / \tau)}{\sum_j \exp(\text{sim}(z_i^{(1)}, z_j) / \tau)}$$

### Phase 2: Downstream Multi-Task Fine-Tuning (`train.py`)
Fine-tunes the pretrained weights on labeled security and performance targets:
$$\mathcal{L}_{\text{finetune}} = \mathcal{L}_{\text{anomaly}} + \mathcal{L}_{\text{intent}} + \mathcal{L}_{\text{bot}}$$
*   **Sequence Anomaly Head**: Step-level binary classifier for sequence/latency violations.
*   **Business Intent Head**: Session-level intent class predictor (`normal`, `503_cascade`, `credential_stuffing`, `bot_scraping`, `sequence_abuse`).
*   **Bot Detection Head**: Session-level binary classifier separating user-agents from bot behaviors.

---

## 🐳 Containerization, Orchestration & Monitoring

### docker-compose.yml Architecture
Deploys a complete local observability stack:
*   `api-service` (FastAPI serving endpoint, port `8000`)
*   `dashboard` (Streamlit analytics GUI, port `8501`)
*   `redis` (Session cache database, port `6379`)
*   `kafka` & `zookeeper` (Streaming broker, port `9092`)
*   `prometheus` & `grafana` (Metrics scraper and panel, ports `9090` and `3000`)

### Kubernetes YAML manifests (`k8s/`)
Deploys and orchestrates the services in a Kubernetes cluster:
*   `api-deployment.yaml`: Replicates the API service (3 replicas) for load balancing.
*   `dashboard-deployment.yaml`: Replicates the Streamlit dashboard.
*   `redis-deployment.yaml`: Deploys the Redis session cache pod.
*   `ingress.yaml`: Ingress controller rules routing `/api` to the FastAPI service and `/` to the dashboard.

---

## ⚙️ Quick Start Guide

### 1. Data Pipeline Ingestion
```bash
$env:PYTHONPATH="."
python scripts/run_pipeline.py
```

### 2. Pretraining and Fine-Tuning Optimizations
```bash
python scripts/train.py
```

### 3. Model Evaluation
```bash
python scripts/evaluate.py
```

### 4. FastAPI Server Inference
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Query predictions:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @examples/payload.json
```

### 5. Streamlit Dashboard
```bash
streamlit run src/visualization/dashboard.py
```

### 6. Docker Compose up
```bash
$env:DOCKER_HOST="npipe:////./pipe/docker_engine"
docker-compose up -d --build
```

### 7. Kubernetes Deployments
```bash
kubectl apply -f k8s/
```

### 8. Run Verification Suites
```bash
python -m pytest -v
```
All **23 unit test specifications** have passed, validating numerical and computational pipeline correctness.
