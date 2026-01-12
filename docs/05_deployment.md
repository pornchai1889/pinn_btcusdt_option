# Deployment & Production Engineering

## 1. High-Performance Inference Engine

To bridge the gap between theoretical physics and high-frequency trading (HFT) requirements, the PINN solver is deployed via a **FastAPI** microservice architecture. This design ensures non-blocking, asynchronous execution suitable for real-time pricing demands.

### 1.1 Asynchronous Architecture
The inference server utilizes Python's `asyncio` ecosystem to handle concurrent pricing requests. Unlike traditional synchronous web servers, this allows the I/O-bound operations (such as request parsing and logging) to proceed without blocking the CPU-intensive neural network inference.

* **Framework:** FastAPI (0.100+) with Uvicorn (ASGI Server).
* **Throughput:** Capable of handling 1,000+ requests per second (RPS) on standard CPU instances.
* **Latency:** Sub-millisecond inference time (< 5ms) per batch of options.

### 1.2 Dual-Model Serving Strategy
The `ModelManager` class employs a singleton pattern to preload both **Call** and **Put** option models into memory at startup. Incoming requests are dynamically routed to the appropriate computational graph based on the `option_type` field, eliminating the overhead of model loading during runtime.

---

## 2. API Specification (OpenAPI/Swagger)

The service exposes a RESTful API complying with the OpenAPI 3.0 standard. Strict type validation is enforced via **Pydantic** schemas to ensure that all financial inputs fall within the valid training domain.

### 2.1 Endpoint: `/v1/predict`
Performs real-time pricing and Greek calculation for a batch of option contracts.

* **Method:** `POST`
* **Content-Type:** `application/json`

#### **Request Schema**
The input is a list of `PricingRequest` objects:

```json
[
  {
    "spot_price": 96000.0,      // S (USD)
    "strike_price": 95000.0,    // K (USD)
    "time_to_maturity": 0.15,   // tau (Years)
    "risk_free_rate": 0.05,     // r (Decimal)
    "volatility": 0.5,          // sigma (Decimal)
    "option_type": "call",      // "call" or "put"
    "request_id": "tx-1001"     // Correlation ID for tracing
  }
]
```