# Build the model in one stage, ship only what serving needs in the next.
# The training image carries Optuna, Jupyter and the raw CSVs; the runtime image
# needs a booster, a context snapshot and FastAPI.
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
# The tuned hyperparameters travel with the source. Without them the build falls
# back to train.py's defaults, and the image would ship a model that does not
# match the metrics in the README.
COPY models/best_params.json models/best_params.json
RUN python -m src.data.make_dataset --n-auctions 48000 --seed 42 \
 && python -m src.features.leakage_audit \
 && python -m src.models.train --save


FROM python:3.11-slim AS runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 appuser

# Serving needs a much smaller dependency set than training.
RUN pip install --no-cache-dir \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" "pydantic>=2.6" \
        "lightgbm>=4.3" "pandas>=2.1" "numpy>=1.26" "scikit-learn>=1.4" \
        "joblib>=1.3" "pyarrow>=14.0"

COPY --from=builder /build/models/ models/
COPY src/ src/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
