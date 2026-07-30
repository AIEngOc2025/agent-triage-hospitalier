import os

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Minimal Healthcheck")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Minimal server starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
