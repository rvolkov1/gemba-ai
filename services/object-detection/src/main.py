from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="Object Detection API",
    description="API for processing images and detecting objects.",
    version="0.1.0"
)

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Object Detection service is running."}

# You can add more endpoints here for submitting jobs, checking status, etc.

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
