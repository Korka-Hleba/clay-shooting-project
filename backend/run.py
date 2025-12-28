import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    upload_dir = base_dir / "uploads"
    models_dir = base_dir / "models"

    upload_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    print("=" * 60)
    print("CLAY SHOOTING VIDEO ANALYZER - BACKEND SERVER")
    print("=" * 60)
    print(f"Server URL: http://{host}:{port}")
    print(f"API Docs:   http://localhost:{port}/docs")
    print(f"Uploads:    {upload_dir}")
    print(f"Models:     {models_dir}")
    print(f"Database:   detections.db")
    print("=" * 60)
    print("Endpoints:")
    print("  POST /analyze/        - Analyze video file")
    print("  GET  /health/         - Health check")
    print("  GET  /detections/     - Get all detections")
    print("  GET  /export/excel/   - Export to Excel")
    print("  GET  /test-data/      - Get test data")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )