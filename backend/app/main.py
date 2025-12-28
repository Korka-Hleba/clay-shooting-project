from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import uuid
import json
from sqlalchemy.orm import Session

from .config import UPLOAD_DIR, ALLOWED_EXTENSIONS
from .video_processor import video_processor
from .database import get_db, Detection, init_db
from .export import export_to_excel, export_to_csv

app = FastAPI(
    title="Clay Shooting Video Analyzer API",
    description="API для анализа видео стендовой стрельбы с детекцией мишеней",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {
        "service": "Clay Shooting Video Analyzer",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze/ - Загрузить и проанализировать видео",
            "health": "GET /health/ - Проверка состояния сервиса",
            "detections": "GET /detections/ - Получить все детекции",
            "export": "GET /export/excel/ - Экспорт в Excel"
        }
    }


@app.get("/health/")
async def health_check():
    from .model_loader import detector
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "upload_dir": str(UPLOAD_DIR),
        "database": "connected"
    }


@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    file_id = uuid.uuid4().hex[:8]
    safe_filename = file.filename.replace(" ", "_").replace("'", "").replace('"', "")
    saved_filename = f"{file_id}_{safe_filename}"
    file_path = UPLOAD_DIR / saved_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = video_processor.process_video(str(file_path), file.filename)

        saved_detections = []
        for r in results:
            if r.get("target_id"):
                detection = Detection(
                    video_file=file.filename,
                    timestamp=r["time_stamp_sec"],
                    bbox_coords=json.dumps({
                        "x1": r["pos_x"],
                        "y1": r["pos_y"],
                        "x2": r["pos_x"] + r["width"],
                        "y2": r["pos_y"] + r["height"]
                    }),
                    target_state=r["target_state"],
                    confidence=r["confidence"]
                )
                db.add(detection)
                saved_detections.append({
                    "id": detection.id,
                    "video_file": detection.video_file,
                    "timestamp": detection.timestamp,
                    "target_state": detection.target_state,
                    "confidence": detection.confidence
                })

        db.commit()

        broken_targets = [r for r in results if r.get("target_state") == "Мишень разбитая"]

        return {
            "status": "success",
            "original_filename": file.filename,
            "saved_filename": saved_filename,
            "total_frames_analyzed": len(results),
            "total_detections": len([r for r in results if r["target_id"] is not None]),
            "broken_targets_count": len(broken_targets),
            "saved_to_db": len(saved_detections),
            "results": results
        }

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки видео: {str(e)}")


@app.get("/detections/")
async def get_detections(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    detections = db.query(Detection).offset(skip).limit(limit).all()

    return [
        {
            "id": d.id,
            "video_file": d.video_file,
            "timestamp": d.timestamp,
            "bbox": json.loads(d.bbox_coords),
            "target_state": d.target_state,
            "confidence": d.confidence,
            "created_at": d.created_at.isoformat()
        }
        for d in detections
    ]


@app.get("/export/excel/")
async def export_excel(db: Session = Depends(get_db)):
    detections = db.query(Detection).all()

    detections_data = [
        {
            "id": d.id,
            "video_file": d.video_file,
            "timestamp": d.timestamp,
            "bbox": d.bbox_coords,
            "target_state": d.target_state,
            "confidence": d.confidence,
            "created_at": d.created_at.isoformat()
        }
        for d in detections
    ]

    filepath = export_to_excel(detections_data)
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="detections.xlsx"
    )


@app.get("/export/csv/")
async def export_csv(db: Session = Depends(get_db)):
    detections = db.query(Detection).all()

    detections_data = [
        {
            "id": d.id,
            "video_file": d.video_file,
            "timestamp": d.timestamp,
            "bbox": d.bbox_coords,
            "target_state": d.target_state,
            "confidence": d.confidence
        }
        for d in detections
    ]

    filepath = export_to_csv(detections_data)
    return FileResponse(
        filepath,
        media_type="text/csv",
        filename="detections.csv"
    )

    filepath = export_to_excel(broken_data, "broken_targets.xlsx")
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="broken_targets.xlsx"
    )


@app.get("/test-data/")
async def get_test_data(filename: str = "test_video.mp4"):
    test_data = video_processor._get_test_data(filename)

    return {
        "status": "test_data",
        "filename": filename,
        "results": test_data,
        "total_detections": len([r for r in test_data if r["target_id"] is not None]),
        "broken_targets_count": len([r for r in test_data if r.get("target_state") == "Мишень разбитая"])
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)