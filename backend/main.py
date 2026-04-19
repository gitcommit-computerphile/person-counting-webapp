from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="YOLO Person Counting API")

# Load YOLO model once when app starts
model = YOLO("yolov8n.pt")

PERSON_CLASS_ID = 0  # COCO class id for 'person'


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/count-persons")
async def count_persons(file: UploadFile = File(...)):
    # Check whether uploaded file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        # Read file bytes
        contents = await file.read()

        # Open image using PIL
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run YOLO prediction
        results = model.predict(image, conf=0.25, verbose=False)

        person_count = 0
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                if cls_id == PERSON_CLASS_ID:
                    person_count += 1
                    detections.append({
                        "class_name": "person",
                        "confidence": round(conf, 4),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2)
                        }
                    })

        return JSONResponse(content={
            "filename": file.filename,
            "person_count": person_count,
            "detections": detections
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")