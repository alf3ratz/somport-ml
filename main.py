import os
import asyncio
import random
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import cv2
import numpy as np

app = FastAPI()

S3_BUCKET_NAME = "my-local-bucket"
S3_ENDPOINT_URL = os.environ.get('S3_ENDPOINT_URL')  # Для MinIO или LocalStack

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    endpoint_url=S3_ENDPOINT_URL
)


class PredictRequest(BaseModel):
    number: int


def download_video_from_s3(prefix: str) -> List[str]:
    """Скачивает все видео из S3 с указанным префиксом и возвращает список локальных путей к файлам."""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        if 'Contents' not in response:
            raise HTTPException(status_code=404, detail=f"No videos found with prefix {prefix}")

        video_paths = []
        for obj in response['Contents']:
            key = obj['Key']
            local_path = key.split('/')[-1]
            s3_client.download_file(S3_BUCKET_NAME, key, local_path)
            video_paths.append(local_path)
        return video_paths
    except NoCredentialsError:
        raise HTTPException(status_code=400, detail="AWS credentials not found!")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 error: {e}")


def read_video_frames(video_path: str) -> List[np.ndarray]:
    """Читает кадры из видеофайла и возвращает их как список массивов NumPy."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def predict_model(frames: List[np.ndarray]) -> dict:
    """Заглушка для функции предсказания модели."""
    # Здесь можно заменить на реальную модель
    result = {
        "prediction": random.choice(["positive", "negative"]),
        "confidence": random.uniform(0.5, 1.0)
    }
    return result


@app.post("/predict")
async def predict(request: PredictRequest):
    prefix = f"{request.number}/"
    video_paths = download_video_from_s3(prefix)

    all_results = []
    for video_path in video_paths:
        frames = read_video_frames(video_path)
        result = predict_model(frames)
        all_results.append({
            "video_path": video_path,
            "result": result
        })

    return {"results": all_results}


if __name__ == "__main__":
    import uvicorn

    try:
        if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
            raise ValueError("AWS credentials are missing!")

        uvicorn.run(app, host="0.0.0.0", port=8081)
    except KeyboardInterrupt:
        print("Server stopped")