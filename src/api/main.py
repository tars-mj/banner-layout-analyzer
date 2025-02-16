from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from pathlib import Path
from src.models.detector import LayoutDetector
import logging
import imghdr
import aiohttp
import asyncio
from urllib.parse import urlparse

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png'}

app = FastAPI(
    title="Banner Layout Analyzer API",
    description="API for detecting faces and logos in banner images",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "http://localhost:3000",  # Local preview
        "https://frontend-production-683e.up.railway.app",  # Production frontend
        "https://web-production-4e7af.up.railway.app"  # Production backend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_image(file: UploadFile, content: bytes) -> bool:
    """
    Validates if the file is a valid image
    
    Args:
        file: Uploaded file
        content: File content
    
    Returns:
        bool: True if the file is a valid image
    
    Raises:
        HTTPException: If the file is invalid
    """
    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024:.1f}MB"
        )
    
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file extension
    ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed extensions are: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Additional image format check
    img_format = imghdr.what(None, h=content)
    if img_format not in ['jpeg', 'png']:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. File must be a valid JPEG or PNG image"
        )
    
    return True

# Singleton for the model
class ModelSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Update path to use production model
            model_path = Path("models/production/combined_detector_8s.pt")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            cls._instance = LayoutDetector(str(model_path))
        return cls._instance

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

async def validate_url(url: str) -> bool:
    """
    Checks if the given URL is active
    
    Args:
        url: URL to check
    
    Returns:
        bool: True if the URL is active
    """
    try:
        # Check if the URL is valid
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.warning(f"Invalid URL format: {url}")
            return False

        async with aiohttp.ClientSession() as session:
            try:
                async with session.head(url, allow_redirects=True, timeout=5) as response:
                    is_valid = response.status < 400
                    logger.info(f"URL validation result for {url}: {is_valid} (status: {response.status})")
                    return is_valid
            except asyncio.TimeoutError:
                logger.warning(f"Timeout while validating URL: {url}")
                return False
            except aiohttp.ClientError as e:
                logger.warning(f"Client error while validating URL {url}: {str(e)}")
                return False
    except Exception as e:
        logger.error(f"Error validating URL {url}: {str(e)}")
        return False

async def validate_qr_codes(detections: dict) -> dict:
    """
    Checks all URLs in QR codes
    
    Args:
        detections: Detection results from the model
    
    Returns:
        dict: Updated results with URL status information
    """
    try:
        if 'qrcodes' in detections:
            for qr in detections['qrcodes']:
                try:
                    if qr['data'].lower().startswith(('http://', 'https://')):
                        qr['isValidUrl'] = await validate_url(qr['data'])
                    else:
                        qr['isValidUrl'] = None  # Not a URL
                except Exception as e:
                    logger.error(f"Error processing QR code: {str(e)}")
                    qr['isValidUrl'] = None
    except Exception as e:
        logger.error(f"Error validating QR codes: {str(e)}")
    return detections

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read file content
        contents = await file.read()
        
        # Validate file
        validate_image(file, contents)
        
        # Convert image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400, 
                detail="Could not decode image. Please ensure it's a valid image file"
            )
            
        # Check image dimensions
        height, width = image.shape[:2]
        if width * height > 4096 * 4096:  # Max 4K resolution
            raise HTTPException(
                status_code=400,
                detail="Image resolution too high. Maximum supported resolution is 4K (4096x4096)"
            )
        
        # Object detection
        detector = ModelSingleton.get_instance()
        results = detector.detect_all(image)
        
        # Validate URLs in QR codes
        results = await validate_qr_codes(results)
        
        # Log success
        logger.info(f"Successfully processed image: {file.filename} ({width}x{height})")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 