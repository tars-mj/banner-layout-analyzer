export interface BannerForm {
    width: number;
    height: number;
    maxSectionWidth: number;
    showSections: boolean;
    sectionPositions: number[];
    margin: number;
    showMargins: boolean; 
}

export interface BoundingBox {
    bbox: [number, number, number, number]; // [x1, y1, x2, y2]
    confidence: number;
}

export interface QRCodeDetection {
    bbox: [number, number, number, number];
    data: string;
    isValidUrl?: boolean;
}

export interface DetectionResult {
    faces: Array<{
        bbox: [number, number, number, number];
        confidence: number;
    }>;
    logos: Array<{
        bbox: [number, number, number, number];
        confidence: number;
    }>;
    qrcodes: QRCodeDetection[];
}

export interface UploadResponse {
    success: boolean;
    data?: DetectionResult;
    error?: string;
}

export interface SmartSection {
    x: number;
    width: number;
    hasCollision?: boolean;
} 