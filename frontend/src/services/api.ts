import axios from 'axios';
import { DetectionResult } from '../types/detection';

// Get environment
const isDevelopment = import.meta.env.DEV;

// Define fallback URLs
const DEVELOPMENT_URL = 'http://localhost:8000';
const PRODUCTION_URL = 'https://web-production-4e7af.up.railway.app';

// Get API URL from environment or use fallback
const API_URL = import.meta.env.VITE_API_URL || (isDevelopment ? DEVELOPMENT_URL : PRODUCTION_URL);

console.log('Environment:', isDevelopment ? 'development' : 'production');
console.log('API URL from env:', import.meta.env.VITE_API_URL);
console.log('Using API URL:', API_URL);

export const uploadImage = async (file: File): Promise<DetectionResult> => {
    if (!file) {
        throw new Error('No file selected');
    }

    const formDataToSend = new FormData();
    formDataToSend.append('file', file);

    try {
        console.log('Sending request to:', `${API_URL}/api/detect`); // For debugging
        const response = await axios.post<DetectionResult>(
            `${API_URL}/api/detect`,
            formDataToSend,
            {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            }
        );
        return response.data;
    } catch (error) {
        console.error('Upload error:', error); // For debugging
        if (axios.isAxiosError(error) && error.response) {
            throw new Error(error.response.data.detail || 'Failed to upload image');
        }
        throw new Error('Failed to upload image');
    }
}; 