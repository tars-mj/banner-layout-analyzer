import axios from 'axios';
import { DetectionResult } from '../types/detection';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const uploadImage = async (file: File): Promise<DetectionResult> => {
    if (!file) {
        throw new Error('No file selected');
    }

    const formDataToSend = new FormData();
    formDataToSend.append('file', file);

    try {
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
        if (axios.isAxiosError(error) && error.response) {
            throw new Error(error.response.data.detail || 'Failed to upload image');
        }
        throw new Error('Failed to upload image');
    }
}; 