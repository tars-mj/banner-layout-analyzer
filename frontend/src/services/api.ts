import axios from 'axios';
import { DetectionResult, AuthStatus, LoginResponse, LoginCredentials } from '../types/detection';

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

// Lokalne przechowywanie klucza API
const API_KEY_STORAGE_KEY = 'banner_analyzer_api_key';
const USERNAME_STORAGE_KEY = 'banner_analyzer_username';

export const getStoredApiKey = (): string | null => {
    return localStorage.getItem(API_KEY_STORAGE_KEY);
};

export const getStoredUsername = (): string | null => {
    return localStorage.getItem(USERNAME_STORAGE_KEY);
};

export const storeApiKey = (apiKey: string, username: string): void => {
    localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
    localStorage.setItem(USERNAME_STORAGE_KEY, username);
};

export const clearStoredApiKey = (): void => {
    localStorage.removeItem(API_KEY_STORAGE_KEY);
    localStorage.removeItem(USERNAME_STORAGE_KEY);
};

// Funkcja do tworzenia konfiguracji nagłówków z kluczem API
const getAuthHeaders = () => {
    const apiKey = getStoredApiKey();
    return {
        'X-API-Key': apiKey || ''
    };
};

// Funkcja do logowania
export const login = async (credentials: LoginCredentials): Promise<LoginResponse> => {
    try {
        console.log('Logging in with username:', credentials.username);
        const response = await axios.post<LoginResponse>(
            `${API_URL}/api/login`,
            null,
            {
                params: {
                    username: credentials.username,
                    password: credentials.password
                }
            }
        );
        
        // Zapisz klucz API i nazwę użytkownika
        storeApiKey(response.data.api_key, response.data.username);
        
        return response.data;
    } catch (error) {
        console.error('Login error:', error);
        if (axios.isAxiosError(error) && error.response) {
            throw new Error(error.response.data.detail || 'Login failed');
        }
        throw new Error('Login failed');
    }
};

// Funkcja do wylogowania
export const logout = async (): Promise<void> => {
    try {
        const headers = getAuthHeaders();
        await axios.post(`${API_URL}/api/logout`, null, { headers });
        clearStoredApiKey();
    } catch (error) {
        console.error('Logout error:', error);
    }
};

export const checkAuth = async (authKey: string | null): Promise<AuthStatus> => {
    try {
        // Najpierw spróbuj użyć przechowywanego klucza API
        const storedApiKey = getStoredApiKey();
        const apiKey = authKey || storedApiKey;
        
        if (!apiKey) {
            return { status: 'no_key', authorized: false };
        }
        
        console.log('Checking auth with key:', apiKey.substring(0, 8) + '...');
        const url = `${API_URL}/api/auth?auth_key=${apiKey}`;
        const response = await axios.get(url);
        
        return {
            status: response.data.status,
            authorized: response.data.authorized
        };
    } catch (error) {
        console.error('Auth check error:', error);
        return {
            status: 'error',
            authorized: false
        };
    }
};

export const uploadImage = async (file: File): Promise<DetectionResult> => {
    if (!file) {
        throw new Error('No file selected');
    }

    const formDataToSend = new FormData();
    formDataToSend.append('file', file);

    try {
        console.log('Sending request to:', `${API_URL}/api/detect`);
        const headers = getAuthHeaders();
        
        const response = await axios.post<DetectionResult>(
            `${API_URL}/api/detect`,
            formDataToSend,
            {
                headers: {
                    ...headers,
                    'Content-Type': 'multipart/form-data',
                },
            }
        );
        return response.data;
    } catch (error) {
        console.error('Upload error:', error);
        if (axios.isAxiosError(error) && error.response) {
            throw new Error(error.response.data.detail || 'Failed to upload image');
        }
        throw new Error('Failed to upload image');
    }
}; 