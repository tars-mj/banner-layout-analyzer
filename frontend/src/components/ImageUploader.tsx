import React, { useCallback } from 'react';
import { Button, Typography, Stack } from '@mui/material';

interface ImageUploaderProps {
    onImageSelect: (file: File) => void;
    onUpload: () => void;
    isLoading: boolean;
    isFormValid: boolean;
    selectedFile: File | null;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ 
    onImageSelect, 
    onUpload,
    isLoading,
    isFormValid,
    selectedFile
}) => {
    const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            onImageSelect(file);
        }
    }, [onImageSelect]);

    return (
        <Stack 
            spacing={3} 
            alignItems="center"
            sx={{ width: '100%', py: 4 }}
        >
            <Stack spacing={1} alignItems="center">
                <Button
                    variant="contained"
                    component="label"
                    disabled={isLoading}
                    size="large"
                    sx={{ 
                        minWidth: 200,
                        fontSize: '1.1rem'
                    }}
                >
                    {isLoading ? 'Processing...' : 'Select image'}
                    <input
                        type="file"
                        hidden
                        accept="image/jpeg,image/png"
                        onChange={handleFileChange}
                    />
                </Button>
                <Typography 
                    variant="body1" 
                    sx={{ 
                        height: '1.5rem',
                        visibility: selectedFile ? 'visible' : 'hidden'
                    }}
                >
                    {selectedFile ? `Selected file: ${selectedFile.name}` : 'Placeholder'}
                </Typography>
            </Stack>
            <Button
                variant="contained"
                color="primary"
                disabled={!isFormValid || !selectedFile || isLoading}
                onClick={onUpload}
                size="large"
                sx={{ 
                    minWidth: 200,
                    fontSize: '1.1rem'
                }}
            >
                Analyze image
            </Button>
            <Typography variant="body2" color="text.secondary">
                Supported formats: JPEG, PNG (max 1MB)
            </Typography>
        </Stack>
    );
}; 