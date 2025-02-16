import React from 'react';
import { TextField, FormControlLabel, Switch, Stack, Box, Typography } from '@mui/material';
import { BannerForm } from '../types/detection';

interface BannerFormProps {
    formData: BannerForm;
    onChange: (formData: BannerForm) => void;
    showBoundingBoxes: boolean;
    onBoundingBoxesChange: (show: boolean) => void;
}

export const BannerFormComponent: React.FC<BannerFormProps> = ({ 
    formData, 
    onChange,
    showBoundingBoxes,
    onBoundingBoxesChange
}) => {
    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value, checked } = event.target;
        
        if (name === 'showSections' || name === 'showMargins') {
            onChange({
                ...formData,
                [name]: checked
            });
            return;
        }

        onChange({
            ...formData,
            [name]: Number(value)
        });
    };

    return (
        <Box>
            {/* Header with title and switches */}
            <Box sx={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                mb: 3
            }}>
                <Typography variant="h6" sx={{ color: '#1976d2' }}>
                    Banner settings
                </Typography>
                <Stack direction="row" spacing={2}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={formData.showSections}
                                onChange={handleChange}
                                name="showSections"
                                color="primary"
                            />
                        }
                        label="Sections"
                    />
                    <FormControlLabel
                        control={
                            <Switch
                                checked={formData.showMargins}
                                onChange={handleChange}
                                name="showMargins"
                                color="primary"
                            />
                        }
                        label="Margins"
                    />
                    <FormControlLabel
                        control={
                            <Switch
                                checked={showBoundingBoxes}
                                onChange={(e) => onBoundingBoxesChange(e.target.checked)}
                                color="primary"
                            />
                        }
                        label="Detections"
                    />
                </Stack>
            </Box>

            {/* Input fields */}
            <Stack 
                direction={{ xs: 'column', sm: 'row' }} 
                spacing={4} 
                alignItems="center"
            >
                <TextField
                    label="Banner width (cm)"
                    type="number"
                    name="width"
                    value={formData.width || ''}
                    onChange={handleChange}
                    size="small"
                    fullWidth
                    sx={{ flex: 1 }}
                />
                <TextField
                    label="Banner height (cm)"
                    type="number"
                    name="height"
                    value={formData.height || ''}
                    onChange={handleChange}
                    size="small"
                    fullWidth
                    sx={{ flex: 1 }}
                />
                <TextField
                    label="Maximum division width (cm)"
                    type="number"
                    name="maxSectionWidth"
                    value={formData.maxSectionWidth || ''}
                    onChange={handleChange}
                    size="small"
                    fullWidth
                    sx={{ flex: 1 }}
                />
                <TextField
                    label="Margin (cm)"
                    type="number"
                    name="margin"
                    value={formData.margin || ''}
                    onChange={handleChange}
                    size="small"
                    fullWidth
                    sx={{ flex: 1 }}
                />
            </Stack>
        </Box>
    );
}; 