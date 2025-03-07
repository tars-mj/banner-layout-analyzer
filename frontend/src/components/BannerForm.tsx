import React from 'react';
import { TextField, Box, Typography } from '@mui/material';
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

    // Toggle switch style for the blue switches on top
    const ToggleSwitch = ({ checked, label, onChange }: { 
        checked: boolean, 
        label: string, 
        onChange: (event: React.MouseEvent<HTMLDivElement>) => void 
    }) => (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box 
                onClick={onChange}
                sx={{ 
                    width: 36, 
                    height: 20, 
                    bgcolor: checked ? '#2196f3' : '#e0e0e0', 
                    borderRadius: 10,
                    position: 'relative',
                    cursor: 'pointer',
                    transition: 'background-color 0.3s',
                    display: 'flex',
                    alignItems: 'center',
                    px: 0.5
                }}
            >
                <Box 
                    sx={{ 
                        width: 16, 
                        height: 16, 
                        bgcolor: 'white', 
                        borderRadius: '50%',
                        position: 'absolute',
                        left: checked ? 'calc(100% - 18px)' : '2px',
                        transition: 'left 0.3s'
                    }} 
                />
            </Box>
            <Typography variant="body2" sx={{ color: checked ? '#2196f3' : '#757575' }}>
                {label}
            </Typography>
        </Box>
    );

    return (
        <Box sx={{ width: '100%' }}>
            {/* Top header with logos and title - mocno poszerzony */}
            <Box sx={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                mb: 4,
                width: '100%',
                position: 'relative' // Dodaję pozycjonowanie względne
            }}>
                {/* Logo po lewej */}
                <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    width: '15%', // Używam procentów zamiast stałej szerokości
                    pl: 4 // Zwiększam padding, aby zachować symetrię z prawym logo
                }}>
                    <img src="/opinion.svg" alt="Opinion Logo" style={{ height: '40px', maxWidth: '100%' }} />
                </Box>
                
                {/* Środkowa sekcja */}
                <Box sx={{ 
                    textAlign: 'center',
                    width: '70%', // Stała szerokość dla środkowej części
                }}>
                    <Typography variant="h6" sx={{ color: '#1976d2', mb: 2 }}>
                        Banner settings
                    </Typography>
                    
                    {/* Switches as blue controls */}
                    <Box sx={{ 
                        display: 'flex', 
                        justifyContent: 'center', 
                        gap: 5
                    }}>
                        <ToggleSwitch 
                            checked={formData.showSections}
                            label="Sections"
                            onChange={() => onChange({...formData, showSections: !formData.showSections})}
                        />
                        <ToggleSwitch 
                            checked={formData.showMargins}
                            label="Margins"
                            onChange={() => onChange({...formData, showMargins: !formData.showMargins})}
                        />
                        <ToggleSwitch 
                            checked={showBoundingBoxes}
                            label="Detections"
                            onChange={() => onBoundingBoxesChange(!showBoundingBoxes)}
                        />
                    </Box>
                </Box>
                
                {/* Logo po prawej */}
                <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'flex-end',
                    width: '15%', // Używam procentów zamiast stałej szerokości
                    pr: 8 // Zwiększam padding z prawej strony, aby logo było dalej od krawędzi
                }}>
                    <img src="/pp.svg" alt="PP Logo" style={{ height: '40px', maxWidth: '100%' }} />
                </Box>
            </Box>
            
            {/* Input fields in one row - tylko 2 pola, mniejsze i wyśrodkowane */}
            <Box sx={{ 
                display: 'flex',
                justifyContent: 'center',
                gap: 3,
                mb: 3
            }}>
                <TextField
                    label="Maximum division width (cm)"
                    type="number"
                    name="maxSectionWidth"
                    value={formData.maxSectionWidth || ''}
                    onChange={handleChange}
                    size="small"
                    sx={{ width: '250px' }}
                    InputProps={{ sx: { borderRadius: 1 } }}
                />
                <TextField
                    label="Margin (cm)"
                    type="number"
                    name="margin"
                    value={formData.margin || ''}
                    onChange={handleChange}
                    size="small"
                    sx={{ width: '250px' }}
                    InputProps={{ sx: { borderRadius: 1 } }}
                />
            </Box>
        </Box>
    );
}; 