import React from 'react';
import { Paper, Typography, List, ListItem, ListItemIcon, ListItemText, Box, Tooltip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import { DetectionResult } from '../types/detection';

interface DetectionChecklistProps {
  detections: DetectionResult;
}

export const DetectionChecklist: React.FC<DetectionChecklistProps> = ({ detections }) => {
  // Sprawdź, czy w QR kodach jest niedziałający link
  const hasInvalidQrUrl = detections.qrcodes.some(qr => qr.isValidUrl === false);
  
  // Lista wszystkich możliwych elementów do detekcji
  const items = [
    { name: 'Faces', detected: detections.faces.length > 0, supported: true },
    { name: 'Logo', detected: detections.logos.length > 0, supported: true },
    { 
      name: 'QR Code', 
      detected: detections.qrcodes.length > 0, 
      supported: true,
      hasWarning: hasInvalidQrUrl,
      warningText: 'Warning: The URL in the QR code is not valid or cannot be accessed'
    },
    { name: 'Disclaimer', detected: false, supported: false },
    { name: 'CTA', detected: false, supported: false },
    { name: 'Headliner', detected: false, supported: false },
    { name: 'Website address', detected: false, supported: false },
    { name: 'Phone number', detected: false, supported: false }
  ];

  return (
    <Paper 
      elevation={0}
      sx={{ 
        p: 2, 
        maxWidth: 300,
        bgcolor: 'rgba(255, 255, 255, 0.9)',
        borderRadius: 1,
        boxShadow: '0 2px 5px rgba(0,0,0,0.1)'
      }}
    >
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 500, color: '#333' }}>
        Detection Results
      </Typography>
      
      <List dense disablePadding>
        {items.map((item, index) => (
          <ListItem key={index} dense sx={{ py: 0.5 }}>
            <ListItemIcon sx={{ minWidth: 36 }}>
              {item.supported ? (
                item.detected ? (
                  <CheckCircleIcon sx={{ color: '#4caf50' }} />
                ) : (
                  <CancelIcon sx={{ color: '#d32f2f', opacity: 0.7 }} />
                )
              ) : (
                <HourglassEmptyIcon sx={{ color: '#9e9e9e' }} />
              )}
            </ListItemIcon>
            <ListItemText 
              primary={item.supported ? item.name : `${item.name} (coming soon)`} 
              primaryTypographyProps={{ 
                sx: { 
                  color: item.supported 
                    ? (item.detected ? '#333' : '#777') 
                    : '#9e9e9e',
                  fontWeight: item.detected ? 500 : 400,
                  fontSize: item.supported ? 'inherit' : '0.85rem'
                } 
              }} 
            />
            
            {/* Ikona ostrzegawcza dla QR kodów z niedziałającym linkiem */}
            {item.hasWarning && (
              <Tooltip 
                title={item.warningText || ''} 
                arrow 
                placement="right"
              >
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <WarningAmberIcon 
                    sx={{ 
                      color: '#ff9800', 
                      fontSize: '1.2rem',
                      ml: 1
                    }} 
                  />
                </Box>
              </Tooltip>
            )}
          </ListItem>
        ))}
      </List>
    </Paper>
  );
}; 