import { useState } from 'react'
import { Container, Box, Alert, Snackbar, Paper, IconButton } from '@mui/material'
import { ImageUploader } from './components/ImageUploader'
import { ImageViewer } from './components/ImageViewer'
import { BannerFormComponent } from './components/BannerForm'
import { uploadImage } from './services/api'
import { DetectionResult, BannerForm } from './types/detection'
import ReplayIcon from '@mui/icons-material/Replay';

export const App = () => {
  const [selectedImage, setSelectedImage] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [detections, setDetections] = useState<DetectionResult>({
    faces: [],
    logos: [],
    qrcodes: []
  })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [formData, setFormData] = useState<BannerForm>({
    width: 0,
    height: 0,
    maxSectionWidth: 0,
    showSections: true,
    sectionPositions: [],
    margin: 0,
    showMargins: true
  })
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true)

  const isFormValid = formData.width > 0 && 
                     formData.height > 0 && 
                     formData.maxSectionWidth > 0 &&
                     selectedFile !== null;

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
  };

  const handleImageUpload = async () => {
    if (!isFormValid || !selectedFile) {
      setError('Please select a file and fill in all required fields');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Create URL for preview
      const imageUrl = URL.createObjectURL(selectedFile);
      console.log('Created image URL:', imageUrl);
      
      // Upload and process image
      const result = await uploadImage(selectedFile);
      console.log('Upload result:', result);
      
      // Set image and detections only after successful processing
      setSelectedImage(imageUrl);
      setDetections(result);
      
      console.log('Updated state with:', {
        imageUrl,
        detections: result,
        selectedFile: selectedFile.name
      });
    } catch (err) {
      console.error('Upload error:', err);
      setError(err instanceof Error ? err.message : 'Failed to process image');
      setSelectedImage('');
      setDetections({
        faces: [],
        logos: [],
        qrcodes: []
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFormChange = (newFormData: BannerForm) => {
    setFormData(newFormData);
  };

  const handleSectionPositionsChange = (positions: number[]) => {
    setFormData(prev => ({
      ...prev,
      sectionPositions: positions
    }));
  };

  const handleReset = () => {
    setSelectedImage('');
    setSelectedFile(null);
    setDetections({
      faces: [],
      logos: [],
      qrcodes: []
    });
    setFormData({
      width: 0,
      height: 0,
      maxSectionWidth: 0,
      showSections: true,
      sectionPositions: [],
      margin: 0,
      showMargins: true
    });
  };

  return (
    <Box sx={{ 
      height: '100vh',
      bgcolor: '#f5f5f5',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <Container 
        maxWidth="lg" 
        sx={{
          flex: 1,
          py: 3
        }}
      >
        {/* Top Form Section */}
        <Paper 
          elevation={3}
          sx={{
            p: 3,
            mb: 4,
            bgcolor: 'white',
            borderRadius: 2
          }}
        >
          <BannerFormComponent
            formData={formData}
            onChange={handleFormChange}
            showBoundingBoxes={showBoundingBoxes}
            onBoundingBoxesChange={setShowBoundingBoxes}
          />
        </Paper>

        {/* Main Content Section */}
        <Box sx={{ 
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 'calc(100vh - 300px)',
          gap: 2
        }}>
          {!selectedImage ? (
            <Paper 
              elevation={0}
              sx={{
                p: 6,
                width: '100%',
                maxWidth: 600,
                bgcolor: 'white',
                borderRadius: 2,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}
            >
              <ImageUploader 
                onImageSelect={handleFileSelect}
                onUpload={handleImageUpload}
                isLoading={isLoading}
                isFormValid={isFormValid}
                selectedFile={selectedFile}
              />
            </Paper>
          ) : (
            <Box 
              sx={{
                width: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 3
              }}
            >
              <ImageViewer 
                imageUrl={selectedImage}
                detections={detections}
                formData={formData}
                onSectionPositionsChange={handleSectionPositionsChange}
                showBoundingBoxes={showBoundingBoxes}
              />
              <IconButton 
                onClick={handleReset}
                sx={{ 
                  bgcolor: 'white',
                  boxShadow: 2,
                  p: 2,
                  transition: 'all 0.2s ease-in-out',
                  '&:hover': {
                    bgcolor: 'white',
                    boxShadow: 4,
                    transform: 'scale(1.1)'
                  }
                }}
                size="large"
              >
                <ReplayIcon fontSize="large" />
              </IconButton>
            </Box>
          )}
        </Box>

        <Snackbar 
          open={!!error} 
          autoHideDuration={6000} 
          onClose={() => setError(null)}
        >
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        </Snackbar>
      </Container>
    </Box>
  )
}

export default App
