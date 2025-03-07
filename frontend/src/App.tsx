import { useState, useEffect } from 'react'
import { Container, Box, Alert, Snackbar, Paper, IconButton, Typography, Button } from '@mui/material'
import { ImageUploader } from './components/ImageUploader'
import { ImageViewer } from './components/ImageViewer'
import { BannerFormComponent } from './components/BannerForm'
import { DetectionChecklist } from './components/DetectionChecklist'
import { LoginForm } from './components/LoginForm'
import { uploadImage, checkAuth, login, logout, getStoredUsername } from './services/api'
import { DetectionResult, BannerForm, AuthStatus, LoginCredentials } from './types/detection'
import ReplayIcon from '@mui/icons-material/Replay'
import LogoutIcon from '@mui/icons-material/Logout'

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
    height: 500,
    maxSectionWidth: 300,
    showSections: true,
    sectionPositions: [],
    margin: 20,
    showMargins: true
  })
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true)
  const [authStatus, setAuthStatus] = useState<AuthStatus>({ status: 'checking', authorized: false })
  const [loginError, setLoginError] = useState<string | null>(null)
  const [isLoginLoading, setIsLoginLoading] = useState(false)
  const [username, setUsername] = useState<string | null>(null)

  // Sprawdza autoryzację przy starcie aplikacji
  useEffect(() => {
    const verifyAuth = async () => {
      try {
        setAuthStatus({ status: 'checking', authorized: false });
        
        // Sprawdź autoryzację
        const status = await checkAuth(null); // Użyj null, aby funkcja używała zapisanego klucza
        setAuthStatus(status);
        
        // Jeśli zalogowany, pobierz nazwę użytkownika
        if (status.authorized) {
          setUsername(getStoredUsername());
        }
      } catch (err) {
        console.error('Auth error:', err);
        setAuthStatus({ status: 'error', authorized: false });
      }
    };

    verifyAuth();
  }, []);

  const isFormValid = formData.maxSectionWidth > 0 && selectedFile !== null;

  const handleLogin = async (credentials: LoginCredentials) => {
    setIsLoginLoading(true);
    setLoginError(null);
    
    try {
      await login(credentials);
      setAuthStatus({ status: 'success', authorized: true });
      setUsername(credentials.username);
    } catch (err) {
      console.error('Login error:', err);
      setLoginError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setIsLoginLoading(false);
    }
  };
  
  const handleLogout = async () => {
    try {
      await logout();
      setAuthStatus({ status: 'no_key', authorized: false });
      setUsername(null);
      handleReset(); // Reset aplikacji przy wylogowaniu
    } catch (err) {
      console.error('Logout error:', err);
      setError('Logout failed');
    }
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    
    // Odczytaj szerokość obrazu
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);
    
    img.onload = () => {
      // Aktualizuj formData z szerokością obrazu
      setFormData(prev => ({
        ...prev,
        width: img.width // Ustawiam szerokość na podstawie wymiarów obrazu
      }));
      
      // Zwolnij URL obiektu
      URL.revokeObjectURL(objectUrl);
    };
    
    img.src = objectUrl;
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
      height: 500,
      maxSectionWidth: 300,
      showSections: true,
      sectionPositions: [],
      margin: 20,
      showMargins: true
    });
  };

  // Renderuj ekran logowania, jeśli użytkownik nie jest zalogowany
  if (authStatus.status === 'checking') {
    return (
      <Box sx={{ height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Typography variant="h6">Checking authorization...</Typography>
      </Box>
    );
  }

  if (!authStatus.authorized) {
    return (
      <Box sx={{ 
        height: '100vh', 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        bgcolor: '#f5f5f5'
      }}>
        <LoginForm 
          onLogin={handleLogin}
          isLoading={isLoginLoading}
          error={loginError}
        />
      </Box>
    );
  }

  // Główna aplikacja - gdy użytkownik jest zalogowany
  return (
    <Box sx={{ 
      minHeight: '100vh',
      bgcolor: '#f5f5f5',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      <Container 
        disableGutters
        maxWidth={false} 
        sx={{
          flex: 1,
          p: 0,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'auto'
        }}
      >
        {/* Top Form Section - przylega do krawędzi */}
        <Paper 
          elevation={0}
          square
          sx={{
            pt: 4,
            pb: 4,
            px: 3,
            mb: 4,
            bgcolor: 'white',
            borderRadius: 0,
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            m: 0,
            width: '100%',
            position: 'relative',
            boxSizing: 'border-box'
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
          minHeight: { xs: 'auto', md: 'calc(100vh - 300px)' },
          gap: 2,
          position: 'relative',
          px: 2,
          pb: 10,
          boxSizing: 'border-box'
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
              <Box 
                sx={{
                  width: '100%',
                  display: 'flex', 
                  flexDirection: { xs: 'column', md: 'row' },
                  alignItems: { xs: 'center', md: 'flex-start' },
                  justifyContent: 'center',
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
                <DetectionChecklist 
                  detections={detections}
                />
              </Box>
              
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
        
        {/* Przycisk wylogowania w lewym dolnym rogu */}
        <Box 
          sx={{ 
            position: 'fixed', 
            bottom: 20, 
            left: 20, 
            zIndex: 1000
          }}
        >
          <Button
            variant="contained"
            color="primary"
            startIcon={<LogoutIcon />}
            onClick={handleLogout}
            sx={{ 
              px: 3, 
              py: 1, 
              borderRadius: 2,
              boxShadow: 3,
              textTransform: 'none',
              fontSize: '1rem'
            }}
          >
            {username ? `Logout (${username})` : 'Logout'}
          </Button>
        </Box>
      </Container>
    </Box>
  );
}

export default App
