import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Paper, 
  CircularProgress, 
  Grid, 
  FormControl, 
  FormLabel, 
  RadioGroup, 
  FormControlLabel, 
  Radio,
  AppBar,
  Toolbar,
  useTheme,
  useMediaQuery,
  Divider,
  alpha
} from '@mui/material';
import ModelSelector from './components/ModelSelector';
import InputForm from './components/InputForm';
import PredictionResult from './components/PredictionResult';
import axios from 'axios';
import './styles.css';
import MedicalInformationIcon from '@mui/icons-material/MedicalInformation';

// Define the API base URL
const API_BASE_URL = 'http://localhost:5030';

function App() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState({});
  const [features, setFeatures] = useState({});
  const [selectedModel, setSelectedModel] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [predictionMode, setPredictionMode] = useState('single'); // 'single', 'ensemble', or 'high-accuracy'

  // Fetch models and features on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        const modelsResponse = await axios.get(`${API_BASE_URL}/api/models?_t=${timestamp}`);
        const featuresResponse = await axios.get(`${API_BASE_URL}/api/features?_t=${timestamp}`);
        
        // Process models data to ensure consistent structure
        const processedModels = {};
        Object.entries(modelsResponse.data).forEach(([modelName, modelDetails]) => {
          processedModels[modelName] = {
            accuracy: modelDetails.accuracy,
            roc_auc: modelDetails.roc_auc || 0,
            is_best: modelDetails.is_best,
            // Ensure parameters are available under a consistent key name
            parameters: modelDetails.parameters || modelDetails.params || {}
          };
        });
        
        setModels(processedModels);
        setFeatures(featuresResponse.data);
        
        // Set the best model as default or the one with highest ROC AUC if no best is marked
        let bestModel = Object.entries(processedModels).find(
          ([_, details]) => details.is_best
        );
        
        if (!bestModel) {
          // If no best model is marked, select the one with highest ROC AUC
          bestModel = Object.entries(processedModels).reduce((best, [name, details]) => {
            if (!best || (details.roc_auc && details.roc_auc > best[1].roc_auc)) {
              return [name, details];
            }
            return best;
          }, null);
          
          // If no model has ROC AUC, fall back to accuracy
          if (!bestModel || !bestModel[1].roc_auc) {
            bestModel = Object.entries(processedModels).reduce((best, [name, details]) => {
              if (!best || details.accuracy > best[1].accuracy) {
                return [name, details];
              }
              return best;
            }, null);
          }
        }
        
        if (bestModel) {
          setSelectedModel(bestModel[0]);
        }
        
        setError(null);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load models or features. Please check if the backend server is running.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
    setPrediction(null);
  };

  const handlePredictionModeChange = (event) => {
    setPredictionMode(event.target.value);
    setPrediction(null);
  };

  const handleSubmit = async (patientData) => {
    try {
      setLoading(true);
      console.log('Submitting prediction with data:', patientData);
      console.log('Prediction mode:', predictionMode);
      
      // Add a timestamp to avoid caching issues
      const timestamp = new Date().getTime();
      
      let response;
      
      if (predictionMode === 'ensemble') {
        // Use the ensemble endpoint
        response = await axios.post(`${API_BASE_URL}/api/predict-ensemble?_t=${timestamp}`, {
          parameters: patientData,
          timestamp: timestamp
        });
      } else if (predictionMode === 'high-accuracy') {
        // Use the direct ensemble endpoint instead of high-accuracy-ensemble
        response = await axios.post(`${API_BASE_URL}/api/direct-ensemble?_t=${timestamp}`, {
          parameters: patientData,
          timestamp: timestamp
        });
      } else {
        // Use the single model endpoint
        response = await axios.post(`${API_BASE_URL}/api/predict?_t=${timestamp}`, {
          model_name: selectedModel,
          parameters: patientData,
          timestamp: timestamp
        });
      }
      
      console.log('Prediction response:', response.data);
      setPrediction(response.data);
      setError(null);
      
      // Scroll to top when prediction is received
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      console.error('Error making prediction:', err);
      setError('Failed to make a prediction. Please try again.');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <AppBar position="static" elevation={0} sx={{ 
        backgroundColor: alpha(theme.palette.primary.main, 0.95),
        backdropFilter: 'blur(8px)'
      }}>
        <Toolbar>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <MedicalInformationIcon sx={{ mr: 1.5, fontSize: 30 }} />
            <Typography variant="h6" component="div" sx={{ fontWeight: 500 }}>
              Alzheimer's Prediction Tool
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>
      
      <Container maxWidth="lg" className="fade-in">
        <Box sx={{ my: 4 }}>
          <Box sx={{ textAlign: 'center', mb: 4 }} className="slide-up">
            <Typography variant="h3" component="h1" gutterBottom color="primary">
              Alzheimer's Disease Prediction
            </Typography>
            
            <Typography variant="subtitle1" gutterBottom sx={{ maxWidth: '800px', mx: 'auto', mb: 3, color: 'text.secondary' }}>
              This application predicts the likelihood of Alzheimer's disease based on patient parameters using tuned ML models.
            </Typography>
            
            <Divider sx={{ width: '100px', mx: 'auto', mb: 4 }} />
          </Box>

          {error && (
            <Paper sx={{ 
              p: 3, 
              my: 2, 
              bgcolor: alpha(theme.palette.error.main, 0.08),
              borderLeft: `4px solid ${theme.palette.error.main}`,
              borderRadius: 1
            }}>
              <Typography color="error">{error}</Typography>
            </Paper>
          )}

          {loading && !error ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
              <CircularProgress />
            </Box>
          ) : (
            <>
              <Grid container spacing={4}>
                {/* Left side: Model selector and prediction result */}
                <Grid item xs={12} md={4}>
                  <Box sx={{ mb: 3, p: 2, bgcolor: alpha(theme.palette.primary.main, 0.03), borderRadius: 2 }}>
                    <FormControl component="fieldset">
                      <FormLabel component="legend" sx={{ fontWeight: 500, color: 'primary.main', mb: 1 }}>
                        Prediction Mode
                      </FormLabel>
                      <RadioGroup
                        value={predictionMode}
                        onChange={handlePredictionModeChange}
                      >
                        <FormControlLabel value="single" control={<Radio />} label="Single Model" />
                        <FormControlLabel value="ensemble" control={<Radio />} label="Voting Ensemble" />
                        <FormControlLabel value="high-accuracy" control={<Radio />} label="Weighted Ensemble" />
                      </RadioGroup>
                    </FormControl>
                  </Box>
                  
                  {predictionMode === 'single' && (
                    <Box className="slide-up" sx={{ mb: 3 }}>
                      <ModelSelector 
                        models={Object.fromEntries(
                          Object.entries(models).filter(([key]) => key !== 'high_accuracy_ensemble')
                        )} 
                        selectedModel={selectedModel} 
                        onChange={handleModelChange} 
                      />
                    </Box>
                  )}
                  
                  {predictionMode === 'ensemble' && (
                    <Paper sx={{ 
                      p: 3, 
                      mb: 3, 
                      bgcolor: alpha(theme.palette.primary.main, 0.03),
                      borderLeft: `4px solid ${theme.palette.primary.main}`,
                      borderRadius: 2,
                      transition: 'all 0.3s ease'
                    }} className="slide-up">
                      <Typography variant="h6" gutterBottom color="primary.main">
                        Voting Ensemble
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                        Using weighted majority voting across all available models for more reliable predictions.
                      </Typography>
                    </Paper>
                  )}
                  
                  {predictionMode === 'high-accuracy' && (
                    <Paper sx={{ 
                      p: 3, 
                      mb: 3, 
                      bgcolor: alpha(theme.palette.secondary.main, 0.03),
                      borderLeft: `4px solid ${theme.palette.secondary.main}`,
                      borderRadius: 2,
                      transition: 'all 0.3s ease'
                    }} className="slide-up">
                      <Typography variant="h6" gutterBottom color="secondary.main">
                        Weighted Ensemble
                      </Typography>
                      <Typography variant="body2" gutterBottom sx={{ color: 'text.secondary' }}>
                        Using an optimal weighted ensemble of our best models for more accurate and reliable predictions.
                      </Typography>
                      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          <strong>Accuracy:</strong> 90%
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          <strong>ROC AUC:</strong> 96%
                        </Typography>
                      </Box>
                    </Paper>
                  )}
                  
                  {prediction && (
                    <Box className="slide-up">
                      <PredictionResult prediction={prediction} />
                    </Box>
                  )}
                </Grid>
                
                {/* Right side: Input form */}
                <Grid item xs={12} md={8}>
                  <Paper 
                    elevation={0}
                    sx={{ 
                      p: { xs: 2, md: 3 }, 
                      borderRadius: 2,
                      boxShadow: '0 4px 20px rgba(0,0,0,0.05)',
                      border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
                    }}
                  >
                    <InputForm 
                      onPredict={handleSubmit}
                      loading={loading}
                      featureDescriptions={features}
                    />
                  </Paper>
                </Grid>
              </Grid>
            </>
          )}
        </Box>
        
        <Box 
          component="footer" 
          sx={{ 
            py: 3, 
            mt: 4, 
            textAlign: 'center',
            borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            color: 'text.secondary'
          }}
        >
          <Typography variant="body2">
            © {new Date().getFullYear()} Alzheimer's Disease Prediction Tool
          </Typography>
        </Box>
      </Container>
    </>
  );
}

export default App;