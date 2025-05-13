import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  alpha,
  useTheme,
  LinearProgress
} from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import AssessmentIcon from '@mui/icons-material/Assessment';

// Dictionary for mapping numeric codes to human-readable values (same as in PatientForm)
const valueMappings = {
  Gender: {
    0: 'Male',
    1: 'Female'
  },
  Ethnicity: {
    0: 'Caucasian',
    1: 'African American',
    2: 'Asian',
    3: 'Other'
  },
  EducationLevel: {
    0: 'None',
    1: 'High School',
    2: 'Bachelor\'s',
    3: 'Higher'
  },
  Smoking: {
    0: 'No',
    1: 'Yes'
  },
  FamilyHistoryAlzheimers: {
    0: 'No',
    1: 'Yes'
  },
  CardiovascularDisease: {
    0: 'No',
    1: 'Yes'
  },
  Diabetes: {
    0: 'No',
    1: 'Yes'
  },
  Depression: {
    0: 'No',
    1: 'Yes'
  },
  HeadInjury: {
    0: 'No',
    1: 'Yes'
  },
  Hypertension: {
    0: 'No',
    1: 'Yes'
  },
  MemoryComplaints: {
    0: 'No',
    1: 'Yes'
  },
  BehavioralProblems: {
    0: 'No',
    1: 'Yes'
  },
  Confusion: {
    0: 'No',
    1: 'Yes'
  },
  Disorientation: {
    0: 'No',
    1: 'Yes'
  },
  PersonalityChanges: {
    0: 'No',
    1: 'Yes'
  },
  DifficultyCompletingTasks: {
    0: 'No',
    1: 'Yes'
  },
  Forgetfulness: {
    0: 'No',
    1: 'Yes'
  }
};

const PredictionResult = ({ prediction }) => {
  const theme = useTheme();
  console.log("Prediction data:", prediction);
  
  // Check if prediction is high risk based on prediction value or risk level
  const isHighRisk = prediction.prediction === 'Positive' || 
                    (prediction.risk_level && prediction.risk_level.includes('High'));
  
  // Check if this is an ensemble prediction
  const isEnsemble = prediction.prediction_type === 'ensemble';
  
  // Check if this is the weighted ensemble
  const isWeightedEnsemble = prediction.model && 
    (prediction.model.name === 'High-Accuracy Ensemble' || 
     prediction.model.name === 'Weighted Ensemble');
  
  // Helper function to format model name nicely
  const formatModelName = (modelName) => {
    const modelMappings = {
      'rf': 'Random Forest',
      'xgboost': 'XGBoost',
      'logistic': 'Logistic Regression',
      'svm': 'Support Vector Machine',
      'knn': 'K-Nearest Neighbors',
      'nn': 'Neural Network'
    };
    
    return modelMappings[modelName] || (prediction.model && prediction.model.name) || modelName || "Unknown";
  };
  
  // Calculate confidence percentage for display
  const confidence = isEnsemble ? 
    Math.round(prediction.confidence * 100) : 
    (prediction.probability !== undefined ? Math.round(prediction.probability * 100) : null);
  
  return (
    <Card sx={{ 
      mb: 3, 
      bgcolor: isHighRisk 
        ? alpha(theme.palette.error.main, 0.04)
        : alpha(theme.palette.success.main, 0.04),
      borderLeft: isHighRisk 
        ? `4px solid ${theme.palette.error.main}` 
        : `4px solid ${theme.palette.success.main}`,
      borderRadius: 2,
      boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
      overflow: 'hidden'
    }}>
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        p: 2, 
        bgcolor: isHighRisk 
          ? alpha(theme.palette.error.main, 0.08)
          : alpha(theme.palette.success.main, 0.08)
      }}>
        <AssessmentIcon 
          sx={{ 
            mr: 1.5, 
            color: isHighRisk ? theme.palette.error.main : theme.palette.success.main 
          }} 
        />
        <Typography variant="h6" sx={{ fontWeight: 500, color: isHighRisk ? 'error.main' : 'success.main' }}>
          Prediction Results
        </Typography>
      </Box>
      
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between', 
          mb: 3,
          p: 2,
          borderRadius: 1,
          bgcolor: isHighRisk 
            ? alpha(theme.palette.error.main, 0.08)
            : alpha(theme.palette.success.main, 0.08)
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {isHighRisk ? (
              <WarningIcon sx={{ mr: 1, color: theme.palette.error.main }} />
            ) : (
              <CheckCircleIcon sx={{ mr: 1, color: theme.palette.success.main }} />
            )}
            <Typography variant="h6" color={isHighRisk ? 'error' : 'success'} sx={{ fontWeight: 'bold' }}>
              {prediction.risk_level || (isHighRisk ? 'High Risk' : 'Low Risk')}
            </Typography>
          </Box>
          
          {isEnsemble && confidence !== null && (
            <Chip 
              label={`${confidence}% Confidence`}
              color={isHighRisk ? 'error' : 'success'}
              variant="outlined"
              sx={{ fontWeight: 500 }}
            />
          )}
        </Box>
        
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          p: 2, 
          borderRadius: theme.shape.borderRadius,
          bgcolor: alpha(theme.palette.background.default, 0.5),
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        }}>
          {isWeightedEnsemble ? (
            // Simplified display for Weighted Ensemble
            <>
              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold', color: 'text.primary' }}>
                Ensemble Prediction
              </Typography>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Prediction:
                </Typography>
                <Typography variant="body2" fontWeight="bold" color={isHighRisk ? 'error.main' : 'success.main'}>
                  {prediction.prediction || 'Unknown'}
                </Typography>
              </Box>
            </>
          ) : isEnsemble ? (
            // Standard display for Voting Ensemble
            <>
              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold', color: 'text.primary' }}>
                Ensemble Prediction
              </Typography>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Prediction:
                </Typography>
                <Typography variant="body2" fontWeight="bold" color={isHighRisk ? 'error.main' : 'success.main'}>
                  {prediction.prediction || 'Unknown'}
                </Typography>
              </Box>
              
              {confidence !== null && (
                <>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1, alignItems: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      Confidence:
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {confidence}%
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 2, mt: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={confidence} 
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        bgcolor: alpha(theme.palette.primary.main, 0.1),
                        '& .MuiLinearProgress-bar': {
                          bgcolor: isHighRisk ? theme.palette.error.main : theme.palette.success.main,
                          borderRadius: 4
                        }
                      }}
                    />
                  </Box>
                </>
              )}
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Model consensus:
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {prediction.positive_votes}/{prediction.total_votes} models
                </Typography>
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              <Accordion sx={{ 
                boxShadow: 'none', 
                bgcolor: 'transparent',
                '&:before': {
                  display: 'none',
                },
              }}>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel1a-content"
                  id="panel1a-header"
                  sx={{ 
                    p: 0,
                    minHeight: 'auto',
                    '& .MuiAccordionSummary-content': {
                      margin: 0
                    }
                  }}
                >
                  <Typography variant="body2" sx={{ color: theme.palette.primary.main, fontWeight: 500 }}>
                    View Individual Model Predictions
                  </Typography>
                </AccordionSummary>
                <AccordionDetails sx={{ p: 0, pt: 1 }}>
                  <List dense disablePadding sx={{ 
                    bgcolor: alpha(theme.palette.background.paper, 0.5),
                    borderRadius: 1,
                    mt: 1
                  }}>
                    {prediction.individual_models && prediction.individual_models.map((model, index) => (
                      <ListItem 
                        key={index} 
                        divider={index < prediction.individual_models.length - 1}
                        sx={{ px: 2, py: 1 }}
                      >
                        <ListItemText
                          primary={
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {model.display_name}
                            </Typography>
                          }
                          secondary={
                            <Typography 
                              component="span" 
                              variant="body2" 
                              sx={{ 
                                color: model.prediction === 'Positive' ? 'error.main' : 'success.main',
                                fontWeight: 500
                              }}
                            >
                              {model.prediction}
                            </Typography>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>
            </>
          ) : (
            // Display for single model
            <>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Model:
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {prediction.model ? prediction.model.name : formatModelName(prediction.model_name)}
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Prediction:
                </Typography>
                <Typography variant="body2" fontWeight="bold" color={isHighRisk ? 'error.main' : 'success.main'}>
                  {prediction.prediction || 'Unknown'}
                </Typography>
              </Box>
            </>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default PredictionResult; 