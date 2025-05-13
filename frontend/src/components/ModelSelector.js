import React from 'react';
import { 
  Card, 
  CardContent, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Typography, 
  Box,
  Chip,
  Collapse,
  Divider,
  Grid,
  alpha,
  useTheme,
  Paper,
  LinearProgress
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import InfoIcon from '@mui/icons-material/Info';
import MemoryIcon from '@mui/icons-material/Memory';
import TimelineIcon from '@mui/icons-material/Timeline';
import NeuralNetworkIcon from '@mui/icons-material/Hub';  // Using Hub icon for neural network

// Dictionary for mapping numeric codes to human-readable values (same as in other components)
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

const ModelSelector = ({ models, selectedModel, onChange }) => {
  const theme = useTheme();
  const [expandedModel, setExpandedModel] = React.useState(null);

  if (!models || Object.keys(models).length === 0) {
    return null;
  }

  const handleExpandClick = (modelName) => {
    setExpandedModel(expandedModel === modelName ? null : modelName);
  };

  const formatParameterValue = (param, value) => {
    // Check if this parameter has a human-readable mapping
    if (valueMappings[param] && valueMappings[param][value] !== undefined) {
      return valueMappings[param][value];
    }
    
    // Otherwise format as before
    if (typeof value === 'number') {
      // Format number to 3 decimal places if it has decimals
      return Number.isInteger(value) ? value : value.toFixed(3);
    }
    return value;
  };

  // Get model display name
  const getModelDisplayName = (name) => {
    switch(name) {
      case 'rf': return 'Random Forest';
      case 'logistic': return 'Logistic Regression';
      case 'xgboost': return 'XGBoost';
      case 'svm': return 'SVM';
      case 'knn': return 'KNN';
      case 'nn': return 'Neural Network';
      default: return name;
    }
  };
  
  // Get model icon
  const getModelIcon = (name) => {
    switch(name) {
      case 'nn': 
        return <NeuralNetworkIcon fontSize="small" sx={{ color: theme.palette.secondary.main }} />;
      default:
        return <MemoryIcon fontSize="small" sx={{ color: theme.palette.primary.main }} />;
    }
  };

  return (
    <Card sx={{ 
      borderRadius: theme.shape.borderRadius,
      boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
      border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
      overflow: 'hidden'
    }}>
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        p: 2, 
        bgcolor: alpha(theme.palette.primary.main, 0.05)
      }}>
        <MemoryIcon color="primary" sx={{ mr: 1.5 }} />
        <Typography variant="h6" color="primary" sx={{ fontWeight: 500 }}>
          ML Model Selection
        </Typography>
      </Box>
      
      <CardContent sx={{ p: 3 }}>
        <FormControl fullWidth variant="outlined" sx={{ mb: 3 }}>
          <InputLabel id="model-select-label">Select Prediction Model</InputLabel>
          <Select
            labelId="model-select-label"
            id="model-select"
            value={selectedModel}
            label="Select Prediction Model"
            onChange={onChange}
            sx={{
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: alpha(theme.palette.primary.main, 0.2),
              },
              '&:hover .MuiOutlinedInput-notchedOutline': {
                borderColor: theme.palette.primary.main,
              },
              '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                borderColor: theme.palette.primary.main,
              },
            }}
          >
            {Object.entries(models).map(([name, details]) => {
              // Get display name using our helper function
              const displayName = getModelDisplayName(name);
              
              return (
                <MenuItem key={name} value={name} sx={{ py: 1.5 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {details.is_best && <CheckCircleIcon color="success" fontSize="small" sx={{ mr: 1 }} />}
                      {getModelIcon(name)}
                      <Typography sx={{ ml: 1, fontWeight: 500 }}>{displayName}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Chip 
                        label={`Acc: ${(details.accuracy * 100).toFixed(1)}%`} 
                        size="small" 
                        color={details.is_best ? "success" : "primary"}
                        variant={details.is_best ? "filled" : "outlined"}
                        sx={{ mr: 1, fontWeight: 500 }}
                      />
                      {details.roc_auc && (
                        <Chip 
                          label={`ROC: ${(details.roc_auc * 100).toFixed(1)}%`} 
                          size="small" 
                          color={details.is_best ? "success" : "secondary"}
                          variant={details.is_best ? "filled" : "outlined"}
                          sx={{ fontWeight: 500 }}
                        />
                      )}
                    </Box>
                  </Box>
                </MenuItem>
              );
            })}
          </Select>
        </FormControl>
        
        {selectedModel && models[selectedModel] && (
          <Box sx={{ mt: 2 }} className="fade-in">
            {models[selectedModel].is_best && (
              <Paper sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                p: 2, 
                bgcolor: alpha(theme.palette.success.main, 0.08), 
                borderRadius: theme.shape.borderRadius,
                mb: 3,
                borderLeft: `4px solid ${theme.palette.success.main}`
              }}>
                <CheckCircleIcon color="success" fontSize="small" sx={{ mr: 1.5 }} />
                <Typography variant="body2" color="success.dark" sx={{ fontWeight: 500 }}>
                  This is the best performing model based on ROC AUC score
                </Typography>
              </Paper>
            )}
            
            {/* Model metrics display */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom sx={{ mb: 2, fontWeight: 500 }}>
                Model Performance
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={6}>
                  <Box sx={{ 
                    p: 2,
                    borderRadius: theme.shape.borderRadius,
                    bgcolor: alpha(theme.palette.primary.main, 0.05),
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column'
                  }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 1 }}>
                      Accuracy
                    </Typography>
                    <Typography variant="h6" color="primary" sx={{ mb: 1, fontWeight: 500 }}>
                      {(models[selectedModel].accuracy * 100).toFixed(1)}%
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={models[selectedModel].accuracy * 100} 
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        bgcolor: alpha(theme.palette.primary.main, 0.1),
                        '& .MuiLinearProgress-bar': {
                          bgcolor: theme.palette.primary.main,
                          borderRadius: 3
                        }
                      }}
                    />
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ 
                    p: 2,
                    borderRadius: theme.shape.borderRadius,
                    bgcolor: alpha(theme.palette.secondary.main, 0.05),
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column'
                  }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 1 }}>
                      ROC AUC
                    </Typography>
                    <Typography variant="h6" color="secondary" sx={{ mb: 1, fontWeight: 500 }}>
                      {models[selectedModel].roc_auc 
                        ? (models[selectedModel].roc_auc * 100).toFixed(1) + '%' 
                        : 'N/A'}
                    </Typography>
                    {models[selectedModel].roc_auc && (
                      <LinearProgress 
                        variant="determinate" 
                        value={models[selectedModel].roc_auc * 100} 
                        sx={{
                          height: 6,
                          borderRadius: 3,
                          bgcolor: alpha(theme.palette.secondary.main, 0.1),
                          '& .MuiLinearProgress-bar': {
                            bgcolor: theme.palette.secondary.main,
                            borderRadius: 3
                          }
                        }}
                      />
                    )}
                  </Box>
                </Grid>
              </Grid>
            </Box>
            
            {/* Model parameters section */}
            {models[selectedModel].parameters && Object.keys(models[selectedModel].parameters).length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Box 
                  onClick={() => handleExpandClick(selectedModel)}
                  sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                    p: 1.5,
                    borderRadius: theme.shape.borderRadius,
                    bgcolor: alpha(theme.palette.info.main, 0.05),
                    '&:hover': {
                      bgcolor: alpha(theme.palette.info.main, 0.1),
                    }
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <InfoIcon fontSize="small" color="info" sx={{ mr: 1 }} />
                    <Typography variant="body2" color="info.main" sx={{ fontWeight: 500 }}>
                      Model Parameters
                    </Typography>
                  </Box>
                  {expandedModel === selectedModel ? <ExpandLessIcon color="info" /> : <ExpandMoreIcon color="info" />}
                </Box>
                
                <Collapse in={expandedModel === selectedModel}>
                  <Box sx={{ 
                    mt: 2, 
                    p: 2, 
                    borderRadius: theme.shape.borderRadius,
                    bgcolor: alpha(theme.palette.background.default, 0.5),
                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
                  }}>
                    <Grid container spacing={2}>
                      {Object.entries(models[selectedModel].parameters).map(([param, value]) => (
                        <Grid item xs={6} key={param}>
                          <Box sx={{ mb: 1 }}>
                            <Typography variant="caption" color="text.secondary" display="block">
                              {param}:
                            </Typography>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {formatParameterValue(param, value)}
                            </Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                </Collapse>
              </Box>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ModelSelector; 