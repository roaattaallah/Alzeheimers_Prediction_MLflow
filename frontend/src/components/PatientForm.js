import React, { useState, useEffect } from 'react';
import { 
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  MenuItem,
  Button,
  FormControlLabel,
  Checkbox,
  Box,
  Tooltip,
  IconButton
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

// Dictionary for mapping numeric codes to human-readable values
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

// Initial values for a standard patient
const initialPatientData = {
  Age: 52.0,
  Gender: 0.0,
  Ethnicity: 0.0,
  EducationLevel: 3.0,
  BMI: 23.5,
  Smoking: 0.0,
  AlcoholConsumption: 1.0,
  PhysicalActivity: 3.0,
  DietQuality: 5.0,
  SleepQuality: 5.0,
  FamilyHistoryAlzheimers: 0.0,
  CardiovascularDisease: 0.0,
  Diabetes: 0.0,
  Depression: 0.0,
  HeadInjury: 0.0,
  Hypertension: 0.0,
  SystolicBP: 115.0,
  DiastolicBP: 75.0,
  CholesterolTotal: 150.0,
  CholesterolLDL: 70.0,
  CholesterolHDL: 65.0,
  CholesterolTriglycerides: 80.0,
  MMSE: 30.0,
  FunctionalAssessment: 10.0,
  MemoryComplaints: 0.0,
  BehavioralProblems: 0.0,
  ADL: 10.0,
  Confusion: 0.0,
  Disorientation: 0.0,
  PersonalityChanges: 0.0,
  DifficultyCompletingTasks: 0.0,
  Forgetfulness: 0.0
};

const PatientForm = ({ features, onSubmit }) => {
  // State initialization
  const [formData, setFormData] = useState(initialPatientData);
  const [missingValues, setMissingValues] = useState([]);
  
  // Version stamp to verify which version is loaded
  console.log("PatientForm Version: REBUILT - NO BUTTONS - " + new Date().toISOString());

  // Initialize all fields from API feature descriptions
  useEffect(() => {
    if (features && Object.keys(features).length > 0) {
      const initialFormData = {...formData};
      
      // For each feature from the API, ensure we have a value in our form
      Object.keys(features).forEach(feature => {
        if (!(feature in initialFormData)) {
          initialFormData[feature] = getDefaultValueForFeature(feature);
        }
      });
      
      setFormData(initialFormData);
    }
  }, [features]);

  // Helper function to get a default value based on feature type
  const getDefaultValueForFeature = (feature) => {
    if (isBinaryField(feature)) return 0;
    if (feature.includes("Cholesterol")) return 150;
    if (feature === "SystolicBP") return 120;
    if (feature === "DiastolicBP") return 80;
    if (feature === "BMI") return 25;
    return 0;
  };

  if (!features || Object.keys(features).length === 0) {
    return null;
  }

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
    
    // Remove from missing values if it was marked as missing
    if (missingValues.includes(name)) {
      setMissingValues(missingValues.filter(item => item !== name));
    }
  };

  const toggleMissingValue = (name) => {
    if (missingValues.includes(name)) {
      setMissingValues(missingValues.filter(item => item !== name));
    } else {
      setMissingValues([...missingValues, name]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Create a copy of the form data
    const submissionData = { ...formData };
    
    // Set missing values to null
    missingValues.forEach(field => {
      submissionData[field] = null;
    });
    
    onSubmit(submissionData);
  };

  // Helper function to determine if a field should be a select or text input
  const isBinaryField = (fieldName) => {
    return fieldName in valueMappings && Object.keys(valueMappings[fieldName]).length === 2;
  };

  const isCategoricalField = (fieldName) => {
    return fieldName in valueMappings && Object.keys(valueMappings[fieldName]).length > 2;
  };

  const isScaleField = (fieldName) => {
    const scaleFields = {
      'AlcoholConsumption': 3,
      'PhysicalActivity': 3,
      'DietQuality': 5,
      'SleepQuality': 5
    };
    return fieldName in scaleFields ? scaleFields[fieldName] : false;
  };

  // Create menu items for a scale
  const createScaleMenuItems = (max) => {
    const items = [];
    for (let i = 0; i <= max; i++) {
      items.push(<MenuItem key={i} value={i}>{i}</MenuItem>);
    }
    return items;
  };

  // Create menu items for categorical fields
  const createCategoricalMenuItems = (fieldName) => {
    if (!valueMappings[fieldName]) return null;
    
    return Object.entries(valueMappings[fieldName]).map(([value, label]) => (
      <MenuItem key={value} value={parseInt(value, 10)}>{label}</MenuItem>
    ));
  };

  const renderFeatureInputs = () => {
    return Object.keys(features).map(feature => {
      // Skip rendering if feature isn't in our form data or API features
      if (!(feature in formData)) return null;
      
      if (isBinaryField(feature)) {
        return (
          <Grid item xs={12} sm={6} md={4} key={feature}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
              <TextField
                name={feature}
                label={feature}
                fullWidth
                select
                value={missingValues.includes(feature) ? '' : formData[feature]}
                onChange={handleInputChange}
                disabled={missingValues.includes(feature)}
                variant="outlined"
                margin="normal"
              >
                {createCategoricalMenuItems(feature)}
              </TextField>
              <Tooltip title={features[feature] || `${feature} value`}>
                <IconButton size="small" sx={{ mt: 2.5, ml: 1 }}>
                  <HelpOutlineIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <FormControlLabel
              control={
                <Checkbox
                  checked={missingValues.includes(feature)}
                  onChange={() => toggleMissingValue(feature)}
                />
              }
              label="Missing value"
            />
          </Grid>
        );
      } else if (isCategoricalField(feature)) {
        return (
          <Grid item xs={12} sm={6} md={4} key={feature}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
              <TextField
                name={feature}
                label={feature}
                fullWidth
                select
                value={missingValues.includes(feature) ? '' : formData[feature]}
                onChange={handleInputChange}
                disabled={missingValues.includes(feature)}
                variant="outlined"
                margin="normal"
              >
                {createCategoricalMenuItems(feature)}
              </TextField>
              <Tooltip title={features[feature] || `${feature} value`}>
                <IconButton size="small" sx={{ mt: 2.5, ml: 1 }}>
                  <HelpOutlineIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <FormControlLabel
              control={
                <Checkbox
                  checked={missingValues.includes(feature)}
                  onChange={() => toggleMissingValue(feature)}
                />
              }
              label="Missing value"
            />
          </Grid>
        );
      } else if (isScaleField(feature)) {
        const maxScale = isScaleField(feature);
        return (
          <Grid item xs={12} sm={6} md={4} key={feature}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
              <TextField
                name={feature}
                label={feature}
                fullWidth
                select
                value={missingValues.includes(feature) ? '' : formData[feature]}
                onChange={handleInputChange}
                disabled={missingValues.includes(feature)}
                variant="outlined"
                margin="normal"
              >
                {createScaleMenuItems(maxScale)}
              </TextField>
              <Tooltip title={`${features[feature]} (Scale 0-${maxScale})`}>
                <IconButton size="small" sx={{ mt: 2.5, ml: 1 }}>
                  <HelpOutlineIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <FormControlLabel
              control={
                <Checkbox
                  checked={missingValues.includes(feature)}
                  onChange={() => toggleMissingValue(feature)}
                />
              }
              label="Missing value"
            />
          </Grid>
        );
      } else {
        return (
          <Grid item xs={12} sm={6} md={4} key={feature}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
              <TextField
                name={feature}
                label={feature}
                fullWidth
                type="number"
                value={missingValues.includes(feature) ? '' : formData[feature]}
                onChange={handleInputChange}
                disabled={missingValues.includes(feature)}
                InputProps={{ 
                  inputProps: { 
                    step: feature === 'BMI' ? 0.1 : 1 
                  } 
                }}
                variant="outlined"
                margin="normal"
              />
              <Tooltip title={features[feature] || `${feature} value`}>
                <IconButton size="small" sx={{ mt: 2.5, ml: 1 }}>
                  <HelpOutlineIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <FormControlLabel
              control={
                <Checkbox
                  checked={missingValues.includes(feature)}
                  onChange={() => toggleMissingValue(feature)}
                />
              }
              label="Missing value"
            />
          </Grid>
        );
      }
    });
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Patient Parameters (Rebuilt)
        </Typography>
        
        <Typography variant="body2" color="text.secondary" paragraph>
          Enter the patient's information below. Missing values will be handled appropriately.
        </Typography>
        
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            {renderFeatureInputs()}
            
            <Grid item xs={12} sx={{ mt: 2 }}>
              <Button 
                type="submit" 
                variant="contained" 
                color="primary" 
                fullWidth
                size="large"
              >
                Make Prediction
              </Button>
            </Grid>
          </Grid>
        </form>
      </CardContent>
    </Card>
  );
};

export default PatientForm;