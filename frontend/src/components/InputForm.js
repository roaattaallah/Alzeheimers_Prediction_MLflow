import React from 'react';
import PatientForm from './PatientForm_new';

const InputForm = ({ onPredict, loading, featureDescriptions }) => {
  return (
    <PatientForm 
      features={featureDescriptions} 
      onSubmit={onPredict} 
    />
  );
};

export default InputForm;