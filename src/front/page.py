import sys
import streamlit as st
import pandas as pd

sys.path.append('src/')
from api.models.ModelPipeline import ModelPipeline
from api.models.CONSTANTS.FEATURES_CONSTANTS import TIMESTAMP, AGE, GENDER, TYPES, SUBTYPES

model = ModelPipeline('src/api/ML/models/model_pipeline.pkl')

st.title('SAMU - Detector de possível óbito antes ou durante o atendimento')

selected_timestamp = st.selectbox('Qual o horário da ocorrência', TIMESTAMP.keys())
selected_age = st.selectbox('Qual a idade do paciente', AGE.keys())
selected_gender = st.selectbox('Qual o gênero do paciente', GENDER.keys())
selected_type = st.selectbox('Qual o tipo de ocorrência', TYPES)
selected_subtype = st.selectbox('Qual o subtipo de ocorrência', SUBTYPES)

# mapping by keys
timestamp_value = TIMESTAMP[selected_timestamp]
age_value = AGE[selected_age]
gender_value = GENDER[selected_gender]

# dnt have keys
type_value = selected_type
subtype_value = selected_subtype


X_input = pd.DataFrame({
    'timestamp': [timestamp_value],
    'type': [type_value],
    'subtype': [subtype_value],
    'gender': [gender_value],
    'age': [age_value],
})

if st.button('Detectar risco'):
    prediction = model.predict(X_input)

    if prediction[0] == 1:
        st.markdown('<p style="color:red;">O paciente tem <strong>RISCO</strong> de óbito antes ou durante o atendimento</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green;"><strong>Não há</strong> risco de óbito antes ou durante o atendimento.</p>', unsafe_allow_html=True)
    
    

