

# SAMU Life-Saving Outcome Predictor

This project is an MVP for software quality and intelligence as part of the Postgraduate Program in Software Engineering at PUC-RJ.

The goal of this model is to predict the likelihood of survival or death in emergency cases handled by SAMU, either before or during the first aid process. By analyzing features such as the time of occurrence, patient's age, gender, and type of incident, the model forecasts whether the case may result in a fatality. Based on these predictions, the administrative department can make more informed decisions on how to handle each situation, increasing the chances of saving lives.


## First steps to follow this project:

1 - Clone git repository: 
```bash
   git clone https://github.com/gabrielsacampos/ML_SAMU.git
```
2 - Create env:
```bash
    python3 -m env venv
```
3 - Activate the env:
```bash
    source env/bin/activate
```
4 - Install requirements:
```bash
    pip install -r requirements.txt
```

## Testing:
We have implemented three simple test cases:

1 - ```test_processed_data()``` – This test verifies if the data is correctly preprocessed before being used for training the model.
2 - ```test_model_pipeline_recall()``` – This test ensures that the model's recall score is above 0.70, indicating acceptable performance in identifying positive cases.
3 - ```test_prediction()``` – This test checks if the model's predictions are correctly returning a binary outcome (1 or 0).

We can running the tests with a very simple command:
```bash
    pytest
```

## Running the front end with Streamlit.
[Streamlit](https://streamlit.io) is a Python framework that transforms scripts into interactive web apps, making it perfect for MVP testing by enabling rapid prototyping and effortless data visualization.

To launch the server and interface, use the following command:
```bash
    streamlit run src/front/page.py
```

By default, the application will be accessible at: http://localhost:8501.


## Notebooks:
You can explore the open notebooks for the [EDA/Preprocessing](https://colab.research.google.com/drive/1m1SU9tJjR-4h8C8Pa5n34Ltnh-bKdVIv?usp=sharing) of the dataset, as well as the [model](https://colab.research.google.com/drive/15O3XhrYAsjmZBHoPkyH7qZGGTdYah9vD?usp=sharing) construction process.