from transformers import pipeline
import pandas as pd

def load_finbert():
    """
    Descarga e inicializa el modelo FinBERT pre-entrenado.
    """
    print("Cargando modelo FinBERT (puede tardar un poco la primera vez)...")
    # 'ProsusAI/finbert' es el estándar de la industria para textos financieros
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return sentiment_pipeline

def analyze_sentiment(news_list, model_pipeline):
    """
    Pasa una lista de textos por el modelo y devuelve el sentimiento.
    """
    print(f"Analizando {len(news_list)} titulares...")
    results = model_pipeline(news_list)
    return results

if __name__ == "__main__":
    # 1. Cargamos el modelo
    finbert = load_finbert()
    
    # 2. Creamos unos titulares de prueba (Mock data)
    sample_news = [
        "Apple reports record-breaking revenue for the third quarter.",
        "Inflation rises unexpectedly, causing massive market panic.",
        "Microsoft announces minor routine updates to its cloud infrastructure.",
        "Tesla stock plunges 10% after unexpected production delays."
    ]
    
    # 3. Hacemos la predicción
    predictions = analyze_sentiment(sample_news, finbert)
    
    # 4. Mostramos los resultados de forma elegante
    print("\n--- RESULTADOS DEL ANÁLISIS ---")
    for text, pred in zip(sample_news, predictions):
        print(f"Titular: {text}")
        print(f"Sentimiento: {pred['label'].upper()} | Confianza: {pred['score']:.4f}\n")