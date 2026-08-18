import pandas as pd
from sentiment_analysis import load_finbert, analyze_sentiment
import os

def map_sentiment(label):
    """Convierte etiquetas de texto a valores numéricos para poder hacer medias."""
    if label == 'positive': return 1.0
    elif label == 'negative': return -1.0
    else: return 0.0

def main():
    print("1. Cargando datos financieros históricos...")
    stock_path = "data/AAPL_historical.csv"
    if not os.path.exists(stock_path):
        print("Error: No se encuentra el CSV de Apple. Ejecuta data_collection.py primero.")
        return
        
    df_stock = pd.read_csv(stock_path)
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    
    print("2. Generando noticias (Simulación de un dataset real)...")
    # Usamos fechas de principios de 2023 que coinciden con nuestros precios
    news_data = [
        {"Date": "2023-01-03", "Headline": "Apple faces massive supply chain issues in China factories."},
        {"Date": "2023-01-04", "Headline": "Apple announces record-breaking holiday sales numbers."},
        {"Date": "2023-01-05", "Headline": "Tech stocks rally as inflation slows down globally."},
        {"Date": "2023-01-05", "Headline": "New iPhone features leaked, investors remain optimistic."},
        {"Date": "2023-01-06", "Headline": "Analysts downgrade Apple stock citing weak global demand."}
    ]
    df_news = pd.DataFrame(news_data)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    
    print("3. Analizando el sentimiento con FinBERT...")
    finbert = load_finbert()
    predictions = analyze_sentiment(df_news['Headline'].tolist(), finbert)
    
    df_news['Sentiment_Label'] = [pred['label'] for pred in predictions]
    df_news['Sentiment_Score'] = df_news['Sentiment_Label'].apply(map_sentiment)
    
    # Si hay varias noticias el mismo día, hacemos la media matemática
    daily_sentiment = df_news.groupby('Date')['Sentiment_Score'].mean().reset_index()
    
    print("\n4. Cruzando datos (Data Engineering)...")
    # Unimos los precios con el sentimiento
    df_merged = pd.merge(df_stock, daily_sentiment, on='Date', how='left')
    
    # Rellenamos los días que no tienen noticias con 0 (Neutral)
    df_merged['Sentiment_Score'] = df_merged['Sentiment_Score'].fillna(0)
    
    # EL TOQUE QUANT: Desplazamos (shift) el sentimiento 1 día hacia adelante.
    # El sentimiento de ayer es el que usamos para predecir el movimiento de hoy.
    df_merged['Prev_Day_Sentiment'] = df_merged['Sentiment_Score'].shift(1)
    
    # Borramos la primera fila porque al desplazar se queda vacía (NaN)
    df_merged = df_merged.dropna()
    
    print("\n--- RESULTADO FINAL (Primeras filas cruzadas) ---")
    print(df_merged[['Date', 'Close', 'Daily_Return', 'Sentiment_Score', 'Prev_Day_Sentiment']].head(6))
    
    df_merged.to_csv("data/merged_data.csv", index=False)
    print("\nDatos guardados en data/merged_data.csv")

if __name__ == "__main__":
    main()