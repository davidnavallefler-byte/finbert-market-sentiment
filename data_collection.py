import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, start_date, end_date, save_path="data"):
    """
    Descarga datos históricos de una acción y calcula el retorno diario.
    """
    print(f"Descargando datos para {ticker} desde {start_date} hasta {end_date}...")
    
    # Usar Ticker().history() evita el problema de MultiIndex de las versiones nuevas
    stock_obj = yf.Ticker(ticker)
    stock = stock_obj.history(start=start_date, end=end_date)
    
    if stock.empty:
        raise ValueError(f"No se encontraron datos para {ticker}.")
        
    # En .history(), 'Close' ya está ajustado (es lo que necesitamos)
    stock['Daily_Return'] = stock['Close'].pct_change()
    
    # Etiquetar si el día fue positivo (1) o negativo (0)
    stock['Target'] = (stock['Daily_Return'] > 0).astype(int)
    
    # Resetear el índice para que la fecha sea una columna normal
    stock.reset_index(inplace=True)
    
    # Quitar la zona horaria de la fecha (fundamental para luego cruzar con noticias)
    stock['Date'] = pd.to_datetime(stock['Date']).dt.tz_localize(None)
    
    # Crear carpeta si no existe y guardar
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    file_path = os.path.join(save_path, f"{ticker}_historical.csv")
    stock.to_csv(file_path, index=False)
    print(f"Datos guardados exitosamente en {file_path}")
    
    return stock

if __name__ == "__main__":
    # Prueba descargando datos de Apple del último año
    df = fetch_stock_data("AAPL", start_date="2023-01-01", end_date="2024-01-01")
    print(df[['Date', 'Close', 'Daily_Return', 'Target']].head())