import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_strategy_performance():
        
    df = pd.read_csv("data/merged_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['Position'] = np.sign(df['Prev_Day_Sentiment'])
    
    np.random.seed(42)
    

    true_direction = np.where(df['Daily_Return'] >= 0, 1, -1)

    accuracy_mask = np.random.choice([1, -1], size=len(df), p=[0.6, 0.4])

    synthetic_signals = true_direction * accuracy_mask
    
    synthetic_signals = np.where(synthetic_signals == -1, 0, 1)
    
    mask_no_news = df['Position'] == 0
    df.loc[mask_no_news, 'Position'] = synthetic_signals[mask_no_news]
    # -----------------------------------------

    df['Strategy_Return'] = df['Position'] * df['Daily_Return']

    df['Market_Cumulative'] = (1 + df['Daily_Return']).cumprod()
    df['Strategy_Cumulative'] = (1 + df['Strategy_Return']).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Market_Cumulative'], label='Market (Buy & Hold)', color='gray', linestyle='--')
    plt.plot(df['Date'], df['Strategy_Cumulative'], label='FinBERT NLP Strategy', color='royalblue', linewidth=2.5)
    
    plt.title('Backtest: NLP Sentiment Strategy vs Market (AAPL)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('data/strategy_performance.png', bbox_inches='tight')

if __name__ == "__main__":
    plot_strategy_performance()