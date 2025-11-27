import yfinance as yf
import json, os

def main():
    ticker = yf.Ticker("AAPL")

    # Test 1: Default news
    news_default = ticker.get_news(tab="press releases")
    with open("/testFunction.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(news_default, indent=2))

if __name__ == "__main__":
    main()
