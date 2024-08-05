# Import relevant libraries
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf;
import numpy as np
import requests
import lxml.html
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import random
import time
import json
import numpy as np  # Import the numpy library
import os
from flask_cors import CORS

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'StockInfo.txt')

# Construct the path to the StockInfo.txt file

# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/trending', methods=['GET'])
def get_trending():
    def get_yahoo_finance_data(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Debug: Print the response status and URL
        print(f"Fetching data from: {url} - Status Code: {response.status_code}")
        
        if response.status_code == 200:
            # Find all links within the table rows
            links = soup.select('#fin-scr-res-table a')
            print(f"Number of links found: {len(links)}")
            
            if not links:
                print("No links found on the page.")
                return []

            data_array = []
            for link in links:
                # Extract the symbol using more robust parsing
                href = link.get('href')
                if href:
                    parts = href.split('/')
                    if len(parts) > 2:
                        symbol = parts[2]  # Assuming the symbol is the third part in the href
                        data_array.append(symbol)
                    else:
                        print(f"Unexpected href format: {href}")
                else:
                    print("No href found in link.")
                
            return data_array
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return []

    # URL for top gainers and top losers
    url_gainers = 'https://finance.yahoo.com/gainers'
    url_losers = 'https://finance.yahoo.com/losers'

    # Fetch data
    top_gainers = get_yahoo_finance_data(url_gainers)
    top_losers = get_yahoo_finance_data(url_losers)

    print("Top Gainers:", top_gainers)
    print("Top Losers:", top_losers)

    return jsonify({
        'Trending': top_gainers,
        'Losing': top_losers
    })


@app.route('/stock_info', methods=['GET'])
def get_stock_info():
    # Retrieve the stock ticker from the query parameters
    stock_ticker = request.args.get('ticker')
    
    stock = yf.Ticker(stock_ticker)
    # Get the intraday data for the current day
    intraday_data = stock.history(period='1d', interval='1m')

    # Access the most recent closing price (current value)
    current_value = intraday_data['Close'].iloc[-1]

    # Access the closing price from yesterday (second-to-last data point)
    historical_data = stock.history(period='2d', interval='1d')
    yesterday_close = historical_data['Close'].iloc[-2]

    # Calculate the change in dollars
    change_in_dollars = current_value - yesterday_close

    # Calculate the percent change
    percent_change = (change_in_dollars / yesterday_close) * 100

    return jsonify({
        'Stock': stock_ticker,
        'Value': current_value,
        'dChange': change_in_dollars,
        'pChange': percent_change
    })
        
 
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    def calculate_sector_distribution(stock_list):
        sector_counts = {}
        sector_info = {}

        # Read sector information from StockInfo.txt
        with open("StockInfo.txt", "r") as file:
            for line in file:
                parts = line.strip().split(", ")
                if len(parts) >= 3:
                    sector = parts[-1]  # Get the last part as sector
                    sector_counts[sector] = sector_counts.get(sector, 0)
                    sector_info[parts[0]] = sector  # Store sector information for each stock

        # Count occurrences of each sector in the input stock list
        for stock in stock_list:
            sector = sector_info.get(stock, "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Calculate the percentage distribution of sectors
        total_stocks = len(stock_list)
        sector_distribution = {sector: (count / total_stocks) * 100 for sector, count in sector_counts.items()}

        return sector_distribution, sector_info

    def pick_stocks_based_on_distribution(sector_distribution, total_stocks=20, existing_stocks=[]):
        picked_stocks = []

        # Pick stocks based on sector distribution percentages
        for sector, percentage in sector_distribution.items():
            num_stocks = int(total_stocks * (percentage / 100))
            with open("StockInfo.txt", "r") as file:
                stocks_in_sector = [line.split(", ")[0] for line in file if line.strip().endswith(sector)]
                
                # Exclude stocks that are already in existing_stocks
                filtered_stocks = [stock for stock in stocks_in_sector if stock not in existing_stocks]
                
                picked_stocks.extend(random.sample(filtered_stocks, min(num_stocks, len(filtered_stocks))))

        return picked_stocks

    def get_stats(ticker):
        stock = yf.Ticker(ticker)
        # Get the intraday data for the current day
        intraday_data = stock.history(period='1d', interval='1m')

        # Access the most recent closing price (current value)
        current_value = intraday_data['Close'].iloc[-1]
        return [ticker, current_value]
    
    def get_change(ticker):
        stock = yf.Ticker(ticker)
        
        # Get the intraday data for the current day
        intraday_data = stock.history(period='1d', interval='1m')

        # Access the most recent closing price (current value)
        current_value = intraday_data['Close'].iloc[-1]

        # Access the closing price from yesterday (second-to-last data point)
        historical_data = stock.history(period='2d', interval='1d')
        yesterday_close = historical_data['Close'].iloc[-2]

        # Calculate the change in dollars
        change_in_dollars = current_value - yesterday_close

        # Calculate the percent change
        percent_change = (change_in_dollars / yesterday_close) * 100
        
        return change_in_dollars, percent_change
    
    
    # Printing out the array of arrays received from the URL
    array_of_arrays_str = request.args.get('arrayOfArrays')
    FullStock_list = json.loads(array_of_arrays_str)
    print("Array of arrays received:", FullStock_list)

    stock_list = [stock[0] for stock in FullStock_list]

    # Calculate the total price and count of stocks
    total_price = sum(stock[1] for stock in FullStock_list)
    total_stocks = len(FullStock_list)

    # Calculate the average price
    average_price = total_price / total_stocks

    # Calculate sector distribution and sector information
    sector_distribution, sector_info = calculate_sector_distribution(stock_list)

    # Pick stocks based on sector distribution
    picked_stocks = pick_stocks_based_on_distribution(sector_distribution, existing_stocks=stock_list)

    # Fetch stats for the picked stocks using multithreading
    start_time = time.time()

    stats_array = []

    # Fetch stats for each picked stock individually
    for stock in picked_stocks:
        stats = get_stats(stock)
        stats_array.append(stats)

    # Sort the stats_array based on the absolute difference between each stock's price and the average_price
    sorted_stats = sorted(stats_array, key=lambda x: abs(x[1] - average_price))

    # Select the top 8 closest stocks
    closest_stocks = sorted_stats[:8]

    end_time = time.time()
    print("Average price of each stock:", average_price)
    print(f"The program took {end_time - start_time:.2f} seconds.")
    print("Information for the closest stocks:")
    # Initialize an empty array to store the information for each stock
    stock_info_array = []

    # Iterate over each stock in closest_stocks
    for stock in closest_stocks:
        ticker, price = stock
        dchange, pchange = get_change(ticker)
        sector = sector_info.get(ticker, "Unknown")
        
        # Append the information for the current stock to the stock_info_array
        stock_info_array.append([ticker, price, dchange, pchange])

    print(stock_info_array)

    # Return the array of arrays for the closest stocks
    return jsonify(stock_info_array)
    

@app.route('/SingleRecommendation', methods=['GET'])
def get_SingleRecommendations():
    def calculate_sector_distribution(stock_list):
        sector_counts = {}
        sector_info = {}

        # Read sector information from StockInfo.txt
        with open("StockInfo.txt", "r") as file:
            for line in file:
                parts = line.strip().split(", ")
                if len(parts) >= 3:
                    sector = parts[-1]  # Get the last part as sector
                    sector_counts[sector] = sector_counts.get(sector, 0)
                    sector_info[parts[0]] = sector  # Store sector information for each stock

        # Count occurrences of each sector in the input stock list
        for stock in stock_list:
            sector = sector_info.get(stock, "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Calculate the percentage distribution of sectors
        total_stocks = len(stock_list)
        sector_distribution = {sector: (count / total_stocks) * 100 for sector, count in sector_counts.items()}

        return sector_distribution, sector_info

    def pick_stocks_based_on_distribution(sector_distribution, total_stocks=20, existing_stocks=[]):
        picked_stocks = []

        # Pick stocks based on sector distribution percentages
        for sector, percentage in sector_distribution.items():
            num_stocks = int(total_stocks * (percentage / 100))
            with open("StockInfo.txt", "r") as file:
                stocks_in_sector = [line.split(", ")[0] for line in file if line.strip().endswith(sector)]
                
                # Exclude stocks that are already in existing_stocks
                filtered_stocks = [stock for stock in stocks_in_sector if stock not in existing_stocks]
                
                picked_stocks.extend(random.sample(filtered_stocks, min(num_stocks, len(filtered_stocks))))

        return picked_stocks

    def get_stats(ticker):
        stock = yf.Ticker(ticker)
        # Get the intraday data for the current day
        intraday_data = stock.history(period='1d', interval='1m')

        # Access the most recent closing price (current value)
        current_value = intraday_data['Close'].iloc[-1]
        return [ticker, current_value]
    
    def get_change(ticker):
        stock = yf.Ticker(ticker)
        
        # Get the intraday data for the current day
        intraday_data = stock.history(period='1d', interval='1m')

        # Access the most recent closing price (current value)
        current_value = intraday_data['Close'].iloc[-1]

        # Access the closing price from yesterday (second-to-last data point)
        historical_data = stock.history(period='2d', interval='1d')
        yesterday_close = historical_data['Close'].iloc[-2]

        # Calculate the change in dollars
        change_in_dollars = current_value - yesterday_close

        # Calculate the percent change
        percent_change = (change_in_dollars / yesterday_close) * 100
        
        return change_in_dollars, percent_change
    
    
    # Printing out the array of arrays received from the URL
    array_of_arrays_str = request.args.get('arrayOfArrays')
    FullStock_list = json.loads(array_of_arrays_str)
    print("Array of arrays received:", FullStock_list)

    stock_list = [stock[0] for stock in FullStock_list]

    # Calculate the total price and count of stocks
    total_price = sum(stock[1] for stock in FullStock_list)
    total_stocks = len(FullStock_list)

    # Calculate the average price
    average_price = total_price / total_stocks

    # Calculate sector distribution and sector information
    sector_distribution, sector_info = calculate_sector_distribution(stock_list)

    # Pick stocks based on sector distribution
    picked_stocks = pick_stocks_based_on_distribution(sector_distribution, existing_stocks=stock_list)

    # Fetch stats for the picked stocks using multithreading
    start_time = time.time()

    stats_array = []

    # Fetch stats for each picked stock individually
    for stock in picked_stocks:
        stats = get_stats(stock)
        stats_array.append(stats)


    # Sort the stats_array based on the absolute difference between each stock's price and the average_price
    sorted_stats = sorted(stats_array, key=lambda x: abs(x[1] - average_price))

    # Select the top 1 closest stocks
    closest_stocks = sorted_stats[:1]

    end_time = time.time()
    print("Average price of each stock:", average_price)
    print(f"The program took {end_time - start_time:.2f} seconds.")
    print("Information for the closest stocks:")
    # Initialize an empty array to store the information for each stock
    stock_info_array = []

    # Iterate over each stock in closest_stocks
    for stock in closest_stocks:
        ticker, price = stock
        dchange, pchange = get_change(ticker)
        sector = sector_info.get(ticker, "Unknown")
        
        # Append the information for the current stock to the stock_info_array
        stock_info_array.append([ticker, price, dchange, pchange])

    print(stock_info_array)

    # Return the array of arrays for the closest stocks
    return jsonify(stock_info_array)


    

# Define a route for fetching sentiment analysis based on a stock ticker
@app.route('/sentiment', methods=['GET'])
def get_sentiment():
    # Retrieve the stock ticker from the query parameters
    stock_ticker = request.args.get('ticker')
    
    stock = yf.Ticker(stock_ticker)
    # chartData = stock.history(period='1095d', interval='1d')
    chartData = stock.history(period='10950d', interval='1d')
    close_prices = chartData['Close']
    close_prices_list = close_prices.tolist()
    chartPointCt =  len(close_prices_list)


    news_articles = stock.news
    

        
    GetStockInfo = yf.Ticker(stock_ticker)

    long_business_summary = "LBS"
    if 'longBusinessSummary' in GetStockInfo.info:
        long_business_summary = GetStockInfo.info['longBusinessSummary']
    else:
        print("No 'longBusinessSummary' found in the dictionary.")# get all key value pairs that are available


    historical_data = stock.history(period='2d', interval='1d')

    # Access the closing price from yesterday (second-to-last data point)
    yesterday_close = float(historical_data['Close'].iloc[-2])

    # Get the intraday data for the current day (as you've already done)
    intraday_data = stock.history(period='1d', interval='1m')

    # Access the most recent closing price (current value)
# Access the most recent closing price (current value)
    current_value = float(intraday_data['Close'].iloc[-1])  # Convert to float

    # Calculate the change in dollars
    change_in_dollars = current_value - yesterday_close

    # Calculate the percent change
    percent_change = (change_in_dollars / yesterday_close) * 100

    article_links = []
    article_texts = []
    article_titles = []
    article_publishers = []
    article_dates = []
    articleCt = 0

    if(news_articles):

        for i, article in enumerate(news_articles):
            if i >= 15:
                break

            try:
                # Fetch the article content using requests and BeautifulSoup
                response = requests.get(article['link'])
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find and print the article body text
                article_body = soup.find('div', class_='caas-body')
                if article_body:
                    article_title = article['title']
                    article_text = article_body.get_text()
                    article_texts.append((article_title, article_text))
                    articleCt = articleCt + 1
                    article_links.append(article['link'])
                    article_publishers.append(article['publisher'])
                    providerpublishtime = datetime.utcfromtimestamp( article['providerPublishTime'])
                    current_time = datetime.utcnow()
                    time_difference = current_time - providerpublishtime

                    # If the article was published on the same day, display the time in hours
                    if time_difference.days == 0:
                        hours_ago = time_difference.seconds // 3600  # Convert seconds to hours
                        if hours_ago == 0:
                            # If less than an hour ago, calculate and display minutes
                            minutes_ago = (time_difference.seconds % 3600) // 60
                            if minutes_ago == 0:
                                formatted_time = "Just now"
                            elif minutes_ago == 1:
                                formatted_time = "1 minute ago"
                            else:
                                formatted_time = f"{minutes_ago} minutes ago"
                        elif hours_ago == 1:
                            formatted_time = "1 hour ago"
                        else:
                            formatted_time = f"{hours_ago} hours ago"
                    elif time_difference.days == 1:
                        formatted_time = "1 day ago"
                    else:
                        # If not the same day, display days
                        formatted_time = f"{time_difference.days} days ago"

                    article_dates.append(formatted_time)
                    #print("-" * 50)
                else:
                    print("Article body not found on the page.")

            except Exception as e:
                print(f"Error processing article: {str(e)}")
                
    nltk.download('vader_lexicon')
    vader = SentimentIntensityAnalyzer() # or whatever you want to call it

    data = []
    scores_list = []

    if(article_texts):
        # Iterate through the list of article texts and titles
        for i, (article_title, article_text) in enumerate(article_texts):
            # Concatenate the title and text, treating the title as the first sentence
            combined_text = article_title + ' ' + article_text
            article_titles.append(article_title)
            # Analyze the sentiment of the combined text
            sentiment_scores = vader.polarity_scores(combined_text)

            # Extract sentiment scores
            compound_score = sentiment_scores['compound']
            positive_score = sentiment_scores['pos']
            negative_score = sentiment_scores['neg']
            neutral_score = sentiment_scores['neu']
            scores = [
                ('positive_score', positive_score),
                ('negative_score', negative_score),
                ('neutral_score', neutral_score),
                ('compound_score', compound_score)
            ]
            scores_list.append(scores)

            # Determine sentiment based on the compound score
            if compound_score >= 0.05:
                sentiment = 'pos'
            elif compound_score <= -0.05:
                sentiment = 'neg'
            else:
                sentiment = 'neu'

            # Append data to the list
            data.append([article_title, sentiment_scores, compound_score, sentiment])

    combined_triplets = None
    mean_neu = 0
    mean_pos = 0
    mean_neg = 0
    overall_score = 'nothing'
    average_compound_score = 0
    if (data):
        # Create a DataFrame from the data list
        df = pd.DataFrame(data, columns=['Article Title', 'Sentiment Scores', 'Compound Score', 'Sentiment'])

        # Calculate the mean values for 'neg', 'neu', and 'pos'
        mean_neg = df['Sentiment Scores'].apply(lambda x: x['neg']).mean()
        mean_neu = df['Sentiment Scores'].apply(lambda x: x['neu']).mean()
        mean_pos = df['Sentiment Scores'].apply(lambda x: x['pos']).mean()
        
        # Calculate the mean 'Compound Score'
        average_compound_score = df['Compound Score'].mean()


        # Derive the overall sentiment score based on average polarity
        if average_compound_score >= 0.05:
            overall_score = 'positive'
        elif average_compound_score <= -0.05:
            overall_score = 'negative'
        else:
            overall_score = 'neutral'

        combined_triplets = [{'href': link, 'title': title, 'score': score, 'info': info, 'date' : date} for link, title, score, info, date in zip(article_links, article_titles, scores_list, article_publishers, article_dates)]

        # Assuming mean_neg, mean_neu, mean_pos, and average_compound_score are potentially nullable variables

    if combined_triplets is None:
    # Set combined_triplets to a default value
        combined_triplets = []  # Or any other appropriate default value

    if (not data and not news_articles and not article_texts):
        return jsonify({
            'Stock' : stock_ticker,
            'Value' : current_value,
            'yClose' : yesterday_close,
            'dChange' : change_in_dollars,
            'pChange' : percent_change,
            'LBS' : long_business_summary,
            'Close Prices': close_prices_list,
            'ChartPointCt' : chartPointCt,
        })

    return jsonify({
        'Stock' : stock_ticker,
        'Value' : current_value,
        'Count' : articleCt,
        'Articles' : combined_triplets,
        'yClose' : yesterday_close,
        'dChange' : change_in_dollars,
        'pChange' : percent_change,
        'LBS' : long_business_summary,
        'Neg': mean_neg,
        'Neu': mean_neu,
        'Pos': mean_pos,
        'Compound Score': average_compound_score,
        'Overall Sentiment': overall_score,
        'Close Prices': close_prices_list,
        'ChartPointCt' : chartPointCt,
    })


# Running app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
