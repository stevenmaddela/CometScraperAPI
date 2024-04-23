# Import relevant libraries
import pandas as pd
from bs4 import BeautifulSoup
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
import numpy as np  # Import numpy library
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
    url = 'https://finance.yahoo.com/gainers'

    trendingArray = []
    ytext = requests.get(url).text
    yroot = lxml.html.fromstring(ytext)
    for x in yroot.xpath('//*[@id="fin-scr-res-table"]//a'):
        trendingArray.append(x.attrib['href'].split("/")[-1].split("?")[0])

    url2 = 'https://finance.yahoo.com/losers'

    losingArray = []
    ytext = requests.get(url2).text
    yroot = lxml.html.fromstring(ytext)
    for x in yroot.xpath('//*[@id="fin-scr-res-table"]//a'):
        losingArray.append(x.attrib['href'].split("/")[-1].split("?")[0])

    return jsonify({
        'Trending' : trendingArray,
        'Losing' : losingArray
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
        with open(file_path, 'r') as file:
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
            with open(file_path, 'r') as file:
                stocks_in_sector = [line.split(", ")[0] for line in file if line.strip().endswith(sector)]
                
                # Exclude stocks that are already in existing_stocks
                filtered_stocks = [stock for stock in stocks_in_sector if stock not in existing_stocks]
                
                picked_stocks.extend(random.sample(filtered_stocks, min(num_stocks, len(filtered_stocks))))

        return picked_stocks

    def get_stats(ticker):
        info = yf.Tickers(ticker).tickers[ticker].info
        return [ticker, info['currentPrice']]
    
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
    average_price = total_price / total_stocks if total_price / total_stocks != 0 else 1

    # Calculate sector distribution and sector information
    sector_distribution, sector_info = calculate_sector_distribution(stock_list)

    # Pick stocks based on sector distribution
    picked_stocks = pick_stocks_based_on_distribution(sector_distribution, existing_stocks=stock_list)

    # Fetch stats for the picked stocks using multithreading
    start_time = time.time()

    stats_array = []

    # Fetch stats for the picked stocks using multithreading
    with ThreadPoolExecutor() as executor:
        for stats in executor.map(get_stats, picked_stocks):
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



        
    GetStockInfo = yf.Ticker(stock_ticker)

    long_business_summary = "LBS"
    if 'longBusinessSummary' in GetStockInfo.info:
        long_business_summary = GetStockInfo.info['longBusinessSummary']
    else:
        print("No 'longBusinessSummary' found in the dictionary.")# get all key value pairs that are available


  
    
    return jsonify({
        'Stock' : stock_ticker,
        'LBS' : long_business_summary,
        'Close Prices': close_prices_list,
        'ChartPointCt' : chartPointCt,
    })




# Running app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
