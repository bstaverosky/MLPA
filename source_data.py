#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 21:15:52 2023

@author: bstaverosky
"""

import requests
from bs4 import BeautifulSoup

# Send a GET request to the webpage
url = "https://www.gurufocus.com/economic_indicators/5728/us-inflation-rate"
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table containing the data
table = soup.find('table')

# Extract the table headers
headers = [header.text for header in table.find_all('th')]

# Extract the table rows
rows = []
for row in table.find_all('tr'):
    rows.append([cell.text.strip() for cell in row.find_all('td')])

# Print the headers and data
print(headers)
for row in rows:
    print(row)

import yfinance as yf

# Define the ticker symbol for the S&P 500 index
ticker_symbol = "^GSPC"

# Download historical data for the ticker symbol
data = yf.download(ticker_symbol, period="1d")

# Calculate the aggregate earnings yield
earnings_yield = data["Earnings"] / data["Close"]

# Print the last 10 entries of the earnings yield
print(earnings_yield.tail(10))


### SHILLER DATA ###

import urllib.request

def download_file(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        print("File downloaded successfully!")
    except Exception as e:
        print("Error occurred while downloading the file:", e)

# Example usage
file_url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
save_location = "/home/bstaverosky/Documents/projects/MLPA/shiller_data.xls"

download_file(file_url, save_location)
