#
# Chain of responsibility to implement a three step pipeline
#

# Need to locate or create some good data to pass through this chain of responsibility.  For now, we will just pass a string message through the chain od responsibility.  The message will be processed by each handler in the chain, and each handler will decide whether to handle the message or pass it on to the next handler in the chain.

# Inspired by Chain Of Responsibility Design Pattern | Python Example https://www.youtube.com/watch?v=QOW1IN8i8J8
# 
# and Refactoring a 500-Line Method with the Pipeline Pattern https://www.youtube.com/watch?v=RfknMfzTUbo&t=496s

import csv
from functools import wraps
import io
import os
from abc import ABC, abstractmethod
from typing import Any
from unittest import result
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastparquet as fp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import logging
import time

from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logging.info("%s completed in %.3f ms",
                         func.__name__,
                         elapsed * 1000)
    return wrapper


class Handler(ABC):                         #handler is an abstract class that defines the interface for handling requests and for setting the next handler in the chain.
    def __init__(self, successor=None):
        self.successor = successor

    @abstractmethod
    def process_stage(self, message):
        pass

    def next_stage(self, message):
        self.make_entry(message)

        if(self.__successor is None):
            return
        else:
            self.__successor.process_stage(message)

# 
# Get logger going
#

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@timer
class BronzeHandler(Handler):
    def process_stage(self, source: str):
        if source == "bronze":
            print("BronzeHandler: Handling bronze source.")
        elif source == "any":
            print("BronzeHandler: Handling any source.")
        
        try:
            source:str = source

            df = pd.read_csv(source, delimiter="|")  # Specifies delimiter as '|'
            print("DataFrame loaded successfully!")
            print(df.head())  # Show first 5 rows

            # Filter out null values from the DataFrame
            null_count = df['Account'].isnull().sum()
            print(f"Null values in 'Account' column: {null_count}")

            df = df[df['Account'].notnull()]
            print(df.head())  # Show first 5 rows

            # Filter out null values from the DataFrame
            null_count = df['Account'].isnull().sum()
            print(f"Null values in 'Account' column: {null_count}")

            df = df[df['Account'].notnull()]
            print(df.head())  # Show first 5 rows

            # Filter not a number values an Account column

            null_count = df.isnull().sum().sum()
        
            print(f"Null values in {source}: {null_count}")

            # Drop rows with non-numeric values in Account and BilledAmount columns

            df["Account"] = pd.to_numeric(df["Account"], errors="coerce")
            df.dropna(subset=["Account"], inplace=True)
        
            df["BilledAmount"] = pd.to_numeric(df["BilledAmount"], errors="coerce")
            df.dropna(subset=["BilledAmount"], inplace=True)
            print(df.head())  # Show first 5 rows

            # Write the 'cleansed' DataFrame to a CSV file

            df.to_csv('bronze.csv', index=False)

        except FileNotFoundError:
            print(f"Error: File '{source}' not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{source}' is empty.")
        except pd.errors.ParserError as e:
            print(f"Parsing error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        if self.successor is not None:
            self.successor.process_stage('bronze.csv')

@timer
class SilverHandler(Handler):
    def process_stage(self, source: str):
        if source == "silver":
            print("SilverHandler: Handling silver source.")
        elif source == "any":
            print("SilverHandler: Handling any source.")
    
        try:
            bronze:str = source

            df = pd.read_csv(bronze)  # Specifies delimiter as ','
            print(f"DataFrame loaded successfully from {bronze}!")
            print(df.head())  # Show first 5 rows

            # Filter not a unreasonable values in BilledAmount column

            amount = pd.to_numeric(df['BilledAmount'], errors='coerce')
            df.drop(df[amount > 10000000].index, inplace=True)

            # Write the 'cleansed' DataFrame to a CSV file

            df.to_csv('silver.csv', index=False)

        except FileNotFoundError:
            print(f"Error: File '{bronze}' not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{bronze}' is empty.")
        except pd.errors.ParserError as e:
            print(f"Parsing error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        if self.successor is not None:
            self.successor.process_stage('silver.csv')
#
#   Create the classes that implement the chain of responsibility.
#
@timer
class GoldHandler(Handler):
    def process_stage(self, source: str):
        if source == "gold":
            print("GoldHandler: Handling gold source.")
        elif source == "any":
            print("GoldHandler: Handling any source.")

        try:
            silver:str = source

            df = pd.read_csv(silver)  # Specifies delimiter as ','
            print(f"DataFrame loaded successfully from {silver} in GoldHandler!")
            logger.info(f"DataFrame loaded successfully from {silver} in GoldHandler!")
            print(df.head())  # Show first 5 rows

            # Split the incomming dataframe to two dataframes
            pymtdf = df[['Account','DatePaid', 'AmtPaid']].copy()
            pymtdf.to_csv('goldpymt.csv', index=False)

            df.drop(columns=['DatePaid', 'AmtPaid'], inplace=True, errors='ignore')  # Will run smoothly even if 'xxxxx' isn't a column

            # Write the 'cleansed' DataFrame to a CSV file as gold.csv

            df.to_csv('gold.csv', index=False)
            df.to_parquet('gold.parquet', index=False)

        except FileNotFoundError:
            print(f"Error: File '{silver}' not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{silver}' is empty.")
        except pd.errors.ParserError as e:
            print(f"Parsing error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e} {e.__traceback__}")

        gold:str = "gold.csv"
        if self.successor is not None:
            self.successor.process_stage(gold)

#
#   Wireup the chain of responsibility
#

goldHandler1 = GoldHandler(successor = None)
silverHandler1 = SilverHandler(successor=goldHandler1)
bronzeHandler1 = BronzeHandler(successor=silverHandler1)

#
#  send a message (file) to the chain
#

bronzeHandler1.process_stage("source.csv")

#
#   Now truy to read the gold.parquet file and display the first 5 rows
#

park = pq.read_table('gold.parquet')
print(park.to_pandas().head())  # Show first 5 rows
#
#   Now try to read the gold.parquet file using pandas and display the first 5 rows
#   Having saved the gold.parquet file, we can read it back in using pandas and display the first 5 rows. This is a common operation when working with Parquet files, as they are often used for efficient data storage and retrieval.
#   This also means we can start additional work from here as opposed to doing all the processing needed to get to gold.
#
df = pd.read_parquet('gold.parquet')
print(df.head())  # Show first 5 rows
#
#   Graph the number of unique accounts by state using a bar chart.  This is from ChatGPT.
#
account_counts = (
    df.groupby("State")["Account"]
      .nunique()
      .sort_index()
)

plt.figure(figsize=(14, 6))
account_counts.plot(kind="bar")

plt.title("Unique Accounts by State")
plt.xlabel("State")
plt.ylabel("Number of Accounts")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.savefig("AccountsByState.png", dpi=300, bbox_inches="tight")  # Save the figure with high resolution and tight layout


#
# BilledAmount by State using a bar chart.  This is from ChatGPT.
#

billed_by_state = (
    df.groupby("State")["BilledAmount"]
      .sum()
      .sort_index()
)

plt.figure(figsize=(16, 6))
billed_by_state.plot(kind="bar")

plt.title("Total Billed Amount by State")
plt.xlabel("State")
plt.ylabel("Total Billed Amount ($)")

plt.xticks(rotation=45)

plt.gca().yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: f'${y:,.0f}')
)

plt.tight_layout()
plt.show()