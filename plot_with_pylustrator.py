import sys
import pandas as pd
import matplotlib.pyplot as plt
import pylustrator
import numpy as np

def main(filename, top_n=9, start_date_str="2010-01-01", end_date_str="2024-12-31"):
    try:
        # Load the file based on its extension
        df = pd.read_csv(filename) if filename.endswith('.csv') else pd.read_excel(filename)
        
        # Ensure 'created_at' is converted to datetime
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], utc=True)  # Make datetime objects timezone-aware

        # Set static start and end dates for testing
        start_date = pd.to_datetime(start_date_str, utc=True)
        end_date = pd.to_datetime(end_date_str, utc=True)
        
        # Filter data within the specified range
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]

        # Start Pylustrator for interactive plot adjustments
        pylustrator.start()

        # Separate figures for each plot
        # Plot 1: Daily averages
        plt.figure(figsize=(10, 5))
        result_columns = [col for col in df.columns if col.startswith('result_')]
        for result_column in result_columns:
            df[result_column] = pd.to_numeric(df[result_column], errors='coerce')
            daily_avg = df.groupby(df['created_at'].dt.date)[result_column].mean()
            plt.plot(daily_avg.index, daily_avg.values, label=result_column)
        plt.title('Daily Average of Results')
        plt.ylabel('Average')
        plt.xlabel('Date')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot 2: Logarithmic scale
        plt.figure(figsize=(10, 5))
        for result_column in result_columns:
            plt.plot(daily_avg.index, daily_avg.values, label=result_column)
        plt.yscale('log')
        plt.title('Logarithmic Daily Average of Results')
        plt.ylabel('Log Average')
        plt.xlabel('Date')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot 3: Monthly counts for top 'n' locations in bar chart format
        plt.figure(figsize=(10, 5))
        if 'tweet_location' in df.columns:
            top_locations = df['tweet_location'].value_counts().nlargest(top_n).index
            monthly_counts = df[df['tweet_location'].isin(top_locations)].groupby([pd.Grouper(key='created_at', freq='M'), 'tweet_location']).size().unstack().fillna(0)
            monthly_counts.plot(kind='bar', stacked=True)
            plt.title('Monthly Counts of Top Locations')
            plt.ylabel('Counts')
            plt.xlabel('Month')
            plt.xticks(rotation=90)
            plt.legend(title='Location')
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Failed to process the file: {e}")

if __name__ == "__main__":
    # Usage: script.py <filename> <top_n> <start_date> <end_date>
    args = sys.argv[1:]
    if len(args) >= 4:
        main(*args)
    else:
        print("Usage: script.py <filename> <top_n> <start_date> <end_date>")
