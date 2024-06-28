import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import tweepy
import pandas as pd
from geopy.geocoders import Nominatim
from tkinter import filedialog


class DataGenerationPage(tk.Frame):
    def __init__(self, parent, styles):
        tk.Frame.__init__(self, parent)
        self.grid_columnconfigure(1, weight=1) 
        
        # Title Label
        tk.Label(self, text="Data Generation from Twitter", **styles.title_style).grid(row=0, columnspan=2, sticky='ew', pady=10, padx=5)

        # Bearer Token
        tk.Label(self, text="Twitter API v2 Bearer Token", **styles.label_style).grid(row=1, column=0, sticky='ew', padx=5)
        self.token_entry = tk.Entry(self, show="*", **styles.entry_style)
        self.token_entry.grid(row=1, column=1, sticky='ew', padx=5)

        # Keywords
        tk.Label(self, text="Keywords", **styles.label_style).grid(row=2, column=0, sticky='ew', padx=5)
        self.keywords_entry = tk.Entry(self, **styles.entry_style)
        self.keywords_entry.grid(row=2, column=1, sticky='ew', padx=5)

        # Include Retweets and Quotes
        self.include_retweets_var = tk.BooleanVar()
        tk.Checkbutton(self, text="Include retweets", variable=self.include_retweets_var, **styles.checkbutton_style).grid(row=3, column=0, sticky='ew', padx=5)
        self.include_quotes_var = tk.BooleanVar()
        tk.Checkbutton(self, text="Include quotes", variable=self.include_quotes_var, **styles.checkbutton_style).grid(row=3, column=1, sticky='ew', padx=5)

        # Geo-Location Entries
        tk.Label(self, text="Coordinate X (Latitude)", **styles.label_style).grid(row=4, column=0, sticky='ew', padx=5)
        self.geo_x_entry = tk.Entry(self, **styles.entry_style)
        self.geo_x_entry.grid(row=4, column=1, sticky='ew', padx=5)

        tk.Label(self, text="Coordinate Y (Longitude)", **styles.label_style).grid(row=5, column=0, sticky='ew', padx=5)
        self.geo_y_entry = tk.Entry(self, **styles.entry_style)
        self.geo_y_entry.grid(row=5, column=1, sticky='ew', padx=5)

        tk.Label(self, text="Radius (km)", **styles.label_style).grid(row=6, column=0, sticky='ew', padx=5)
        self.radius_entry = tk.Entry(self, **styles.entry_style)
        self.radius_entry.grid(row=6, column=1, sticky='ew', padx=5)

        # Date Range with Date Entry
        tk.Label(self, text="Start Date", **styles.label_style).grid(row=7, column=0, sticky='ew', padx=5)
        self.start_date_entry = DateEntry(self, selectmode='day', year=2020, month=1, day=1, date_pattern='y-mm-dd', **styles.entry_style)
        self.start_date_entry.grid(row=7, column=1, sticky='ew', padx=5)

        tk.Label(self, text="End Date", **styles.label_style).grid(row=8, column=0, sticky='ew', padx=5)
        self.end_date_entry = DateEntry(self, selectmode='day', year=2020, month=1, day=1, date_pattern='y-mm-dd', **styles.entry_style)
        self.end_date_entry.grid(row=8, column=1, sticky='ew', padx=5)

        # Number of Tweets
        tk.Label(self, text="Number of Tweets", **styles.label_style).grid(row=9, column=0, sticky='ew', padx=5)
        self.tweets_number_spinbox = tk.Spinbox(self, from_=0, to=1000000, **styles.entry_style)
        self.tweets_number_spinbox.grid(row=9, column=1, sticky='ew', padx=5)

        # Language Dropdown
        tk.Label(self, text="Language", **styles.label_style).grid(row=10, column=0, sticky='ew', padx=5)
        self.language_combobox = ttk.Combobox(self, values=["English (en)", "German (de)", "French (fr)", "Turkish (tr)"], **styles.entry_style)
        self.language_combobox.grid(row=10, column=1, sticky='ew', padx=5)
        
        # Search and Download Button
        self.search_download_button = tk.Button(self, text="Search and Download Tweets", command=self.search_and_download, **styles.button_style)
        self.search_download_button.grid(row=11, column=0, columnspan=2, pady=20, padx=5, sticky='e')


    def search_and_download(self):
        bearer_token = self.token_entry.get()
        if not bearer_token:
            messagebox.showerror("Error", "Bearer Token is required.")
            return

        client = tweepy.Client(bearer_token=bearer_token)
        
        # Simple API call to verify token
        try:
            # Using a public endpoint to verify the token
            # For example, fetching recent tweets from a well-known public account or using a public metrics endpoint
            response = client.get_users_tweets(id="44196397", max_results=1)  # Twitter ID for @Twitter
            if response.data is None:
                raise Exception("No data returned; possible invalid token.")
        except tweepy.errors.Unauthorized as e:
            messagebox.showerror("Authentication Error", f"Invalid bearer token. Please check your credentials. Error: {e}")
            return
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while verifying the token: {e}")
            return

        geolocator = Nominatim(user_agent="geoapiExercises")  # Initialize the geolocator

        # Additional setup for fetching tweets
        tweet_fields = ['created_at', 'public_metrics', 'author_id', 'text']
        user_fields = ['username', 'name', 'public_metrics', 'description', 'verified', 'created_at', 'location']
        expansions = ['author_id']

        # Prepare the query
        keywords = self.keywords_entry.get()
        geo_x = self.geo_x_entry.get()
        geo_y = self.geo_y_entry.get()
        radius = self.radius_entry.get()
        start_date = self.start_date_entry.get_date()
        end_date = self.end_date_entry.get_date()

        if start_date > end_date:
            messagebox.showerror("Error", "Start date must not be later than end date.")
            return

        include_retweets = self.include_retweets_var.get()
        include_quotes = self.include_quotes_var.get()
        num_tweets = int(self.tweets_number_spinbox.get())
        language = self.language_combobox.get().split(" ")[-1].strip("()")

        query = keywords if keywords else "*"
        if geo_x and geo_y and radius:
            query += f" point_radius:[{geo_y} {geo_x} {radius}km]"
        if not include_retweets:
            query += " -is:retweet"
        if not include_quotes:
            query += " -is:quote"
        query += f" lang:{language}"

        try:
            paginator = tweepy.Paginator(client.search_recent_tweets, query=query,
                                        start_time=start_date.isoformat(), end_time=end_date.isoformat(),
                                        tweet_fields=tweet_fields, expansions=expansions,
                                        user_fields=user_fields, max_results=100)
            tweets = list(paginator.flatten(limit=num_tweets))

            if not tweets:
                messagebox.showerror("No Data", "No tweets found matching your criteria.")
                print("No tweets found. Please adjust your search criteria.")
            else:
                # Process tweet and user data
                data = []
                for tweet in tweets:
                    user = client.get_user(id=tweet.author_id, user_fields=user_fields).data
                    location_name = None
                    if tweet.geo: 
                        latitude, longitude = tweet.geo['coordinates']['coordinates']  # Update based on actual structure
                        location = geolocator.reverse((latitude, longitude), exactly_one=True)
                        location_name = location.address if location else "Not found"

                    row = {
                        'created_at': tweet.created_at,
                        'tweet_location': location_name,
                        'text': tweet.text,
                        'retweets': tweet.public_metrics['retweet_count'],
                        'replies': tweet.public_metrics['reply_count'],
                        'likes': tweet.public_metrics['like_count'],
                        'quote_count': tweet.public_metrics['quote_count'],
                        'author_id': tweet.author_id,
                        'username': user.username,
                        'name': user.name,
                        'author_followers': user.public_metrics['followers_count'],
                        'author_listed': user.public_metrics['listed_count'],
                        'author_following': user.public_metrics['following_count'],
                        'author_tweets': user.public_metrics['tweet_count'],
                        'author_description': user.description,
                        'author_verified': user.verified,
                        'author_created_at': user.created_at,
                        'author_location': user.location
                    }
                    data.append(row)

                df = pd.DataFrame(data)

                file_path = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel Files', '*.xlsx'), ('CSV Files', '*.csv')])

                if file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                elif file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)

                messagebox.showinfo("Success", f"Downloaded {len(df)} tweets and saved to '{file_path}'")
                print(f"Downloaded {len(df)} tweets. Data saved to '{file_path}'.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            print(f"An error occurred: {e}")