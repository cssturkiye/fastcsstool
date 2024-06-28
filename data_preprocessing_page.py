import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import pandas as pd
import tkinter.filedialog
import random
import extract_bot_features as ebf
from tkinter.font import Font
import tkinter.messagebox
from sklearn.model_selection import train_test_split
from tkinter import messagebox


class DataPreprocessingPage(tk.Frame):
    def __init__(self, parent, styles):
        tk.Frame.__init__(self, parent)
        self.grid_columnconfigure(1, weight=1)  # Make the second column expandable

        # Import Data section
        tk.Label(self, text="Data Preprocessing", **styles.title_style).grid(row=0, columnspan=2, sticky='ew', pady=10, padx=5)
        self.import_data_button = tk.Button(self, text="Import Data", command=self.import_data, **styles.button_style)
        self.import_data_button.grid(row=1, columnspan=2, sticky='w', padx=5)

        # Manual Filtering section
        tk.Label(self, text="Manuel Filtering", **styles.title_style).grid(row=2, columnspan=2, sticky='ew', pady=10, padx=5)
        tk.Label(self, text="Exclude these keywords (comma separated)", **styles.label_style).grid(row=3, column=0, sticky='ew', padx=5)
        self.exclude_keywords_entry = tk.Entry(self, **styles.entry_style)
        self.exclude_keywords_entry.grid(row=3, column=1, sticky='ew', padx=5)
        
        tk.Label(self, text="Minimum Tweet Length (chars)", **styles.label_style).grid(row=4, column=0, sticky='ew', padx=5)
        self.min_tweet_length_spinbox = tk.Spinbox(self, from_=0, to=500, **styles.entry_style)
        self.min_tweet_length_spinbox.grid(row=4, column=1, sticky='ew', padx=5)
        self.min_tweet_length_spinbox.delete(0, 'end')
        self.min_tweet_length_spinbox.insert(0, 10)
        
        tk.Label(self, text="Maximum Tweet Length (chars)", **styles.label_style).grid(row=5, column=0, sticky='ew', padx=5)
        self.max_tweet_length_spinbox = tk.Spinbox(self, from_=0, to=500, **styles.entry_style)
        self.max_tweet_length_spinbox.grid(row=5, column=1, sticky='ew', padx=5)
        self.max_tweet_length_spinbox.delete(0, 'end')
        self.max_tweet_length_spinbox.insert(0, 280)

        # A.I. Based Filtering section
        tk.Label(self, text="A.I. Based Filtering", **styles.title_style).grid(row=6, columnspan=2, sticky='ew', pady=10, padx=5)
        tk.Label(self, text="Select Trained A.I. Model", **styles.label_style).grid(row=7, column=0, sticky='ew', padx=5)
        self.model_selection_combobox = ttk.Combobox(self, values=["No AI Filter", "Filter Bots"], **styles.entry_style)
        self.model_selection_combobox.current(1)
        self.model_selection_combobox.grid(row=7, column=1, sticky='ew', padx=5)
        
        tk.Label(self, text="Filter if model output equals ...", **styles.label_style).grid(row=8, column=0, sticky='ew', padx=5)
        self.model_output_entry = tk.Entry(self, **styles.entry_style)
        self.model_output_entry.grid(row=8, column=1, sticky='ew', padx=5)
        self.model_output_entry.insert(0, "bot")

        tk.Label(self, text="(Ex: bot or human)", **styles.label_style).grid(row=9, column=1, sticky='ew', padx=5)


        # Export Data section
        tk.Label(self, text="Export Data", **styles.title_style).grid(row=10, columnspan=2, sticky='ew', pady=10, padx=5)
        tk.Label(self, text="Sample Random Train Data", **styles.label_style).grid(row=11, column=0, sticky='ew', padx=5)
        self.test_train_ratio = tk.Spinbox(self, from_=0, to=1, increment=0.01, **styles.entry_style)
        self.test_train_ratio.grid(row=11, column=1, sticky='ew', padx=5)
        # set test_train_ratio default value to 0.15
        self.test_train_ratio.delete(0, 'end')
        self.test_train_ratio.insert(0, 0.15)
        
        self.filter_save_button = tk.Button(self, text="Filter and Export Train/Raw Data", command=self.filter_and_save, **styles.button_style)
        self.filter_save_button.grid(row=12, columnspan=2, sticky='e', pady=10, padx=5)

        self.filename = None

    def import_data(self):
        self.filename = tkinter.filedialog.askopenfilename(title="Select file", filetypes=[("Excel or CSV files", "*.csv *.xlsx")])
        if self.filename:
            file_extension = self.filename.split('.')[-1]
            if file_extension == 'csv':
                self.df = pd.read_csv(self.filename)
            elif file_extension == 'xlsx':
                self.df = pd.read_excel(self.filename)
            
            required_columns = ['created_at', 'tweet_location', 'text', 'retweets', 'replies', 'likes', 'quote_count', 'author_id', 'username', 'name', 'author_followers', 'author_listed', 'author_following', 'author_tweets', 'author_description', 'author_verified', 'author_created_at', 'author_location']
            if not all(column in self.df.columns for column in required_columns):
                error_message = "Error: The following columns are missing: " + ', '.join(set(required_columns) - set(self.df.columns))
                print(error_message)
                tkinter.messagebox.showerror("Error", error_message)
                

            print(self.df.head())
            print("Data imported successfully.")
            messagebox.showinfo("Success", "Data imported successfully.")
        else:
            print("No file selected.")
            messagebox.showinfo("Warning", "No file selected.")


    def apply_manual_filters(self):
        keywords = self.exclude_keywords_entry.get().split(',')
        min_length = int(self.min_tweet_length_spinbox.get())
        max_length = int(self.max_tweet_length_spinbox.get())

        # Filter by keywords
        if keywords[0]:
            print(f"Excluding tweets containing any of these keywords: {keywords}")
            pattern = '|'.join(keywords)
            self.df = self.df[~self.df['text'].str.contains(pattern, na=False)]
        else:
            print("No keywords provided.")
        # Filter by length
        self.df = self.df[self.df['text'].str.len().between(min_length, max_length)]
        
        print("Manual filters applied.")


    def apply_ai_filters(self):
        selected_model = self.model_selection_combobox.get()
        type_to_filter = self.model_output_entry.get()

        print(f"Selected model: {selected_model}, Type to be Filtered: {type_to_filter}")

        if selected_model == "Filter Bots":
            # Apply model to each row and filter
            try:
                self.df = ebf.process_df(self.df)
            # except the KeyError exception
            except KeyError:
                print("Error: The model could not be applied. Please make sure the data is in the correct format.")
                tkinter.messagebox.showerror("Error", "The model could not be applied. Please make sure the data is in the correct format.")
            
            if type_to_filter == "bot":
                # filter out the rows that have "is_bot" == True
                self.df = self.df[self.df["is_bot"] == False]
            else:
                # filter out the rows that have "is_bot" == False
                self.df = self.df[self.df["is_bot"] == True]
            print("AI filters applied based on model output.")

    def apply_filters(self):
        try:
            self.disable_buttons()
            self.apply_manual_filters()
            self.apply_ai_filters()
            self.enable_buttons()
            # messagebox.showinfo("Success", "Data filtered and ready for export.")
        except Exception as e:
            self.enable_buttons()
            messagebox.showerror("Error", str(e))

    def filter_and_save(self):
        try:
            # Apply manual and AI filters
            original_row_count = len(self.df)
            self.apply_filters()

            if not hasattr(self, 'df') or self.df.empty:
                messagebox.showerror("Error", "No data to export")
                print("No data to filter. Please import data first.")
                return


            # Define the columns to save
            columns_to_save     = ['created_at', 'tweet_location', 'author_location', 'author_verified', 'text']
            columns_to_save_bot = ['created_at', 'tweet_location', 'author_location', 'author_verified', 'text', 'is_bot', 'bot_prob']


            # Select only the columns to save
            if self.model_selection_combobox.get() == "Filter Bots":
                df_to_save = self.df[columns_to_save_bot]
            else:
                df_to_save = self.df[columns_to_save]
            

            df_to_save = df_to_save.sample(frac=1).reset_index(drop=True)

            train_df, test_df = train_test_split(df_to_save, test_size=float(self.test_train_ratio.get()), random_state=42)

            # Save the DataFrames as CSV files
            # This data will be analysed in the Analyze Data page so it is not a train data
            final_fname = self.filename.split('/')[-1].split('.')[0]
            train_df.to_csv(f"{final_fname}_raw.csv", index=False)

            # This is not a actual test data, it is just a whole data including train-test-valid data.
            test_df.to_csv(f"{final_fname}_train.csv", index=False)

            # Calculate how many rows remain after filtering
            filtered_row_count = len(self.df)
            rows_filtered = original_row_count - filtered_row_count

            print("original_row_count: ", original_row_count)
            print("filtered_row_count: ", filtered_row_count)

            success_message = f"Data filtered and saved using _train and _raw suffixes successfully.\nFiltering results: {rows_filtered} rows were filtered out."
            messagebox.showinfo("Success", success_message)
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def disable_buttons(self):
        self.import_data_button.config(state='disabled')
        self.filter_save_button.config(state='disabled')

    def enable_buttons(self):
        self.import_data_button.config(state='normal')
        self.filter_save_button.config(state='normal')