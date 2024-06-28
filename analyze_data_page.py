import tkinter as tk
from tkinter import ttk, filedialog
from tkcalendar import DateEntry
import tkinter.filedialog as fd

import os
import re
import glob
import flaml
import numpy as np
import pandas as pd
import emoji
from pprint import pprint
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from flaml import AutoML
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pylustrator

import datetime
import matplotlib.ticker as ticker

import nltk


import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud


class AnalyzeDataPage(tk.Frame):
    def __init__(self, parent, styles):
        super().__init__(parent)
        self.styles = styles

        try:
            nltk.download('stopwords')
        except:
            print("Not able to download stopwords, check internet connection")
        
        # Configure the grid system
        self.grid_columnconfigure(0, weight=1, uniform="group1")
        self.grid_columnconfigure(1, weight=1, uniform="group1")

        # Title Label
        tk.Label(self, text="Analyze Data", **styles.title_style).grid(row=0, column=0, columnspan=2, sticky='ew', pady=10, padx=5)

        # --- Model and Data Selection ---
        tk.Label(self, text="Model and Data Selection", **styles.title_style).grid(row=1, column=0, sticky='ew', padx=5, pady=20)
        tk.Label(self, text="Import the trained model and the data to be analysed and press start analyzing the data by pressing start model inference button.", wraplength=400, justify=tk.LEFT, **styles.label_style).grid(row=2, column=0, sticky='ew', padx=5)
        tk.Label(self, text="Import trained model for inference", **styles.label_style).grid(row=3, column=0, sticky='w', padx=5, pady=(10, 0))
        tk.Button(self, text="Import Model", command=self.import_model, **styles.button_style).grid(row=4, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self, text="Import filtered data to analyze", **styles.label_style).grid(row=5, column=0, sticky='w', padx=5)
        tk.Button(self, text="Import Data", command=self.import_files, **styles.button_style).grid(row=6, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self, text="Analyze filtered data with selected model", **styles.label_style).grid(row=7, column=0, sticky='w', padx=5)
        tk.Button(self, text="Start Model Inference", command=self.start_inference, **styles.button_style).grid(row=8, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self, text="Export analyzed data with analyzed results", **styles.label_style).grid(row=9, column=0, sticky='w', padx=5)
        tk.Button(self, text="Export Analyze Results as .xlsx", command=self.export_results, **styles.button_style).grid(row=10, column=0, sticky='w', padx=5, pady=5)

        # --- Graphical Data Analysis ---
        tk.Label(self, text="Graphical Data Analysis", **styles.title_style).grid(row=1, column=1, sticky='ew', padx=5, pady=20)
        tk.Label(self, text="Import the analyse results, pick a date range, select top <n> locations to show and create wordclouds for desired langauges.", wraplength=400, justify=tk.LEFT, **styles.label_style).grid(row=2, column=1, sticky='ew', padx=5)
        tk.Label(self, text="Import analysed results for graphical analysis.", **styles.label_style).grid(row=3, column=1, sticky='w', padx=5, pady=(10, 0))
        tk.Button(self, text="Import Results as .xlsx", command=self.import_results, **styles.button_style).grid(row=4, column=1, sticky='w', padx=5, pady=5)
        tk.Label(self, text="Start Date", **styles.label_style).grid(row=5, column=1, sticky='w', padx=5)
        self.start_date_picker = DateEntry(self, **styles.entry_style)
        self.start_date_picker.set_date(datetime.date(2010, 1, 1))
        self.start_date_picker.grid(row=6, column=1, padx=5, pady=5, sticky='ew')
        tk.Label(self, text="End Date", **styles.label_style).grid(row=7, column=1, sticky='w', padx=5)
        self.end_date_picker = DateEntry(self, **styles.entry_style)
        self.end_date_picker.set_date(datetime.date.today())
        self.end_date_picker.grid(row=8, column=1, padx=5, pady=5, sticky='ew')
        tk.Label(self, text="Top Locations (n)", **styles.label_style).grid(row=9, column=1, sticky='w', padx=5)
        self.top_locations_entry = tk.Entry(self, **styles.entry_style)
        self.top_locations_entry.grid(row=10, column=1, sticky='ew', padx=5, pady=5)
        self.top_locations_entry.insert(0, "9")  # Default value for top locations
        tk.Label(self, text="Language (it is required for removing the stop-words)", **styles.label_style).grid(row=11, column=1, sticky='w', padx=5)
        #self.language_selection_combobox = ttk.Combobox(self, state='readonly', **styles.entry_style)
        self.language_selection_combobox = ttk.Combobox(self, state='readonly', **self.styles.ttk_entry_style)
        self.language_selection_combobox['values'] = ('Turkish', 'English', 'German', 'French', 'Spanish')
        self.language_selection_combobox.current(0)  # Set default selection to the first entry
        self.language_selection_combobox.grid(row=12, column=1, sticky='ew', padx=5, pady=5)
        tk.Button(self, text="Plot Time Chart", command=self.plot_time_chart, **styles.button_style).grid(row=13, column=1, sticky='w', padx=5, pady=5)
        tk.Button(self, text="Plot Log Time Chart", command=self.plot_log_time_chart, **styles.button_style).grid(row=14, column=1, sticky='w', padx=5, pady=5)
        tk.Button(self, text="Plot Location Distributions", command=self.plot_location_distributions, **styles.button_style).grid(row=15, column=1, sticky='w', padx=5, pady=5)
        tk.Button(self, text="Plot Wordcloud", command=self.plot_wordcloud, **styles.button_style).grid(row=16, column=1, sticky='w', padx=5, pady=5)

        self.df = None
        self.model = None
        self.label_encoder = None
        self.stopwords_dict = self.load_custom_stopwords()



    def import_results(self):
        filename = filedialog.askopenfilename(title="Open Results File", filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv") ])
        if filename:
            try:
                self.df_results = pd.read_csv(filename) if filename.endswith('.csv') else pd.read_excel(filename)
                result_columns = [col for col in self.df_results.columns if col.startswith('result_')]
                if result_columns:
                    self.results_column = result_columns[0]  # Assume the first result column if multiple
                    tk.messagebox.showinfo("Success", "Results file loaded successfully.")
                    print(self.df_results.head())  # Print for verification

                else:
                    tk.messagebox.showwarning("Warning", "No result columns found in the file.")
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to load the results file: {e}")


    def preprocess_text(self, text):
        # Lowercase the text
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Replace usernames with a placeholder
        text = re.sub(r'(@\w+\s*)+', '"kullanıcıadı" ', text)
        # Replace emojis with text
        text = emoji.demojize(text)
        return text

    def start_inference(self):
        print("Start model inference clicked")

        if self.df is not None and self.model is not None and self.label_encoder is not None:
            try:

                if hasattr(self.model, 'predict'):
                    # Extracting the file name without extension for result column naming
                    model_name = os.path.splitext(os.path.basename(self.model_filename))[0]
                    result_column = f"result_{model_name}"

                    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
                    X = np.array(model.encode(self.df['text']))

                    self.df[result_column] = self.label_encoder.inverse_transform( self.model.predict(X) )

                    print(self.df[['text', result_column]].head())
                    tk.messagebox.showinfo("Success", "Model inference completed successfully.")
                else:
                    tk.messagebox.showwarning("Warning", "Loaded model does not support prediction.")
            except Exception as e:
                print("Error", f"An error occurred during model inference: {e}")
                tk.messagebox.showerror("Inference Error", f"An error occurred during model inference: {e}")
        else:
            tk.messagebox.showwarning("Warning", "Please load both a model and a data file before starting inference.")


    def import_model(self):
        print("Import model clicked")
        self.model_filename = fd.askopenfilename(title="Open Model File", filetypes=[("Pickle files", "*.pkl")])
        if self.model_filename:
            try:
                with open(self.model_filename, 'rb') as file:
                    self.model, self.label_encoder = pickle.load(file)  # Load both model and label encoder
                print(f"Model '{self.model_filename}' loaded successfully.")
                tk.messagebox.showinfo("Success", f"Model '{os.path.basename(self.model_filename)}' loaded successfully.")
            except Exception as e:
                print("Error", f"An error occurred during model import: {e}")
                tk.messagebox.showerror("Model Import Error", f"An error occurred during model import: {e}")
        else:
            print("No model file selected.")

    def import_files(self):
        print("Import files clicked")
        self.filename = fd.askopenfilename(title="Open CSV", filetypes=[("CSV files", "*.csv")])
        if self.filename:
            try:
                self.df = pd.read_csv(self.filename, keep_default_na=False)
                self.df['text'] = self.df['text'].apply(self.preprocess_text)
                print(self.df.tail())
                tk.messagebox.showinfo("Success", f"CSV file '{os.path.basename(self.filename)}' loaded successfully.")
                print("CSV file loaded successfully.")
            except Exception as e:
                print("Error", f"An error occurred during file import: {e}")
                tk.messagebox.showerror("File Import Error", f"An error occurred during file import: {e}")
        else:
            tk.messagebox.showwarning("Warning", "No file selected.")


    def export_results(self):
        print("Export results clicked")

        # First check if the DataFrame has been initialized and loaded
        if self.df is not None:
            # Check if there is any column starting with 'result_'
            result_columns = [col for col in self.df.columns if col.startswith('result_')]
            if result_columns:
                try:
                    initial_file = "Analyzed_Results.xlsx"
                    file_path = fd.asksaveasfilename(
                        title="Save Excel File",
                        initialfile=initial_file,
                        defaultextension=".xlsx",
                        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                    )
                    if file_path:
                        self.df.to_excel(file_path, index=False)
                        tk.messagebox.showinfo("Success", f"Analyzed results exported successfully to:\n{file_path}")
                    else:
                        tk.messagebox.showwarning("Warning", "Export cancelled. No file was selected.")
                except Exception as e:
                    print("Error", f"Failed to export results: {e}")
                    tk.messagebox.showerror("Export Error", f"Failed to export results: {e}")
            else:
                tk.messagebox.showwarning("Warning", "No analyzed data to export. Please run inference first.")
        else:
            # If df is None, show an error message
            print("Error", "Data not loaded. Please import data before exporting results.")
            tk.messagebox.showerror("Export Error", "Data not loaded. Please import data before exporting results.")


    def plot_time_chart(self, log_scale=False):
        if hasattr(self, 'df_results') and self.results_column in self.df_results.columns:
            try:
                self.df_results['created_at'] = pd.to_datetime(self.df_results['created_at']).dt.tz_localize(None)
                # Retrieve the dates from the date pickers
                start_date = pd.to_datetime(self.start_date_picker.get_date()).tz_localize(None)
                end_date = pd.to_datetime(self.end_date_picker.get_date()).tz_localize(None)
                
                filtered_df = self.df_results[(self.df_results['created_at'] >= start_date) & (self.df_results['created_at'] <= end_date)]
                daily_counts = filtered_df.groupby(filtered_df['created_at'].dt.date)[self.results_column].value_counts().unstack().fillna(0)

                # Start Pylustrator
                pylustrator.start()

                fig, ax = plt.subplots()
                daily_counts.plot(kind='line', ax=ax)
                ax.set_title('Daily Counts of Results')
                ax.set_xlabel('Date')
                
                # y asix should logaritmic
                if log_scale:
                    ax.set_yscale('log')
                    ax.set_ylabel('Log Counts')
                else:
                    ax.set_ylabel('Counts')

                # Pylustrator provides GUI controls to adjust the plot
                plt.show()
                plt.close('all')

            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to generate graphs: {e}")
        else:
            tk.messagebox.showwarning("Warning", "Please load results data and ensure a results column is present before attempting to show graphs.")

    def plot_log_time_chart(self):
        # Implement the function to plot the logarithmic time chart
        self.plot_time_chart(log_scale=True)
        pass

    def plot_location_distributions(self):
        if hasattr(self, 'df_results') and 'tweet_location' in self.df_results.columns:
            try:

                self.df_results['created_at'] = pd.to_datetime(self.df_results['created_at']).dt.tz_localize(None)
                start_date = pd.to_datetime(self.start_date_picker.get_date()).tz_localize(None)
                end_date = pd.to_datetime(self.end_date_picker.get_date()).tz_localize(None)
                filtered_df = self.df_results[(self.df_results['created_at'] >= start_date) & (self.df_results['created_at'] <= end_date)]
                filtered_df['month'] = filtered_df['created_at'].dt.to_period('M').dt.to_timestamp()
                location_data = filtered_df.groupby(['month', 'tweet_location']).size().unstack(fill_value=0)
                top_n = int(self.top_locations_entry.get() or 9)  # Default to 9 if entry is empty
                top_locations = location_data.sum().nlargest(top_n).index
                location_data = location_data[top_locations]
                pylustrator.start()
                fig, ax = plt.subplots(figsize=(10, 6))
                location_data.plot(kind='bar', ax=ax, stacked=True)

                months_range = len(location_data)
                if months_range > 12:
                    step = months_range // 12
                else:
                    step = 1

                ax.xaxis.set_major_locator(ticker.MultipleLocator(step))  # Set dynamic step based on data range
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: location_data.index[int(x/step)].strftime('%Y-%m') if int(x/step) < len(location_data) else ''))

                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))  # Adds one minor tick between major ticks

                ax.tick_params(axis='x', which='major', length=10, width=1, direction='out', labelrotation=45)
                ax.tick_params(axis='x', which='minor', length=5, width=1, direction='out')  # Minor ticks are shorter

                ax.set_title('Tweet Location Distributions Over Time')
                ax.set_ylabel('Number of Tweets')
                ax.set_xlabel('Month')
                ax.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')

                plt.xticks(rotation=45)
                plt.tight_layout()  # Adjust layout to make room for label rotation

                #% start: automatic generated code from pylustrator
                plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
                import matplotlib as mpl
                getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
                #% end: automatic generated code from pylustrator
                plt.show()
                plt.close('all')

            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to generate location distribution graph: {e}")
        else:
            tk.messagebox.showwarning("Warning", "Please load results data and ensure 'tweet_location' column is present before attempting to plot location distributions.")

    def load_custom_stopwords(self):
        languages = ['turkish', 'english', 'german', 'french', 'spanish']
        stopwords_dict = {}
        base_dir = 'custom_stopwords'

        for language in languages:
            filepath = os.path.join(base_dir, f'{language}.txt')
            custom_stopwords = set()
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as file:
                    custom_stopwords.update([line.strip() for line in file if line.strip()])

            nltk_stopwords = set(stopwords.words(language))
            combined_stopwords = nltk_stopwords.union(custom_stopwords)
            stopwords_dict[language] = combined_stopwords

        return stopwords_dict
    
    def plot_wordcloud(self):
        if hasattr(self, 'df_results') and 'text' in self.df_results.columns:
            try:
                pylustrator.start()

                self.df_results['created_at'] = pd.to_datetime(self.df_results['created_at']).dt.tz_localize(None)
                start_date = pd.to_datetime(self.start_date_picker.get_date()).tz_localize(None)
                end_date = pd.to_datetime(self.end_date_picker.get_date()).tz_localize(None)
                filtered_df = self.df_results[(self.df_results['created_at'] >= start_date) & (self.df_results['created_at'] <= end_date)]
                text = " ".join(tweet for tweet in filtered_df['text'])

                language = self.language_selection_combobox.get().lower()  # Get the selected language from combobox
                try:
                    # current_stopwords = set(stopwords.words(language))
                    current_stopwords = self.stopwords_dict.get(language, set())
                except OSError:
                    tk.messagebox.showerror("Error", f"Stopwords for '{language}' are not available.")
                    return
                wordcloud = WordCloud(stopwords=current_stopwords, background_color="white", width=1600, height=800).generate(text)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title('Tweet Word Cloud')  # Add a title to the plot
                plt.axis("off")

                plt.tight_layout(pad=0)  # Reduce the padding around the plot to minimize white space
                plt.show()
                plt.close('all')

            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to generate word cloud: {e}")
        else:
            tk.messagebox.showwarning("Warning", "Please load results data and ensure 'text' column is present before attempting to plot word cloud.")