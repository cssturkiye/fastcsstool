import tkinter as tk
from tkinter import ttk

class HelpPage(tk.Frame):
    def __init__(self, parent, styles):
        super().__init__(parent)
        self.styles = styles
        
        self.grid_columnconfigure(0, weight=1)

        # Title Label
        tk.Label(self, text="Help & Usage Guide", **styles.title_style).grid(row=0, column=0, sticky='ew', padx=5, pady=10)

        # Creating the Text widget
        text_widget = tk.Text(self, wrap='word', font=styles.label_font, borderwidth=0, highlightthickness=0, bg='#f0f0f0', fg='black', width=80, height=40)
        text_widget.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        # Adding tags for styling
        text_widget.tag_configure('heading', font=(styles.base_font, 12, 'bold'), spacing1=10)
        text_widget.tag_configure('normal', font=(styles.base_font, 12), spacing1=5)
        
        # Help content
        help_contents = [
            ("Data Generation from Twitter", "heading"),
            ("Bearer Token: Enter your valid Twitter bearer token.", "normal"),
            ("Keywords: Set keywords to filter tweets.", "normal"),
            ("Include Options: Choose to include retweets and quotes.", "normal"),
            ("Geo-Location: Specify latitude, longitude, and radius for geographic targeting.", "normal"),
            ("Date Range: Select the start and end dates for your data collection.", "normal"),
            ("Language: Choose the language of the tweets to collect.", "normal"),
            ("Search and Download: Click to begin the data collection process.", "normal"),
            
            ("Data Labeling", "heading"),
            ("Import CSV: Load your data file for labeling.", "normal"),
            ("Labels: Input and update the labels for categorizing data.", "normal"),
            ("Navigation: Navigate through data entries and save your labeling progress.", "normal"),
            
            ("Data Preprocessing", "heading"),
            ("Import Data: Load your dataset.", "normal"),
            ("Manual Filtering: Apply filters like keyword exclusion and tweet length constraints.", "normal"),
            ("AI-Based Filtering: Use AI models to filter data automatically.", "normal"),
            ("Export Data: Save your filtered dataset for further analysis or training.", "normal"),
            
            ("Model Training & Evaluation", "heading"),
            ("Training Data: Load your dataset for model training.", "normal"),
            ("Start Training: Begin the training of your model.", "normal"),
            ("Evaluation: Assess the performance of your model with accuracy, recall, precision, and F1-score metrics.", "normal"),
            ("Save Models: Save your trained models for future use.", "normal"),
            
            ("Analyze Data", "heading"),
            ("Import Data and Model: Load your analysis model and dataset.", "normal"),
            ("Graphical Analysis: Perform and visualize various analyses like time series and distribution of data points.", "normal"),
            ("Export Analysis Results: Save your analysis results for reporting or documentation purposes.", "normal"),
        ]

        for text, tag in help_contents:
            text_widget.insert('end', text + '\n', tag)
        
        text_widget.configure(state='disabled')