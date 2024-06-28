import tkinter as tk
from tkinter import ttk

class Styles:
    def __init__(self):
        self.base_font = "Helvetica"
        self.title_font = (self.base_font, 16)
        self.label_font = (self.base_font, 12)
        self.entry_font = (self.base_font, 12)
        self.button_font = (self.base_font, 12, "bold")
        self.checkbutton_font = (self.base_font, 12)

        self.title_style = {'font': self.title_font, 'background': '#f0f0f0', 'anchor': 'w'}
        self.label_style = {'font': self.label_font, 'background': '#f0f0f0', 'anchor': 'w'}
        self.entry_style = {'font': self.entry_font}
        self.button_style = {'font': self.button_font, 'relief': tk.FLAT, 'bg': '#bf0d37', 'fg': 'white', 'borderwidth': 1}
        self.checkbutton_style = {'font': self.checkbutton_font, 'background': '#f0f0f0'}

        # Initialize and configure ttk styles
        s = ttk.Style()
        s.configure('TButton', font=self.button_font, background='#bf0d37', foreground='white', relief=tk.FLAT)
        s.configure('TFrame', background='white')
        s.configure('TLabel', background='#f0f0f0', font=self.label_font)
        s.configure('TEntry', font=self.entry_font)
        s.configure('TCheckbutton', font=self.checkbutton_font, background='#f0f0f0')

        s.map('TButton',
              foreground=[('pressed', 'white'), ('active', 'white')],
              background=[('pressed', '!disabled', '#e10d3f'), ('active', '#e10d3f')])

        # Entry style specifically for ttk widgets that are more limited in options
        self.ttk_entry_style = {'font': self.entry_font}

        # Page names mapping
        self.page_names = {
            'DataGenerationPage': 'Data Generation',
            'DataPreprocessingPage': 'Data Preprocessing',
            'DataLabelingPage': 'Data Labeling',
            'ModelTrainingPage': 'Model Training',
            # 'ModelTrainingGridPage': 'Model Training (Grid Search)',
            'AnalyzeDataPage': 'Analyze Data',
            'AboutPage': 'About',
            'HelpPage': 'Help'
        }