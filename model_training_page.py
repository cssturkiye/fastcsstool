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


class ModelTrainingPage(tk.Frame):
    def __init__(self, parent, styles):
        tk.Frame.__init__(self, parent)
        self.styles = styles
        self.grid_columnconfigure(1, weight=1)

        # Title Label
        tk.Label(self, text="Training & Evaluating Models", **styles.title_style).grid(row=0, columnspan=2, sticky='ew', pady=10, padx=5)

        # Training Section
        tk.Label(self, text="Training", **styles.title_style).grid(row=1, columnspan=2, sticky='ew', padx=5)
        tk.Button(self, text="Open Training File (CSV)", command=self.open_training_files, **styles.button_style).grid(row=2, columnspan=2, sticky='w', padx=5, pady=5)
        tk.Button(self, text="Start Training", command=self.start_training, **styles.button_style).grid(row=3, columnspan=2, sticky='w', padx=5, pady=5)

        # Model Evaluation Section
        tk.Label(self, text="Model Evaluation", **styles.title_style).grid(row=4, columnspan=2, sticky='ew', padx=5)
        self.create_evaluation_table()

        # Placeholder for training status
        self.training_status_var = tk.StringVar(value="Status: Not started")
        tk.Label(self, textvariable=self.training_status_var, **styles.label_style).grid(row=5, columnspan=2, sticky='w', padx=5, pady=5)
        
        # Add buttons for saving models and exporting table
        tk.Button(self, text="Save Models", command=self.save_models, **self.styles.button_style).grid(row=7, column=0, sticky='w', padx=5, pady=5)
        tk.Button(self, text="Export Table", command=self.export_table, **self.styles.button_style).grid(row=7, column=1, sticky='w', padx=5, pady=5)

        self.label_columns = None
        self.df = None
        self.trained_models = None

    def save_models(self):
        """Save the trained models to disk with the option to change the file name."""
        if self.trained_models:
            for label, results in self.trained_models.items():
                initial_file = f"{label.split('_label')[0]}.pkl"
                file_path = filedialog.asksaveasfilename(
                    title="Save Model As",
                    initialfile=initial_file,
                    defaultextension=".pkl",
                    filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                    initialdir=os.getcwd() 
                )
                if file_path:
                    with open(file_path, 'wb') as file:
                        pickle.dump((results['model'], results['label_encoder']), file)  # Save both model and label encoder
                    tk.messagebox.showinfo("Success", f"Model '{label}' saved successfully at {file_path}.")
                else:
                    tk.messagebox.showwarning("Warning", "No file selected. Model not saved.")
        else:
            tk.messagebox.showwarning("Warning", "No models to save.")


    def export_table(self):
        """Export the evaluation results to an Excel file with error handling and customizable file name."""
        data = []
        for child in self.evaluation_table.get_children():
            data.append(self.evaluation_table.item(child)["values"])
        if data:
            df = pd.DataFrame(data, columns=['Model Name', 'Accuracy', 'Recall', 'Precision', 'F1-Score'])
            try:
                initial_file = "Evaluation_Results.xlsx"
                file_path = filedialog.asksaveasfilename(
                    title="Save Excel File",
                    initialfile=initial_file,
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                    initialdir=os.getcwd()  # Set initial directory to current working directory or any specific path
                )
                if file_path:
                    df.to_excel(file_path, index=False)
                    tk.messagebox.showinfo("Success", f"Table exported successfully to:\n{file_path}")
                else:
                    tk.messagebox.showwarning("Warning", "Export cancelled. No file was selected.")
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to export table: {e}")
        else:
            tk.messagebox.showwarning("Warning", "No data available to export.")



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

    def open_training_files(self):
        print("Open training files clicked")

        self.filename = fd.askopenfilename(title="Open CSV", filetypes=[("CSV files", "*.csv")])
        if self.filename:
            try:

                self.df = pd.read_csv(self.filename, keep_default_na=False)
                self.label_columns = [col for col in self.df.columns if col.endswith('_label')]

                # Apply the preprocess_text function to the text column
                self.df['text'] = self.df['text'].apply(self.preprocess_text)
                self.df = self.df.dropna(subset=self.label_columns)

                # TODO: data labeling bolumune bu alanlar mutlaka eklensin
                for label in self.label_columns:
                    self.df = self.df[(self.df[label] != "not-sure") & (self.df[label] != "bad-data")]

                print(self.df.tail())
                print("CSV file loaded successfully.")
                tk.messagebox.showinfo("Success", "CSV file loaded successfully.")

            except Exception as e:
                print("Training Error", f"An error occurred during training: {e}")
                self.training_status_var.set("Status: Training failed")
                tk.messagebox.showerror("Training Error", f"An error occurred during training: {e}")
        else:
            print("No file selected.")

    def create_evaluation_table(self):
        self.evaluation_table = ttk.Treeview(self, columns=('Model Name', 'Accuracy', 'Recall', 'Precision', 'F1-Score'), show='headings')
        self.evaluation_table.grid(row=6, columnspan=2, sticky='nsew', padx=5, pady=5)
        self.evaluation_table.heading('Model Name', text='Model Name')
        self.evaluation_table.heading('Accuracy', text='Accuracy')
        self.evaluation_table.heading('Recall', text='Recall')
        self.evaluation_table.heading('Precision', text='Precision')
        self.evaluation_table.heading('F1-Score', text='F1-Score')

        # Set the width of the columns and align to center
        column_width = 100  # Adjust as needed
        for col in self.evaluation_table['columns']:
            self.evaluation_table.column(col, width=column_width, stretch=tk.NO, anchor=tk.CENTER)


    def start_training(self):
        print("Start training clicked")
        self.training_status_var.set("Status: Training in progress...")

        self.disable_buttons()

        # update the ui
        self.update_idletasks()

        # Clear previous entries in the table
        for item in self.evaluation_table.get_children():
            self.evaluation_table.delete(item)
        
        try:

            trained_models = {}

            for label in self.label_columns:
                trained_models[label] = self.train_and_evaluate(texts=self.df['text'], labels=self.df[label])

            # iterate trained_models and add to the table
            for label, results in trained_models.items():
                # round the values to 2 decimal places
                self.evaluation_table.insert("", "end",
                    values=(
                        label.split("_label")[0], 
                        round(results['accuracy'], 2), 
                        round(results['recall'], 2), 
                        round(results['precision'], 2),
                        round(results['f1_score'], 2)
                    )
                )

            self.trained_models = trained_models

            # Update the training status
            self.training_status_var.set("Status: Training completed")

        except Exception as e:
            print("Training Error", f"An error occurred during training: {e}")
            self.training_status_var.set("Status: Training failed")
            tk.messagebox.showerror("Training Error", f"An error occurred during training: {e}")

        self.enable_buttons()


    def train_and_evaluate(self, texts, labels, test_size=0.15, random_state=42):
        # Encode texts using SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        X = np.array(model.encode(texts))

        # Encode labels using LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels)
        y = np.array(y).reshape(-1, 1)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Initialize the AutoML instance
        automl = AutoML()

        # Specify automl goal and constraint
        automl_settings = {
            "time_budget": 300,  # in seconds
            "metric": 'macro_f1',
            "task": 'classification',  # can be 'classification', 'regression', or 'ranking'
            "log_file_name": 'automl.log',  # flaml log file
        }

        # Train with the training set
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

        # Predict on the test set
        y_pred = automl.predict(X_test)
        y_pred_proba = automl.predict_proba(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Print evaluation results
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"F1 Score: {f1}")

        # Return the trained model and evaluation results
        results = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "model": automl,
            "label_encoder": le  # Save the label encoder
        }
        return results
    

    def disable_buttons(self):
        """Disable buttons during training to prevent multiple processes."""
        self.winfo_children()[3].config(state='disabled')
        self.winfo_children()[4].config(state='disabled')

    def enable_buttons(self):
        """Re-enable buttons after training to allow new operations."""
        self.winfo_children()[3].config(state='normal') 
        self.winfo_children()[4].config(state='normal')