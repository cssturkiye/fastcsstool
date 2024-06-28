import os
import re
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import joblib
import emoji

class ModelTrainingGridPage(tk.Frame):
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

        # Note about grid search
        note_style = self.styles.label_style.copy()
        note_style["font"] = ('Helvetica', 10, 'italic')
        tk.Label(self, text="Note: Grid search is way more slower than Gaussian search with a very similar performance; thus, this page will be obsolete.",
                 **note_style).grid(row=8, columnspan=2, sticky='ew', padx=5, pady=5)

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
                        joblib.dump((results['model'], results['label_encoder']), file)  # Save both model and label encoder
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
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'(@\w+\s*)+', '"kullanıcıadı" ', text)
        text = emoji.demojize(text)
        return text

    def open_training_files(self):
        self.filename = filedialog.askopenfilename(title="Open CSV", filetypes=[("CSV files", "*.csv")])
        if self.filename:
            try:
                self.df = pd.read_csv(self.filename, keep_default_na=False)
                self.label_columns = [col for col in self.df.columns if col.endswith('_label')]
                self.df['text'] = self.df['text'].apply(self.preprocess_text)
                self.df = self.df.dropna(subset=self.label_columns)
                print(self.df.tail())
                print("CSV file loaded successfully.")
                messagebox.showinfo("Success", "CSV file loaded successfully.")
            except Exception as e:
                print(f"An error occurred while loading the CSV file: {e}")
                messagebox.showerror("Error", f"An error occurred while loading the CSV file: {e}")
        else:
            print("No file selected.")
            messagebox.showwarning("Warning", "No file selected.")

    def create_evaluation_table(self):
        self.evaluation_table = ttk.Treeview(self, columns=('Model Name', 'Accuracy', 'Recall', 'Precision', 'F1-Score'), show='headings')
        self.evaluation_table.grid(row=6, columnspan=2, sticky='nsew', padx=5, pady=5)
        self.evaluation_table.heading('Model Name', text='Model Name')
        self.evaluation_table.heading('Accuracy', text='Accuracy')
        self.evaluation_table.heading('Recall', text='Recall')
        self.evaluation_table.heading('Precision', text='Precision')
        self.evaluation_table.heading('F1-Score', text='F1-Score')

        # Set the width of the columns and align to center
        column_width = 100  
        for col in self.evaluation_table['columns']:
            self.evaluation_table.column(col, width=column_width, stretch=tk.NO, anchor=tk.CENTER)

    def start_training(self):
        print("Start training clicked")
        self.training_status_var.set("Status: Training in progress...")

        self.disable_buttons()
        self.update_idletasks()

        # Clear previous entries in the table
        for item in self.evaluation_table.get_children():
            self.evaluation_table.delete(item)

        try:
            self.trained_models = {}

            for label in self.label_columns:
                model, accuracy, recall, precision, f1_score = self.train_and_evaluate(self.df['text'], self.df[label])
                self.trained_models[label] = {"model": model}

                self.evaluation_table.insert("", "end",
                    values=(
                        label.split("_label")[0],
                        round(accuracy, 2),
                        round(recall, 2),
                        round(precision, 2),
                        round(f1_score, 2)
                    )
                )

            self.training_status_var.set("Status: Training completed")
        except Exception as e:
            print(f"An error occurred during training: {e}")
            self.training_status_var.set("Status: Training failed")
            messagebox.showerror("Training Error", f"An error occurred during training: {e}")

        self.enable_buttons()

    def train_and_evaluate(self, texts, labels, test_size=0.15, random_state=42):
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        X = np.array(model.encode(texts))

        le = LabelEncoder()
        y = le.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        classifiers = {
            "AdaBoost": AdaBoostClassifier(),
            "Bagging": BaggingClassifier(),
            "ExtraTrees": ExtraTreesClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "RandomForest": RandomForestClassifier(),
            "Ridge": RidgeClassifier(),
            "GaussianNB": GaussianNB(),
            "KNN": KNeighborsClassifier(),
            "MLP": MLPClassifier(),
            "LinearSVC": LinearSVC(),
            "SVC": SVC(),
            "DecisionTree": DecisionTreeClassifier()
        }

        param_grids = {
            "AdaBoost": {"n_estimators": [50, 100, 200]},
            "Bagging": {"n_estimators": [10, 50, 100]},
            "ExtraTrees": {"n_estimators": [50, 100, 200]},
            "GradientBoosting": {"learning_rate": [0.01, 0.1, 0.2], "n_estimators": [50, 100, 200]},
            "RandomForest": {"n_estimators": [50, 100, 200], "max_features": ["sqrt", "log2"]},
            "Ridge": {"alpha": [0.1, 1.0, 10.0]},
            "GaussianNB": {},
            "KNN": {"n_neighbors": [3, 5, 7]},
            "MLP": {"hidden_layer_sizes": [(100,), (100, 100)], "activation": ["relu", "tanh"]},
            "LinearSVC": {"C": [0.1, 1.0, 10.0]},
            "SVC": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
            "DecisionTree": {"max_depth": [None, 10, 20]}
        }

        best_model = None
        best_score = 0

        for name, classifier in classifiers.items():
            grid_search = GridSearchCV(classifier, param_grids[name], scoring='accuracy', cv=10, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_estimator = grid_search.best_estimator_
            y_pred = best_estimator.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f"{name} - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}")

            if accuracy > best_score:
                best_score = accuracy
                best_model = best_estimator

        return best_model, accuracy, recall, precision, f1

    def disable_buttons(self):
        """Disable buttons during training to prevent multiple processes."""
        self.winfo_children()[3].config(state='disabled')
        self.winfo_children()[4].config(state='disabled')

    def enable_buttons(self):
        """Re-enable buttons after training to allow new operations."""
        self.winfo_children()[3].config(state='normal')
        self.winfo_children()[4].config(state='normal')
