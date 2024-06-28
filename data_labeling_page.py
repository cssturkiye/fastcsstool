import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import re
import random
import pandas as pd
import tkinter.filedialog as fd
from tkinter import messagebox

class DataLabelingPage(tk.Frame):
    def __init__(self, parent, styles):
        tk.Frame.__init__(self, parent)
        self.styles = styles
        self.df = None
        self.filename = None
        self.current_index = None 
        self.radio_button_frames = []
        self.label_vars = []


        # Configure the grid
        self.grid_columnconfigure(1, weight=1)

        # Label Management Section
        tk.Label(self, text="Label Management", **styles.title_style).grid(row=0, columnspan=2, sticky='ew',  pady=10, padx=5)

        # Import Data Section
        self.import_data_label = tk.StringVar(value="You should import the data to be labeled.")
        tk.Label(self, textvariable=self.import_data_label, **styles.label_style).grid(row=1, columnspan=2, sticky='w', padx=5, pady=5)
        import_button = tk.Button(self, text="Import CSV", command=self.import_csv, **styles.button_style)
        import_button.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        
        # Label Text Entry
        self.label_text_notes = tk.StringVar(value="Insert the comma seperated labels here. Ex: label_1,label_2,...label_n ")
        tk.Label(self, textvariable=self.label_text_notes, **styles.label_style).grid(row=3, columnspan=2, sticky='w', padx=5, pady=5)

        self.label_text = tk.Text(self, height=4, **styles.entry_style)
        self.label_text.grid(row=4, columnspan=2, sticky='ew', padx=5)
        self.label_text.bind("<KeyPress>", self.validate_label_text)  # Bind key press to validate input

        # Update Labels Button
        tk.Button(self, text="Update Labels", command=self.update_labels, **styles.button_style).grid(row=5, columnspan=2, sticky='w', padx=5, pady=5)

        # Set a maximum width for the large Text box
        max_text_width = 50  # Adjust as needed
        self.document_text = tk.Text(self, height=11, width=max_text_width, **styles.entry_style)
        self.document_text.grid(row=6, columnspan=2, sticky='nsew', padx=5)
        self.document_text.insert('1.0', "Please import training data to see it in here...")
        self.document_text.config(state='disabled')

        # Total Labeled Documents
        self.total_labeled_var = tk.StringVar(value="Total Labeled Documents: 0")
        tk.Label(self, textvariable=self.total_labeled_var, **styles.label_style).grid(row=7, columnspan=2, sticky='w', padx=5, pady=5)

        # Radio Button Frames Placeholder
        self.radio_buttons_frame = tk.Frame(self)
        self.radio_buttons_frame.grid(row=8, columnspan=2, sticky='ew', padx=5)
        
        # Save Label Button
        self.save_label_button = tk.Button(self, text="Save Label and Go to Next", command=self.save_label, **styles.button_style)
        self.save_label_button.grid(row=9, columnspan=2, sticky='e', padx=30, pady=10)


        # Create right-click context menu for text widget
        self.text_context_menu = tk.Menu(self, tearoff=0)
        self.text_context_menu.add_command(label="Cut", command=self.cut_text)
        self.text_context_menu.add_command(label="Copy", command=self.copy_text)
        self.text_context_menu.add_command(label="Paste", command=self.paste_text)

        self.label_text.bind("<Button-3>", self.show_context_menu)

        # Load labels from file or set default
        self.load_labels()

    def import_csv(self):
        self.filename = fd.askopenfilename(title="Open CSV", filetypes=[("CSV files", "*.csv")])
        if self.filename:
            self.df = pd.read_csv(self.filename, keep_default_na=False)
            self.load_next_text()
            print("CSV file loaded successfully.")
        else:
            print("No file selected.")

    def load_labels(self):
        try:
            with open('labels.txt', 'r') as file:
                labels = file.read().strip()
            if labels:
                self.label_text.delete('1.0', tk.END)
                self.label_text.insert('1.0', labels)
                self.update_labels()
            else:
                self.label_text.insert('1.0', 'label_1,label_2,label_3')
        except FileNotFoundError:
            self.label_text.insert('1.0', 'label_1,label_2,label_3')


    def get_current_labels(self):
        # Return a list of lines from the label_text widget
        label_text = self.label_text.get("1.0", tk.END).strip()
        return [line.strip() for line in label_text.split('\n') if line.strip()]


    def load_next_text(self):
        label_columns = [col for col in self.df.columns if col.startswith('model_') and col.endswith('_label')]

        # Initialize columns if not present (this might not be necessary if ensured at data load)
        for col in label_columns:
            if col not in self.df.columns:
                self.df[col] = pd.NA  # Ensuring columns are initialized correctly

        # Check if any entry is still completely unlabeled
        # Here we ensure to check for not just NA but also potentially empty strings or other placeholders
        unlabeled_mask = self.df[label_columns].apply(lambda x: pd.isna(x) | (x == ''), axis=1).all(axis=1)
        labeled_count = (~unlabeled_mask).sum()
        self.total_labeled_var.set(f"Total Labeled Documents: {labeled_count}")

        if unlabeled_mask.any():
            current_row = self.df[unlabeled_mask].iloc[0]
            self.current_index = self.df[unlabeled_mask].index[0]
            self.document_text.config(state='normal')
            self.document_text.delete('1.0', tk.END)
            self.document_text.insert('1.0', current_row['text'])
            self.document_text.config(state='disabled')
        else:
            messagebox.showinfo("Info", "All data are already labeled or no data found.")




    def show_context_menu(self, event):
        """Show the context menu."""
        try:
            # Check if the text widget is disabled, if so, only show 'Paste'
            if self.label_text.cget('state') == tk.DISABLED:
                self.text_context_menu.entryconfig("Cut", state="disabled")
                self.text_context_menu.entryconfig("Copy", state="disabled")
                self.text_context_menu.entryconfig("Paste", state="normal")
            else:
                self.text_context_menu.entryconfig("Cut", state="normal")
                self.text_context_menu.entryconfig("Copy", state="normal")
                self.text_context_menu.entryconfig("Paste", state="normal")
            # Show the context menu
            self.text_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.text_context_menu.grab_release()

    def cut_text(self):
        """Cut the selected text."""
        self.label_text.event_generate("<<Cut>>")

    def copy_text(self):
        """Copy the selected text."""
        self.label_text.event_generate("<<Copy>>")

    def paste_text(self):
        """Paste from clipboard."""
        self.label_text.event_generate("<<Paste>>")


    def validate_label_text(self, event=None):
        # Allowed control characters
        allowed_controls = {'BackSpace', 'Return', 'Left', 'Right', 'Up', 'Down', 'Delete', 'Home', 'End'}
        
        # If the character is a control key, allow it
        if event.keysym in allowed_controls:
            return
        
        # If the character is a non-control, check if it is valid
        allowed_characters = re.compile(r'^[\w,\n]+$')
        if event.char and not allowed_characters.match(event.char):
            # Disallow the character by returning 'break'
            return "break"


    def update_labels(self):
        label_text = self.label_text.get("1.0", tk.END).strip()
        with open('labels.txt', 'w') as file:
            file.write(label_text)

        # Destroy existing radio button frames and clear the list
        for frame in self.radio_button_frames:
            frame.destroy()
        self.radio_button_frames.clear()
        self.label_vars.clear()

        lines = label_text.split('\n')
        row = 8
        for line in lines:
            if line.strip():  # Only create frames for non-empty lines
                frame = tk.Frame(self)
                frame.grid(row=row, columnspan=2, sticky='ew', padx=5)
                self.radio_button_frames.append(frame)

                label_var = tk.StringVar(value="none")
                self.label_vars.append(label_var)

                labels = [label.strip() for label in line.split(',')]
                for label in labels:
                    if label:
                        radio_button = tk.Radiobutton(frame, text=label, variable=label_var, value=label, **self.styles.checkbutton_style)
                        radio_button.pack(side='left', padx=5)
                row += 1

        # Move the save button to below the last set of radio buttons
        self.save_label_button.grid(row=row, columnspan=2, sticky='e', padx=30, pady=10)


            
    def save_label(self):
        if self.df is not None and self.current_index is not None:
            all_labeled = True
            for i, var in enumerate(self.label_vars):
                selected_label = var.get()
                column_name = f'model_{i+1}_label'
                if selected_label != "none":
                    self.df.at[self.current_index, column_name] = selected_label
                    var.set("none")  # Reset the variable after saving

            # Save the DataFrame and update the counter
            self.df.to_csv(self.filename, index=False)
            messagebox.showinfo("Label Saved", "Your labels have been saved and data updated in the file.")
            self.load_next_text()  # Load the next unlabeled text
        else:
            messagebox.showerror("Error", "No data loaded. Please import a CSV file first.")
