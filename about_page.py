import tkinter as tk
from tkinter import ttk

class AboutPage(tk.Frame):
    def __init__(self, parent, styles):
        super().__init__(parent)
        tk.Label(self, text="About Fast CSS Tool", **styles.title_style).grid(row=0, column=0, columnspan=2, sticky='w', pady=10, padx=5)
        
        about_text = ("Welcome to Fast CSS Tool, a pioneering software designed to empower social scientists with "
                      "advanced data analytics capabilities. This tool integrates artificial intelligence and machine learning "
                      "to facilitate the analysis of digital datasets, making it accessible even to non-programmers. "
                      "Whether you're analyzing social media data or complex network structures, Fast CSS Tool simplifies "
                      "your research process.")
        
        tk.Label(self, text=about_text, wraplength=867, justify="left" , **styles.label_style).grid(row=1, column=0, sticky="w", padx=10)
        

        self.logo_image = tk.PhotoImage(file='images/about_fast_css_tool.png') 
        logo_label = tk.Label(self, image=self.logo_image)
        logo_label.image = self.logo_image
        logo_label.grid(row=2, column=0, padx=10, pady=10)