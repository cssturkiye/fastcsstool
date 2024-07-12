import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk

from styles import Styles
from data_generation_page import DataGenerationPage
from data_preprocessing_page import DataPreprocessingPage
from data_labeling_page import DataLabelingPage
from model_training_page import ModelTrainingPage
# from model_training_grid_page import ModelTrainingGridPage
from analyze_data_page import AnalyzeDataPage
from about_page import AboutPage
from help_page import HelpPage

class MainApplication:
    def __init__(self, root, styles):
        self.root = root
        self.styles = styles

        self.root.set_theme("arc")
        self.root.title('Fast CSS Tool')
        self.root.geometry("1100x700")
        self.root.resizable(width=False, height=False)

        # self.iconbitmap('images/icon.ico')
        root.iconbitmap("images/icon.ico")


        self.side_menu = ttk.Frame(self.root, width=200, height=500)
        self.side_menu.pack(side="left", fill="y")
        self.side_menu.pack_propagate(False)

        self.logo_image = tk.PhotoImage(file='images/fast_css_tool_logo.png')
        logo_label = ttk.Label(self.side_menu, image=self.logo_image)
        logo_label.pack(padx=10, pady=50)

        self.content_area = ttk.Frame(self.root)
        self.content_area.pack(side="right", fill="both", expand=True)

        self.pages = {}
        self.page_classes = [
            DataGenerationPage, 
            DataPreprocessingPage, 
            DataLabelingPage, 
            ModelTrainingPage, 
            # ModelTrainingGridPage, 
            AnalyzeDataPage, 
            AboutPage, 
            HelpPage
        ]

        for page_class in self.page_classes:
            class_name = page_class.__name__
            page_name = styles.page_names.get(class_name, class_name)
            btn = ttk.Button(self.side_menu, text=page_name, command=lambda pc=page_class: self.show_page(pc))
            btn.pack(fill="x")

        self.show_page(AnalyzeDataPage)  # Show the About page by default

    def show_page(self, page_class):
        if page_class not in self.pages:
            self.pages[page_class] = page_class(self.content_area, self.styles)
        for page in self.pages.values():
            page.pack_forget()
        self.pages[page_class].pack(fill="both", expand=True)

if __name__ == "__main__":
    root = ThemedTk()
    styles = Styles()  # Instantiate the styles
    app = MainApplication(root, styles)
    root.mainloop()
