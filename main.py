import tkinter as tk
from tkinter import ttk

from utils.Distribution_window import analysis
from utils.Augmentation_window import augmentation

def main() ->int :

  # Create a Tkinter window
  window = tk.Tk()
  window.title("LeafFliction")
  window.geometry("800x600")
  
  # Création du widget Notebook
  notebook = ttk.Notebook(window)

  # Création des onglets
  onglet1 = tk.Frame(notebook)
  onglet2 = tk.Frame(notebook)
  onglet3 = tk.Frame(notebook)
  onglet4 = tk.Frame(notebook)

  # Ajout des onglets au widget Notebook
  notebook.add(onglet1, text="Analysis of the Data Set")
  notebook.add(onglet2, text="Data augmentation")
  notebook.add(onglet3, text="Image Transformation")
  notebook.add(onglet4, text="Classification")

  # Affichage du widget Notebook
  notebook.pack(expand=True, fill="both")

  #Define option for selection
  leafFont = tk.LabelFrame(window, text="Leaf")
  leafFont.pack(padx=100, pady=1)
  leafVar = tk.StringVar()
  leafVar.set("Select the Leaf")
  options = ["Apples", "Grapes"]
  leafSelect = ttk.Combobox(leafFont, textvariable=leafVar, values=options, state="readonly")
  leafSelect.pack()

  #Onglet Analysis / Distribution
  buttonAnalyse = tk.Button(onglet1, text="Analysis of the Data Set", command=lambda:analysis(leafVar.get(), expVar.get()))

  #Onglet Augmentation
  expVar = tk.IntVar(value=0)
  tk.Radiobutton(onglet2, text="Flip", font=("Arial", 12), variable=expVar, value=0).pack(anchor="w")
  tk.Radiobutton(onglet2, text="Rotate", font=("Arial", 12), variable=expVar, value=1).pack(anchor="w")
  tk.Radiobutton(onglet2, text="Contrast", font=("Arial", 12), variable=expVar, value=2).pack(anchor="w")
  tk.Radiobutton(onglet2, text="Brightness", font=("Arial", 12), variable=expVar, value=3).pack(anchor="w")
  tk.Radiobutton(onglet2, text="Shear", font=("Arial", 12), variable=expVar, value=4).pack(anchor="w")
  tk.Radiobutton(onglet2, text="Projection", font=("Arial", 12), variable=expVar, value=5).pack(anchor="w")
  tk.Radiobutton(onglet2, text="Blur", font=("Arial", 12), variable=expVar, value=6).pack(anchor="w")
  buttonAugmentation = tk.Button(onglet2, text="Data augmentation", command=lambda:augmentation(leafVar.get(), expVar.get()))

  buttonAnalyse.pack()
  buttonAugmentation.pack()

  # Start the main event loop
  window.mainloop()
  return 0

if __name__ == "__main__" :
  SystemExit(main()) 
