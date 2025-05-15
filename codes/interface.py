import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
from playsound import playsound
import threading
from PIL import Image, ImageTk
import os

# Charger le modèle
model = load_model("cuicui/modeles/saved_model.keras")

# Charger le fichier CSV contenant les noms des classes
df = pd.read_csv("cuicui/ouiseau/bird_songs_metadata.csv")

# Liste des classes d'oiseaux
class_names = df["name"].unique()

# Variable globale
selected_file = None
current_image = None

# Fonction de reconnaissance
def recognize_bird(wave_file: str, model: tf.keras.Model, class_names: list) -> str:
    """
    Reconnaît l'oiseau à partir d'un fichier audio.
    :param wave_file: Chemin vers le fichier audio
    :param model: Modèle de reconnaissance
    :param class_names: Liste des noms de classes
    :return: Nom de l'oiseau reconnu
    """
    audio_data, sample_rate = librosa.load(wave_file, duration=3)
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = tf.expand_dims(mel_spec, axis=0)
    prediction = model.predict(mel_spec)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_names[predicted_class]


# Interface Tkinter
def browse_file():
    global selected_file
    file_path = filedialog.askopenfilename(filetypes=[("Fichiers audio", "*.wav")])
    if file_path:
        selected_file = file_path
        try:
            prediction = recognize_bird(file_path, model, class_names)
            result_label.config(text=f"Oiseau reconnu : {prediction}")
            # Afficher l'image de l'oiseau
            show_bird_image(prediction)
            # Jouer le son de l'oiseau reconnu
            play_button.config(state = tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue :\n{str(e)}")

# Fonction pour jouer un fichier audio
def play_audio():
    global selected_file
    if selected_file: 
        try:
            threading.Thread(target=playsound, args=(selected_file,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire le fichier audio :\n{str(e)}")

def show_bird_image(bird_name):
    global current_image
    # Chemin de l'image de l'oiseau
    image_path = f"cuicui/oiso/{bird_name}.jpg"
    print(f"Chemin de l'image : {image_path}")

    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((200, 200))  # Redimensionner pour l’affichage
            current_image = ImageTk.PhotoImage(image)
            image_label.config(image=current_image, text="")
            image_label.image = current_image  # Attacher l'image au widget
        except Exception as e:
            print(f"Erreur lors du chargement de l'image : {e}")
            image_label.config(image='', text="Erreur lors du chargement de l'image")
    else:
        print(f"L'image {image_path} est introuvable.")
        image_label.config(image='', text="Image non trouvée")


# Créer la fenêtre principale
root = tk.Tk()
root.title("Reconnaissance d'oiseaux")
root.geometry("500x600")

title_label = tk.Label(root, text="Reconnaissance d'oiseaux par audio", font=("Arial", 14))
title_label.pack(pady=10)

browse_button = tk.Button(root, text="Choisir un fichier audio (.wav)", command=browse_file)
browse_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

image_label = tk.Label(root, text="Image de l'oiseau")
image_label.pack(pady=20)

play_button = tk.Button(root, text="Jouer le son de l'oiseau", command=play_audio(), state=tk.DISABLED)
play_button.pack(pady=10)



root.mainloop()
