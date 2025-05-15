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
import pygame

# Initialiser le mixer de pygame
pygame.mixer.init()

# Charger le modèle de reconnaissance d'oiseaux
model = load_model("cuicui/modeles/saved_model.keras")

# Charger le fichier CSV contenant les métadonnées des chants d'oiseaux
df = pd.read_csv("cuicui/ouiseau/bird_songs_metadata.csv")

# Liste des noms des espèces d'oiseaux
class_names = df["name"].unique()

# Variables globales
selected_file = None  # Fichier audio sélectionné
current_image = None  # Image actuellement affichée

# Fonction pour reconnaître l'espèce d'oiseau à partir d'un fichier audio
def recognize_bird(wave_file: str, model: tf.keras.Model, class_names: list) -> str:
    """
    Reconnaît l'espèce d'oiseau à partir d'un fichier audio.
    :param wave_file: Chemin vers le fichier audio
    :param model: Modèle de reconnaissance
    :param class_names: Liste des noms des espèces d'oiseaux
    :return: Nom de l'espèce d'oiseau reconnue
    """
    audio_data, sample_rate = librosa.load(wave_file, duration=3)  # Charger le fichier audio
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)  # Calculer le spectrogramme Mel
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convertir en échelle logarithmique
    mel_spec = tf.expand_dims(mel_spec, axis=0)  # Ajouter une dimension pour correspondre à l'entrée du modèle
    prediction = model.predict(mel_spec)  # Faire une prédiction avec le modèle
    predicted_class = np.argmax(prediction, axis=1)[0]  # Obtenir l'indice de la classe prédite
    return class_names[predicted_class]  # Retourner le nom de l'espèce prédite

# Fonction pour parcourir et sélectionner un fichier audio
def browse_file():
    global selected_file
    file_path = filedialog.askopenfilename(filetypes=[("Fichiers audio", "*.wav")])  # Ouvrir une boîte de dialogue pour sélectionner un fichier
    if file_path:
        selected_file = file_path
        try:
            # Reconnaître l'espèce d'oiseau
            prediction = recognize_bird(file_path, model, class_names)
            result_label.config(text=f"Oiseau reconnu : {prediction}")

            # Afficher l'image de l'oiseau
            show_bird_image(prediction)

            # Activer les boutons de contrôle de l'audio
            play_button.config(state=tk.NORMAL)
            pause_button.config(state=tk.NORMAL)
            unpause_button.config(state=tk.NORMAL)
            stop_button.config(state=tk.NORMAL)

            # Activer le bouton pour jouer un son aléatoire de l'espèce reconnue
            random_sound_button.config(state=tk.NORMAL)
            random_sound_button.config(command=lambda: play_random_audio(prediction))

        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue :\n{str(e)}")

# Fonction pour jouer un fichier audio
def play_audio():
    global selected_file
    if selected_file:
        print(f"Chemin du fichier audio : {selected_file}")  # Debugging
        try:
            pygame.mixer.music.load(selected_file)  # Charger le fichier audio
            pygame.mixer.music.play()  # Jouer le fichier audio
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire le fichier audio :\n{str(e)}")
    else:
        print("Aucun fichier audio sélectionné.")

# Fonction pour mettre en pause l'audio
def pause_audio():
    try:
        pygame.mixer.music.pause()  # Mettre en pause l'audio
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de mettre en pause l'audio :\n{str(e)}")

# Fonction pour reprendre l'audio
def unpause_audio():
    try:
        pygame.mixer.music.unpause()  # Reprendre l'audio
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de reprendre l'audio :\n{str(e)}")

# Fonction pour arrêter l'audio
def stop_audio():
    try:
        pygame.mixer.music.stop()  # Arrêter l'audio
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible d'arrêter l'audio :\n{str(e)}")

# Fonction pour afficher l'image de l'oiseau reconnu
def show_bird_image(bird_name):
    global current_image
    # Construire le chemin de l'image de l'oiseau
    image_path = f"cuicui/oiso/{bird_name}.jpg"

    if os.path.exists(image_path):  # Vérifier si l'image existe
        try:
            image = Image.open(image_path)  # Charger l'image
            image = image.resize((200, 200))  # Redimensionner pour l'affichage
            current_image = ImageTk.PhotoImage(image)
            image_label.config(image=current_image, text="")  # Mettre à jour l'image dans l'interface
            image_label.image = current_image  # Attacher l'image au widget
        except Exception as e:
            print(f"Erreur lors du chargement de l'image : {e}")
            image_label.config(image='', text="Erreur lors du chargement de l'image")
    else:
        print(f"L'image {image_path} est introuvable.")
        image_label.config(image='', text="Image non trouvée")

# Fonction pour jouer un son aléatoire de l'espèce reconnue
def play_random_audio(predicted_bird):
    """
    Joue un son aléatoire pour l'espèce d'oiseau reconnue.
    :param predicted_bird: Nom de l'espèce d'oiseau reconnue
    """
    global df

    # Filtrer les fichiers audio pour l'espèce reconnue
    filtered_df = df[df["name"] == predicted_bird]

    try:
        # Sélectionner un fichier audio aléatoire
        random_song = filtered_df.sample(1).iloc[0]
        random_file_path = f"cuicui/ouiseau/wavfiles/{random_song['filename']}"
        threading.Thread(target=playsound, args=(random_file_path,), daemon=True).start()  # Jouer le fichier audio dans un thread séparé

    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de lire un fichier audio :\n{str(e)}")

# ==================================================Créer la fenêtre principale===============================
root = tk.Tk()
root.title("Reconnaissance d'oiseaux")
root.geometry("500x550")

# Ajouter les widgets à l'interface
title_label = tk.Label(root, text="Reconnaissance d'oiseaux par audio", font=("Arial", 14))
title_label.pack(pady=10)

browse_button = tk.Button(root, text="Choisir un fichier audio (.wav)", command=browse_file)
browse_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

image_label = tk.Label(root, text="Image de l'oiseau")
image_label.pack(pady=20)

random_sound_button = tk.Button(root, text="Jouer un son aléatoire de l'espèce détectée", state=tk.DISABLED)
random_sound_button.pack(pady=10)

# Créer un conteneur pour les boutons de contrôle de l'audio
audio_control_frame = tk.Frame(root)
audio_control_frame.pack(pady=10)

# Bouton pour jouer l'audio
play_button = tk.Button(audio_control_frame, text="Jouer le son de l'oiseau", command=play_audio, state=tk.DISABLED)
play_button.pack(side=tk.LEFT, padx=5)

# Bouton pour mettre en pause l'audio
pause_button = tk.Button(audio_control_frame, text="Pause", command=pause_audio, state=tk.DISABLED)
pause_button.pack(side=tk.LEFT, padx=5)

# Bouton pour reprendre l'audio
unpause_button = tk.Button(audio_control_frame, text="Reprendre", command=unpause_audio, state=tk.DISABLED)
unpause_button.pack(side=tk.LEFT, padx=5)

# Bouton pour arrêter l'audio
stop_button = tk.Button(audio_control_frame, text="Arrêter", command=stop_audio, state=tk.DISABLED)
stop_button.pack(side=tk.LEFT, padx=5)

# Lancer la boucle principale de l'interface
root.mainloop()
