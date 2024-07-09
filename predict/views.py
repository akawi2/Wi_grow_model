from django.shortcuts import render

# Create your views here.

import io
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf


def preprocess_image(image):
    # Charger le modèle TensorFlow
    model = tf.keras.models.load_model('detectionfr.keras')

    # Dictionnaire de mappage des indices de prédiction aux étiquettes
    label_map = {0: 'brulure des feuilles', 1: 'Feuilles saines', 2: 'rouille des feuilles', 3: 'Tache foliaire'}


    # Redimensionner selon les besoins de votre modèle (ici 224x224)
    img = image.resize((32, 32))
    # Convertir l'image en tableau numpy et normaliser les pixels
    img = np.array(img) / 255.0
    # Ajouter une dimension pour correspondre à la forme attendue par le modèle
    img = np.expand_dims(img, axis=0)
    return img


@csrf_exempt
def predict_image(request):
    # Charger le modèle TensorFlow
    model = tf.keras.models.load_model('detectionfr.keras')

    # Dictionnaire de mappage des indices de prédiction aux étiquettes
    label_map = {0: 'brulure des feuilles', 1: 'Feuilles saines', 2: 'rouille des feuilles', 3: 'Tache foliaire'}


    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file part'}, status=400)

        file = request.FILES['file']

        if not file.name:
            return JsonResponse({'error': 'No selected file'}, status=400)

        try:
            img = Image.open(io.BytesIO(file.read()))
            img = preprocess_image(img)

            # Faire la prédiction
            predictions = model.predict(img)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_label = label_map[predicted_index]

            # Retourner la prédiction sous forme de texte
            return JsonResponse({'prediction': predicted_label})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)
