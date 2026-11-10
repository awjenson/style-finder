"""
Module for image processing, encoding, and similarity matching.
"""

import base64
from io import BytesIO
import traceback
import numpy as np
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, resnet50


class ImageProcessor:
    def __init__(
        self,
        image_size=(224, 224),
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=weights)
        except Exception:
            self.model = resnet50(pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    def encode_image(self, image_input, is_url=True):
        try:
            if is_url:
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                if isinstance(image_input, Image.Image):
                    image = image_input.convert("RGB")
                else:
                    image = Image.open(image_input).convert("RGB")

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(input_tensor)

            feature_vector = features.cpu().numpy().flatten()
            return {"base64": base64_string, "vector": feature_vector}
        except Exception as e:
            print(f"Error encoding image: {e}")
            traceback.print_exc()
            return {"base64": None, "vector": None}

    def find_closest_match(self, user_vector, dataset):
        try:
            if user_vector is None:
                return None, None

            valid_df = dataset.dropna(subset=['Embedding']).reset_index(drop=True)
            if valid_df.empty:
                return None, None

            dataset_vectors = np.vstack(valid_df['Embedding'].values)
            user_vec = np.asarray(user_vector).reshape(1, -1)

            similarities = cosine_similarity(user_vec, dataset_vectors)
            closest_index = int(np.argmax(similarities))
            similarity_score = float(similarities[0][closest_index])

            closest_row = valid_df.iloc[closest_index]
            return closest_row, similarity_score
        except Exception as e:
            print(f"Error finding closest match: {e}")
            traceback.print_exc()
            return None, None
