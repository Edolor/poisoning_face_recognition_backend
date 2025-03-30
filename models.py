from attacks import AdversarialAttacks
import torch
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image as PILImage
from scipy.spatial.distance import euclidean
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn

# Defining Class That Handles Machine Learning Operations

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class FaceRecognitionModel(nn.Module):
    """Custom Face Recognition Model"""

    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.feature_extractor = InceptionResnetV1(
            pretrained='vggface2', classify=False)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        embeddings = self.feature_extractor(x)
        return embeddings
        # return self.fc(embeddings)


class MachineKernel:
    def __init__(self):
        self.training_statistics = []
        self.adv_kernel = AdversarialAttacks()

    def train_model(self, model, train_loader, criterion, optimizer, num_epochs=5):
        """Function handling model training"""

        model.train()  # Start training of model

        for epoch in range(num_epochs):  # Loop for given EPOCH
            total_loss = 0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader):  # Show loader
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()  # Backpropagate the error
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            accuracy = 100 * correct / total

            # Store statistics for later runs
            self.training_statistics.append(
                {
                    "accuracy": accuracy,
                    "epoch": epoch+1,
                    "loss": total_loss/len(train_loader)
                }
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    def evaluate_model(self, model, val_loader, epsilon, attack_type):
        """Method that evaluates an attack for different types a model in multiple ways [FGSM, PGD]"""

        model.eval()  # Begin model evaluation

        correct = 0
        total = 0
        for images, labels in val_loader:
            # If an adversarial attack is specified, compute it with gradients enabled.
            if epsilon is not None:
                if attack_type == "fgsm":
                    with torch.enable_grad():
                        images = self.adv_kernel.fgsm_attack(
                            images, model, epsilon)
                elif attack_type == "pgd":
                    with torch.enable_grad():
                        images = self.adv_kernel.pgd_attack(
                            images, model, epsilon, 0.05, num_iter=20)

            # Now, no_grad can be used for evaluation
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

    def get_embedding(self, image, model):
        """Function that (squishes our code)"""
        print(image.shape, "SHAPE")

        with torch.no_grad():
            embedding = model(image)
        return embedding.squeeze()

    def predict_face(self, image, model, train_embeddings, train_labels):
        """Function that handles (LABEL-MATCHING)"""
        embedding = self.get_embedding(image, model)

        distances = [euclidean(embedding, train_emb)
                     for train_emb in train_embeddings]

        print(image.shape, "--MIDDLE")

        # Fetching the closest in the embedding layer
        min_distance_idx = np.argmin(distances)
        predicted_label = train_labels[min_distance_idx]

        # Compute confidence as an inverse distance similarity (normalized)
        # Convert distances to similarity

        # Compute confidence as an inverse similarity measure
        # Use exponential to convert distances to similarity scores
        # similarity_scores = np.exp(-np.array(distances, dtype=np.float32))
        # Convert distances to similarity
        similarity_scores = 1 / (1 + np.array(distances, dtype=np.float32))
        confidence = similarity_scores[min_distance_idx] / \
            np.sum(similarity_scores)

        train_dataset_classes = np.load('train_dataset_classes.npy')
        return train_dataset_classes[predicted_label], confidence

    def predict_single_image(self, image, model, train_embeddings, train_labels, attack_type, epsilon=None):
        """Handle"""

        # Load and preprocess the image
        # image = PILImage.open(image_path).convert('RGB')
        image = transform(image)

        if image.dim() == 3:
            print("GOT HERE--- DID NOT DIE")
            image = image.unsqueeze(0)  # Add batch dimension

        # Apply adversarial attack if epsilon is set
        if epsilon is not None and epsilon > 0:
            print("APPLYING ATTACK TO IMAGE")
            if attack_type == "fgsm":
                image = self.adv_kernel.fgsm_attack(image, model, epsilon)
            elif attack_type == "pgd":
                image = self.adv_kernel.pgd_attack(
                    image, model, epsilon, alpha=0.05, num_iter=20)
            else:
                raise ValueError(
                    "Invalid attack type. Choose 'fgsm' or 'pgd'.")

        # Perform prediction
        predicted_person, confidence = self.predict_face(
            image, model, train_embeddings, train_labels)

        return predicted_person, confidence
