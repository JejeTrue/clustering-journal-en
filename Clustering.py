import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import RobertaModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import pandas as pd
from sklearn.cluster import KMeans


class RobertaClassifier(nn.Module):
    def __init__(self, num_labels, model_path='C:/Coding/skripsi/model/finetuned_model_roberta_en.pt'):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_labels)
        )
        
        # Load the pre-trained weights
        if model_path:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint['model_state_dict'])
            self.eval()  # Set the model to evaluation mode

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs['last_hidden_state']
        return x

class ArticleDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx]
        }

class EmbeddingRoberta:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_data(self, texts):
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

class ClusteringKmeans:
    def __init__(self):
        self.pca = None
        self.kmeans = None

    def run_clustering(self, embeddings, jurnal_id):
        X = embeddings.reshape(embeddings.shape[0], -1)
        self.pca = joblib.load(f'pca_en/pca_model_{jurnal_id}.pkl')
        X_2d = self.pca.transform(X)

        silhouette_scores = []
        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, random_state=0)
            cluster_labels = kmeans.fit_predict(X_2d)
            silhouette_avg = silhouette_score(X_2d, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        best_k = np.argmax(silhouette_scores) + 2
        df_silhouette = pd.DataFrame({'k': range(2, 7), 'Silhouette Score': silhouette_scores})
        self.kmeans = KMeans(n_clusters=best_k, random_state=0)
        cluster_labels = self.kmeans.fit_predict(X_2d)

        return df_silhouette, best_k, max(silhouette_scores), X_2d, cluster_labels

class InjectData:
    def __init__(self):
        self.df = pd.read_csv('data\combined_data.csv')

    def analyze(self, source_jid, target_jid):
        source_data = self.df[self.df['jid'] == source_jid]
        target_data = self.df[self.df['jid'] == target_jid]

        source_embeddings_2d = np.load(f'embeddings_2d_en/embeddings_2d_{source_jid}.npy')
        target_embeddings_2d = np.load(f'embeddings_2d_en/embeddings_2d_{target_jid}.npy')

        source_kmeans = joblib.load(f'kmeans_en/kmeans_model_{source_jid}.pkl')
        target_kmeans = joblib.load(f'kmeans_en/kmeans_model_{target_jid}.pkl')

        source_pca = joblib.load(f'pca_en/pca_model_{source_jid}.pkl')
        target_pca = joblib.load(f'pca_en/pca_model_{target_jid}.pkl')

        predicted_clusters = target_kmeans.predict(
            target_pca.transform(source_pca.inverse_transform(source_embeddings_2d))
        )

        distances = []
        for i in range(len(source_embeddings_2d)):
            cluster = predicted_clusters[i]
            centroid = target_kmeans.cluster_centers_[cluster]
            distance = np.linalg.norm(source_embeddings_2d[i] - centroid)
            distances.append(distance)

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold = mean_distance + 2 * std_distance

        results = ["In-scope" if d <= threshold else "Out-scope" for d in distances]

        return pd.DataFrame({
            'Journal Name': source_data['journal'].values,
            'Article': source_data['data_cleaned'].values,
            # 'Article Title': source_data['title'].values,
            'Distance to Centroid': distances,
            'Scope': results
        }), source_embeddings_2d, target_embeddings_2d, predicted_clusters