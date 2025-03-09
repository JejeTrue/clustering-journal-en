import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Clustering import ClusteringKmeans, InjectData
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

class App:
    def __init__(self):
        self.clustering = ClusteringKmeans()
        self.inject_data = InjectData()

    def load_journal_data(self):
        df = pd.read_csv('data\combined_data.csv')
        title_unik = list(df.journal.unique())
        journal_id = {title_unik[i]: i + 1 for i in range(len(title_unik))}
        return ['Pilih Jurnal'] + title_unik, journal_id

    @staticmethod
    def plot_silhouette_chart(df_silhouette, best_k):
        plt.figure(figsize=(8, 6))
        bars = plt.bar(df_silhouette['k'], df_silhouette['Silhouette Score'], 
                    color='skyblue', edgecolor='black')
        plt.xlabel("Jumlah Klaster (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score untuk Nilai k yang Berbeda")
        plt.xticks(df_silhouette['k'])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.grid(True, axis='y')
        return plt

    @staticmethod
    def plot_cluster_distribution(X_2d, cluster_labels):
        df_pca = pd.DataFrame(X_2d, columns=['Dimension 1', 'Dimension 2'])
        df_pca['Cluster Label'] = cluster_labels

        cluster_palette = sns.color_palette('tab10', n_colors=len(np.unique(cluster_labels)))
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Cluster Label', 
                       data=df_pca, palette=cluster_palette)

        centroids = []
        for label in np.unique(cluster_labels):
            centroid = np.mean(X_2d[cluster_labels == label], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', c='red', 
                   s=50, label='Centroids')

        plt.title('Representasi Laten PCA dengan Centroid')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        return plt

    @staticmethod
    def plot_inject_analysis(source_embeddings_2d, target_embeddings_2d, predicted_clusters, 
                           source_jid, target_jid):
        plt.figure(figsize=(10, 6))
        plt.scatter(target_embeddings_2d[:, 0], target_embeddings_2d[:, 1], 
                   label='Target Journal Articles', alpha=0.5)
        plt.scatter(source_embeddings_2d[:, 0], source_embeddings_2d[:, 1], 
                   c=predicted_clusters, label='Source Journal Articles (Projected)', 
                   cmap='viridis')

        plt.title(f'Analisis Injeksi: Jurnal {source_jid} ke {target_jid}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        return plt

    @staticmethod
    def reset_session_state():
        for key in list(st.session_state.keys()):
            del st.session_state[key]

    def run(self):
        st.title("Aplikasi Pengelompokan Artikel Ilmiah Menggunakan RoBERTa dan K-means")
        title_unik, journal_id_map = self.load_journal_data()

        st.sidebar.title("Navigasi")
        page = st.sidebar.selectbox("Go to", ["Home", "Inject Data"], 
                                  on_change=self.reset_session_state)

        if page == "Home":
            self.run_home_page(title_unik, journal_id_map)
        elif page == "Inject Data":
            self.run_inject_page()

    def run_home_page(self, title_unik, journal_id_map):
        st.header("Pilihan Jurnal")
        selected_journal = st.selectbox("Pilih Salah Satu Jurnal", title_unik)

        if selected_journal == 'Pilih Jurnal':
            jurnal_id = None
        else:
            jurnal_id = journal_id_map[selected_journal]

        if jurnal_id is not None:
            if st.button("Proses Pengelompokan"):
                embeddings = np.load(f'embeddings_en/embeddings_{jurnal_id}.npy')
                results = self.clustering.run_clustering(embeddings, jurnal_id)
                st.session_state.clustering_results = {
                    'df_silhouette': results[0],
                    'best_k': results[1],
                    'best_silhouette_score': results[2],
                    'X_2d': results[3],
                    'cluster_labels': results[4]
                }

        if 'clustering_results' in st.session_state:
            results = st.session_state.clustering_results

            st.subheader("Analisis Silhouette Score untuk jumlah klaster yang berbeda")
            silhouette_chart = self.plot_silhouette_chart(results['df_silhouette'], 
                                                        results['best_k'])
            st.pyplot(silhouette_chart)

            st.write(f"Nilai k terbaik: {results['best_k']}")
            st.write(f"Silhouette Score terbaik: {results['best_silhouette_score']}")

            st.subheader("Distribusi klaster")
            cluster_plot = self.plot_cluster_distribution(results['X_2d'], 
                                                        results['cluster_labels'])
            st.pyplot(cluster_plot)

    def run_inject_page(self):
        st.title("Inject Data Antara Jurnal-Jurnal")

        title_unik = ['Pilih Jurnal', 'JURNAL OBSESI: JURNAL PENDIDIKAN ANAK USIA DINI',
                      'JURNAL CENDEKIA : JURNAL PENDIDIKAN MATEMATIKA',
                      'INTERNATIONAL JOURNAL OF ELEMENTARY EDUCATION',
                      'JURNAL BISNIS DAN AKUNTANSI',
                      'JURNAL AKUNTANSI DAN KEUANGAN',
                      'JURNAL PENDIDIKAN TEKNIK MESIN UNDIKSHA',
                      'INTERNATIONAL JOURNAL OF BASIC AND APPLIED SCIENCE',
                      'JURNAL KESEHATAN MASYARAKAT',
                      'GADJAH MADA INTERNATIONAL JOURNAL OF BUSINESS',
                      'JURNAL KESEHATAN ANDALAS',
                      'E-JOURNAL OF CULTURAL STUDIES',
                      'E-JURNAL AKUNTANSI']

        journal_map = {title: idx for idx, title in enumerate(title_unik[1:], start=1)}

        source_journal = st.selectbox("Pilih Jurnal Sumber", title_unik)
        target_journal = st.selectbox("Pilih Jurnal Target", title_unik)

        if st.button("Proses Inject"):
            if source_journal == "Pilih Jurnal" or target_journal == "Pilih Jurnal":
                st.error("Mohon pilih jurnal sumber dan target yang valid.")
            elif source_journal == target_journal:
                st.error("Jurnal sumber dan target tidak boleh sama.")
            else:
                source_jid = journal_map[source_journal]
                target_jid = journal_map[target_journal]

                st.write(f"Memulai proses inject dari jurnal {source_journal} ke jurnal {target_journal}...")

                result_df, source_emb_2d, target_emb_2d, pred_clusters = self.inject_data.analyze(
                    source_jid, target_jid)
                
                inject_plot = self.plot_inject_analysis(source_emb_2d, target_emb_2d, 
                                                      pred_clusters, source_jid, target_jid)
                st.pyplot(inject_plot)
                
                st.write("Hasil Analisis:")
                st.dataframe(result_df)

if __name__ == "__main__":
    app = App()
    app.run()