#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版梦境内容语义分析 - 基于Qwen3-Embedding

核心改进：
1. 多种聚类算法比较（K-means, DBSCAN, 谱聚类）
2. 自动参数调优（肘部法则确定K值）
3. 聚类效果评估（轮廓系数等）
4. 改进的主题分析（TF-IDF关键词提取）
5. 增强的可视化
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from typing import List, Dict, Tuple, Optional
import warnings
import os
import config
from collections import Counter
import pandas as pd


class ImprovedDreamAnalyzer:
    """改进版梦境内容语义分析工具"""
    
    def __init__(self, model_name: str = None):
        """初始化梦境分析器"""
        if model_name is None:
            self.model = config.load_model(device='cpu')
        else:
            print(f"Loading specified model: {model_name}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            print("✓ Model loaded successfully!")
        
        self.dreams = []
        self.dream_embeddings = None
        self.cluster_labels = None
        self.reduced_embeddings = None
        self.cluster_results = {}
        
    def load_dream_data(self, dreams: List[str]):
        """加载梦境数据"""
        print(f"\nLoading {len(dreams)} dream descriptions...")
        self.dreams = dreams
        
        # 生成梦境embedding
        print("Generating dream embeddings...")
        self.dream_embeddings = self.model.encode(
            dreams,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print(f"✓ Dream embeddings generated! Shape: {self.dream_embeddings.shape}")
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.scaled_embeddings = self.scaler.fit_transform(self.dream_embeddings)
        
    def analyze_similarity(self):
        """分析梦境相似度"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        print("\nAnalyzing dream similarity...")
        
        # 计算相似度矩阵
        similarity_matrix = util.cos_sim(self.dream_embeddings, self.dream_embeddings)
        
        # 找出最相似的梦境对
        most_similar_pairs = []
        for i in range(len(self.dreams)):
            for j in range(i+1, len(self.dreams)):
                similarity = similarity_matrix[i][j].item()
                if similarity > 0.7:
                    most_similar_pairs.append((i, j, similarity))
        
        # 按相似度排序
        most_similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nTop 10 most similar dream pairs:")
        for i, (idx1, idx2, sim) in enumerate(most_similar_pairs[:10]):
            print(f"{i+1}. Similarity: {sim:.4f}")
            print(f"   Dream {idx1+1}: {self.dreams[idx1][:80]}...")
            print(f"   Dream {idx2+1}: {self.dreams[idx2][:80]}...")
            print()
            
        return similarity_matrix
    
    def reduce_dimensions_tsne(self):
        """使用t-SNE降维（优化版）"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        print("\nReducing dimensions using optimized t-SNE...")
        
        # 自适应PCA降维：不能超过样本数量
        max_pca_components = min(50, len(self.dreams) - 1, self.dream_embeddings.shape[1])
        if max_pca_components < 2:
            # 如果样本太少，直接使用原始特征
            embeddings_for_tsne = self.dream_embeddings
            print(f"  Using original embeddings (n_samples too small for PCA)")
        else:
            # 先使用PCA降维，提高t-SNE效果
            pca = PCA(n_components=max_pca_components)
            embeddings_for_tsne = pca.fit_transform(self.dream_embeddings)
            print(f"  PCA reduced to {max_pca_components} dimensions")
        
        # 自适应perplexity
        perplexity = min(30, len(self.dreams) - 1)
        if perplexity < 5:
            perplexity = 5
            
        tsne = TSNE(n_components=2, random_state=42, 
                   perplexity=perplexity, 
                   n_iter=1000, learning_rate=200)
        self.reduced_embeddings = tsne.fit_transform(embeddings_for_tsne)
        
        print(f"✓ Dimensionality reduction completed! Shape: {self.reduced_embeddings.shape}")
        return self.reduced_embeddings
    
    def find_optimal_k(self, max_k: int = 6):
        """使用肘部法则和轮廓系数找到最优K值（改进版），限制最大K值为6"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        print("\nFinding optimal number of clusters (improved, limited to max 6)...")
        
        # 限制K值范围在2-6之间，优先考虑3-4个聚类
        k_values = range(2, min(max_k + 1, len(self.dreams), 7))  # 最大6个聚类
        
        inertias = []
        silhouette_scores = []
        
        for k in k_values:
            # 使用BisectingKMeans替代普通KMeans，以获得更均衡的聚类
            kmeans = BisectingKMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.dream_embeddings)
            
            inertias.append(kmeans.inertia_)
            
            if len(set(labels)) > 1:
                silhouette_scores.append(silhouette_score(self.dream_embeddings, labels))
            else:
                silhouette_scores.append(0)
        
        # 改进的肘部点检测：寻找曲率变化最大的点
        if len(inertias) >= 3:
            # 计算二阶导数（曲率）
            second_derivatives = np.diff(np.diff(inertias))
            if len(second_derivatives) > 0:
                # 找到曲率变化最大的点（肘部点）
                elbow_point = np.argmax(np.abs(second_derivatives)) + 2
            else:
                elbow_point = 2
        else:
            elbow_point = 2
        
        # 确保肘部点在合理范围内
        elbow_point = min(elbow_point, max(k_values))
        
        # 改进的轮廓系数选择：优先选择3-4个聚类
        silhouette_scores_array = np.array(silhouette_scores)
        
        # 给3-4个聚类更高的权重
        weighted_scores = []
        for i, score in enumerate(silhouette_scores):
            k = k_values[i]
            # 权重：优先3-4个聚类
            if k == 3 or k == 4:
                weight = 1.5  # 最高权重
            elif k == 2 or k == 5:
                weight = 1.2  # 中等权重
            else:
                weight = 1.0  # 默认权重
            weighted_scores.append(score * weight)
        
        best_silhouette_k = k_values[np.argmax(weighted_scores)]
        
        # 如果轮廓系数都很低，默认使用3个聚类
        if np.max(silhouette_scores_array) < 0.1:
            best_silhouette_k = 3
            print(f"  Low silhouette scores detected, defaulting to K = {best_silhouette_k}")
        
        print(f"  Elbow method suggests K = {elbow_point}")
        print(f"  Weighted silhouette method suggests K = {best_silhouette_k}")
        print(f"  Silhouette scores range: {min(silhouette_scores):.3f} to {max(silhouette_scores):.3f}")
        print(f"  Available K values: {list(k_values)}")
        
        return {
            'elbow_k': elbow_point,
            'silhouette_k': best_silhouette_k,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
    
    def cluster_dreams_kmeans_optimized(self, target_clusters: int = None):
        """使用优化版K-means（BisectingKMeans）聚类，默认目标为3-4个聚类"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        # 自动确定最佳K值，但优先考虑3-4个聚类
        optimal_k = self.find_optimal_k(max_k=6)
        
        # 如果指定了目标聚类数量，使用指定值
        if target_clusters is not None:
            n_clusters = target_clusters
            print(f"\nUsing specified number of clusters: {n_clusters}")
        else:
            n_clusters = optimal_k['silhouette_k']
            # 确保聚类数量在3-4之间
            if n_clusters < 3:
                n_clusters = 3
            elif n_clusters > 4:
                # 如果算法建议多于4个聚类，使用3或4中轮廓系数较高的
                if optimal_k['silhouette_scores'][1] > optimal_k['silhouette_scores'][2]:  # 比较K=3和K=4的轮廓系数
                    n_clusters = 3
                else:
                    n_clusters = 4
        
        print(f"\nClustering dreams into {n_clusters} clusters using Bisecting K-means (for better balance)...")
        
        # 使用BisectingKMeans替代普通KMeans，通常能产生更均衡的聚类
        # bisecting_strategy='largest_cluster' 倾向于切分最大的聚类，有助于均衡
        kmeans = BisectingKMeans(n_clusters=n_clusters, random_state=42, n_init=20, bisecting_strategy='largest_cluster')
        labels = kmeans.fit_predict(self.dream_embeddings)
        
        # 评估聚类效果
        silhouette = silhouette_score(self.dream_embeddings, labels)
        davies_bouldin = davies_bouldin_score(self.dream_embeddings, labels)
        
        # 统计每个聚类的梦境数量
        unique, counts = np.unique(labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} dreams")
        
        # 检查聚类分布是否均匀
        min_count = min(counts)
        max_count = max(counts)
        ratio = max_count / min_count
        if ratio > 3:  # 稍微严格一点的检查
            print(f"  ⚠️  Warning: Cluster distribution is still somewhat uneven (ratio: {ratio:.1f}x)")
        else:
            print(f"  ✓ Cluster distribution is relatively balanced (ratio: {ratio:.1f}x)")
            
        print(f"\nClustering metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (越高越好)")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (越低越好)")
        
        # Store results similar to other clustering methods
        self.cluster_results['kmeans'] = {
            'labels': labels,
            'model': kmeans,
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
        
        self.cluster_labels = labels
        return labels
    
    def cluster_dreams_dbscan_optimized(self):
        """使用优化版DBSCAN聚类"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        print("\nClustering dreams using optimized DBSCAN...")
        
        # 自动确定eps参数
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=min(5, len(self.dreams)))
        neighbors_fit = neighbors.fit(self.dream_embeddings)
        distances, indices = neighbors_fit.kneighbors(self.dream_embeddings)
        distances = np.sort(distances[:, -1])
        
        # 使用k-distance图的拐点确定eps
        eps = np.percentile(distances, 70)  # 使用70%分位数
        
        # 自适应min_samples
        min_samples = max(2, min(5, len(self.dreams) // 20))
        
        print(f"  Auto-selected parameters: eps={eps:.3f}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(self.dream_embeddings)
        
        # 统计聚类结果
        unique, counts = np.unique(labels, return_counts=True)
        print("\nDBSCAN clustering results:")
        n_clusters = 0
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                print(f"  Noise points: {count} dreams")
            else:
                print(f"  Cluster {cluster_id}: {count} dreams")
                n_clusters += 1
        
        print(f"\nFound {n_clusters} clusters with {np.sum(labels == -1)} noise points")
        
        # 评估聚类效果（排除噪声点）
        valid_mask = labels != -1
        if np.sum(valid_mask) > 1 and len(set(labels[valid_mask])) > 1:
            silhouette = silhouette_score(self.dream_embeddings[valid_mask], labels[valid_mask])
            davies_bouldin = davies_bouldin_score(self.dream_embeddings[valid_mask], labels[valid_mask])
        else:
            silhouette = 0
            davies_bouldin = float('inf')
        
        print(f"\nClustering metrics (excluding noise):")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        self.cluster_results['dbscan'] = {
            'labels': labels,
            'model': dbscan,
            'eps': eps,
            'min_samples': min_samples,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
        
        return labels
    
    def cluster_dreams_hierarchical(self, n_clusters: int = None):
        """使用层次聚类，默认目标为3-4个聚类"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        # 如果没有指定聚类数量，默认使用3-4个聚类
        if n_clusters is None:
            # 根据数据量决定：小数据集用3个，大数据集用4个
            if len(self.dreams) < 30:
                n_clusters = 3
            else:
                n_clusters = 4
        
        # 确保聚类数量在合理范围内（2-6）
        n_clusters = max(2, min(n_clusters, 6))
        
        print(f"\nClustering dreams into {n_clusters} clusters using Hierarchical Clustering...")
        
        # 使用层次聚类
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='ward'
        )
        labels = hierarchical.fit_predict(self.dream_embeddings)
        
        # 评估聚类效果
        silhouette = silhouette_score(self.dream_embeddings, labels)
        davies_bouldin = davies_bouldin_score(self.dream_embeddings, labels)
        
        # 统计每个聚类的梦境数量
        unique, counts = np.unique(labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} dreams")
        
        # 检查聚类分布是否均匀
        if len(counts) > 1:
            min_count = min(counts)
            max_count = max(counts)
            if max_count / min_count > 5:  # 如果最大聚类是最小聚类的5倍以上
                print(f"  ⚠️  Warning: Cluster distribution is uneven (ratio: {max_count/min_count:.1f}x)")
        
        print(f"\nClustering metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (越高越好)")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (越低越好)")
        
        self.cluster_results['hierarchical'] = {
            'labels': labels,
            'model': hierarchical,
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
        
        self.cluster_labels = labels
        return labels
    
    def cluster_dreams_manual_k(self, n_clusters: int = 4):
        """手动指定聚类数量的K-means聚类，默认4个聚类"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        # 确保聚类数量在合理范围内（2-6）
        n_clusters = max(2, min(n_clusters, 6))
            
        print(f"\nClustering dreams into {n_clusters} clusters using manual K-means...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(self.dream_embeddings)
        
        # 评估聚类效果
        silhouette = silhouette_score(self.dream_embeddings, labels)
        davies_bouldin = davies_bouldin_score(self.dream_embeddings, labels)
        
        # 统计每个聚类的梦境数量
        unique, counts = np.unique(labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} dreams")
        
        # 检查聚类分布是否均匀
        if len(counts) > 1:
            min_count = min(counts)
            max_count = max(counts)
            if max_count / min_count > 5:  # 如果最大聚类是最小聚类的5倍以上
                print(f"  ⚠️  Warning: Cluster distribution is uneven (ratio: {max_count/min_count:.1f}x)")
        
        print(f"\nClustering metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (越高越好)")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (越低越好)")
        
        self.cluster_results['manual_kmeans'] = {
            'labels': labels,
            'model': kmeans,
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
        
        self.cluster_labels = labels
        return labels
    
    def analyze_cluster_themes(self, labels=None, top_n_words=10):
        """分析每个聚类的主题特征（使用TF-IDF）"""
        if labels is None:
            if self.cluster_labels is None:
                raise ValueError("Please perform clustering first")
            labels = self.cluster_labels
        
        print(f"\nAnalyzing cluster themes using TF-IDF...")
        
        cluster_themes = {}
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
                
            cluster_dreams = [self.dreams[i] for i in range(len(self.dreams)) 
                            if labels[i] == cluster_id]
            
            if len(cluster_dreams) < 2:
                continue
                
            print(f"\n--- Cluster {cluster_id} ({len(cluster_dreams)} dreams) ---")
            
            # 使用TF-IDF提取关键词
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
                tfidf_matrix = vectorizer.fit_transform(cluster_dreams)
                feature_names = vectorizer.get_feature_names_out()
                
                # 计算每个词的TF-IDF总分
                tfidf_scores = tfidf_matrix.sum(axis=0).A1
                word_scores = list(zip(feature_names, tfidf_scores))
                word_scores.sort(key=lambda x: x[1], reverse=True)
                
                top_words = [word for word, score in word_scores[:top_n_words]]
                print(f"Top {top_n_words} keywords: {', '.join(top_words)}")
                
                # 显示代表性梦境
                print("Representative dreams:")
                for i, dream in enumerate(cluster_dreams[:2]):
                    print(f"  {i+1}. {dream[:150]}...")
                
                cluster_themes[cluster_id] = {
                    'keywords': top_words,
                    'sample_dreams': cluster_dreams[:2],
                    'tfidf_scores': dict(word_scores[:top_n_words])
                }
                
            except Exception as e:
                print(f"Error analyzing cluster {cluster_id}: {e}")
                # 回退到简单词频统计
                all_words = []
                for dream in cluster_dreams:
                    words = dream.lower().split()
                    all_words.extend([w for w in words if len(w) > 3])
                
                word_freq = Counter(all_words)
                common_words = word_freq.most_common(top_n_words)
                top_words = [word for word, freq in common_words]
                
                print(f"Top {top_n_words} words (frequency): {', '.join(top_words)}")
                
                cluster_themes[cluster_id] = {
                    'keywords': top_words,
                    'sample_dreams': cluster_dreams[:2],
                    'word_frequencies': dict(common_words)
                }
        
        return cluster_themes
    
    def visualize_clustering_results(self, labels=None, title="梦境内容语义聚类分析"):
        """可视化聚类结果"""
        if labels is None:
            if self.cluster_labels is None:
                raise ValueError("Please perform clustering first")
            labels = self.cluster_labels
        
        if self.reduced_embeddings is None:
            self.reduce_dimensions_tsne()
            
        print(f"\nCreating clustering visualization...")
        
        # 准备数据
        df = pd.DataFrame({
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'cluster': [f"Cluster {label}" if label != -1 else "Noise" for label in labels],
            'dream': [dream[:100] + "..." if len(dream) > 100 else dream for dream in self.dreams]
        })
        
        # 创建交互式散点图
        fig = px.scatter(
            df, x='x', y='y', color='cluster',
            hover_data={'dream': True, 'cluster': True},
            title=title,
            width=1200, height=800,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(
            marker=dict(size=12, opacity=0.8, line=dict(width=1, color='white')),
            selector=dict(mode='markers')
        )
        
        fig.update_layout(
            title_font=dict(size=20, family='Arial, sans-serif'),
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            hovermode='closest',
            template='plotly_white',
            legend=dict(title="Dream Clusters", font=dict(size=12))
        )
        
        return fig
    
    def compare_clustering_algorithms(self, include_hierarchical=True, include_manual=True):
        """比较不同聚类算法的效果（包括更多算法）"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        print("\n" + "="*80)
        print("Comparing Clustering Algorithms (Extended)")
        print("="*80)
        
        # 运行聚类算法
        algorithms = [
            ('K-means (Auto)', self.cluster_dreams_kmeans_optimized),
            ('DBSCAN', self.cluster_dreams_dbscan_optimized)
        ]
        
        # 添加层次聚类
        if include_hierarchical:
            algorithms.append(('Hierarchical', lambda: self.cluster_dreams_hierarchical()))
        
        # 添加手动K-means（4个聚类）
        if include_manual:
            algorithms.append(('K-means (4 clusters)', lambda: self.cluster_dreams_manual_k(4)))
        
        results = []
        for name, method in algorithms:
            print(f"\n--- Running {name} ---")
            try:
                labels = method()
                if labels is not None:
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_points = np.sum(labels == -1) if -1 in labels else 0
                    
                    # 获取算法名称的键
                    if 'Auto' in name:
                        key = 'kmeans'
                    elif 'DBSCAN' in name:
                        key = 'dbscan'
                    elif 'Hierarchical' in name:
                        key = 'hierarchical'
                    elif '4 clusters' in name:
                        key = 'manual_kmeans'
                    else:
                        key = name.lower().replace(' ', '_')
                    
                    if key in self.cluster_results:
                        result = self.cluster_results[key]
                        results.append({
                            'Algorithm': name,
                            'Clusters': n_clusters,
                            'Noise': noise_points,
                            'Silhouette': result.get('silhouette', 0),
                            'Davies-Bouldin': result.get('davies_bouldin', float('inf'))
                        })
            except Exception as e:
                print(f"Error running {name}: {e}")
        
        # 创建比较表格
        if results:
            df = pd.DataFrame(results)
            print("\n" + "="*80)
            print("Clustering Algorithm Comparison")
            print("="*80)
            print(df.to_string(index=False))
            
            # 找出最佳算法（基于轮廓系数）
            if len(df) > 0:
                best_idx = df['Silhouette'].idxmax()
                best_algorithm = df.loc[best_idx, 'Algorithm']
                best_silhouette = df.loc[best_idx, 'Silhouette']
                print(f"\n✓ Best algorithm based on Silhouette Score: {best_algorithm} ({best_silhouette:.4f})")
                
                # 根据聚类数量给出建议
                best_clusters = df.loc[best_idx, 'Clusters']
                print(f"  Creates {best_clusters} clusters")
                
                # 如果最佳算法聚类数量太少，建议使用层次聚类
                if best_clusters < 4 and include_hierarchical:
                    print(f"  ⚠️  Low number of clusters detected. Consider using Hierarchical clustering for more granular results.")
            
            return df
        else:
            print("No clustering results available for comparison")
            return None
    
    def run_complete_analysis(self, dreams: List[str], output_dir="dream_analysis_results"):
        """运行完整的分析流程"""
        print("=" * 80)
        print("开始完整梦境分析流程")
        print("=" * 80)
        
        # 1. 加载数据
        self.load_dream_data(dreams)
        
        # 2. 分析相似度
        print("\n[2/6] 分析梦境相似度...")
        self.analyze_similarity()
        
        # 3. 比较聚类算法
        print("\n[3/6] 比较聚类算法...")
        comparison_df = self.compare_clustering_algorithms()
        
        # 4. 分析聚类主题
        print("\n[4/6] 分析聚类主题...")
        if self.cluster_labels is not None:
            themes = self.analyze_cluster_themes()
            
            # 保存主题分析结果
            os.makedirs(output_dir, exist_ok=True)
            themes_path = os.path.join(output_dir, "cluster_themes.txt")
            with open(themes_path, "w", encoding="utf-8") as f:
                f.write("梦境聚类主题分析报告\n")
                f.write("="*50 + "\n\n")
                
                for cluster_id, theme_info in themes.items():
                    f.write(f"聚类 {cluster_id}:\n")
                    f.write(f"  关键词: {', '.join(theme_info['keywords'])}\n")
                    f.write(f"  示例梦境:\n")
                    for i, dream in enumerate(theme_info['sample_dreams']):
                        f.write(f"    {i+1}. {dream[:200]}...\n")
                    f.write("\n")
            
            print(f"✓ 主题分析结果已保存到 {themes_path}")
        
        # 5. 创建可视化
        print("\n[5/6] 创建聚类可视化...")
        if self.cluster_labels is not None:
            fig = self.visualize_clustering_results(
                title="梦境内容语义聚类分析 (改进版)"
            )
            
            # 保存HTML文件
            viz_path = os.path.join(output_dir, "dream_clusters.html")
            fig.write_html(viz_path)
            print(f"✓ 可视化已保存到 {viz_path}")
        
        # 6. 保存聚类比较结果
        print("\n[6/6] 保存分析结果...")
        if comparison_df is not None:
            results_path = os.path.join(output_dir, "clustering_comparison.csv")
            comparison_df.to_csv(results_path, index=False)
            print(f"✓ 聚类比较结果已保存到 {results_path}")
        
        print("\n" + "=" * 80)
        print("✓ 完整梦境分析流程完成!")
        print("=" * 80)
        print(f"\n所有结果已保存到目录: {output_dir}")
        print("包含文件:")
        print("  - cluster_themes.txt - 聚类主题分析")
        print("  - dream_clusters.html - 交互式可视化")
        print("  - clustering_comparison.csv - 聚类算法比较")
        
        return output_dir
