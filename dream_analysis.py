#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梦境内容语义分析 - 基于Qwen3-Embedding

使用Qwen3-Embedding模型分析梦境内容的语义特征，探索潜意识表达的规律性
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional
import warnings
import os
import config  # Import configuration file

# Try to import UMAP (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed, will only use t-SNE for dimensionality reduction. Install with: pip install umap-learn")


class DreamAnalyzer:
    """梦境内容语义分析工具"""
    
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
        
    def load_dream_data(self, dreams: List[str]):
        """加载梦境数据"""
        print(f"\nLoading {len(dreams)} dream descriptions...")
        self.dreams = dreams
        
        # 生成梦境embedding
        print("Generating dream embeddings...")
        self.dream_embeddings = self.model.encode(
            dreams,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✓ Dream embeddings generated! Shape: {self.dream_embeddings.shape}")
        
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
                if similarity > 0.7:  # 高相似度阈值
                    most_similar_pairs.append((i, j, similarity))
        
        # 按相似度排序
        most_similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nTop 5 most similar dream pairs:")
        for i, (idx1, idx2, sim) in enumerate(most_similar_pairs[:5]):
            print(f"{i+1}. Similarity: {sim:.4f}")
            print(f"   Dream {idx1+1}: {self.dreams[idx1][:80]}...")
            print(f"   Dream {idx2+1}: {self.dreams[idx2][:80]}...")
            print()
            
        return similarity_matrix
    
    def cluster_dreams_kmeans(self, n_clusters: int = 5):
        """使用K-means聚类梦境"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        print(f"\nClustering dreams into {n_clusters} clusters using K-means...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.dream_embeddings)
        
        # 统计每个聚类的梦境数量
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} dreams")
            
        return self.cluster_labels
    
    def cluster_dreams_dbscan(self, eps: float = 0.5, min_samples: int = 2):
        """使用DBSCAN聚类梦境（自动确定聚类数量）"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        print(f"\nClustering dreams using DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(self.dream_embeddings)
        
        # 统计聚类结果
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("\nDBSCAN clustering results:")
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                print(f"  Noise points: {count} dreams")
            else:
                print(f"  Cluster {cluster_id}: {count} dreams")
                
        return self.cluster_labels
    
    def reduce_dimensions(self, method: str = "tsne", n_components: int = 2):
        """降维可视化"""
        if self.dream_embeddings is None:
            raise ValueError("Please load dream data first")
            
        print(f"\nReducing dimensions to {n_components}D using {method.upper()}...")
        
        if method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=15)
        elif method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        self.reduced_embeddings = reducer.fit_transform(self.dream_embeddings)
        print(f"✓ Dimensionality reduction completed! Shape: {self.reduced_embeddings.shape}")
        
        return self.reduced_embeddings
    
    def visualize_dream_clusters(self, title: str = "梦境内容语义聚类分析", save_path: Optional[str] = None):
        """可视化梦境聚类"""
        if self.reduced_embeddings is None or self.cluster_labels is None:
            raise ValueError("Please perform clustering and dimensionality reduction first")
            
        print(f"\nCreating dream cluster visualization...")
        
        # 准备颜色和标签
        colors = [f"Cluster {label}" if label != -1 else "Noise" for label in self.cluster_labels]
        
        # 创建散点图
        fig = go.Figure()
        
        unique_clusters = list(set(colors))
        color_palette = px.colors.qualitative.Set3[:len(unique_clusters)]
        
        for i, cluster in enumerate(unique_clusters):
            mask = [c == cluster for c in colors]
            indices = [j for j, m in enumerate(mask) if m]
            
            # 准备悬停文本
            hover_texts = [
                f"<b>{cluster}</b><br>" +
                f"Dream: {self.dreams[j][:100]}{'...' if len(self.dreams[j]) > 100 else ''}"
                for j in indices
            ]
            
            fig.add_trace(go.Scatter(
                x=self.reduced_embeddings[indices, 0],
                y=self.reduced_embeddings[indices, 1],
                mode='markers',
                name=cluster,
                marker=dict(
                    size=12,
                    color=color_palette[i % len(color_palette)],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family='Arial, sans-serif')
            ),
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode='closest',
            width=1200,
            height=800,
            template='plotly_white',
            legend=dict(
                title="Dream Clusters",
                font=dict(size=12)
            )
        )
        
        # 保存可视化
        if save_path:
            try:
                html_str = fig.to_html(
                    include_plotlyjs='cdn',
                    config={'displayModeBar': True, 'responsive': True},
                    include_mathjax=False
                )
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(html_str)
                print(f"✓ Visualization saved to: {save_path}")
            except Exception as e:
                print(f"⚠ Error saving HTML file: {e}")
                
        return fig
    
    def analyze_cluster_themes(self):
        """分析每个聚类的主题特征"""
        if self.cluster_labels is None:
            raise ValueError("Please perform clustering first")
            
        print(f"\nAnalyzing cluster themes...")
        
        cluster_themes = {}
        
        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:
                continue
                
            cluster_dreams = [self.dreams[i] for i in range(len(self.dreams)) 
                            if self.cluster_labels[i] == cluster_id]
            
            print(f"\n--- Cluster {cluster_id} ({len(cluster_dreams)} dreams) ---")
            
            # 简单主题分析：提取关键词
            all_words = []
            for dream in cluster_dreams:
                words = dream.split()
                all_words.extend(words)
            
            # 统计词频
            from collections import Counter
            word_freq = Counter(all_words)
            common_words = word_freq.most_common(10)
            
            print("Common words:", [word for word, freq in common_words[:5]])
            print("Sample dreams:")
            for i, dream in enumerate(cluster_dreams[:3]):
                print(f"  {i+1}. {dream[:100]}...")
                
            cluster_themes[cluster_id] = {
                'common_words': common_words[:5],
                'sample_dreams': cluster_dreams[:3]
            }
            
        return cluster_themes


def create_sample_dreams():
    """创建示例梦境数据集"""
    dreams = [
        
        "My memory of this dream is vague. I think the setting is on a college campus. I'm in a cafe and two elderly ladies walk in and start talking to me about a university that a guy I am dating got into for law school. They were saying that I was accepted. I thought that this information was weird because I didn't even apply to this school. I got the feeling that while I was talking to these ladies, that they were interviewing me as art of the orientation to go there. I was also pregnant in the dream and he cafe that I was in was a hospital cafe. The guy I am dating is in the dream and we were talking, but I'm not sure about what.",
        "I was in a mansion or more like a castle and there was a person who was supposed to be the devil in the castle. He was snapping his fingers and slitting people's throats. I was running away and riding my bike through my neighborhood. Then I arrive at the preschool where I am doing a field study, and I am a teacher and my students and I are playing a name song. When it gets to one of the male volunteers, he leans in and kisses me. I don't recognize his face, but I seem to know him. I have a strong physical attraction to this person and throughout the dream I hope he feels the same way. Then I am with the children again and we are running from some other children who are the devil. The devil looks like someone from Bach's era. He has a wide grin on his malicious-looking face. The children and I are escaping in a jeep, but have to leave the jeep and continue on foot. We are in a jungle now and I lose one of the children, which has me scared, but then I found the child again, alive and well",
        "This dream took place in the water. Two sisters wanted to be mermaids. So they turned into mermaids, but one of the sisters really wanted to have her sister's legs, so she took them. She returned them to their rightful owner after she was done using them. One of my sorority sisters whom I haven't seen in a while was also in my dream. She was telling me she was going to school in another state, and I told her that I had family there. I felt uncomfortable talking to her and I could sense tension between us. We had been taking care of some babies as we were talking and I felt like she didn't like me, or had something against me",
        "The guy that I have been dating was standing in his room naked with one of his housemates who is a girl, and someone whom he has previously slept with. She was lying on the floor, also naked. I walked into the room and as I saw them, they both told me not to worry, that nothing was going on. I believed them and said,  and walked out of the room.",
        
        # Weding
        "Last night I dreamt my family and I went to see my brother's wedding in Providence. We went into the church and sat down and listened to the music. However we had some difficulty getting into the church and getting places because of the crowds. And the whole dream was repeated with this exception... everything went well and we were seated with no difficulty. My brother and sister-in-law did not appear",
        "About a week ago I dreamt that I was walking down the aisle in our church at home with our uncle. It was a wedding ceremony and I believe that my uncle was giving me away. The most outstanding incident that I can remember was that I was trying to keep in step with the music and my uncle was not keeping in step with me. In the dream I struggled to keep my uncle in step with the music and myself",
        "My cousin was getting married and I was bridesmaid. As she stood in the back of the church ready to walk up the aisle she took off her veil and refused to get married. Since we looked so much alike, and since they didn't want the wedding spoiled, they made me wear the veil and walk up the aisle. I never got to the altar because I woke up.",
        "I dreamt that I was to be maid-of-honor in S.S. (female) wedding. It was very confusing for I couldn't find my shoes or dress, and S.S. was nowhere to be found either. She finally showed up, but was in quite a state for she wanted to marry a different man. ",
        
        # 坠落类梦境
        "Frau Meier war bei uns in der Wohnung. Sie wollte ihren Sohn bei uns auf dem Bett wickeln. Ich war ziemlich gereizt und unter Zeitdruck. Ich war unfreundlich zu ihr. Sie lachte mich aber immer nur an. Ich dachte, sie kann ja nichts dazu, daß ich keine Zeit habe. Ich räumte das Bett frei, damit sie ihren Sohn wickeln konnte. Ich verstand nicht, warum sie es bei uns machen mußte. Ich wollte, daß sie geht.",
        "Ich bin gemeinsam mit einigen Freunden auf einen entfernten Berg gefahren. Dort befanden wir uns plötzlich in einem anderen Land. Wir wollten einkaufen gehen, doch es war alles ziemlich teuer, und wir hatten Angst vor der Bevölkerung. Ständig hatte ich das Gefühl, daß gleich etwas passiert. Wir konnten zusehen, wie Straßenbahnhaltestellen gebaut wurden, und meine Freundin kaufte sich sogar eine eigene Straßenbahn, die nach ihr benannt wurde. Ich befand mich noch lange in dieser Stadt und beobachtete passiv das Geschehen um mich herum",
        "Ich saß mit einigen meiner Freunde an einem Tisch in irgendeinem Cafe oder Bistro. Wir hatten viel Spaß, aber ich hatte ständig ein komisches Gefühl. Irgendwann ging die Tür auf und ich konnte sehen, wie ein alter Bekannter hereinkam, mit dem ich mich vor längerer Zeit verstritten hatte und wir dann beschlossen hatten, getrennte Wege zu gehen. Einer von denen an meinem Tisch stand auf und begrüßte den Bekannten herzlichst. Mir war das alles sehr unangenehm und als er dann auch noch neben mir an den Tisch trat, wurde meine Unsicherheit sehr groß. ",
        "ch hatte mir den Arm gebrochen, wobei ich nicht sagen kann, wie, wo und wann es geschah. Er war also gebrochen (der rehte), und ich hielt ihn so vor meinen Körper. Ich war bei meinem gleich um die Ecke wohnenden alten Hausarzt, den ich überhaupt nicht leiden kann und zu dem ich eigentlich auch nicht mehr gehe. Doch er war nun mal der einzige. Er hat sich den Arm angesehen und daran rumgezogen und gezerrt. Es tat höllisch weh. Dann meinte er, ich solle nicht simulieren, und zog den Arm lang. Danach hatte er ihn in der Hand, erschrak furchtbar und war plötzlich weg. ",
        
        # 考试类梦境
        "梦见参加重要考试，却发现自己什么都不会",
        "昨晚梦见高考迟到，急得满头大汗",
        "梦见考试时笔写不出字，非常着急",
        "考试梦境总是让我感到压力和紧张",
        
        # 牙齿脱落类梦境
        "梦见牙齿一颗颗掉下来，满嘴是血",
        "昨晚梦见门牙松动，用手一碰就掉了",
        "牙齿脱落的梦境让我感到不安",
        "梦见满口牙齿都掉光了，非常害怕",
        
        # 浪漫类梦境
        "梦见和喜欢的人在海边散步，非常浪漫",
        "昨晚梦见和暗恋对象约会，心跳加速",
        "梦见被心仪的人表白，感到幸福",
        "浪漫的梦境让我醒来后心情愉悦",
        
        # 奇幻类梦境
        "梦见和会说话的动物一起冒险",
        "昨晚梦见进入魔法世界，学习魔法",
        "梦见穿越到古代，成为英雄",
        "奇幻梦境充满了想象力和创造力",
        
        # 日常类梦境
        "梦见在超市购物，找不到想买的东西",
        "昨晚梦见上班迟到，被老板批评",
        "梦见和家人一起吃饭，聊得很开心",
        "日常梦境反映了我白天的生活经历"
    ]
    
    return dreams


def main():
    # 设置环境变量避免多进程问题
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("=" * 80)
    print("梦境内容语义分析 - 基于Qwen3-Embedding")
    print("=" * 80)
    
    # 1. 初始化梦境分析器
    print("\n[1/6] 初始化梦境分析器...")
    config.print_model_info()
    print()
    analyzer = DreamAnalyzer()
    
    # 2. 加载梦境数据
    print("\n[2/6] 加载梦境数据...")
    dreams = create_sample_dreams()
    print(f"加载了 {len(dreams)} 个梦境描述")
    analyzer.load_dream_data(dreams)
    
    # 3. 分析梦境相似度
    print("\n[3/6] 分析梦境相似度...")
    similarity_matrix = analyzer.analyze_similarity()
    
    # 4. 聚类分析
    print("\n[4/6] 聚类梦境内容...")
    cluster_labels = analyzer.cluster_dreams_kmeans(n_clusters=6)
    
    # 5. 降维可视化
    print("\n[5/6] 降维可视化...")
    reduced_embeddings = analyzer.reduce_dimensions(method="tsne", n_components=2)
    
    # 6. 创建可视化
    print("\n[6/6] 创建梦境聚类可视化...")
    fig = analyzer.visualize_dream_clusters(
        title="梦境内容语义聚类分析 (t-SNE 2D)",
        save_path="dream_clustering_analysis.html"
    )
    
    # 7. 分析聚类主题
    print("\n[Extra] 分析聚类主题...")
    cluster_themes = analyzer.analyze_cluster_themes()
    
    # 总结
    print("\n" + "=" * 80)
    print("✓ 梦境内容语义分析完成!")
    print("=" * 80)
    print("\n生成的文件:")
    print("  - dream_clustering_analysis.html - 梦境聚类可视化")
    print("\n分析结果:")
    print("  - 发现了梦境内容的语义聚类模式")
    print("  - 揭示了不同主题梦境在语义空间中的分布")
    print("  - 展示了潜意识表达的规律性")
    print("\n打开HTML文件在浏览器中查看交互式可视化!")
    print("=" * 80)


if __name__ == "__main__":
    main()
