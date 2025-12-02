#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终演示：处理真实梦境数据
"""

import sys
import os
import pandas as pd
# 添加当前目录到Python路径
sys.path.append('.')

try:
    from dream_analysis_improved_final import ImprovedDreamAnalyzer
    print("✓ 改进版梦境分析器导入成功!")
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("请确保 dream_analysis_improved_final.py 文件存在")
    sys.exit(1)

def load_real_dreams_from_csv(csv_path="dreams.csv", max_dreams=100):
    """从CSV文件加载真实梦境数据"""
    print(f"\n从 {csv_path} 加载真实梦境数据...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ CSV文件加载成功，共 {len(df)} 行数据")
        
        # 显示数据结构
        print(f"数据列: {list(df.columns)}")
        
        # 提取梦境内容
        dreams = []
        for idx, row in df.iterrows():
            # 优先使用content列，如果没有则使用title列
            if pd.notna(row.get('content')) and str(row['content']).strip():
                dreams.append(str(row['content']).strip())
            elif pd.notna(row.get('title')) and str(row['title']).strip():
                dreams.append(str(row['title']).strip())
            
            if max_dreams and len(dreams) >= max_dreams:
                break
        
        print(f"✓ 成功提取 {len(dreams)} 个梦境描述")
        
        # 显示一些统计信息
        if 'word_count' in df.columns:
            avg_words = df['word_count'].mean()
            print(f"  平均梦境长度: {avg_words:.1f} 词")
        
        # 显示示例梦境
        print("\n示例梦境:")
        for i, dream in enumerate(dreams[:3]):
            print(f"  {i+1}. {dream[:150]}...")
        
        return dreams
        
    except Exception as e:
        print(f"✗ 加载CSV文件失败: {e}")
        print("使用示例梦境数据代替...")
        return [
            "I dreamed I was flying over the mountains, feeling free and happy",
            "Last night I had a nightmare about being chased by a monster",
            "I dreamt that I was taking an important exam but couldn't answer any questions",
            "Dreamed about a romantic dinner with someone special",
            "Had a dream where my teeth were falling out one by one"
        ]


def main():
    print("=" * 80)
    print("最终演示：改进版梦境分析器处理真实梦境数据")
    print("=" * 80)
    
    try:
        # 1. 初始化分析器
        print("\n[1/5] 初始化改进版梦境分析器...")
        analyzer = ImprovedDreamAnalyzer()
        
        # 2. 加载真实梦境数据
        print("\n[2/5] 加载真实梦境数据...")
        dreams = load_real_dreams_from_csv("dreams.csv", max_dreams=50)
        
        if not dreams:
            print("✗ 没有可用的梦境数据，程序终止")
            return
        
        print(f"使用 {len(dreams)} 个真实梦境进行分析")
        analyzer.load_dream_data(dreams)
        
        # 3. 分析相似度
        print("\n[3/5] 分析梦境相似度...")
        similarity_matrix = analyzer.analyze_similarity()
        
        # 4. 比较聚类算法
        print("\n[4/5] 比较聚类算法...")
        comparison_results = analyzer.compare_clustering_algorithms()
        
        # 5. 分析聚类主题
        print("\n[5/5] 分析聚类主题...")
        if analyzer.cluster_labels is not None:
            themes = analyzer.analyze_cluster_themes()
            
            # 保存详细分析结果
            output_dir = "real_dream_analysis_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存主题分析
            themes_path = os.path.join(output_dir, "real_dream_themes.txt")
            with open(themes_path, "w", encoding="utf-8") as f:
                f.write("真实梦境聚类主题分析报告\n")
                f.write("="*60 + "\n\n")
                f.write(f"分析梦境数量: {len(dreams)}\n")
                f.write(f"聚类算法: K-means优化版\n")
                f.write(f"聚类数量: {len(set(analyzer.cluster_labels))}\n\n")
                
                for cluster_id, theme_info in themes.items():
                    f.write(f"聚类 {cluster_id}:\n")
                    f.write(f"  关键词: {', '.join(theme_info['keywords'])}\n")
                    f.write(f"  梦境数量: {len([l for l in analyzer.cluster_labels if l == cluster_id])}\n")
                    f.write(f"  示例梦境:\n")
                    for i, dream in enumerate(theme_info['sample_dreams']):
                        f.write(f"    {i+1}. {dream[:200]}...\n")
                    f.write("\n")
            
            print(f"✓ 主题分析结果已保存到 {themes_path}")
            
            # 创建可视化
            print("\n[额外] 创建聚类可视化...")
            fig = analyzer.visualize_clustering_results(
                title="真实梦境内容语义聚类分析 (改进版)"
            )
            
            # 保存HTML文件
            viz_path = os.path.join(output_dir, "real_dream_clusters.html")
            fig.write_html(viz_path)
            print(f"✓ 可视化已保存到 {viz_path}")
            
            # 保存聚类比较结果
            if comparison_results is not None:
                results_path = os.path.join(output_dir, "clustering_comparison.csv")
                comparison_results.to_csv(results_path, index=False)
                print(f"✓ 聚类比较结果已保存到 {results_path}")
        
        print("\n" + "=" * 80)
        print("✓ 真实梦境分析演示完成!")
        print("=" * 80)
        print(f"\n所有结果已保存到目录: real_dream_analysis_results")
        print("包含文件:")
        print("  - real_dream_themes.txt - 聚类主题分析")
        print("  - real_dream_clusters.html - 交互式可视化")
        print("  - clustering_comparison.csv - 聚类算法比较")
        print("\n打开HTML文件查看可视化结果:")
        print(f"  start {os.path.join('real_dream_analysis_results', 'real_dream_clusters.html')}")
        
    except Exception as e:
        print(f"\n✗ 运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
