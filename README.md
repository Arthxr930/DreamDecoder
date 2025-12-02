# DreamDecoder: Automated Dream Content Analysis

## Project Overview

DreamDecoder is an AI-powered prototype for automated dream content analysis. The project aims to transform subjective dream experiences into structured, analyzable data through advanced natural language processing and machine learning techniques.

**Vision**: To create a complete "dream interpretation pipeline" that can automatically analyze dream content, identify patterns, and provide insights into psychological states.

## Project Status

- **Current Implementation**: **Text Clustering** - Grouping similar dreams based on semantic content
- **Future Expansion**: **Label Mapping** + **Psychological State Inference** - Adding semantic labels and psychological interpretations

## Current Features (Implemented)

### 1. Dream Data Processing
- Loads dream text datasets from CSV files or direct input
- Supports real dream datasets with preprocessing capabilities
- Handles multilingual dream descriptions

### 2. Semantic Embedding Generation
- Uses state-of-the-art Qwen3-Embedding models (4B/8B variants)
- Generates high-dimensional semantic embeddings (4096 dimensions)
- Supports both CPU and GPU acceleration
- Automatic model downloading with China-optimized mirrors

### 3. Advanced Clustering Algorithms
- **Optimized K-means (BisectingKMeans)**: Automatically determines optimal cluster count (3-4 clusters)
- **DBSCAN**: Density-based clustering with automatic parameter tuning
- **Hierarchical Clustering**: Ward linkage for balanced cluster distribution
- **Manual K-means**: Fixed cluster count for controlled experiments

### 4. Comprehensive Analysis Pipeline
- **Dream Similarity Analysis**: Identifies similar dream pairs and patterns
- **Cluster Theme Extraction**: Uses TF-IDF to extract keywords for each cluster
- **Visualization**: Interactive 2D t-SNE plots with Plotly
- **Algorithm Comparison**: Evaluates multiple clustering methods with metrics

### 5. Evaluation Metrics
- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Index**: Evaluates cluster quality
- **Cluster Distribution Analysis**: Ensures balanced grouping

## Not Yet Implemented (Future Plans)

### 1. Semantic Label Generation
- **Goal**: Automatically assign meaningful labels to each cluster
- **Example**: "Flying dreams", "Chase nightmares", "Exam anxiety dreams"
- **Approach**: LLM-based label generation or predefined taxonomy mapping

### 2. Psychological State Inference
- **Goal**: Map dream clusters to psychological states
- **Example**: "Freedom → Relaxation", "Pressure → Anxiety", "Loss → Stress"
- **Approach**: Psychological theory integration with empirical validation

### 3. Complete Dream Interpretation Pipeline
- **Goal**: End-to-end system: Dream → Clustering → Labeling → Psychological State
- **Vision**: Provide actionable insights for dream analysis and psychological research

## Project Positioning

- **Current Status**: **Prototype/Research Project**
- **Focus**: Demonstrating clustering effectiveness on dream content
- **Future Direction**: Building comprehensive dream analysis tools for psychology and self-reflection

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ GPU memory for 4B model, 16GB+ for 8B model

### Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd embeddings-dreams
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure model settings (optional):
Edit `config.py` to choose between 4B (recommended) and 8B models.

### For Users in China (Faster Downloads)
```bash
pip install modelscope
```

## Usage Examples

### 1. Basic Dream Analysis

```python
from dream_analysis_improved_final import ImprovedDreamAnalyzer

# Initialize analyzer
analyzer = ImprovedDreamAnalyzer()

# Load dream data
dreams = [
    "I dreamed about flying in the sky with birds",
    "I had a nightmare about falling from a tall building",
    "I dreamed about being in a beautiful garden with flowers",
    # Add more dreams...
]

analyzer.load_dream_data(dreams)

# Run complete analysis
output_dir = analyzer.run_complete_analysis(dreams)
```

### 2. Real Dream Dataset Analysis

```bash
python final_demo_real_dreams.py
```

This script:
- Loads real dreams from `dreams.csv`
- Performs similarity analysis
- Compares clustering algorithms
- Extracts cluster themes
- Generates interactive visualizations
- Saves results to `real_dream_analysis_results/`

### 3. Individual Analysis Steps

```python
# 1. Analyze dream similarity
similarity_matrix = analyzer.analyze_similarity()

# 2. Find optimal cluster count
optimal_k = analyzer.find_optimal_k(max_k=6)

# 3. Cluster dreams with optimized K-means
labels = analyzer.cluster_dreams_kmeans_optimized()

# 4. Analyze cluster themes
themes = analyzer.analyze_cluster_themes()

# 5. Create visualization
fig = analyzer.visualize_clustering_results()
fig.write_html("dream_clusters.html")
```

## Project Structure

```
embeddings-dreams/
├── dream_analysis_improved_final.py    # Main analysis class (ImprovedDreamAnalyzer)
├── final_demo_real_dreams.py           # Demo with real dream data
├── dream_analysis_improved.py          # Previous version
├── dream_analysis.py                   # Basic version
├── config.py                           # Model configuration
├── requirements.txt                    # Python dependencies
├── dreams.csv                          # Example dream dataset
├── README.md                           # This file
├── test_clustering_fix.py              # Clustering test scripts
├── test_improved_clustering.py         # Improved clustering tests
├── demo_improved_analysis.py           # Demo scripts
├── clustering_test_results/            # Test outputs
├── real_dream_analysis_results/        # Analysis outputs
└── transform/                          # Data transformation utilities
```

## Output Files

When running the complete analysis, the system generates:

1. **Cluster Themes Report** (`cluster_themes.txt`):
   - Keywords for each cluster
   - Sample dreams from each cluster
   - TF-IDF scores for important terms

2. **Interactive Visualization** (`dream_clusters.html`):
   - 2D t-SNE plot of dream clusters
   - Hover to view dream content
   - Color-coded by cluster assignment

3. **Algorithm Comparison** (`clustering_comparison.csv`):
   - Performance metrics for each algorithm
   - Cluster counts and noise points
   - Recommendations for best algorithm

## Technical Details

### Model Architecture
- **Base Model**: Qwen3-Embedding (4B or 8B parameters)
- **Embedding Dimension**: 4096
- **Languages Supported**: 100+ languages
- **MTEB Multilingual Score**: 69.45 (4B) / 70.58 (8B, #1 ranked)

### Clustering Optimization
- **Automatic K Selection**: Uses elbow method and silhouette scores
- **Cluster Balance**: Prioritizes 3-4 balanced clusters
- **Parameter Tuning**: Adaptive parameters based on dataset size
- **Multiple Algorithms**: Comparative analysis for best results

### Visualization
- **Dimensionality Reduction**: t-SNE with adaptive perplexity
- **Interactive Features**: Hover, zoom, pan in HTML output
- **Cluster Coloring**: Distinct colors for easy identification

## Future Development Roadmap

### Phase 1: Enhanced Clustering (Current)
- [x] Multiple clustering algorithm support
- [x] Automatic parameter optimization
- [x] Comprehensive evaluation metrics
- [x] Interactive visualization

### Phase 2: Semantic Labeling (Next)
- [ ] LLM-based cluster label generation
- [ ] Dream category taxonomy development
- [ ] Cross-cultural dream pattern analysis

### Phase 3: Psychological Inference
- [ ] Psychological state mapping framework
- [ ] Empirical validation with dream journals
- [ ] Longitudinal dream pattern tracking

### Phase 4: Application Development
- [ ] Web interface for dream journaling
- [ ] Mobile app for dream recording
- [ ] API for research integration

## Example Analysis

### Input Dreams:
```
1. "I was flying over mountains, feeling free"
2. "Being chased by a shadowy figure in a dark alley"
3. "Taking an exam I didn't study for"
4. "Floating peacefully in a warm ocean"
5. "Losing all my teeth one by one"
```

### Output Clusters:
- **Cluster 0 (Freedom/Flight)**: Dreams 1, 4
  - Keywords: flying, free, floating, peaceful
- **Cluster 1 (Anxiety/Stress)**: Dreams 2, 3, 5
  - Keywords: chased, exam, losing, teeth, shadowy

### Psychological Insight:
- **Freedom Cluster**: Associated with positive emotions, relaxation
- **Anxiety Cluster**: Associated with stress, fear, pressure

## Dependencies

Core requirements:
- `sentence-transformers>=3.0.0`
- `torch>=2.0.0`
- `transformers>=4.34.0`
- `scikit-learn>=1.3.0`
- `plotly>=5.18.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`

See `requirements.txt` for complete list.

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Use 4B model instead of 8B
   - Enable half-precision: `model.half()`
   - Reduce batch size in `model.encode()`

2. **Slow Downloads**:
   - Install ModelScope: `pip install modelscope`
   - Uses China-optimized mirrors automatically

3. **Empty Dream Dataset**:
   - Check `dreams.csv` file exists and has content
   - Use example dreams as fallback

4. **Import Errors**:
   - Ensure all dependencies are installed
   - Check Python version (3.8+ required)

## Research Applications

DreamDecoder can be used for:
- **Psychological Research**: Dream pattern analysis across populations
- **Sleep Studies**: Correlation between dream content and sleep quality
- **Cultural Studies**: Cross-cultural dream pattern comparison
- **Therapeutic Applications**: Tracking dream themes in therapy
- **Creative Writing**: Inspiration from dream pattern analysis

## Contributing

This is a research prototype. Contributions are welcome in:
- Improved clustering algorithms
- Semantic labeling approaches
- Psychological inference models
- User interface development
- Dataset collection and curation

## License

MIT License

The Qwen3-Embedding models are licensed under Apache 2.0.

## Citation

If you use DreamDecoder in your research, please cite:

```bibtex
@software{dreamdecoder2024,
  title = {DreamDecoder: Automated Dream Content Analysis},
  author = {DreamDecoder Team},
  year = {2024},
  url = {https://github.com/your-repo/dreamdecoder}
}
```

## Acknowledgments

- Qwen Team for the Qwen3-Embedding models
- Scikit-learn for clustering algorithms
- Plotly for interactive visualizations
- All contributors to open-source NLP tools

---

**Note**: This is a research prototype. The psychological interpretations are illustrative examples and should not be used for clinical diagnosis without professional validation.

**Project Goal**: To bridge the gap between subjective dream experiences and objective computational analysis, ultimately contributing to better understanding of the human mind through dream content.
