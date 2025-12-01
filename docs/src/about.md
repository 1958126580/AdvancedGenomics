# About AdvancedGenomics.jl

- **FarmCPU & BLINK**: Implements multi-locus models that iteratively control for false positives, significantly boosting statistical power.
- **Post-GWAS Analysis**: Includes fine-mapping tools to calculate Posterior Inclusion Probabilities (PIP) and pathway enrichment analysis to interpret biological mechanisms.
- **Complex Traits**: Supports logistic regression for binary traits and burden tests for rare variant analysis.

### 2. Comprehensive Genomic Selection

The package offers a versatile toolkit for predicting breeding values and complex phenotypes.

- **Kernel Methods**: Flexible RKHS regression with custom kernels to model additive, dominance, and epistatic effects.
- **Bayesian Suite**: A complete collection of Bayesian linear regression models (BayesA/B/C/R/Lasso) for variable selection and shrinkage.
- **Machine Learning Integration**: Seamlessly integrates Random Forests and Gradient Boosting Machines (GBM) to capture complex non-linear interactions that traditional linear models miss.

### 3. Deep Learning & AI

Pioneering the use of AI in genomics.

- **Genomic Transformers**: Adapts the self-attention mechanism from NLP to DNA sequences, allowing the model to learn context-dependent marker effects.
- **Explainable AI (XAI)**: Generates saliency maps to visualize which parts of the genome are driving the model's predictions, making "black box" models interpretable.

### 4. Multi-Omics & Systems Biology

- **Data Integration**: Native support for integrating Transcriptomics, Metabolomics, and Epigenomics data to build holistic prediction models.
- **Causal Inference**: Implements Mendelian Randomization (MR) techniques (IVW, Egger regression) to infer causal relationships between molecular traits and phenotypes.

### 5. Breeding Optimization

Designed for practical application in breeding programs.

- **Optimal Contribution Selection (OCS)**: Solves the complex optimization problem of maximizing genetic gain while constraining inbreeding, ensuring sustainable long-term genetic progress.
- **Mating Designs**: Tools to simulate and evaluate different mating strategies.

## Performance

AdvancedGenomics.jl is engineered for speed.

- **GPU Acceleration**: Critical matrix operations, such as GRM construction and GWAS scans, can be offloaded to NVIDIA GPUs, achieving speedups of up to 50x compared to CPU-only implementations.
- **Parallel Computing**: All major algorithms are multi-threaded, fully utilizing modern multi-core processors.
- **Memory Efficiency**: Smart memory management and support for memory-mapped files allow analysis of datasets larger than available RAM.
