# PrivacyGuard: Comprehensive Privacy–Utility Analysis with Multi-Attack Evaluation

**Course:** CIS 545 — Data Security & Privacy · University of Michigan–Dearborn  
**Team:** Kumi A., Oluwamayowa O., Soundarya Lakshmi Rajendran, Hussein Ghosn, Alistair Clarke  
**Category:** Differential Privacy (Enhanced)

## Project Scope

This project extends standard Adult dataset DP analysis by implementing four membership inference attacks for thorough privacy testing and performing a multi-dimensional privacy–utility analysis that incorporates fairness metrics, demographic effects, and trade-off visualization. The goal is to rigorously assess how DP mechanisms withstand attacks and how privacy impacts utility and fairness across demographic groups.

## Core Objectives

- Implement four membership inference attacks  
- Evaluate DP-SGD and PATE  
- Analyze multi-objective privacy–utility and fairness trade-offs  
- Build an interactive dashboard  
- Produce an attack-resistance matrix across race, gender, and age  

## Technical Approach

### Attack Suite

- Confidence-based (Shokri et al., 2017)  
- Label-only (Choquette-Choo et al., 2021)  
- Metric-based loss analysis (Song & Shmatikov)  
- Adaptive thresholding across privacy budgets and class imbalance  

**Research Question:**  
Which attacks are most effectively mitigated by DP-SGD, and which remain resilient under strict ε-privacy?

### Dataset & Preprocessing

- Adult Income dataset (48k records)  
- Split by race (5), sex (2), and age (4)  
- Sensitive-attribute tagging for fairness analysis  
- Member/non-member partitioning with demographic preservation  
- Sensitivity estimates guide DP noise calibration  

### Defense Mechanisms

- DP-SGD with Gaussian and Laplace noise (ε ∈ {0.1, 0.5, 1, 2, 5})  
- PATE (Private Aggregation of Teacher Ensembles)  
- Adaptive clipping for dynamic gradient control  

## Evaluation & Analysis

### Multi-Dimensional Utility

- Per-class accuracy  
- Rare-class sensitivity  
- Calibration reliability  
- Confidence distribution  
- Feature importance under DP noise  
- Visual decision-boundary comparison  

### Fairness Assessment

- Demographic parity  
- Equalized odds  
- Equal opportunity  
- Per-group vulnerability to membership attacks  
- Analysis of DP impact on minority-group utility  

### Privacy–Utility Optimization

- Pareto-front analysis of accuracy vs. ε  
- “Knee-point” selection for optimal balance  
- Privacy budget scheduling with early-stopping evaluation  

## Expected Outcomes

### Attack-Resistance Matrix

| Attack     | Baseline | ε=5 | ε=1 | ε=0.5 | PATE |
|------------|----------|-----|-----|-------|------|
| Confidence | 0.72     | 0.65| 0.54| 0.51  | 0.53 |
| Label-Only | 0.68     | 0.62| 0.55| 0.52  | 0.54 |
| Metric     | 0.70     | 0.64| 0.56| 0.53  | 0.52 |
| Adaptive   | 0.69     | 0.63| 0.55| 0.52  | 0.53 |

**Avg AUC:** Baseline 0.70 → DP/PATE ≈ 0.53 → Effective privacy strengthening

### Fairness Findings

| Defense | Overall | Male | Female | White | Black | DP Gap |
|---------|---------|------|--------|-------|--------|--------|
| Baseline | 85.2%  | 86.1%| 83.4%  | 86.3% | 81.9%  | 0.044  |
| ε=1     | 77.8%  | 79.1%| 75.8%  | 79.4% | 73.2%  | 0.062  |
| ε=0.5   | 73.2%  | 74.8%| 70.9%  | 75.1% | 68.5%  | 0.066  |

**Insight:** Differential privacy narrows attack success but widens demographic performance gaps — critical for ethical deployment.

## Deliverables

- 24 privacy-attack experiments with statistical validation  
- Fairness analysis showing demographic impacts  
- Interactive Streamlit/Plotly dashboard for ε-tuning and “what-if” exploration  
- Publication-ready attack-resistance matrix  
- Practical ε-selection guidelines for private ML deployment  

## Team Member Roles

- **Hussein Ghosn**: Data pipeline and Fairness  
- **Soundarya L. Rajendran**: Experiments and Analysis  
- **Alistair Clarke**: Attack Implementation  
- **Oluwamayowa O.**: Privacy Tracking  
- **Kumi Acheampong**: Model Implementation  

## Project Timeline

| Week         | Focus         | Key Tasks                                                                 |
|--------------|---------------|---------------------------------------------------------------------------|
| Nov 4–10     | Implementation| DP-SGD + PATE, four attacks, fairness metrics, orchestration setup       |
| Nov 11–17    | Experiments   | 24 runs, fairness metric collection, matrix generation                   |
| Nov 18–Dec 1 | Analysis      | Pareto & fairness deep dive, dashboard development                       |
| Dec 2–8      | Finalization  | Dashboard polish, report, presentation rehearsal                         |
| Dec 10       | Demo Day      | Live dashboard and final report submission                               |

## Success Criteria

- **MVP**: ≥ 2 attacks + DP-SGD (3 ε) + fairness metrics + 6-page report + presentation  
- **Target**: All attacks + PATE + interactive dashboard + statistical testing  
- **Stretch**: Adaptive clipping, second dataset, extended fairness study  

## Impact & Value

Establishes an empirical benchmark of DP mechanisms across diverse attacks, integrates multiple defenses with fairness and visual analytics, and provides practical ε-selection guidance through an interactive dashboard for private ML deployment.

## References

- Abadi et al., CCS 2016  
- Shokri et al., S&P 2017  
- Choquette-Choo et al., NeurIPS 2021  
- Song & Shmatikov
