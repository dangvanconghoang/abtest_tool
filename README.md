Experiment AB Testing Framework
This repository is a fork of an existing AB Testing Framework (expan), customized for internal use with additional features and improvements. The tool is designed to handle AB test executions, statistical analyses, and results generation with support for multiple metrics and hypothesis corrections.

Features
1. Primary and Secondary Metric Support: Allows designation of a primary metric for AB testing. Supports analysis of secondary metrics with hypothesis correction.
2. Hypothesis Correction: Supports multiple hypothesis correction methods such as Benjamini-Hochberg (BH) to reduce false discovery rates.
3. Sample Size Estimation: Dynamically estimates the required sample size based on Minimum Detectable Effect (MDE), alpha, beta, and test endpoint type (binary/normal).
4. Variant Combination Handling: Automatically generates all relevant variant combinations, ensuring flexibility in test designs.
5. Custom Traffic Weighting:Validates traffic balance and applies custom weights during the experiment evaluation.
6. Precision Calculation: Calculates precision metrics and provides progress toward expected precision.
7. Flexible Test Configuration: Configuration-driven approach allows easy setup of test parameters such as alpha, beta, MDE, and traffic weights.
8. Generates confidence intervals using non-parametric bootstrap sampling.
9. Proportion Tests: Calculates test statistics and confidence intervals for comparing two proportions.
10. Chi-Square Tests: Evaluates differences between observed and expected frequencies.
11. Sequential Sampling: Determines the number of samples needed for sequential tests.

Acknowledgments
This tool was forked and adapted from an existing AB testing framework to suit internal requirements. Special thanks to the original contributors for their foundational work.
