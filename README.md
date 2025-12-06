# SURVMETH 640 | Machine Learning for Survey and Data Scientists

Welcome to my SURVMETH 640 coursework repository! This collection documents my comprehensive journey through machine learning methods specifically designed for survey and data science applications, featuring lecture materials, hands-on coding assignments, and a capstone project on Premier League soccer analytics.

## Course Overview

SURVMETH 640 is an advanced graduate course that bridges traditional survey methodology with modern machine learning techniques. The course explores how data scientists can leverage machine learning algorithms for prediction, classification, and pattern recognition in survey and observational data contexts. Through a structured 12-week curriculum, the course covers foundational concepts through advanced ensemble methods, model interpretability, and ethical considerations in ML applications.

## Repository Structure

This repository contains a complete collection of course materials organized by topic, including weekly lecture slides, practical coding demonstrations, homework assignments, and a comprehensive final project.

### Lecture Materials

The course is organized into 12 modules, each covering essential machine learning topics:

**Module 01: Introduction to Machine Learning**
* 01-Introduction.pdf: Course overview and ML foundations
* Introduction to ML.pdf: Comprehensive introduction
* 01_MLTrees.pdf: Decision tree fundamentals
* introduction.Rmd: R markdown tutorial

**Module 02: ML Basics and Data Handling**
* 02-01-MLBasics.pdf: Core ML concepts
* 02-02-BiasVarianceTradeoff.pdf: Understanding model complexity
* 02-03-DataSplitting.pdf: Train/test split strategies
* 02-RAndRStudio.pdf: R programming environment setup
* 02_MLTrainTestSplit.pdf: Practical data splitting
* Week2-MLbasics.pptx: Lecture slides
* ml-basics.Rmd: Hands-on R examples

**Module 03: K-Nearest Neighbors and Model Evaluation**
* 03-01-kNN.pdf: K-NN algorithm theory
* 03-02-PerformanceMeasures.pdf: Metrics for model assessment
* 03-Lecture-kNN.pdf: Detailed lecture notes
* 03-Lecture.pdf: Additional materials
* 03_MLEvaluation.pdf: Comprehensive evaluation strategies
* K-NN and Performance Metrics.pdf: Combined resource
* knn.Rmd: Practical implementation

**Module 04: Regularized Regression**
* 04-01-IntroRegularizedRegression.pdf: Regularization concepts
* 04-02-StepwiseSelection.pdf: Feature selection methods
* 04-03-LassoRidgeRegression.pdf: L1 and L2 regularization
* 04_MLOverview.pdf: ML landscape overview
* Regularized Regression and Tuning.pdf: Complete guide
* regularized-regression-1.Rmd: Lasso and Ridge coding
* regularized-regression-2.Rmd: Advanced regularization

**Module 05: Elastic Net and Hyperparameter Tuning**
* 05-01-Introduction.pdf: Advanced regression intro
* 05-02-ElasticNetGroupLasso.pdf: Combined regularization
* 05-03-Tuning.pdf: Hyperparameter optimization strategies

**Module 06: Decision Trees**
* 06-01-IntroductionTrees.pdf: Tree-based model foundations
* 06-02-TreePruning.pdf: Preventing overfitting
* 06-03-CTREE.pdf: Conditional inference trees
* 06-04-ModelBasedRecursivePartitioning.pdf: Advanced partitioning
* Decision Trees.pdf: Comprehensive tree guide
* trees-1.Rmd: Basic tree implementation
* trees-2.Rmd: Advanced tree techniques

**Module 07: Ensemble Methods**
* 07-01-IntroductionEnsemble.pdf: Ensemble learning concepts
* 07-02-Bagging.pdf: Bootstrap aggregating
* 07-03-RandomForests.pdf: Random forest algorithm
* 07-04-ExtraTrees.pdf: Extremely randomized trees
* Ensemble Methods.pdf: Complete ensemble guide
* ensemble-class.R: Class exercises
* ensemble.qmd: Quarto implementation

**Module 08: Boosting Algorithms**
* 08-01-AdaBoost.pdf: Adaptive boosting
* 08-02-GradientBoosting.pdf: Gradient boosting basics
* 08-03-GradientBoosting2.pdf: Advanced gradient boosting
* Boosting.pdf: Comprehensive boosting resource

**Module 09: Advanced Boosting**
* 09-01-ExtremeGradientBoosting.pdf: XGBoost algorithm
* 09-02-ModelBasedBoosting.pdf: Statistical boosting methods

**Module 10: Model Interpretability**
* 10-01-VariableImportance.pdf: Feature importance metrics
* 10-02-PDPICEALE.pdf: Partial dependence plots, ICE, and ALE
* 10-03-SurrogateModels.pdf: Interpretable approximations
* Interpretable ML.pdf: Complete interpretability guide
* interpretable-ml.Rmd: Practical interpretation examples

**Module 11: Bias and Fairness**
* 11-01-MLBias.pdf: Algorithmic bias identification
* 11-02-MLFairness.pdf: Fairness metrics and mitigation
* Bias and Fairness.pdf: Ethics in ML
* bias-fairness.R: Fairness analysis code

**Module 12: Advanced Topics**
* 12-01-Stacking.pdf: Model stacking techniques
* 12-02-OverUndersampling.pdf: Handling imbalanced data
* Neural Networks.pdf: Introduction to neural networks
* ml-toolbox.Rmd: Advanced ML toolkit
* mltoolbox.R: Utility functions

### Course Assignments

Four comprehensive homework assignments demonstrating progressive skill development:

* **assignment1.Rmd**: Introduction to ML concepts and basic implementations
* **assignment2.Rmd**: Regularized regression and model selection
* **assignment3.qmd**: Tree-based methods and ensemble learning
* **assignment4-1.qmd + assignment4-1.pdf**: Advanced ensemble methods and model evaluation

### Final Project: Premier League Soccer Analytics

A comprehensive machine learning project analyzing English Premier League data:

**Project Files**
* Project Proposal.qmd: Initial project proposal
* Project-Proposal.pdf: Rendered proposal document
* Project_Soccer.qmd: Complete project analysis
* Project_Soccer.pdf: Final project report

**Project Data**
* Epl_tables_raw_1992-2021.json: Premier League standings data
* Income_expense_raw_1992-2021.json: Club financial data
* Income_expenditure_table_posItions_1992-2021.csv: Combined dataset
* 92-21-income_expenditure_table_positions.csv: Alternative format
* premier_league_raw_data.RData: Consolidated R data file

The project investigates relationships between club financial expenditures and league performance, applying various machine learning techniques including regularized regression, tree-based models, and ensemble methods to predict team success.

### Supporting Materials

**Code Examples and Demonstrations**
* bootstrapping-example.R: Resampling methods
* caretList_example.R: Caret package workflows
* neural-networks.R: Neural network implementation
* dcgan.ipynb: Deep learning generative model

**Documentation**
* 2025 web-scraping-apis-syllabus.pdf: Web scraping course syllabus
* 367.pdf: Additional reference material
* DESCRIPTION: Package description file

**Tests**
* tests/ folder: Unit tests and validation code

## Skills and Techniques Covered

Through this coursework, I've developed expertise in:

**Machine Learning Fundamentals**
* Supervised learning algorithms
* Bias-variance tradeoff
* Cross-validation and model selection
* Training/test/validation splitting
* Performance metrics and model evaluation

**Regression Methods**
* Linear regression
* Ridge regression (L2 regularization)
* Lasso regression (L1 regularization)
* Elastic Net
* Group Lasso
* Stepwise selection methods

**Classification Algorithms**
* K-Nearest Neighbors
* Logistic regression
* Decision trees (CART, CTREE)
* Support vector machines

**Tree-Based Methods**
* Classification and regression trees
* Tree pruning strategies
* Conditional inference trees
* Model-based recursive partitioning

**Ensemble Learning**
* Bagging (Bootstrap Aggregating)
* Random Forests
* Extra Trees
* AdaBoost
* Gradient Boosting Machines
* XGBoost (Extreme Gradient Boosting)
* Model-based boosting
* Stacking

**Model Evaluation and Selection**
* Confusion matrices
* Accuracy, precision, recall, F1-score
* ROC curves and AUC
* Mean squared error, RMSE
* R-squared and adjusted R-squared
* Cross-validation (k-fold, leave-one-out)

**Model Interpretability**
* Variable importance measures
* Partial dependence plots
* Individual conditional expectation (ICE) plots
* Accumulated local effects (ALE) plots
* Surrogate models
* LIME and SHAP values

**Special Topics**
* Handling imbalanced data
* Over and under-sampling techniques
* Algorithmic bias detection
* Fairness metrics and mitigation
* Neural networks fundamentals
* Web scraping and APIs for data collection

## Programming Tools and Packages

All implementations use R programming with key packages:
* **caret**: Comprehensive ML workflow
* **glmnet**: Regularized regression
* **randomForest**: Random forest implementation
* **xgboost**: Gradient boosting
* **rpart**: Decision trees
* **e1071**: Support vector machines
* **iml**: Interpretable ML
* **DALEX**: Model explanation
* **mlr3**: Modern ML framework
* **tidyverse**: Data manipulation and visualization

## Course Application

SURVMETH 640 uniquely positions machine learning within the survey methodology context, addressing practical challenges such as:
* Predicting survey response propensities
* Identifying data quality issues
* Adaptive survey designs
* Auxiliary data integration
* Variance estimation with ML models
* Survey weight calibration

## Academic Note

This repository represents my independent coursework completed for educational purposes at the University of Michigan. All analyses demonstrate the application of machine learning techniques to real-world data science problems, with particular emphasis on methodology appropriate for survey and observational data contexts.
