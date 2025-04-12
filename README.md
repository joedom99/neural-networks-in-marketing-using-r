# Neural Networks in Marketing Using R

This repository contains a hands-on example of using a neural network to predict marketing campaign success in R. The model is trained on a synthetic dataset of 10,000 marketing campaigns and compared with a logistic regression model to highlight differences in performance and interpretability.

![R](https://img.shields.io/badge/R-4.3.1-blue?logo=r)
![License](https://img.shields.io/badge/license-MIT-green)
![Made%20With](https://img.shields.io/badge/Made%20with-RStudio-blue?logo=rstudio)
![Project Type](https://img.shields.io/badge/type-Tutorial%20Project-lightgrey)

> 🔗 This project supports the blog post:  
> **[Using Neural Networks in Marketing: A Hands-On R Example](#)**  
> *(Add link when published)*

---

## 📁 What's Included

- `marketing_campaigns_10000.csv` — A synthetic dataset with 10,000 campaign records
- `neural_network_marketing.R` — Full R script to:
  - Load and preprocess the dataset
  - Train a neural network model using `neuralnet`
  - Evaluate predictions and model performance
  - Compare with logistic regression

---

## 🧠 Model Inputs

Each campaign includes:

- **Budget** (normalized)
- **Audience Size** (normalized)
- **Duration** (normalized)
- **Platform** (one-hot encoded: Google, Instagram, LinkedIn, TikTok)

The outcome variable is **Success** (1 = campaign met goal, 0 = did not).

---

## 📊 Tools Used

- `tidyverse` for data manipulation
- `fastDummies` for one-hot encoding
- `neuralnet` for training the neural network
- Base `glm()` for logistic regression

---

## 🚀 How to Run

1. Clone the repository
2. Set your default working directory so that the .CSV is in the same folder as the R script
3. Open the R script in R Studio
4. Make sure the required packages are installed. You can uncomment those lines in the script to run them.
5. Run the script to train the models and view the output.

---

## 📚 Learn More

This repo supports the educational content on marketing data science at https://blog.marketingdatascience.ai

---

## 🧑‍💻 Author

Joe Domaleski (LinkedIn) - https://www.linkedin.com/in/joedom

---
