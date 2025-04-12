# =======================================================
# Neural Network Example in R for Marketing Campaigns
# =======================================================
# Created by: Joe Domaleski
# Marketing Data Science example

# -------------------------------------------------------
# STEP 1: Load required libraries
# -------------------------------------------------------
# Only run install.packages() once if not already installed
# install.packages("tidyverse")
# install.packages("readr")
# install.packages("neuralnet")
# install.packages("fastDummies")

library(tidyverse)    # Load core data manipulation and visualization packages
library(readr)        # Load functions for fast CSV file reading
library(neuralnet)    # Load the package for training and visualizing neural networks
library(fastDummies)  # Load helper functions to create one-hot encoded dummy variables

# -------------------------------------------------------
# STEP 2: Load and preview the dataset
# -------------------------------------------------------
# Load your synthetic 10,000-row marketing campaign dataset
data <- read_csv("marketing_campaigns_10000.csv")

# Quickly inspect structure and data types
head(data)
glimpse(data)

# -------------------------------------------------------
# STEP 3: Generate summary statistics
# -------------------------------------------------------
# Overall stats for numeric variables
summary_stats <- data %>%
  summarise(
    Count = n(),
    Budget_Mean = mean(Budget),
    Budget_SD = sd(Budget),
    AudienceSize_Mean = mean(AudienceSize),
    AudienceSize_SD = sd(AudienceSize),
    Duration_Mean = mean(Duration),
    Duration_SD = sd(Duration),
    Revenue_Mean = mean(EstimatedRevenue),
    Revenue_SD = sd(EstimatedRevenue),
    Success_Rate = mean(Success)
  )
print("=== Overall Summary Statistics ===")
print(summary_stats)

# Grouped summary by advertising platform
success_by_platform <- data %>%
  group_by(Platform) %>%
  summarise(
    Count = n(),
    Avg_Budget = mean(Budget),
    Avg_Duration = mean(Duration),
    Avg_Audience = mean(AudienceSize),
    Avg_Revenue = mean(EstimatedRevenue),
    Success_Rate = mean(Success)
  )
print("=== Summary by Platform ===")
print(success_by_platform)

# -------------------------------------------------------
# STEP 4: One-hot encode the categorical 'Platform' column
# -------------------------------------------------------
# Convert 'Platform' into binary columns for each platform (minus one for baseline)
data_encoded <- dummy_cols(data,
                           select_columns = "Platform",
                           remove_first_dummy = TRUE,
                           remove_selected_columns = TRUE)

# -------------------------------------------------------
# STEP 5: Normalize numeric input variables
# -------------------------------------------------------
# Normalize inputs to a 0–1 scale to help the neural network train efficiently
data_scaled <- data_encoded %>%
  mutate(across(c(Budget, AudienceSize, Duration, EstimatedRevenue),
                ~ (. - min(.)) / (max(.) - min(.))))

# -------------------------------------------------------
# STEP 6: Split the data into training and test sets
# -------------------------------------------------------
set.seed(123)
train_indices <- sample(1:nrow(data_scaled), 0.8 * nrow(data_scaled))
train_data <- data_scaled[train_indices, ]
test_data <- data_scaled[-train_indices, ]

# -------------------------------------------------------
# STEP 7: Define the neural network model formula
# -------------------------------------------------------
# Note: EstimatedRevenue is excluded to prevent leakage
formula <- as.formula("Success ~ Budget + AudienceSize + Duration +
                      Platform_Google + Platform_Instagram + Platform_LinkedIn + Platform_TikTok")

# -------------------------------------------------------
# STEP 8: Train the neural network
# -------------------------------------------------------
# Using 3 hidden neurons and a binary (logistic) output
nn_model <- neuralnet(formula,
                      data = train_data,
                      hidden = 3,
                      linear.output = FALSE,
                      stepmax = 1e6)

# -------------------------------------------------------
# STEP 9: Visualize the trained network
# -------------------------------------------------------
plot(nn_model)

# -------------------------------------------------------
# STEP 10: Predict using the test set
# -------------------------------------------------------
test_inputs <- select(test_data, -Success)
predictions <- compute(nn_model, test_inputs)$net.result
predicted_class <- ifelse(predictions > 0.5, 1, 0)
actual <- test_data$Success

# -------------------------------------------------------
# STEP 11: Evaluate model performance
# -------------------------------------------------------

# Accuracy
accuracy <- mean(predicted_class == actual)
cat("Accuracy:", round(accuracy, 3), "\n")

# Confusion matrix
conf_matrix <- table(Predicted = predicted_class, Actual = actual)
print("=== Confusion Matrix ===")
print(conf_matrix)

# Precision, Recall, and F1 Score
TP <- conf_matrix["1", "1"]
TN <- conf_matrix["0", "0"]
FP <- conf_matrix["1", "0"]
FN <- conf_matrix["0", "1"]

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * precision * recall / (precision + recall)

cat("Precision:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1 Score:", round(f1_score, 3), "\n")

# =======================================================
# STEP 12: Logistic Regression Comparison
# =======================================================

# Train logistic regression model (same formula, same inputs)
log_model <- glm(formula,
                 data = train_data,
                 family = binomial)

# View logistic regression model summary
summary(log_model)

# Predict probabilities on test set
log_probs <- predict(log_model, newdata = test_data, type = "response")

# Convert probabilities to binary class predictions
log_predicted_class <- ifelse(log_probs > 0.5, 1, 0)

# Actual values
log_actual <- test_data$Success

# Accuracy
log_accuracy <- mean(log_predicted_class == log_actual)
cat("Logistic Regression Accuracy:", round(log_accuracy, 3), "\n")

# Confusion Matrix
log_conf_matrix <- table(Predicted = log_predicted_class, Actual = log_actual)
print("=== Logistic Regression Confusion Matrix ===")
print(log_conf_matrix)

# Precision, Recall, F1
TP <- log_conf_matrix["1", "1"]
TN <- log_conf_matrix["0", "0"]
FP <- log_conf_matrix["1", "0"]
FN <- log_conf_matrix["0", "1"]

log_precision <- TP / (TP + FP)
log_recall <- TP / (TP + FN)
log_f1_score <- 2 * log_precision * log_recall / (log_precision + log_recall)

cat("Logistic Regression Precision:", round(log_precision, 3), "\n")
cat("Logistic Regression Recall:", round(log_recall, 3), "\n")
cat("Logistic Regression F1 Score:", round(log_f1_score, 3), "\n")

