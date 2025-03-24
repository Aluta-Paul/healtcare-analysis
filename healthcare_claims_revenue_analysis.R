
### ============================================================
### Healthcare Claims & Revenue Analysis
### Predictive Analytics for Claims & Revenue Forecasting
### ============================================================

# Load Required Libraries
library(tidyverse)
library(lubridate)
library(caret)
library(forecast)
library(prophet)
library(xgboost)
library(rpart)
library(rpart.plot)
library(plotly)
library(scales)

### ============================================================
### 1. Load & Inspect Data
### ============================================================

# Load datasets
claims_data <- read.csv("healthcare_claims.csv", header = TRUE)
visits_revenue_data <- read.csv("hospital_visits_revenue.csv", header = TRUE)

# Convert Date columns to proper format
visits_revenue_data <- visits_revenue_data %>%
  mutate(Date.of.Admit = as.Date(Date.of.Admit, format="%m/%d/%Y"),
         Date.of.Discharge = as.Date(Date.of.Discharge, format="%m/%d/%Y"))

# View structure
glimpse(claims_data)
glimpse(visits_revenue_data)

### ============================================================
### 2. Data Cleaning & Preprocessing
### ============================================================

# Handle missing values
visits_revenue_data[visits_revenue_data == ""] <- NA
colSums(is.na(visits_revenue_data))

# Compute Length of Stay (LOS)
visits_revenue_data$Length.of.Stay <- as.numeric(visits_revenue_data$Date.of.Discharge - visits_revenue_data$Date.of.Admit)

# Address negative LOS due to data entry errors
visits_revenue_data <- visits_revenue_data %>%
  mutate(Length.of.Stay = ifelse(Length.of.Stay < 0, abs(Length.of.Stay), Length.of.Stay))

# Remove unnecessary columns
visits_revenue_data <- visits_revenue_data %>% select(-c(count.x, count.y, count))

### ============================================================
### 3. Exploratory Data Analysis (EDA)
### ============================================================

# Revenue Distribution Plot
ggplot(visits_revenue_data, aes(x = Revenue)) +
  geom_histogram(binwidth = 5000, fill = "steelblue", color = "black") +
  scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(title = "Revenue Distribution", x = "Revenue", y = "Count") +
  theme_minimal()

# Claim Status Breakdown
ggplot(claims_data, aes(x = Claim.Status, fill = Claim.Status)) +
  geom_bar() +
  labs(title = "Claim Status Breakdown", x = "Claim Status", y = "Count") +
  theme_minimal()

### ============================================================
### 4. Predicting Claim Denials
### ============================================================

# Select Features
claims_data <- claims_data %>% select(Billed.Amount, AR.Status, Insurance.Type, Follow.up.Required, Reason.Code, Claim.Status)

# Convert categorical variables to factors
claims_data <- claims_data %>% mutate(
  Claim.Status = as.factor(Claim.Status),
  AR.Status = as.factor(AR.Status),
  Insurance.Type = as.factor(Insurance.Type),
  Follow.up.Required = as.factor(Follow.up.Required),
  Reason.Code = as.factor(Reason.Code)
)

# Split Data
set.seed(123)
split <- createDataPartition(claims_data$Claim.Status, p = 0.8, list = FALSE)
train_data <- claims_data[split, ]
test_data <- claims_data[-split, ]

# Train Logistic Regression Model
log_model <- glm(Claim.Status ~ ., data = train_data, family = binomial)

# Train Decision Tree Model
tree_model <- rpart(Claim.Status ~ ., data = train_data, method = "class")

# Train XGBoost Model
train_matrix <- model.matrix(Claim.Status ~ . -1, data = train_data)
test_matrix <- model.matrix(Claim.Status ~ . -1, data = test_data)
dtrain <- xgb.DMatrix(data = train_matrix, label = as.numeric(train_data$Claim.Status) - 1)
dtest <- xgb.DMatrix(data = test_matrix, label = as.numeric(test_data$Claim.Status) - 1)
xgb_model <- xgb.train(params = list(objective = "binary:logistic", eval_metric = "logloss"), 
                       data = dtrain, nrounds = 100)

### ============================================================
### 5. Model Evaluation
### ============================================================

# Predictions & Confusion Matrices
log_preds <- predict(log_model, test_data, type = "response")
log_class <- ifelse(log_preds > 0.5, 1, 0)
tree_preds <- predict(tree_model, test_data, type = "class")
xgb_preds <- predict(xgb_model, dtest)
xgb_class <- ifelse(xgb_preds > 0.5, 1, 0)

# Compare Model Performance
log_results <- confusionMatrix(as.factor(log_class), test_data$Claim.Status)
tree_results <- confusionMatrix(as.factor(tree_preds), test_data$Claim.Status)
xgb_results <- confusionMatrix(as.factor(xgb_class), test_data$Claim.Status)

extract_metrics <- function(conf_mat) {
  data.frame(Accuracy = conf_mat$overall["Accuracy"],
             Sensitivity = conf_mat$byClass["Sensitivity"],
             Specificity = conf_mat$byClass["Specificity"],
             Kappa = conf_mat$overall["Kappa"],
             Balanced_Accuracy = conf_mat$byClass["Balanced Accuracy"])
}

model_results <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "XGBoost"),
  rbind(extract_metrics(log_results), extract_metrics(tree_results), extract_metrics(xgb_results))
)

print(model_results)

### ============================================================
### 6. Revenue Forecasting (ARIMA vs Prophet)
### ============================================================

# Aggregate Revenue by Date
daily_revenue <- visits_revenue_data %>% group_by(Date.of.Admit) %>% summarise(Revenue = sum(Revenue))

# Fit ARIMA Model
ts_revenue <- ts(daily_revenue$Revenue, frequency = 365)
arima_model <- auto.arima(ts_revenue)
arima_forecast <- forecast(arima_model, h = 30)

# Fit Prophet Model
prophet_data <- daily_revenue %>% rename(ds = Date.of.Admit, y = Revenue)
prophet_model <- prophet(prophet_data)
future_dates <- make_future_dataframe(prophet_model, periods = 30)
prophet_forecast <- predict(prophet_model, future_dates)

# Compare Forecasts
comparison_df <- data.frame(Date = tail(future_dates$ds, 30),
                            ARIMA = as.numeric(arima_forecast$mean),
                            Prophet = tail(prophet_forecast$yhat, 30))

ggplot(comparison_df, aes(x = Date)) +
  geom_line(aes(y = ARIMA, color = "ARIMA"), linewidth = 1) +
  geom_line(aes(y = Prophet, color = "Prophet"), linewidth = 1, linetype = "dashed") +
  labs(title = "Comparison: ARIMA vs Prophet Forecast", x = "Date", y = "Predicted Revenue", color = "Model") +
  theme_minimal()
