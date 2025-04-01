library(mlbench)
library(randomForest)
library(caret)
library(caretEnsemble)
library(titanic)
library(PRROC)
library(pROC)

titanic <- titanic_train[complete.cases(titanic_train),]
titanic[, c(2:3,5,12)] <- lapply(titanic[, c(2:3,5,12)], as.factor)
titanic$Pclass <- factor(paste('Class',titanic$Pclass, sep = '_'))
titanic$Survived <- factor(titanic$Survived, 
                           labels = c('Not_survived', 'Survived'))

set.seed(3225)
train <- sample(1:nrow(titanic), 0.8*nrow(titanic))
titanic_train <- titanic[train,]
titanic_test <- titanic[-train,]

ctrl <- trainControl(method = "cv",
                     number = 5,
                     index = createFolds(titanic_train$Survived, 5),
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = "final",
                     returnData = TRUE)

treemod <- train(Survived ~ Pclass + Sex + Age + Fare,
                 data=titanic_train,
                 trControl = ctrl,
                 method = 'ranger',
                 metric = 'ROC')

# Building models
mods <- c('ranger','xgbTree', 'glmnet')
model_list <- caretList(Survived ~ Pclass + Sex + Age + Fare,
                        data=titanic_train,
                        trControl = ctrl,
                        metric = "ROC",
                        methodList = mods)
dotplot(resamples(model_list), metric = 'ROC')
plot(model_list$xgbTree)

rf <- model_list$ranger
enet <- model_list$glmnet
plot(enet)

# Performance
update(treemod, param = list(mtry = 2, splitrule = 'extratrees', min.node.size = 1))

c_rf <- predict(treemod, newdata = titanic_test)
p_rf <- predict(treemod, newdata = titanic_test, type = "prob")

rf_roc <- roc(titanic_test$Survived, p_rf$Survived)
plot(rf_roc, col="blue")

caret::confusionMatrix(c_rf, titanic_test$Survived, 
                       positive = 'Survived')
# Accuracy: 0.7762
# Sensitivity: 0.4839

sex0 <- titanic_test[,'Sex'] == 'female'
caret::confusionMatrix(c_rf[sex0], titanic_test$Survived[sex0], positive = 'Survived')
female_roc <- roc(titanic_test$Survived[sex0], p_rf$Survived[sex0])
# Accuracy: 0.7193
# Sensitivity: 0.6364

sex1 <- titanic_test[,'Sex'] == 'male'
caret::confusionMatrix(c_rf[sex1], titanic_test$Survived[sex1], positive = 'Survived')
male_roc <- roc(titanic_test$Survived[sex1], p_rf$Survived[sex1])
# Accuracy: 0.814
# Sens: 0.1111


ggroc(list(female = female_roc, 
           male = male_roc)) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color="darkgrey", linetype="dashed")

## Choosing another model
update(treemod, param = list(mtry = 3, splitrule = 'extratrees', min.node.size = 1))

c_rf <- predict(treemod, newdata = titanic_test)
p_rf <- predict(treemod, newdata = titanic_test, type = "prob")

rf_roc2 <- roc(titanic_test$Survived, p_rf$Survived)
plot(rf_roc, col="blue")

ggroc(list(first = rf_roc, 
           second = rf_roc2)) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color="darkgrey", linetype="dashed")
sex0 <- titanic_test[-train,'Sex'] == 'female'
caret::confusionMatrix(c_rf[sex0], titanic_test$Survived[sex0], positive = 'Survived')
female_roc <- roc(titanic_test$Survived[sex0], p_rf$Survived[sex0])
# Acc: 0.7429
# Sens: 0.4375

sex1 <- titanic_test[-train,'Sex'] == 'male'
caret::confusionMatrix(c_rf[sex1], titanic_test$Survived[sex1], positive = 'Survived')
male_roc <- roc(titanic_test$Survived[sex1], p_rf$Survived[sex1])
# Acc: 0.787
# Sens: 0.5

ggroc(list(female = female_roc, 
           male = male_roc)) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color="darkgrey", linetype="dashed")
## Choosing another model

treemod
update(treemod, param = list(mtry = 2, splitrule = 'gini', min.node.size = 1))

c_rf <- predict(treemod, newdata = titanic_test)
p_rf <- predict(treemod, newdata = titanic_test, type = "prob")

rf_roc2 <- roc(titanic_test$Survived, p_rf$Survived)
plot(rf_roc, col="blue")

ggroc(list(first = rf_roc, 
           second = rf_roc2)) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color="darkgrey", linetype="dashed")

caret::confusionMatrix(c_rf, titanic_test$Survived, positive = 'Survived')
# Accuracy: 0.7762

sex0 <- titanic_test[-train,'Sex'] == 'female'
caret::confusionMatrix(c_rf[sex0], titanic_test$Survived[sex0], positive = 'Survived')
female_roc <- roc(titanic_test$Survived[sex0], p_rf$Survived[sex0])
# Acc: 0.7429
# Sens: 0.4375

sex1 <- titanic_test[-train,'Sex'] == 'male'
caret::confusionMatrix(c_rf[sex1], titanic_test$Survived[sex1], positive = 'Survived')
male_roc <- roc(titanic_test$Survived[sex1], p_rf$Survived[sex1])
# Acc: 0.787
# Sens: 0.5

ggroc(list(female = female_roc, 
           male = male_roc)) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color="darkgrey", linetype="dashed")

# Using Elastic Net
c_rf <- predict(enet, newdata = titanic_test)
p_rf <- predict(enet, newdata = titanic_test, type = "prob")

rf_roc3 <- roc(titanic_test$Survived, p_rf$Survived)
# plot(rf_roc, col="blue")

ggroc(list(first = rf_roc, 
           second = rf_roc2,
           enet = rf_roc3)) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color="darkgrey", linetype="dashed")

caret::confusionMatrix(c_rf, titanic_test$Survived, positive = 'Survived')
# Accuracy: 0.7972
# Sens: 0.7258

sex0 <- titanic_test[-train,'Sex'] == 'female'
caret::confusionMatrix(c_rf[sex0], titanic_test$Survived[sex0], positive = 'Survived')
female_roc <- roc(titanic_test$Survived[sex0], p_rf$Survived[sex0])
# Acc: 0.8
# Sens: 0.75

sex1 <- titanic_test[-train,'Sex'] == 'male'
caret::confusionMatrix(c_rf[sex1], titanic_test$Survived[sex1], positive = 'Survived')
male_roc <- roc(titanic_test$Survived[sex1], p_rf$Survived[sex1])
# Acc: 0.7963
# Sens: 0.7174

ggroc(list(female = female_roc, 
           male = male_roc)) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color="darkgrey", linetype="dashed")

# Elastic net

c_en <- predict(enet, newdata = titanic_test)
p_en <- predict(enet, newdata = titanic_test, type = "prob")

en_roc <- roc(titanic_test$Survived, p_en$Survived)
plot(en_roc, col="blue")

caret::confusionMatrix(c_en, titanic_test$Survived, 
                       positive = 'Survived')

sex0 <- titanic_test[,'Sex'] == 'female'
caret::confusionMatrix(c_en[sex0], titanic_test$Survived[sex0], positive = 'Survived')
female_roc <- roc(titanic_test$Survived[sex0], p_en$Survived[sex0])
# Accuracy: 0.7193
# Sensitivity: 0.6364

sex1 <- titanic_test[,'Sex'] == 'male'
caret::confusionMatrix(c_en[sex1], titanic_test$Survived[sex1], positive = 'Survived')
male_roc <- roc(titanic_test$Survived[sex1], p_en$Survived[sex1])

ggroc(list(female = female_roc, 
           male = male_roc)) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color="darkgrey", linetype="dashed")

