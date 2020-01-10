###HW6
###Brian Keafer

###Library Loads
library(tidyverse) 
library(lubridate)
library(ggplot2)
library(car)
library(MASS)
library(forcats)
library(elasticnet)
library(caret)
library(glmnet)
library(glmnetUtils)
library(pls)
library(earth)
library(rgl)

###Sales Prediction Problem

###Read Data Sets
dfTrain <- read_csv("Train.csv")
dfTest <- read_csv("Test.csv")

####Create Unique IDs for Training and Test Data
trainIds <- unique(dfTrain$custId)  
testIds <- unique(dfTest$custId)

###combine the test and train data 
combined_DF <- dfTrain %>% dplyr::select(-revenue) %>% bind_rows(dfTest)
head(combined_DF)

###View Structure of Combined Dataset
str(combined_DF) 

#Feature Engineer Dates
tf_maxdate = max(ymd(combined_DF$date))
tf_mindate = min(ymd(combined_DF$date))

###Transformations
trans_combine_DF <- combined_DF %>% 
  mutate(date=ymd(date)) %>%                                              
  mutate(country = fct_lump(fct_explicit_na(country), prop = 0.1)) %>%         
  mutate(subContinent = fct_lump(fct_explicit_na(subContinent), prop = 0.05)) %>%
  mutate(region = fct_lump(fct_explicit_na(region), prop = 0.01)) %>%
  mutate(networkDomain = fct_lump(fct_explicit_na(networkDomain), prop  = 0.01)) %>%
  mutate(source= fct_lump(fct_explicit_na(source), prop = 0.05)) %>%
  mutate(medium = fct_lump(fct_explicit_na(medium), prop = 0.05)) %>%
  mutate(browser = fct_lump(fct_explicit_na(browser), prop = 0.1)) %>%
  mutate(operatingSystem = fct_lump(fct_explicit_na(operatingSystem), prop = 0.05)) %>%
  mutate(deviceCategory = fct_lump(fct_explicit_na(deviceCategory), prop = 0.1)) %>%
  mutate(campaign = fct_lump(fct_explicit_na(campaign), prop = 0.1)) %>%
  group_by(custId) %>%                                   
  summarize(                                            
    unique_date_num = length(unique(date)),
    maxVisitNum = max(visitNumber, na.rm = TRUE),
    browser = first(browser),
    operatingSystem = first(operatingSystem),
    deviceCategory = first(deviceCategory),
    subContinent = first(subContinent),
    country = first(country),
    region = first(region),
    networkDomain = first(networkDomain),
    source = first(source),
    medium = first(medium),
    isMobile = mean(ifelse(isMobile == TRUE, 1 , 0)),
    bounce_sessions = sum(ifelse(is.na(bounces) == TRUE, 0, 1)),
    pageviews_sum = sum(pageviews, na.rm = TRUE),
    pageviews_mean = mean(ifelse(is.na(pageviews), 0, pageviews)),
    pageviews_min = min(ifelse(is.na(pageviews), 0, pageviews)),
    pageviews_max = max(ifelse(is.na(pageviews), 0, pageviews)),
    pageviews_median = median(ifelse(is.na(pageviews), 0, pageviews)),
    session_cnt = NROW(visitStartTime))
  
str(trans_combine_DF)


###get the transformed train data
train_trans_DF <-trans_combine_DF %>% filter(custId %in% trainIds)

###Explore Missingness
train_trans_DF %>% dplyr::select(-c(custId)) %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()

###Calculate target variable
outcome<-dfTrain %>% 
  group_by(custId) %>%
  summarize(
    transactionRevenue = sum(revenue)
  ) %>% 
  mutate(logSumRevenue = log(transactionRevenue+1)) %>% dplyr::select(-transactionRevenue)

###Join target variable to the aggregated train data
train_trans_DF <- train_trans_DF %>% inner_join(outcome, by = "custId")
head(train_trans_DF)

###Retrieve test data
test_trans_DF <-trans_combine_DF %>% filter(custId %in% testIds)
head(test_trans_DF)

########################################Linear Model 2.0######################################
###Initial Model
lm_fit <- train(logSumRevenue~.,
             data=train_trans_DF,
             method="lm")

###Summary of Model
lm_fit
lm_fit$finalModel

###Train Control
fit_control <- trainControl(method="cv",number=5)

lm_fit <- train(logSumRevenue~.,
             data=train_trans_DF,
             method="lm",
             trControl=fit_control)

lm_fit
summary(lm_fit)
##########################################Lasso Model#######################################

###Check for Columns with Zero Variance
remove_cols <- nearZeroVar(train_trans_DF, names = TRUE)

###Initial Model

lasso_fit <- train(logSumRevenue~.,
                          data=train_trans_DF,
                          method="lasso",
                          trControl=fit_control)

###Summary of Model
lasso_fit
plot(lasso_fit)


###Optimizing Hyperparameters (fraction)
lassoGrid <- expand.grid(fraction=seq(0.7,0.99,length=100))

lasso_fit <- train(logSumRevenue~.,
                   data=train_trans_DF,
                   method="lasso",
                   trControl=fit_control,
                   tuneGrid=lassoGrid)

lasso_fit

plot(lasso_fit)

###Optimize Part 2
lassoGrid <- expand.grid(fraction=seq(0.2,0.9,length=100))

lasso_fit <- train(logSumRevenue~.,
                   data=train_trans_DF,
                   method="lasso",
                   trControl=fit_control,
                   tuneGrid=lassoGrid)
lasso_fit

plot(lasso_fit)

###########################################Ridge Model#############################################
###Initial Model
ridge_fit <- train(logSumRevenue~.,
                   data=train_trans_DF,
                   method="ridge",
                   trControl=fit_control)

ridge_fit

plot(ridge_fit)

###Optimizing Hyperparameters (lambda)
ridgeGrid <- expand.grid(lambda=seq(0.001,0.003,length=10))

ridge_fit <- train(logSumRevenue~.,
                   data=train_trans_DF,
                   method="ridge",
                   trControl=fit_control,
                   tuneGrid=ridgeGrid)

ridge_fit

plot(ridge_fit)


##########################################Elasticnet Model##############################################

###Initial Model
enet_fit <- train(logSumRevenue~.,
                   data=train_trans_DF,
                   method="enet",
                   trControl=fit_control)

enet_fit

plot(enet_fit)

###Optimizing Hyperparameters (lambda and fraction)
enetGrid <- expand.grid(lambda=seq(0.001,0.1,length=10), fraction=seq(.2, .9, length = 50))

enet_fit <- train(logSumRevenue~.,
                   data=train_trans_DF,
                   method="enet",
                   trControl=fit_control,
                   tuneGrid=enetGrid)

enet_fit
###Visualizations
plot(enet_fit)
plot(enet_fit, metric = "Rsquared")
plot(enet_fit, plotType = "level")

########################################Partial Least Squares Model#################################
###Initial Model
pls_fit <- train(logSumRevenue~.,
                  data=train_trans_DF,
                  method="pls",
                  trControl=fit_control)

pls_fit

plot(pls_fit)

###Optimizing Hyperparameters (ncomp)
plsGrid <- expand.grid(ncomp=seq(1,50))

pls_fit <- train(logSumRevenue~.,
                  data=train_trans_DF,
                  method="pls",
                  trControl=fit_control,
                  tuneGrid=plsGrid)

pls_fit
plot(pls_fit)

##############################################MARS Model##############################################
###Initial Model
MARS_fit <- train(logSumRevenue~.,
                 data=train_trans_DF,
                 method="earth",
                 trControl=fit_control)

MARS_fit
plot(MARS_fit)


###Optimizing Hyperparameters (ncomp)
MARSGrid <- expand.grid(degree=1,nprune=seq(14,16))

MARS_fit <- train(logSumRevenue~.,
                 data=train_trans_DF,
                 method="earth",
                 trControl=fit_control,
                 tuneGrid=MARSGrid)
summary(MARS_fit)
MARS_fit

##############################################Predict and Export####################################

#apply model to the test data
results01<-tibble(custId=test_trans_DF$custId, predRevenue=predict(MARS_fit, test_trans_DF))
results01
#add a rule: if the predictions are negative, just make them 0
final01 <-results01 %>% mutate(predRevenue = replace(predRevenue, which(predRevenue<0), 0))

#write out results
write.csv(final01, 'MARS_model.csv', row.names = F)


























