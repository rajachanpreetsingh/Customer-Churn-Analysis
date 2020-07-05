####################################################################################################

#############Data Science Project: Customer Churn Analysis using R Programming######################
#############Authors: Saad Hasan, Departmentof Systems and Computer Engineering#####################
#############Author: Chanpreet Singh, School of IT##################################################
#############Special Thanks to our Supervisor for helping us in almost everypart of the project#####
#############Supervisor: Prof. Olga Baysal, Department of Computer Science##########################
#############Course Code: DATA-5000#################################################################

####################################################################################################

#Importing Libraries
#Custoemr Churn Prediction whether a customer is loose left the telecom company or not
library(tensorflow)
library(ggplot2)
library(MASS)
library(e1071)
library(caret)
library(caTools)
library("pROC")
library(ggcorrplot)
library(tidyverse)
library(dplyr)
library(magrittr)
library(keras)
library(fastDummies)
library(neuralnet)
library(cowplot)
library(h2o)
library(ggpubr)
library(corrplot)
library(mlbench)
library(ElemStatLearn)
library(randomForest)
library(gbm)
library(ClusterR)
library(cluster)
library(ROCR)
library(keras)
library(factoextra)

#reading file and understanding data
set.seed(123)
dataframe <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", header=TRUE)
head(dataframe)
#Checking the dimension of the data to understant more
dim(dataframe)
#checking the summary of the data
summary(dataframe$OnlineBackup)
summary(dataframe$gender)
summary(dataframe$SeniorCitizen)
summary(dataframe$Partner)
summary(dataframe$tenure)
summary(dataframe$Dependents)
summary(dataframe$PhoneService)
summary(dataframe$MultipleLines)
summary(dataframe$InternetService)
summary(dataframe$OnlineSecurity)
summary(dataframe$DeviceProtection)
summary(dataframe$TechSupport)
summary(dataframe$StreamingTV)
summary(dataframe$StreamingMovies)
summary(dataframe$Contract)
summary(dataframe$PaperlessBilling)
summary(dataframe$PaymentMethod)
summary(dataframe$MonthlyCharges)
summary(dataframe$TotalCharges)

#So only 11 values are missing in TotalCharges column
# We need to simply remove those rows
dataframe1 <- na.omit(dataframe)
summary(dataframe1$TotalCharges)

#data pre-processing
#so majority of the customers are leaving the company based om mny resons.
#we need to find out the reason responsible for the customers to leave 
prop.table(table(dataframe1$Churn))*100
#Visualization using histogram
#Data Visualization for churn 
options(repr.plot.width = 6, repr.plot.height = 4)
dataframe1 %>% 
group_by(Churn) %>% 
summarise(Count = n())%>% 
mutate(percent = prop.table(Count)*100)%>%
ggplot(aes(reorder(Churn, -percent), percent), fill = Churn) +
        geom_col(fill = c("#1D2588", "#0C0304")) +
        geom_text(aes(label = sprintf("%.2f%%", percent)), hjust = 0.01,vjust = -0.5, size =3) + 
        theme_bw() +  
        xlab("Churn") + 
        ylab("Percent") +
        ggtitle("Churn Percent of Customers")
#Data Transformation
head(dataframe1)
#Checking the corelation between continous varaibles: Monthly Charges, Total charges and tenure
ggscatter(dataframe1, x = "MonthlyCharges", y = "TotalCharges", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Monthly Charges in Dollars", ylab = "Total Charges in Dollars")
dataframe1 <- transform(dataframe1, tenure = as.numeric(tenure))
sapply(dataframe1, class)
cor(x = dataframe1$TotalCharges, y = dataframe1$MonthlyCharges, 
                    method = c("pearson", "kendall", "spearman"))
res <- cor.test(dataframe1$MonthlyCharges, dataframe1$TotalCharges,
                                                method = "pearson")
res$p.value
res$estimate
# R is 0.65 so both features are extremley corelated
ggscatter(dataframe1, x = "MonthlyCharges", y = "tenure", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Monthly Charges in Dollars", ylab = "Total Charges in Dollars")

#Data Visualization
options(repr.plot.width = 22, repr.plot.height = 10)
plot_grid(ggplot(dataframe1, aes(x=InternetService,fill=Churn)) + 
            geom_bar(position = 'fill') +
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)), 
            ggplot(dataframe1, aes(x=OnlineSecurity,fill=Churn)) + 
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
            ggplot(dataframe1, aes(x=StreamingMovies,fill=Churn)) + 
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
            ggplot(dataframe1, aes(x=InternetService,fill=Churn)) + 
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
            ggplot(dataframe1, aes(x=Partner,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
             ggplot(dataframe1, aes(x=SeniorCitizen,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),                                       
             ggplot(dataframe1, aes(x=PhoneService,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
            ggplot(dataframe1, aes(x=gender,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)), 
            ggplot(dataframe1, aes(x=PaperlessBilling,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
            ggplot(dataframe1, aes(x=MultipleLines,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)), 
             ggplot(dataframe1, aes(x=DeviceProtection,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
             ggplot(dataframe1, aes(x=Contract,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
            ggplot(dataframe1, aes(x=TechSupport,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
             ggplot(dataframe1, aes(x=PaymentMethod,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
            ggplot(dataframe1, aes(x=StreamingTV,fill=Churn)) +
                                  geom_bar(position = 'fill') +
                                 # theme_bw()+
          scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          align = "h")
#dataframe1 %>% 
#mutate(Gen=factor(gender)) %>% 
#ggplot(aes( x=gender, fill=Churn)) +
#geom_bar(stat="count") +
#labs(title= "Churn by Gender", y="No of Customers") +
#theme_minimal()

#Data-PreProcessing
# Categorial variabes should be in numbers so that algortihm will understand 
# Now We give the value Male is 0 and Female is 1
#ataframe1$gender <- as.numeric( 
 #as.character( 
dataframe1$gender <- as.numeric(
    as.character(
     factor( 
       dataframe1$gender, 
       levels = c("Female", "Male"), 
       labels = c("1", "2")))) 
dataframe1$Partner <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$Partner, 
      levels = c("Yes", "No"), 
      labels = c("1", "0")))) 
dataframe1$PhoneService <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$PhoneService, 
      levels = c("Yes", "No"), 
      labels = c("1", "0")))) 
dataframe1$Dependents <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$Dependents, 
      levels = c("Yes", "No"), 
      labels = c("1", "0")))) 
dataframe1$MultipleLines <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$MultipleLines, 
      levels = c("Yes", "No", "No phone service"), 
      labels = c("1", "0", "0")))) 
dataframe1$OnlineBackup <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$OnlineBackup, 
      levels = c("Yes", "No", "No internet service"), 
      labels = c("1", "0", "0")))) 
dataframe1$InternetService <- as.numeric( 
 as.character( 
   factor( 
     dataframe1$InternetService, 
     levels = c("DSL", "Fiber optic", "No"), 
     labels = c("1", "2", "0")))) 
dataframe1$OnlineSecurity <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$OnlineSecurity, 
      levels = c("No", "No internet service", "Yes"), 
      labels = c("0", "0", "1")))) 
dataframe1$TechSupport <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$TechSupport, 
      levels = c("No", "No internet service", "Yes"), 
      labels = c("0", "0", "1")))) 
dataframe1$StreamingTV <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$StreamingTV, 
      levels = c("No", "No internet service", "Yes"), 
      labels = c("0", "0", "1")))) 
dataframe1$StreamingMovies <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$StreamingMovies, 
      levels = c("No", "No internet service", "Yes"), 
      labels = c("0", "0", "1")))) 
dataframe1$DeviceProtection <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$DeviceProtection, 
      levels = c("No", "No internet service", "Yes"), 
      labels = c("0", "0", "1")))) 
dataframe1$PaperlessBilling <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$PaperlessBilling, 
      levels = c("No", "Yes"), 
      labels = c("0", "1"))))  
dataframe1$Churn <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$Churn, 
      levels = c("No", "Yes"), 
      labels = c("0", "1")))) 
dataframe1$Contract <- as.numeric( 
  as.character( 
    factor( 
      dataframe1$Contract, 
      levels = c("Month-to-month", "One year", "Two year"), 
      labels = c("1", "2", "3")))) 
dataframe1$PaymentMethod <- as.numeric( 
 as.character( 
   factor( 
     dataframe1$PaymentMethod, 
    levels = c("Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"), 
     labels = c("1", "2", "3", "4"))))
dataframe2 <- dataframe1[,2:21]
#replacing 0 by 1 and it is greater than
with(dataframe2, dataframe2$tenure <-
       ifelse(tenure == 0 & TotalCharges > 18 & TotalCharges < 100, "1",
       ifelse(tenure == 0 & TotalCharges > 100 & TotalCharges < 500, "5",
       ifelse(tenure == 0 & TotalCharges > 500 & TotalCharges < 1000, "10",
       ifelse(tenure == 0 & TotalCharges > 1000 & TotalCharges < 1500, "20",
       ifelse(tenure == 0 & TotalCharges > 1500 & TotalCharges < 2000, "25",
       ifelse(tenure == 0 & TotalCharges > 2000 & TotalCharges < 2500, "30",
       ifelse(tenure == 0 & TotalCharges > 2500 & TotalCharges < 3000, "40",
       ifelse(tenure == 0 & TotalCharges > 3000 & TotalCharges < 4000, "50",
       ifelse(tenure == 0 & TotalCharges > 4000 & TotalCharges < 5000, "60",
       ifelse(tenure == 0 & TotalCharges > 5000 & TotalCharges < 6000, "65",
       ifelse(tenure == 0 & TotalCharges > 6000, "70"
             ,tenure))))))))))))
dataframe2$new <- dataframe2$tenure * dataframe2$MonthlyCharges 
head(dataframe2)
summary(dataframe2$tenure)
# Drop 'DV' which 'Polarity' Column
matrix1c <- dataframe2 %>% select (-new)

# Get Correlation Matrix 'M'
M <- cor(matrix1c)
corrplot(M, type = 'upper', title = 'Correlation Matrix for IVs')
#creating dummy variables for gender, internet services and paymentmethod
#dataframe2 <- dummy_cols(dataframe1, select_columns = c("gender",
                                            #            "PaymentMethod", "InternetService", "Contract"))
#Custoemr id, gender , payment method and internetservice because we have already made dummy variables
#So we remove these features
#dataframe3 <- dataframe2 %>%
 # select("SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
  #       "OnlineSecurity", "gender_Female", "gender_Male",
   #      "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    #     "PaymentMethod_Bank transfer (automatic)","PaymentMethod_Credit card (automatic)",
     #    "InternetService_DSL", "InternetService_Fiber optic","InternetService_No",
      #   "MonthlyCharges", "TotalCharges", "Contract_Month-to-month",
       #  "Contract_One year", "Contract_Two year","Churn")
#at<- function(dataframe1) {
#at<-dummy_cols(dataframe1, select_columns = "gender")
#at<-dummy_cols(dataframe1, select_columns = "PaymentMethod")
   # return(dat)
#head(dataframe3)
#anomaly <- dataframe3$MonthlyCharges > dataframe3$TotalCharges
#dataframe3[anomaly, c("MonthlyCharges", "TotalCharges")] <- NA
#summary(dataframe3)
#dim(dataframe3)
#We need to normalize data for the assesment
#dataframe4 <- dataframe3
#dataframe4 <- scale(dataframe3[-22])

#Selecting the features for better classification
# calculate correlation matrix
#correlationMatrix <- cor(dataframe4)
# summarize the correlation matrix
#print(correlationMatrix)
#Dividing the dataset into set and training set
ind <- sample(2,nrow(dataframe2),replace = TRUE, prob = c(0.8, 0.2))
traindata <- dataframe2[ind == 1,] 
testdata  <- dataframe2[ind == 2,]
dim(traindata)
dim(testdata)
head(traindata[20])
#Logistic Regression to find accuracy
classifier <- glm(formula = Churn ~ tenure +
                 #  MonthlyCharges +
                  + TotalCharges 
                #  + PaperlessBilling 
                  + Contract
              #   + SeniorCitizen
                  + InternetService,
             #     + MultipleLines
               #   + OnlineSecurity
            #    + TechSupport, 
                #  + StreamingTV 
                 # + StreamingMovies ,
                  family = "binomial",
                  data = traindata)
summary(classifier)
probability <- predict(classifier, type= "response",
newdata=testdata)
head(probability)
prediction1 <- ifelse(probability > 0.5, 1, 0)
#prediction <- cut(probability, breaks=c(0,0.5,1),
 #                labels=c("No", "Yes"), include.lowest=TRUE)
test.ct <- table(Actual=testdata$Churn, Predicted=prediction1)
test.ct
y <- performance(prediction(probability,testdata$Churn),'tpr','fpr')
plot(y, col = "green")
#g <- roc(Churn ~ probability, data = testdata)
#plot(g))

#Gradient Boosted Machine  
classifier2 <- gbm(formula =  Churn ~  
                         tenure +
                         MonthlyCharges +
                         TotalCharges +
                       # PaperlessBilling +
                         Contract +
               #   + SeniorCitizen
                         InternetService ,
               #   + MultipleLines
                #  + OnlineSecurity 
               #   + TechSupport 
                        # StreamingTV + 
                       #  StreamingMovies ,
                         distribution = "bernoulli" ,
                         n.trees = 500 ,
                         train.fraction = 0.5 ,
                   # distribution = "gausian" ,
                         n.minobsinnode = 10 ,
                  # nTrain = round(Churn*0.8),
                         verbose = TRUE ,   
                         shrinkage = 0.01 ,
                         interaction.depth = 4 ,
                         data = traindata)
summary(classifier2)
gbm.perf(classifier2)
#plotting the Partial Dependence Plot
#Plot of Response variable with lstat variable
#plot(classifier2, i="TotalCharges") 
#Inverse relation with lstat variable
#plot(classifier2,i="MonthlyCharges") 
cor(traindata$MonthlyCharges, traindata$Churn)#poistive correlation coeff-r
cor(traindata$TotalCharges, traindata$Churn)#negative correlation coeff-r
#Looking at the effect of each variables
for(i in 1:length(classifier2$var.names)) {
         plot(classifier2, i.var = i,
         ntrees = gbm.perf(classifier2, plot.it = FALSE),
         type = "response")    
}
#Test set predictions
testpred <- predict(object = classifier2, newdata = testdata,
                    n.trees = gbm.perf(classifier2, plot.it = FALSE),
                    type = "response")
#training set predictions
trainpred <- predict(object = classifier2, newdata = traindata,
                    n.trees = gbm.perf(classifier2, plot.it = FALSE),
                    type = "response")
head(testpred)
teprediction <- ifelse(testpred > 0.5, 1, 0)
#prediction <- cut(probability, breaks=c(0,0.5,1),
 #                labels=c("No", "Yes"), include.lowest=TRUE)
dim(testdata[20])
test.ct2 <- table(testdata$Churn, teprediction)
test.ct2
#Model Evaluation Using ROC Curve
x <- performance(prediction(testpred,testdata$Churn),'tpr','fpr')
plot(x, col ='red')
plot(y, col = 'green', add = TRUE)


#ggplot(df, aes(testdata$Churn)) +                    # basic graphical object
 # geom_line(aes(y=y1), colour="red") +  # first layer
 # geom_line(aes(y=y2), colour="green") 
#plot(x,y1,type="l",col="red")
#lines(x,y2,col="green")
#plot(performance(prediction(testpred,testdata$Churn),'tpr','fpr'), col = 'red')
#plot(performance(prediction(probability,testdata$Churn),'tpr','fpr'), col = 'red', type = '1')
#Plotting ROC together

#g2 <- roc(Churn ~ teprediction, data = testdata)
#plot(g2)

#Classification using Random Forest
traindata2 <- traindata %>%
  select( "tenure", "MonthlyCharges", "TotalCharges",  "InternetService", "Contract", 
         "Churn")
testdata2 <- testdata %>%
  select("tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService",
         "Churn")
#Feature Scaling 
#Feature Scaling 
#traindata2[-6] <- scale(traindata2[-6])
#testdata2[-6] <- scale(testdata2[-6])
#head(traindata2)
traindata2$Churn <- as.character(traindata2$Churn)
traindata2$Churn <- as.factor(traindata2$Churn)
testdata2$Churn <- as.character(testdata2$Churn)
testdata2$Churn <- as.factor(testdata2$Churn)
#Fit Random Forest Model
rf = randomForest(x = traindata2[-6],
                  y = traindata2$Churn,
                   ntree = 1000)
probability6 <- predict(rf,
   newdata=testdata2)
print(rf)
head(probability6)
teprediction1 <- ifelse(probability6 > 0.6, 1, 0)
#prediction <- cut(probability, breaks=c(0,0.5,1),
 #                labels=c("No", "Yes"), include.lowest=TRUE)
test.ct6 <- table(Actual=testdata2$Churn, Predicted=teprediction)
test.ct6

#z <- performance(prediction(probability6,testdata2$Churn),'tpr','fpr')
plot(x, col ='blue')
plot(y, col = 'cyan', add = TRUE)
plot(z, col = 'black', add = TRUE)
#z <- performance(prediction(probability6,testdata2$Churn),'tpr','fpr')
#g3 <- roc(Churn ~ probability6, data = testdata2)#plot(g3)

#Unsupervised learning Using Kmeans Algorithm
# This thing makes company to analyze which customers are important to loose
# Unsupervised Learning using 3 continous variables but we removed total charges as well 
# Becasue Total charges is nothing but the multiplication of tenure and monthly charges
dataframe3 <- dataframe2 %>%
  select( "tenure", "MonthlyCharges")#, "TotalCharges")
dataframe3[, 1:2] <- data.frame(lapply(dataframe3[, 1:2], scl))
#head(dataframe3)
#Using elbow method to find an optimum value of K
set.seed(6)
vec <- vector()
for(i in 1:15) 
    vec[i] <- sum(kmeans(dataframe3, i)$withinss)   # Fitting our dataset with i clusters
    plot(1:15, vec, type = "b",
         main = paste("Clustering of Telco Customers"), xlab = "Number of Clusters", ylab = "vec")
# Apllying K-Means Algortihm
set.seed(25)
kme <- kmeans(dataframe3, 3,
                 iter.max = 1500, nstart = 25 )
# It’s possible to compute the mean of each of the variables in the clusters:
aggregate(dataframe3, by=list(cluster=kme$cluster), mean)
fviz_cluster(kme, data = dataframe3,
             palette = c("#00AFBB","#2E9FDF", "#E7B800", "#FC4E07"),
             ggtheme = theme_minimal(),
             main = "Partitioning Clustering Plot"
             )
#Visualising Clusters
    clusplot(dataframe3,
            kmeans$cluster,
            lines = 0,
            shade = TRUE,
            color = TRUE,
            labels = 4,
            plotchar = TRUE,
            span = TRUE,
            main = paste("Clusters of Customers"),
            xlab = "Monthly Charges",
            ylab = "Tenure")
dataframe4 <- dataframe2 %>%
  select( "tenure", "MonthlyCharges")
dataframe4$one <- 1
dataframe4$two <- 2
dataframe4$three <- 3
head(dataframe4)
#dataframe4 <- transform(dataframe4,
 #               new=
  #              ifelse(tenure > 0 & tenure < 12 & MonthlyCharges > 18 & MonthlyCharges < 50, "1",
   #             ifelse(tenure > 12 & tenure < 24 & MonthlyCharges > 50, "2",
    #            ifelse(tenure > 24 , "3", new
     #           ))))

ClusteringManually <- function(x){
  a <- x[1]
  b <- x[2]
 # c <- x[3]
 # d <- x[4]
 # e <- x[5]
  value <- if(a > 0 & a < 12 & b > 18 & b < 50) 1
      else if(a > 0 & a < 12 & b > 50) 2
      else if(a > 12 & a < 24 & b > 18 & b < 50) 2
      else if(a > 12 & a < 24 & b > 50) 3
      else if(a > 24 & b > 18 & b < 50) 3
      else if(a > 24) 3
  return(value)
}
#ClusteringManually1<- function(x){
#  a <- x[1]
#  b <- x[2]
#  c <- x[3]
#  d <- x[4]
#  e <- x[5]
#  value <- if(a > 0 & a < 12 ) {
#               if(b > 18 & b < 50) 1 
#               else 2 
#          }
#           if(a > 12 & a < 24) {
#               if(b > 18 & b < 50) 2
#               else 3
#      }
#           if(a > 24 & b > 18) 3
#  return(value)
#}          
dataframe4$new <- apply(dataframe4, 1, ClusteringManually)
#dataframe4$new1 <- apply(dataframe4, 1, ClusteringManually1)    
head(dataframe4)  
dataframe5 <- dataframe4[,-c(3,4,5,7)]
head(dataframe5)    
cluscol <- dataframe4[,6]
as.numeric(cluscol)
dim(cluscol)
head(cluscol)
class(cluscol)
dataframe5[dataframe5$new=='NULL'] <- 3
mat <- sapply(dataframe5,unlist)
mat[order(mat[,3]),]
table(Predicted = kme, Actual = cluscol)
#head(cluscol)    

#So we remove these features
# Creating Dummy variables
dataframe3 <- dummy_cols(dataframe2,select_columns = c("gender", "Contract", "InternetService"
                                                     , "PaymentMethod"))
dataframe3 <- dataframe3[,-c(1,15,8,17)]
#Dividing the dataset into set and training set
ind <- sample(2,nrow(dataframe3),replace = TRUE, prob = c(0.8, 0.2))
traindata <- dataframe3[ind == 1,] 
testdata  <- dataframe3[ind == 2,]
dim(traindata)
dim(testdata)
#Feature Scaling 
scl <- function(x){ (x - min(x))/(max(x) - min(x)) }
traindata[, 1:28] <- data.frame(lapply(traindata[, 1:28], scl))
testdata[, 1:28] <- data.frame(lapply(testdata[, 1:28], scl))

traindata1 <- traindata %>%
  select( "tenure", "MonthlyCharges", "TotalCharges","Contract_1",
          "Contract_2", "Contract_3","InternetService_0",
          "InternetService_1","InternetService_2","Churn")

testdata1 <- testdata %>%
  select( "tenure", "MonthlyCharges", "TotalCharges","Contract_1",
          "Contract_2", "Contract_3","InternetService_0",
          "InternetService_1","InternetService_2","Churn")
train_x <- traindata1[,-c(10)]
train_y <- traindata1[,c(10)]
test_x <- testdata1[,-c(10)]
test_y <- testdata1[,c(10)]
train_x <- as.matrix(train_x)
train_y <- as.matrix(train_y)
test_x <- as.matrix(test_x)
test_y <- as.matrix(test_y)
typeof(train_x)
#traindata$TotalCharges <- as.numeric(traindata$TotalCharges)
#traindata$MonthlyCharges <- as.numeric(traindata$MonthlyCharges)
#traindata$tenure <- as.numeric(traindata$tenure)
#traindata$Contract_1 <- as.numeric(traindata$Contract_1)
#traindata$Contract_2 <- as.numeric(traindata$Contract_2)
#traindata$Contract_3 <- as.numeric(traindata$Contract_3)
#traindata$InternetService_0 <- as.numeric(traindata$InternetService_0)
#traindata$InternetService_1 <- as.numeric(traindata$InternetService_1)
#traindata$InternetService_2 <- as.numeric(traindata$InternetService_2)

# Neural network using keras
# create model
model <- keras_model_sequential()
# define and compile the model
model %>% 
  layer_dense(units = 100, activation = 'relu', input_shape = c(9)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 264, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 18, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>%  
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = c('accuracy')
  )
summary(model)
# training model
annmodel <- model %>% fit(train_x,
              train_y,
              epochs = 500,
              batch_size = 100,
              validation_split = 0.10)

# evaluate
annmodel
score = model %>% evaluate(test_x, test_y)
proba <- model %>%
  predict_proba(test_x)
preda <- model %>%
  predict_classes(test_x)
table(Predicted = preda, Actual = test_y)

# Plotting ROC curve for nuerla network
y <- performance(prediction(prob,test_y),'tpr','fpr')
plot(y)

