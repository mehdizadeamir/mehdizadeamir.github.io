# This Code is Written to Examine the Use of ML Models to Predict whether an SCE will occur in
# the next trip.


# [1] Initilization:
# -----------------

if (!require("pacman")) install.packages("pacman") # checks if the packman package is installed locally
pacman::p_load(DT, foreach, magrittr, tidyverse,fst, # important analytic pkgs
               knitr, kableExtra, # to print nice looking tables
               dataPreparation, DataExplorer, zoo, # important for exploratory data analysis
               recipes, caret, caretEnsemble, doParallel, MLmetrics, # bases for our ML approach
               rpart, glmnet, Matrix, naivebayes, nnet, randomForest, # needed packages for ML
               kernlab, # needed for svm
               xgboost, plyr, # needed packages for xgboost
               caTools, pROC, # for classification metrics
               ROSE, DMwR # for sampling methods
)
startTime = Sys.time() # capture code start time
setwd("~/sce prediction_REG500/Code")

source("functions.R") # Our custom built functions in R

set.seed(2020) # setting the seed for reproducibility
sInfo = sessionInfo()
save(sInfo, file="../Results/sessionInfo.RData")



#[2] Loading the Data:
# --------------------
df = read_csv("../Data/REG500_SCELag7.csv")
df$X1 = NULL
df$driver = NULL
df$distance = NULL
colnames(df) = c('intervalTime', 'speedMean', 'speedSD', 'age', 'gender', 'prepInten',
                 'prepProb', 'windSpeed', 'visibility', 'cumDrive', 'dayOfWeek','weekend', 
                 'holiday', 'hourDayCat', 'SCELag7','SCE') # setting colNames



df %<>% mutate_if(is.character, as.factor) # converting strings to factor variables
df$holiday %<>% as.factor()
df$weekend %<>% as.factor()
df$SCE = as.factor(df$SCE)

levels(df$SCE) = list(Yes=1, No=0) # specifiying the levels of the response
levels(df$hourDayCat)= list(rush1 = "6 a.m. - 10 a.m.",
                            midDay ="11 a.m. - 14 p.m.",
                            rush2 ="15 p.m. - 20 p.m.",
                            oNight ="21 p.m. -  5 a.m.") # renaming the Levels of hourDayCat variable



# [3] ETL:
# --------
pdf("../Results/etlPredHistogram.pdf", width = 6.5, height = 6.5)
plot_histogram(df) %>% print()
dev.off() # closing the image object


pdf("../Results/etlBoxPlot.pdf", width = 6.5, height = 6.5)
plot_boxplot(df[,!names(df) %in% c('intervalTime','gender','hourDayCar','dayOfWeek', 'holiday')], 
             by="SCE", ncol=3) %>% print()
dev.off() # closing the image object

pdf("../Results/etlCorrPlot.pdf", width = 6.5, height = 6.5)
plot_correlation(df, maxcat = 2L) %>% print()
dev.off() # closing the image object

# [4] Pred Modeling:
# ------------------
numCores = detectCores()
cl = makePSOCKcluster(numCores , outfile ="../Results/trainLog.txt") # Telling R to run on # cores
registerDoParallel(cl)

df = sample_frac(df, 0.01)

# [A] Create Training and Testing Datasets
trainRowNumbers = createDataPartition(df$SCE,
                                      p=0.8, 
                                      list = F) %>% as.vector()

trainData = df[trainRowNumbers,] # Training Dataset
testData = df[-trainRowNumbers,] # Testing Dataset



# [C] Cross Validation Setup
fitControl = trainControl(
        method = "cv", # k-fold cross validation
        number = 5, # Number of Folds
        sampling = "down", # Down sampling
        search = "random", # Random search for parameter tuning when applicable
        summaryFunction = outputFun, # see functions.R file
        classProbs = T, # should class probabilities be returned
        selectionFunction = "best", # best fold
        savePredictions = "final",
        index = createResample(trainData$SCE, 5))


# [D] Model Training
models = caretList(SCE ~., data = trainData, metric="ROC", 
                   tuneList= list(
                           cart= caretModelSpec(method="rpart", tuneLength=20),
                           glm = caretModelSpec(method = "glm", family= "binomial"),
                           lasso = caretModelSpec(method = "glmnet", family= "binomial", 
                                                  tuneGrid = expand.grid(.alpha=1,
                                                                         .lambda= seq(0.00001, 2, length=20))),
                           nb = caretModelSpec(method="naive_bayes", tuneLength=20),
                           nnet = caretModelSpec(method="avNNet", tuneLength=20),
                           rf = caretModelSpec(method="rf", tuneLength=20),
                           ridge = caretModelSpec(method = "glmnet", family= "binomial", 
                                                  tuneGrid = expand.grid(.alpha=0,
                                                                         .lambda= seq(0.00001, 2, length=20))),
                           svm = caretModelSpec(method="svmRadial", tuneLength=20),
                           xgb = caretModelSpec(method="xgbTree", tuneLength=20) 
                   ),  
                   trControl=fitControl, continue_on_fail = T,
                   preProcess = c("nzv", "center", "scale", "corr")
)


# [F] Results
cvResults = resamples(models) # Cross-validation results for best fold

stopCluster(cl)

fullResultsTrain_class = lapply(models, predict, trainData, type = "raw") 
fullResultsTest_class = lapply(models, predict, testData, type = "raw") 

save(fullResultsTest_class, fullResultsTrain_class,
     file="../Results/fullResults.RData")

save(models, file="../Results/trainedModels.RData")
save(cvResults, file="../Results/cvResults.RData")

pdf("../Results/cvBoxPlot.pdf", width = 6.5, height = 6.5)
scales = list(x=list(relation="free"), y=list(relation="free"))
bwplot(cvResults, scales=scales) %>% print()
dev.off()

modelCorrs = modelCor(resamples(models))

predResults = predSummary(models, testData, responseVarName = 'SCE', lev = "Yes") # see  functions.R

save(predResults,modelCorrs,
     file="../Results/predResults.RData")

confMatrix = lapply(models, predict, newdata = testData, type="raw") %>% 
        lapply(confusionMatrix, testData$SCE) %>% 
        lapply("[[", "table")

timeTaken = Sys.time() - startTime # total computing time

save(confMatrix,timeTaken,
     file="../Results/ConfMatrix.RData")

# [G] Variable Importance

# cart

cart.var = varImp(models$cart)
pdf("../Results/varimp_cart.pdf", width = 6.5, height = 6.5)
plot(varImp(models$cart), main = "Variable importance: Cart")
dev.off()

# glm
glm.var = varImp(models$glm)
pdf("../Results/varimp_glm.pdf", width = 6.5, height = 6.5)
plot(varImp(models$glm), main = "Variable importance: Glm")
dev.off()

# lasso
lasso.var = varImp(models$lasso)
pdf("../Results/varimp_lasso.pdf", width = 6.5, height = 6.5)
plot(varImp(models$lasso), main = "Variable importance: Lasso")
dev.off()

# nb
nb.var = varImp(models$nb)
pdf("../Results/varimp_nb.pdf", width = 6.5, height = 6.5)
plot(varImp(models$nb), main = "Variable importance: Nb")
dev.off()

# nnet
nnet.var = varImp(models$nnet)
pdf("../Results/varimp_nnet.pdf", width = 6.5, height = 6.5)
plot(varImp(models$nnet), main = "Variable importance: Nnet")
dev.off()

# rf
rf.var = varImp(models$rf)
pdf("../Results/varimp_rf.pdf", width = 6.5, height = 6.5)
plot(varImp(models$rf), main = "Variable importance: Rf")
dev.off()

# ridge
ridge.var = varImp(models$ridge)
pdf("../Results/varimp_ridge.pdf", width = 6.5, height = 6.5)
plot(varImp(models$ridge), main = "Variable importance: Ridge")
dev.off()

# svm
svm.var = varImp(models$svm)
pdf("../Results/varimp_svm.pdf", width = 6.5, height = 6.5)
plot(varImp(models$svm), main = "Variable importance: SVM")
dev.off()

# xgb
xgb.var = varImp(models$xgb)
pdf("../Results/varimp_xgb.pdf", width = 6.5, height = 6.5)
plot(varImp(models$xgb), main = "Variable importance: Xgb")
dev.off()

save(cart.var, glm.var, lasso.var, nb.var, nnet.var, rf.var, ridge.var, svm.var, xgb.var,
     file = "../Results/VarImp.RData")

