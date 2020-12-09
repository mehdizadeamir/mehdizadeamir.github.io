# This Code is Written to Examine the Use of ML Models to Predict whether an SCE will occur in
# the next trip.


# [1] Initilization:
# -----------------

if (!require("pacman")) install.packages("pacman") # checks if the packman package is installed locally
pacman::p_load(DT, foreach, magrittr, tidyverse, # important analytic pkgs
               knitr, kableExtra, # to print nice looking tables
               dataPreparation, DataExplorer, # important for exploratory data analysis
               recipes, caret, caretEnsemble, doParallel, MLmetrics, # bases for our ML approach
               rpart, glmnet, Matrix, naivebayes, nnet, randomForest, # needed packages for ML
               kernlab, # needed for svm
               xgboost, plyr, # needed packages for xgboost
               caTools, pROC, # for classification metrics
               lime # for variable importance
)
startTime = Sys.time() # capture code start time
setwd("~/sce prediction_REG500_ensemble/Code")

source("functions_ens.R") # Our custom built functions in R

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
df$holiday %<>% as.factor() # converting the holiday variable to a factor
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
        index = createResample(trainData$SCE, 5) )


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

cvResults = resamples(models) # Cross-validation results for best fold

stopCluster(cl)

save(models, file="../Results/trainedModels.RData")
save(cvResults, file="../Results/cvResults.RData")


# [E] Model Results
pdf("../Results/cvBoxPlot.pdf", width = 6.5, height = 6.5)
scales = list(x=list(relation="free"), y=list(relation="free"))
bwplot(cvResults, scales=scales) %>% print()
dev.off()

modelCorrs = modelCor(resamples(models))

predResults = predSummary(models, testData, responseVarName = 'SCE', lev = "Yes") # see  functions.R

confMatrix = lapply(models, predict, newdata = testData, type="raw") %>% 
        lapply(confusionMatrix, testData$SCE) %>% 
        lapply("[[", "table")


# [F] Ensemble moels


numCores = detectCores()
cl = makePSOCKcluster(numCores , outfile ="../Results/greedy_all.txt") # Telling R to run on # cores
registerDoParallel(cl)

## e_all: all together
ens_all = models[c("cart","glm","lasso", "nb", "nnet", "rf", "ridge", "xgb")]
greedy_ensemble_all = caretEnsemble(
        ens_all, 
        metric="ROC",
        trControl= fitControl,
        preProcess = c("nzv", "center", "scale", "corr"))

ensResults_all = ensResults_fun(greedy_ensemble_all, testData, ensName = "e_all")


results = cbind(predResults, ensResults_all)
stopCluster(cl)

## e1: ridge and xgb

numCores = detectCores()
cl = makePSOCKcluster(numCores , outfile ="../Results/greedy1.txt") # Telling R to run on # cores
registerDoParallel(cl)

ens1 = models[c("ridge", "xgb")]
greedy_ensemble1 = caretEnsemble(
        ens1, 
        metric="ROC",
        trControl= fitControl,
        preProcess = c("nzv", "center", "scale", "corr"))

ensResults1 = ensResults_fun(greedy_ensemble1, testData, ensName = "e1")
results = cbind(results, ensResults1)

## e2: svm cart glm

ens2 = models[c("cart", "glm", "svm")]
greedy_ensemble2 = caretEnsemble(
        ens2, 
        metric="ROC",
        trControl= fitControl,
        preProcess = c("nzv", "center", "scale", "corr"))

ensResults2 = ensResults_fun(greedy_ensemble2, testData, ensName = "e2")
results = cbind(results, ensResults2)

stopCluster(cl)

timeTaken = Sys.time() - startTime # total computing time

save(results, confMatrix, timeTaken, modelCorrs, timeTaken,
     file="../Results/predResults.RData")



