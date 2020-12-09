# Functions to accompany our analysis:
# ------------------------------------

dataTypes = function(df) {
  col.classes = lapply(df, class)
  col.classes.df = unlist(col.classes) %>%  data.frame()
  temp.data = table(col.classes.df) %>% sort(decreasing=T) %>% data.frame()
  return(temp.data)
}


# Output Function for Models
outputFun = function(data, lev = "Yes", model = NULL) {
  defaultS = defaultSummary(data, lev, model) # returns Accuracy and Kappa
  twoClassS = twoClassSummary(data, lev, model) # returns AUC, sensitivity and specificity
  out = c(defaultS, twoClassS)
  out
}


# Data Frame of Prediction Metrics of Interest
predSummary = function(modelsObject, testObject, responseVarName, lev = "Yes"){
  allModelsClassPreds = lapply(modelsObject, predict, newdata = testObject, type="raw") %>% 
    data.frame()
  
  resultsSensSpec = lapply(allModelsClassPreds, confusionMatrix, testObject[[responseVarName]]) %>% 
    lapply("[[", "byClass") %>% as.data.frame() %>% 
    dplyr::slice(c(1,2))
  row.names(resultsSensSpec) = c("Sensitivity","Specificity")
  
  resultsAcc = lapply(allModelsClassPreds, confusionMatrix, testObject[[responseVarName]]) %>% 
    lapply("[[", "overall") %>% as.data.frame() %>% 
    dplyr::slice(c(1))
  row.names(resultsAcc) = "Accuracy"
  
  resultsAUC = lapply(modelsObject, predict, newdata = testObject, type="prob") %>% 
    lapply(function(x) x[, lev]) %>% lapply(colAUC, testObject[[responseVarName]]) %>% 
    data.frame(row.names = "AUC")
  
  resultsGmean = sqrt(resultsSensSpec["Sensitivity",] *resultsSensSpec["Specificity",]  )
  row.names(resultsGmean) = "Gmean"
  
  results = rbind(resultsAUC, resultsAcc, resultsSensSpec, resultsGmean) %>% data.frame()
  results
}



# Ens Results
ensResults_fun = function(modelsObject, testObject, ensName = NULL){
  ensembleResults = predict(modelsObject, testObject, type = "raw")
  
  ensSensSpec = confusionMatrix(ensembleResults, testObject$SCE)$byClass %>% as.data.frame() %>% 
    dplyr::slice(c(1,2))
  
  ensAcc = confusionMatrix(ensembleResults, testObject$SCE)$overall %>% as.data.frame() %>% 
    dplyr::slice(c(1))
  
  ensAUC = predict(modelsObject, testData, type = "prob") 
  ensAUC = colAUC(ensAUC, testObject$SCE) %>% 
    data.frame(row.names = "AUC")
  
  ensGmean = sqrt(ensSensSpec[1,1] * ensSensSpec[2,1]  )
  
  results = rbind(ensAUC, ensAcc, ensSensSpec, ensGmean)
  colnames(results) = ensName
  row.names(results) = c("AUC", "Accuracy", "Sensitivity", "Specificity", "Gmean")
  results
}
