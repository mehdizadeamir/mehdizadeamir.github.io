if (!require("pacman")) install.packages("pacman") # checks if the packman package is installed locally
pacman::p_load(DT, foreach, magrittr, tidyverse,fst, # important analytic pkgs
               knitr, kableExtra, # to print nice looking tables
               dataPreparation, DataExplorer, zoo, # important for exploratory data analysis
               recipes, caret, caretEnsemble, doParallel, MLmetrics, # bases for our ML approach
               rpart, glmnet, Matrix, naivebayes, nnet, randomForest, # needed packages for ML
               kernlab, # needed for svm
               xgboost, plyr, # needed packages for xgboost
               caTools, pROC, # for classification metrics
               ROSE, DMwR) # for sampling methods
library(psych)
library(corrplot)
library(summarytools)
library(dataPreparation)
library(DataExplorer)

df = read_csv("../Data/REG500_SCELag7.csv")
df$X1 = NULL
df$driver = NULL
colnames(df) = c('intervalTime', 'speedMean', 'speedSD', 'distance',
                 'age', 'gender', 'prepInten',
                 'prepProb', 'windSpeed', 'visibility', 'cumDrive', 'dayOfWeek','weekend', 
                 'holiday', 'hourDayCat', 'SCELag7','SCE') # setting colNames


df %<>% mutate_if(is.character, as.factor)
df$SCE = as.factor(df$SCE)
df$weekend = as.factor(df$weekend)
df$holiday = as.factor(df$holiday)

levels(df$SCE) = list(Yes=1, No=0)
levels(df$hourDayCat)= list(rush1 = "6 a.m. - 10 a.m.",
                            midDay ="11 a.m. - 14 p.m.",
                            rush2 ="15 p.m. - 20 p.m.",
                            oNight ="21 p.m. -  5 a.m.")
df  = df %>%
         filter(distance < 40)

descr(df, stats = c("min", "q1", "mean", "med", "q3", "max", "sd", "skewness", "kurtosis"), 
      headings = F) %>% round(digits = 4)

df$distance = NULL

# Histogram
plot_histogram(df,  ncol=3, nrow = 4,
               ggtheme = theme_bw())
ggsave(filename = "../Results/histogram.png", width = 6.5, height = 4, dpi = 600,
       units = 'in', device = 'png')

# Box Plots by SCE
plot_boxplot(df, by="SCE", ncol=3, nrow = 4,  
             geom_boxplot_args = list("outlier.size" = 0.25),
             ggtheme = theme_bw())
ggsave(filename = "../Results/boxplot.png", width = 6.5, height = 4, dpi = 600,
       units = 'in', device = 'png')





df %<>% mutate_if(is.factor, as.integer)


df$SCE = as.numeric(df$SCE)
df$SCE = df$SCE * -1
df$SCE = as.integer(df$SCE)


r0 = df %>%
        mixedCor(c = c(1:4, 6:10,14,15), d = c(5,12,13,16), p = c(11))


pdf(width = 12, height = 9, file = "../Results/correlation_plot.pdf")
corrplot(r0$rho,
         
         method="color",
         
         col = rev(RColorBrewer::brewer.pal(n = 8, name = "RdYlBu")),
         
         type="lower",number.font = 7,
         
         addCoef.col = "black", # Add coefficient of correlation
         
         tl.col="black", tl.srt = 25,
         
         diag=FALSE,
         
         mar = rep(0, 4),
         
         xpd = NA)
dev.off()


