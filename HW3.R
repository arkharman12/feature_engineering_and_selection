# Set the working directory
setwd("/Users/singhh/Downloads/CSCI48900/agriculture-loan-prediction")

######### Census Data #########
######### Question 1 #########

library(tidycensus)
library(tidyverse)

census_api_key("edd6d6ff485a50d0c3b93a373daa2e0dc591e722")

v17 <- load_variables(2017, "acs5", cache = TRUE)
head(v17)

zc <- get_acs(geography = "zcta", variables = c("B00001_001", output="wide"))
head(zc)

# zc <- get_acs(geography = "zcta", variables = c("B00001_001", "B02001_002", "B02008_001", "B02009_001", "B02013_001",
#                                                 "B02014_001", "B02015_001", "	B02016_001", "B03001_001", "B04004_001",
#                                                 "B19127_001", "B19131_001", "B19201_003", "B19202E_001", "B19301H_001",
#                                                 "B19325_006", "B20001_003","B20004_017", "B20005_017", "B20005I_027", "B20005I_032",
#                                                 output="wide"))

# Note: This was making a very long PDF so I commented it out. But it works!

######### Question 2 #########

test <-read.csv(file="train_comp1_2020.csv")
train <-read.csv(file="test_comp1_2020.csv")


# head(aggregate(Zip_Code~AT28+AT33+AT36+COLL+G068+G091+G093+G094+G096+MT28+MT36+RE34+S063, test, FUN="mean"))
# head(aggregate(Zip_Code~AT28+AT33+AT36+COLL+G068+G091+G093+G094+G096+MT28+MT36+RE34+S063, train, FUN="mean"))

# Note: This was making a very long PDF so I commented it out. But it works!

######### Question 3 #########

colnames(zc)
class(zc$estimate)
class(zc$GEOID)
# plot(zc$GEOID, zc$estimate, main = "Scatterplot", xlim=c(00601,00610), ylim=c(-2,10))
mod <- lm(zc$GEOID~zc$estimate)
summary(mod)
attributes(mod)
anova(mod)
# fitted(mod)
3.375e+05+7.144e-02*15
summary(coef(mod))
summary(predict(mod, data.frame(GEOID=c(5, 10, 5)), interval="prediction", level=0.9))

# Variables like CreditScore, TYCornUnits, TYBeanUnits, TYWheatUnits, LYCornUnits, LYBeanUnits, LYWheatUnits
# are most significant in deciding the loan for different loan companies


######### Sparse feature selection #########
######### Question 1 #########

library(vtreat)
vars <- c("Zip_Code", "State")
treatplan <- designTreatmentsZ(train, vars, verbose=FALSE)
scoreFrame <- treatplan$scoreFrame %>%
              select(varName, origName, code)
newvars <- scoreFrame %>%
            filter(code %in% c("clean", "lev")) %>%
            magrittr::use_series(varName)

training.treat <- prepare(treatplan, train, varRestriction = newvars)
test.treat <- prepare(treatplan, test, varRestriction = newvars)


######### Question 2 #########

library(glmnet)  

set.seed(42)  # Set seed for reproducibility

n <- 1000  # Number of observations
p <- 5000  # Number of predictors included in model
real_p <- 15  # Number of true predictors

# Generate the data
x <- matrix(rnorm(n*p), nrow=n, ncol=p)
y <- apply(x[,1:real_p], 1, sum) + rnorm(n)

# Split data into training and testing datasets.
train_rows <- sample(1:n, .66*n)
x.train <- x[train_rows, ]
x.test <- x[-train_rows, ]

y.train <- y[train_rows]
y.test <- y[-train_rows]

# Ridge Regression
alpha0.fit <- cv.glmnet(x.train, y.train, type.measure="mse", 
                        alpha=0, family="gaussian")

alpha0.predicted <- 
  predict(alpha0.fit, s=alpha0.fit$lambda.1se, newx=x.test)

# Test dataset.
mean((y.test - alpha0.predicted)^2)

# Lasso Regression
alpha1.fit <- cv.glmnet(x.train, y.train, type.measure="mse", 
                        alpha=1, family="gaussian")

alpha1.predicted <- 
  predict(alpha1.fit, s=alpha1.fit$lambda.1se, newx=x.test)

mean((y.test - alpha1.predicted)^2)

# These models retain variables from both test and train datasets




