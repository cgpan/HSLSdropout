# NOTE:
#   - This file is to do the data cleaning; based on 50_Data_cleaning_V7.R
#   - Please download the raw HSLS dataset first and then change some file path to your local path
#   - The raw HSLS dataset can found on NCES DataLab: https://nces.ed.gov/datalab/onlinecodebook/session/codebook/ad2422fd-b885-4c91-8f1f-8ecd9d761db0
#   - Please click the download icon on the right side of this page and select the R version of dataset.


##############################################################
##                                                          ##
##  This file focus on the data cleaning                    ##
##  to clean the selected variables                   
##  change some meaningless value into a factor's level
##                                                          ##
##############################################################
cat("Running the data cleaning file...")
cat("Load the dataset...")
load("Your_local_path_towards_public_dataset/hsls_17_student_pets_sr_v1_0.rdata")

df <- hsls_17_student_pets_sr_v1_0
rm(hsls_17_student_pets_sr_v1_0)
cat("Load the dataset...complete")


# ##########################################################
# 2023.10.22. Add tuning section to decide using which variable
# ##########################################################
# choose to use X3 or X4 dropout indicators
using_which_drop_var <- "X4EVERDROP"
# whether to keep Non-response as it is or change it into NA
keep_nr <- F


# the variable about the dropout
# 2023.10.22: The odds is 2576: 20927, no missing value
table(df$X3EVERDROP)
levels(df$X3EVERDROP)
# 2023.10.22: The odds is 2714: 14618, non-response 6168, missing 3
table(df$X4EVERDROP)
levels(df$X4EVERDROP)

# --------------Jun 6th ---------------------
# check the X1SES_U and X1LOCALE: X1SES_U is at the range of -2 ~ 3
hist(df$X1SES_U)
summary(df$X1SES_U[df$X1SES_U>-3])

if (using_which_drop_var == "X3EVERDROP"){
  # Var1: change the outcome variable into numeric
  # 2023.10.22: the drop out case to be 1
  print("Choose to use the X3EVERDROP....")
  df$X3EVERDROP <- as.numeric(df$X3EVERDROP)
  df$X3EVERDROP[which(df$X3EVERDROP ==1)] <- 0
  df$X3EVERDROP[which(df$X3EVERDROP ==2)] <- 1
} else{
  # Var1: change the outcome variable into numeric
  # 2023.10.22: There are 6168 missing case in X4 variable
  # lose too much information.
  print("Choose to use the X4EVERDROP....")
  df$X4EVERDROP <- as.numeric(df$X4EVERDROP)
  print(table(df$X4EVERDROP))
  df$X4EVERDROP[which(df$X4EVERDROP ==1)] <- 0
  df$X4EVERDROP[which(df$X4EVERDROP ==2)] <- 1
  print(dim(df))
  # 2023.10.30: to keep the X4EVERDROP with non-NA responses only
  df <- df[which(df$X4EVERDROP == 0 | df$X4EVERDROP==1),]
  print(dim(df))
  print(table(df$X4EVERDROP, useNA = "always"))
  # 2023.10.30: note since the original HSLS df using the categorical labeled NAs
  # 2023.10.30: the following step wont change anything.
  df <- df[complete.cases(df[,"X4EVERDROP"]),]
  print(table(df$X4EVERDROP, useNA = "always"))
}

# 2023.10.30: adding the parent's and student's expectation
# X1PAREDEXPCT/ X1STUEDEXPCT
table(df$X1PAREDEXPCT)
levels(df$X1PAREDEXPCT)
table(df$X1STUEDEXPCT)
levels(df$X1STUEDEXPCT)
# since there are too many categories in each variable, I choose to convert
# these expectation variable into numeric, higher the value, higher the expectations
df$X1PAREDEXPCT <- as.numeric(df$X1PAREDEXPCT)
df$X1STUEDEXPCT <- as.numeric(df$X1STUEDEXPCT)
table(df$X1PAREDEXPCT)
table(df$X1STUEDEXPCT)
# here I change the level above 10 into NA.
df$X1PAREDEXPCT[which(df$X1PAREDEXPCT >10)] <- NA
df$X1STUEDEXPCT[which(df$X1STUEDEXPCT >10)] <- NA


# Var2: change the gender variable
# 1= male 0=female
table(df$X1SEX)
df$X1SEX <-  as.numeric(df$X1SEX)
df$X1SEX[which(df$X1SEX ==2)] <- 0
df$X1SEX[which(df$X1SEX ==5)] <- NA
table(df$X1SEX)

# Var3: change the 9th grader's math assessment score
hist(df$X1TXMTSCOR)
df$X1TXMTSCOR[which(df$X1TXMTSCOR < 0)] <- NA
summary(df$X1TXMTSCOR)

# Var4: change the school location variable
table(df$X1LOCALE)
levels(df$X1LOCALE)

# Var5: change the region variable
table(df$X1REGION)


# 2023.10.22: here is a concern: Can we use factor in a numeric way?
# Var6: the 9th grader pipeline variable
# NOTE: here I treated the pipeline variable as a continuous variable
table(df$X3THIMATH9)
tempt <- as.numeric(df$X3THIMATH9)
tempt[tempt == 16] <- NA
tempt[tempt == 17] <- NA
df$X3THIMATH9 <- tempt-1
table(df$X3THIMATH9, useNA = "always")
# levels(df$X3THIMATH9)[levels(df$X3THIMATH9) == "Unit non-response"] <- "Non-response"


# var7: the 9th grader pipeline variable in science
# 2023.10.22: decide to take the science pipeline into account
table(df$X3THISCI9)
tempt <- as.numeric(df$X3THISCI9)
tempt[tempt == 8] <- NA
tempt[tempt == 9] <- NA
table(tempt, useNA = "always")
df$X3THISCI9 <- tempt-1
table(df$X3THISCI9)

# var8: the race variable. change some level into NA
table(df$X1RACE)
levels(df$X1RACE)[levels(df$X1RACE) %in% c("Item legitimate skip/NA","Unit non-response")] <- "Non-response"

# var9: the 8th grader's math and science courses and grade
table(df$S1M8)
levels(df$S1M8)[levels(df$S1M8) %in% c("Item legitimate skip/NA","Unit non-response")] <- "Non-response"

table(df$S1S8)
levels(df$S1S8)[levels(df$S1S8) %in% c("Item legitimate skip/NA","Unit non-response")] <- "Non-response"


# var10: math related variables
table(df$X1SCHOOLCLI)
# 2023.10.22: change the negative value -8 and -9 into NAs
df$X1SCHOOLCLI[which(df$X1SCHOOLCLI %in% c(-9,-8))] <- NA

# 2023.10.22: there is a continuous variable can give exact age information
table(df$X1STDOB)
dob_year <- as.numeric(substr(df$X1STDOB,1,4))
head(dob_year)
summary(dob_year)
class(dob_year)
# dob_year <- as.data.frame(dob_year)
dob_year[dob_year < 0 ] <- NA
table(dob_year, useNA = "always")
stu_age <- 2009-dob_year
table(stu_age, useNA = "always")
head(stu_age)
dim(stu_age)
df$X1STAGE <- stu_age

table(df$X1STAGE, useNA = "always")

# 2023.10.22: drop the previous categorical age variable: S1BIRTHYR
# var11: birth year changed into real age

# change some negative value in continuous variables into NA
# ---- on Jun 6th, add the X1SES_U as another outcome variable ----
var_list <- c("X1SES","X1MTHID","X1MTHUTI","X1MTHEFF","X1MTHINT",
              "X1SCHOOLBEL","X1SCHOOLENG","X1SCHOOLCLI","X1SES_U",
              "X1SCIID", "X1SCIUTI", "X1SCIEFF", "X1SCIINT","X2SCHOOLCLI","X2PROBLEM")

# use a for loop to change the negative value into NA
for (var in var_list) {
  df[,var][which(df[,var] < -5)] <- NA
}
cat("The returned the dataset in the memory is called df...")
cat("Data cleaning complete. Now you can move on to next step...")



# var 12: change the survey answers
table(df$X1PAR1EDU)
levels(df$X1PAR1EDU)[levels(df$X1PAR1EDU) =="Item legitimate skip/NA"] <- "Non-response"
levels(df$X1PAR1EDU)[levels(df$X1PAR1EDU) =="Unit non-response"] <- "Non-response"
table(df$X1PAR1EDU)
levels(df$X1PAR2EDU)[levels(df$X1PAR2EDU) =="Item legitimate skip/NA"] <- "Non-response"
levels(df$X1PAR2EDU)[levels(df$X1PAR2EDU) =="Unit non-response"] <- "Non-response"
table(df$X1PAR2EDU)
table(df$X1PAR1OCC2)
levels(df$X1PAR1OCC2)[levels(df$X1PAR1OCC2) =="Item legitimate skip/NA"] <- "Non-response"
levels(df$X1PAR1OCC2)[levels(df$X1PAR1OCC2) =="Unit non-response"] <- "Non-response"

table(df$X1PAR2OCC2)
levels(df$X1PAR2OCC2)[levels(df$X1PAR2OCC2) =="Item legitimate skip/NA"] <- "Non-response"
levels(df$X1PAR2OCC2)[levels(df$X1PAR2OCC2) =="Unit non-response"] <- "Non-response"

table(df$X1HHNUMBER)
levels(df$X1HHNUMBER)[levels(df$X1HHNUMBER) =="Item legitimate skip/NA"] <- "Non-response"
levels(df$X1HHNUMBER)[levels(df$X1HHNUMBER) =="Unit non-response"] <- "Non-response"

# 2023.10.22: add new variable to indicate the school's class and type
# X1CONTROL does not have any missing values.
table(df$X1CONTROL,useNA = "always")



var_list <- c(using_which_drop_var,                          # outcome variables
              "X1SEX","X1RACE","X1STAGE","X1CONTROL",         # demographic variables
              "X1SES","X1SES_U","X1FAMINCOME","X1PAR1EDU","X1PAR2EDU","X1PAR1OCC2","X1PAR2OCC2", # SES-related
              "X1HHNUMBER",                                    # family structure
              "X1SCHOOLENG", "S1NOHWDN", "S1NOPAPER", "S1NOBOOKS", "S1LATE",  # school engagement
              "X1SCHOOLBEL", "S1SAFE","S1PROUD","S1TALKPROB","S1SCHWASTE","S1GOODGRADES", # school belonging
              "S1FRNDGRADES","S1FRNDSCHOOL","S1FRNDCLASS","S1FRNDCLG",     # peer influence there may be more vars
              "X1SCHOOLCLI",
              # "A1CONFLICT","A1ROBBERY","A1VANDALISM","A1DRUGUSE","A1ALCOHOL","A1DRUGSALE", # school climate1
              # "A1WEAPONS","A1PHYSABUSE","A1TENSION","A1BULLY","A1VERBAL","A1MISBEHAVE","A1DISRESPECT",  # school climate2
              "X1TXMTSCOR", # a standardized score for math at first semester of 9
              "X1LOCALE","X1REGION", # school location information
              "S1M8","S1M8GRADE","S1S8","S1S8GRADE", # pre-high school's achievement
              "X1MTHID", "S1MPERSON1", "S1MPERSON2", # Math id
              "X1MTHUTI", "S1MUSELIFE", "S1MUSECLG", "S1MUSEJOB", # Math utility
              "X1MTHEFF", "S1MTESTS","S1MTEXTBOOK","S1MSKILLS","S1MASSEXCL", # Math self-efficacy
              "X1MTHINT", "S1MENJOYING", "S1MENJOYS", "S1MWASTE", "S1MBORING",
              "X1SCIID", "S1SPERSON1", "S1SPERSON2",
              "X1SCIUTI", "S1SUSELIFE", "S1SUSECLG", "S1SUSEJOB",
              "X1SCIEFF", "S1STESTS","S1STEXTBOOK","S1SSKILLS","S1SASSEXCL",
              "X1SCIINT","S1SENJOYING","S1SWASTE","S1SBORING","S1FAVSUBJ","S1LEASTSUBJ","S1SENJOYS",
              "X2SCHOOLCLI","X2PROBLEM",
              "A1RESOURCES","A1HEALTH","A1UNPREP","A1PRNTINV","A1APATHY","A1TCHRABSENT","A1TARDY",
              "A2RESOURCES","A2HEALTH","A2UNPREP","A2PRNTINV","A2APATHY","A2TARDY",
              "A2OTHERBULLY","A2CYBERBULLY",
              "X1PAREDEXPCT","X1STUEDEXPCT") #
class(df$A2HEALTH)

# source("06_functions.r")
# June 1st update, using function to clean the data wont work
# so use the code directly
# in this round, clean all the official missing value into NA
for (var_name in var_list) {
  print(var_name)
  if  (class(df[,var_name]) == "factor"){
    # noted, must separate the factor variables first in case of changing the
    # the factor variable into strings
    levels(df[,var_name])[levels(df[, var_name]) =="Item legitimate skip/NA"] <- "Non-response"
    levels(df[,var_name])[levels(df[, var_name]) =="Unit non-response"] <- "Non-response"
    levels(df[,var_name])[levels(df[,var_name]) == "Component not applicable"] <- "Not applicable"
    levels(df[,var_name])[levels(df[,var_name]) == "Item not administered: abbreviated interview"] <- "Not applicable"
    print(table(df[,var_name]))
  }
}


# -------------May 30, 2023---------------------
never_often_vars <- c("S1NOHWDN", "S1NOPAPER","S1NOBOOKS", "S1LATE") 

table(df$S1NOHWDN)
agree_disagree_vars <- c("S1SAFE","S1PROUD","S1TALKPROB","S1SCHWASTE","S1GOODGRADES",
                         "S1MPERSON1","S1MPERSON2","S1MUSELIFE","S1MUSECLG","S1MUSEJOB",
                         "S1MTESTS","S1MTEXTBOOK","S1MSKILLS","S1MASSEXCL",
                         "S1MENJOYING","S1MWASTE","S1MBORING", 
                         "S1SPERSON1","S1SPERSON2","S1SUSELIFE","S1SUSECLG","S1SUSEJOB",
                         "S1STESTS","S1STEXTBOOK","S1SSKILLS","S1SASSEXCL",
                         "S1SENJOYING","S1SWASTE","S1SBORING")
table(df$S1SAFE)
true_false_vars <- c("S1FRNDGRADES","S1FRNDSCHOOL","S1FRNDCLASS","S1FRNDCLG")

no_prob_serious_4 <- c("A1RESOURCES","A1UNPREP","A1PRNTINV","A1APATHY","A1TARDY",
                       "A2RESOURCES","A2UNPREP","A2PRNTINV","A2APATHY","A2TARDY")
no_prob_serious_3 <- c("A1HEALTH","A1TCHRABSENT","A2HEALTH")

never_daily_5 <- c("A2OTHERBULLY","A2CYBERBULLY")

table(df$A1HEALTH)
# -------------June 1st, 2023---------------------
# change the likert scale into numeric value 

for (var in never_daily_5){
  levels(df[,var])[levels(df[,var] == "Component not applicable")] <- "Not applicable"
  levels(df[,var])[levels(df[,var] == "Item not administered: abbreviated interview")] <- "Not applicable"
  levels(df[,var])[levels(df[,var] == "Item legitimate skip/NA")] <- "Non-response"
  levels(df[,var])[levels(df[,var] == "Unit non-response")] <- "Non-response"
}


for (var in no_prob_serious_4){
  levels(df[,var])[levels(df[,var] == "Item legitimate skip/NA")] <- "Non-response"
  levels(df[,var])[levels(df[,var] == "Unit non-response")] <- "Non-response"
}

for (var in no_prob_serious_3){
  levels(df[,var])[levels(df[,var] == "Item legitimate skip/NA")] <- "Non-response"
  levels(df[,var])[levels(df[,var] == "Unit non-response")] <- "Non-response"
}


for (var in never_often_vars){
  levels(df[,var])[levels(df[,var] == "Item legitimate skip/NA")] <- "Non-response"
  levels(df[,var])[levels(df[,var] == "Unit non-response")] <- "Non-response"
}

for (var in agree_disagree_vars) {
  levels(df[,var])[levels(df[,var] == "Item legitimate skip/NA")] <- "Non-response"
  levels(df[,var])[levels(df[,var] == "Unit non-response")] <- "Non-response"
}

table(df$S1FRNDGRADES)

for (var in true_false_vars) {
  levels(df[,var])[levels(df[,var] == "Item legitimate skip/NA")] <- "Non-response"
  levels(df[,var])[levels(df[,var] == "Unit non-response")] <- "Non-response"
}

var = "S1MENJOYS"
table(df$S1MENJOYS)
levels(df[,var])[levels(df[,var] == "Item legitimate skip/NA")] <- "Non-response"
levels(df[,var])[levels(df[,var] == "Unit non-response")] <- "Non-response"

var = "S1SENJOYS"
levels(df[,var])[levels(df[,var] == "Item legitimate skip/NA")] <- "Non-response"
levels(df[,var])[levels(df[,var] == "Unit non-response")] <- "Non-response"
# now change the self-reported letter grade into numeric values
# the targeting variable is S1S8GRADE S1M8GRADE


# ----------------------------------------------------
# all basic data cleaning task finished at June 1st, 2023
# now slice the dataset only keep the useful variables
# to save memory
# ---------------------------------------------------

df <- df[,c(var_list,"STU_ID")]

# ---------------June 9th Construct the school problem variable ---------------
# explore using the school problem variables to replace the school climate
cor(df$X2SCHOOLCLI,df$X2PROBLEM, use = "pairwise.complete.obs")
# based on the only available variables to construct the school problem variable
# that is the X1PROBLEM
no_prob_serious_4 <- c("A1RESOURCES","A1UNPREP","A1PRNTINV","A1APATHY","A1TARDY",
                       "A2RESOURCES","A2UNPREP","A2PRNTINV","A2APATHY","A2TARDY")
no_prob_serious_3 <- c("A1HEALTH","A1TCHRABSENT","A2HEALTH")
# [0705]subset the original df into a temp df in case of changing the original 
df_temp <- df[,c(no_prob_serious_4,no_prob_serious_3, "X2PROBLEM")]


for (var in no_prob_serious_4){
  df_temp[, var] <- as.character(df_temp[,var])
  df_temp[,var][which(df_temp[,var] == "Not a problem")] <- 0
  df_temp[,var][which(df_temp[,var] == "Minor problem")] <- 1
  df_temp[,var][which(df_temp[,var] == "Moderate problem")] <- 2
  df_temp[,var][which(df_temp[,var] == "Serious problem")] <- 3
  df_temp[,var] <- as.numeric(df_temp[,var])
}

for (var in no_prob_serious_3){
  df_temp[, var] <- as.character(df_temp[,var])
  df_temp[,var][which(df_temp[,var] == "Not a problem")] <- 0
  df_temp[,var][which(df_temp[,var] == "Minor problem")] <- 1
  df_temp[,var][which(df_temp[,var] == "Moderate to serious problem")] <- 2.5
  df_temp[,var] <- as.numeric(df_temp[,var])
}

prob_fit <- lm(X2PROBLEM ~ A2HEALTH + A2UNPREP + A2PRNTINV + A2APATHY + A2TARDY, 
               data = df_temp)
summary(prob_fit)


df_prob <- df_temp[,c("A1HEALTH", "A1UNPREP", "A1PRNTINV", "A1APATHY", "A1TARDY")]
names(df_prob) <- c("A2HEALTH", "A2UNPREP", "A2PRNTINV", "A2APATHY", "A2TARDY")
X1PROBLEM <- predict(prob_fit, newdata = df_prob)
df$X1PROBLEM <- X1PROBLEM 

# July 05: all data has been cleaned.
# But there is still some missing value in most continuous variable
# ---------------------------------------------------------------------
#                                     July 05
#                                 Data Imputation
# ---------------------------------------------------------------------
# 2023.10.30: change the S1M8 and S1S8 into numeric weighted score!!
# Since it is hard to tell the difficulty level of science, here I dropped the science
table(df$S1M8)
table(df$S1M8GRADE)
S1M8GRADE_num <- rep(NA, nrow(df))
S1M8GRADE_num[which(df$S1M8GRADE == "A")] <- 4
S1M8GRADE_num[which(df$S1M8GRADE == "B")] <- 3
S1M8GRADE_num[which(df$S1M8GRADE == "C")] <- 2
S1M8GRADE_num[which(df$S1M8GRADE == "D")] <- 1
S1M8GRADE_num[which(df$S1M8GRADE %in% c("Below D","Class was not graded"))] <- 0

S1M8_weight <- rep(NA, nrow(df))
S1M8_weight[which(df$S1M8 %in% c("Math 8","Advanced or Honors Math 8","Other math course"))] <- 1
S1M8_weight[which(df$S1M8 %in% c("Pre-algebra"))] <- 1.2
S1M8_weight[which(df$S1M8 %in% c("Algebra I including IA and IB","Integrated Math"))] <- 1.4
S1M8_weight[which(df$S1M8 %in% c("Geometry"))] <- 1.6
S1M8_weight[which(df$S1M8 %in% c("Algebra II or Trigonometry"))] <- 1.9

S1MSCORE <- S1M8_weight*S1M8GRADE_num
summary(S1MSCORE)
df$S1MSCORE <- S1MSCORE

# 2023.10.30: Before data imputation, drop the unnecessary variables
dropped_var <- c("X1FAMINCOME","X1PAR1EDU","X1PAR2EDU","X1PAR1OCC2","X1PAR2OCC2", # SES-related
                 "X1HHNUMBER","S1NOHWDN", "S1NOPAPER", "S1NOBOOKS", "S1LATE",
                 "S1SAFE","S1PROUD","S1TALKPROB","S1SCHWASTE","S1GOODGRADES", # school belonging
                 "S1FRNDGRADES","S1FRNDSCHOOL","S1FRNDCLASS","S1FRNDCLG","S1MPERSON1", "S1MPERSON2",
                 "S1MUSELIFE", "S1MUSECLG", "S1MUSEJOB","S1MTESTS","S1MTEXTBOOK","S1MSKILLS","S1MASSEXCL",
                 "S1MENJOYING", "S1MENJOYS", "S1MWASTE", "S1MBORING","S1SPERSON1", "S1SPERSON2",
                 "S1SUSELIFE", "S1SUSECLG", "S1SUSEJOB",
                 "S1STESTS","S1STEXTBOOK","S1SSKILLS","S1SASSEXCL",
                 "S1SENJOYING","S1SWASTE","S1SBORING","S1FAVSUBJ","S1LEASTSUBJ","S1SENJOYS",
                 "A1RESOURCES","A1HEALTH","A1UNPREP","A1PRNTINV","A1APATHY","A1TCHRABSENT","A1TARDY",
                 "A2RESOURCES","A2HEALTH","A2UNPREP","A2PRNTINV","A2APATHY","A2TARDY",
                 "A2OTHERBULLY","A2CYBERBULLY","S1M8","S1MGRADE","S1S8","S1S8GRADE")
df <- df[,-which(names(df) %in% dropped_var)]


# check the missing value distribution
prop.table(table(df$X1RACE))
prop.table(table(df$X1RACE[which(is.na(df$X1SES)==T)]))
# try to use the mean value to impute missing values in all continuous variables
summary(df$X1SES)

# extract all continuous variable
# UPDATE: JULY 12,2023  I change X1PROBLEM to be imputed rather than X2PROBELM
# 2023.10.22 try to use the PMM method to impute the data
# First, change all the Missing category into NA
# 2023.10.22 using a for loop to change all the 'Missing' value into NA
if (keep_nr == T){
  for (i in 1:ncol(df)) {
    df[,i][which(df[,i] == "Missing")] <- NA
  }
} else {
  for (i in 1:ncol(df)) {
    df[,i][which(df[,i] %in% c("Missing", "Non-response"))] <- NA
  }
}

# ----------------------------------------------------
# 2024.01.01 Updation
# ---------------------------------------------------

library(rsample)
library(caret)
library(mice)
# Assuming 'X' is a dataframe of predictors and 'y' is a vector of outcomes
set.seed(42) # Equivalent to random_state in Python
split <- initial_split(df, prop = 0.8, strata = "X4EVERDROP") 

# Extract the training and testing sets
train_data <- training(split)
test_data <- testing(split)

# do imputation on both train and test data
table(train_data$S1M8GRADE, useNA = "always")
summary(train_data$S1MSCORE)


# note, should remove the sch_id first
na_imputation <- function(df_tempt, mi_num = 1){
  # remove the STU_ID
  df_tempt <- df_tempt[, names(df_tempt)!="STU_ID"]
  
  # add the missing proportion
  df_tempt$na_prop <- apply(df_tempt,1, function(row){
    sum(is.na(row))/ length(row)})
  
  # next do school_level imputation using pmm method
  # 2023.10.21. Note this step I drop the school id column
  ## 2023.10.23. The overall PMM running time is from 30-40 mins
  time_0 <- Sys.time()
  pmm_output <- mice(df_tempt, m=mi_num, method = "pmm", print=F)
  time_1 <- Sys.time()
  print(time_1-time_0)
  return(complete(pmm_output))
}

train_data_imp <- na_imputation(train_data, mi_num=5)
test_data_imp <- na_imputation(test_data, mi_num=5)

table(train_data$X1RACE, useNA = "always")
table(train_data_imp$X1RACE, useNA = "always")

# add the STU_ID back to the imputed dataset
train_data_imp$STU_ID <- train_data$STU_ID
test_data_imp$STU_ID <- test_data$STU_ID

# check the balance
summary(train_data_imp$S1MSCORE)
cor(train_data_imp$X1PROBLEM, train_data_imp$X1SCHOOLCLI)

summary(train_data$S1MSCORE)
cor(train_data$X1PROBLEM, train_data$X1SCHOOLCLI, use = "complete.obs")



# note if save it as .csv, the categorical variables level information will be lost!!
# so always save it as a rdata
if (keep_nr == T){
  # save(df,file = "~/Desktop/PhD_Learning/Independent Studies/dropout/01_data/02_processed_data/15_df_1022_unimputed_pmm_nr_kept.rdata")
  # save(df_imp, file = "~/Desktop/PhD_Learning/Independent Studies/dropout/01_data/02_processed_data/14_df_1022_imputed_pmm_nr_kept.rdata")
  write.csv(df, file = "../01_data/02_processed_data/19_df_1022_unimputed_pmm_nr_kept.csv",
            row.names = F)
  write.csv(df_imp, file = "../01_data/02_processed_data/18_df_1022_imputed_pmm_nr_kept.csv",
            row.names = F)
}else{
  # save(df,file = "~/Desktop/PhD_Learning/Independent Studies/dropout/01_data/02_processed_data/17_df_1023_unimputed_pmm_nr_dropped.rdata")
  # save(df_imp, file = "~/Desktop/PhD_Learning/Independent Studies/dropout/01_data/02_processed_data/16_df_1023_imputed_pmm_nr_dropped.rdata")
  write.csv(train_data, file = "../01_data/02_processed_data/24_train_0101_unimputed_pmm_nr_dropped.csv",
            row.names = F)
  write.csv(test_data, file = "../01_data/02_processed_data/24_test_0101_unimputed_pmm_nr_dropped.csv",
            row.names = F)
  write.csv(train_data_imp, file = "../01_data/02_processed_data/25_train_0101_imputed_pmm_nr_dropped.csv",
            row.names = F)
  write.csv(test_data_imp, file = "../01_data/02_processed_data/25_test_0101_imputed_pmm_nr_dropped.csv",
            row.names = F)
}



