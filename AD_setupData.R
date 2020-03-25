# Install packages via CRAN first if required before running the code below
# Load libraries --------------------------------------------------------

library('zoo')
library('tseries')
library('lubridate')
library('tidyverse')
library('reshape2')
library('recipes')



# Set input and output directories

localdir <- T

if (localdir == T) {
  dirname <- "~/Dropbox/" # to setup working directory in my computer
  
} else if (localdir == F) {
  dirname <- "/home/user/j/jrperez/"
  .libPaths(c("/zfs/home/user/j/jrperez/lib/R/3.6.0", .libPaths()))
}



# Set the locality and variable to be analyzed --------------------------------------------------------

site <- c("sandy", "pioneer", "adour", "adourEnv", "garonne", "marseille")[1]
selectVars <- c("Tur", "Cond", "Level", "Ammo", "Ph")[1]

# Set sliding window (in days) according to each locality

if (site == "sandy")  sld_window.seq <- 1 / 2^(4:0)
if (site == "pioneer") sld_window.seq <- 1 / 2^(5:0) 
if (site == "adour" | site == "adourEnv" | site == "marseille") sld_window.seq <- 1 / c(2^(7:2),1) # 6h and then 24h
if (site == "garonne") sld_window.seq <- 1 / c(2^(7:2),1)


# Read-in data and add transformed columns ------------------------------------------------------------



# Read Sandy and Pioneer --------------------------------------------------

# DES data: 12/3/2017 to 12/3/2018 inclusive
if (site == "sandy" | site == "pioneer") {
  setwd(paste0(dirname,"PROJECT_POSTDOC_KERRIE/DATA_Australia"))
  data <- read.csv(paste0("data_",site,".csv"), header = TRUE) %>%
    mutate(Timestamp = dmy_hm(Timestamp)) %>%
    na.omit() %>%
    mutate(
      logTur = log(Tur),
      logCond = log(Cond),
      logLevel = log(Level)
    )
  
  n_tlag <- as.numeric(diff(zoo(data$Timestamp)))
  #which(!is.finite(data$logCond))
  data$logCond[data$logCond == -Inf | is.nan(data$logCond)] <- 0
  
  # https://stackoverflow.com/questions/52470591/creating-a-ts-time-series-with-missing-values-from-a-data-frame
  # https://blog.earo.me/2018/12/20/reintro-tsibble/
  
  # Group anomalies types into classes --------------------------------------
  # NOTE: make sure you run the data reading section again if you changed the PerfClass (on line 14)
  # Those anomaly classes depend on definitions given in Leight (2019)
  
  Class1 <- c("A", "D", "I", "J")
  Class2 <- c("F", "G", "K")
  Class3 <- c("B", "H", "E", "L", "C")
  ClassAll <- union_all(Class1, Class2, Class3)
  PerfClass = 5 # 1 for class1, 2 for class2, 3 for class3, other for combined classes
  
  sort_cases <- function(x){
    
    Class1 <- c("A", "D", "I", "J")
    Class2 <- c("F", "G", "K")
    Class3 <- c("B", "H", "E", "L", "C")
    ClassAll <- union_all(Class1, Class2, Class3)
    
    if (!(x %in% 1:3)) {
      return(ClassAll)
    }
    if (x == 1) {
      return(Class1)
    }
    if (x == 2) {
      return(Class2)
    }
    if (x == 3) {
      return(Class3)
    }
    
  }
  
  Class <- sort_cases(PerfClass)
  
  
  expand <- F
  
  if (expand == T) {
    threshold_tlag <- 60*30 # I assume there is an anomaly when > 60 minutes
    broken_tlag <- which(n_tlag > (median(n_tlag) + threshold_tlag)) 
    
    library(tsibble)
    
    data_expand <- data[1:(broken_tlag[1] - 1),]
    
    for (i in 1:(length(broken_tlag) - 1)) {
      data_expand.tmp <- data[(broken_tlag[i]):(broken_tlag[i] + 1),]
      n_tlag[(broken_tlag[i]):(broken_tlag[i] + 1)]
      
      data_expand2.tmp <- data_expand.tmp %>% complete(Timestamp = seq(min(Timestamp), max(Timestamp), by = threshold_tlag), #median(n_tlag)
                                                       fill = list(value = NA))
      #as_tsibble(index = Timestamp)
      data_expand <- bind_rows(data_expand, data_expand2.tmp)
    }
    data <- bind_rows(data_expand, data[(last(broken_tlag) + 1):nrow(data),])
  }
  
  # There are negative values in sandy [2158 in sandy in Level] and Pioneer [2158 in sandy in Cond]
  data[which(is.na(data$logCond)),]; data[which(is.na(data$logTur)),]; data[which(is.na(data$logLevel)),]
  data[which(is.nan(data$logCond)),]; data[which(is.nan(data$logTur)),]; data[which(is.nan(data$logLevel)),]
  
  data$Tur <- ifelse(is.na(data$Tur), 0, data$Tur)
  data$Cond <- ifelse(is.na(data$Cond), 0, data$Cond)
  data$Level <- ifelse(is.nan(data$Level), 0, data$Level)
  
  # For negative values (NaN)
  data$logTur <- ifelse(is.nan(data$logTur), NA, data$logTur)
  data$logCond <- ifelse(is.nan(data$logCond), NA, data$logCond)
  data$logLevel <- ifelse(is.nan(data$logLevel), NA, data$logLevel)
  
  data <- data[which(is.finite(data$logTur) & is.finite(data$logCond)),] # & is.finite(data$logLevel))
  
  # Impute last finite value
  #data$logCond <- na.locf(data$logCond)
  #data$logTur <- na.locf(data$logTur)
  #data$logLevel <- na.locf(data$logLevel)
  
  data$label_Cond <- ifelse(as.character(data$type_Cond) == 'K' | as.character(data$type_Cond) == 'F', 0, data$label_Cond)
  data$label_Tur <- ifelse(as.character(data$type_Tur) == 'K' | as.character(data$type_Cond) == 'F', 0, data$label_Tur)
  data$label_Level <- ifelse(as.character(data$type_Level) == 'K'  | as.character(data$type_Cond) == 'F', 0, data$type_Level)
  
  data$type_Cond <- recode(data$type_Cond, 'K' = "0", 'F' = "0", .missing = NULL)
  data$type_Tur <-  recode(data$type_Tur, 'K' = "0", 'F' = "0", .missing = NULL)
  data$type_Level <- recode(data$type_Level, 'K' = "0", 'F' = "0", .missing = NULL)
  
  #data$type_Cond <- tidyr::replace_na(data$type_Cond, 'K')
  #data$type_Tur <- tidyr::replace_na(data$type_Tur, 'K')
  #data$type_Level <- tidyr::replace_na(data$type_Level, 'K')
  
  table(data$type_Cond) / length(data$type_Cond); table(data$type_Tur) / length(data$type_Tur); table(data$type_Level) / length(data$type_Level)
  table(data$type_Cond); table(data$type_Tur); table(data$type_Level)
  
  # Check everything is ok
  data[which(is.na(data$logCond)),]; data[which(is.na(data$logTur)),]
  data[which(is.nan(data$logCond)),]; data[which(is.nan(data$logTur)),]
  data[which(is.nan(data$logLevel)),]; data[which(is.nan(data$logLevel)),]
  
  RatiosClass <- setNames(data.frame(matrix(ncol = length(unlist(ClassAll)), nrow = 0)), unlist(ClassAll))
  
  ChosenVars <- c("Timestamp", "Tur", "Cond") 
  
  data$label_Cond[!data$type_Cond %in% unlist(ClassAll)] <- 0
  data$type_Cond[!data$type_Cond %in% unlist(ClassAll)] <- 0
  data$label_Tur[!data$type_Tur %in% unlist(ClassAll)] <- 0
  data$type_Tur[!data$type_Tur %in% unlist(ClassAll)] <- 0
  
  # To select only those anomalies of Class 3
  data$label_Cond3 <- data$label_Cond
  data$label_Tur3 <- data$label_Tur
  data$label_Cond3[!data$type_Tur %in% unlist(Class3)] <- 0
  data$label_Tur3[!data$type_Tur %in% unlist(Class3)] <- 0
  
  data_training <- data %>% 
    dplyr::filter(label_Cond == 0 & label_Tur == 0)
  
  data_test <- data
  
  RNN_trainingdata <- data_training %>% subset(select = c(Timestamp, logTur, logCond, logLevel, label_Tur, label_Cond, label_Level, label_Tur3, label_Cond3, type_Tur, type_Cond, type_Level))
  
  RNN_testdata <- data %>% subset(select = c(Timestamp, logTur, logCond, logLevel, label_Tur, label_Cond, label_Level, label_Tur3, label_Cond3, type_Tur, type_Cond, type_Level))
  
}



# Read Adour ------------------------------------------------------------


if (site == "adour" | site == "adourEnv") {
  setwd(paste0(dirname,"PROJECT_POSTDOC_KERRIE/DATA_France"))
  data_new <- data.frame(read.table(file = "FullAnomalyAdour.csv", header = T, sep = "," ,dec = "."))
  data_new <- data_new[-1,]
  
  data <- data.frame(Timestamp = with(data_new, as.POSIXct(paste(paste0(data_new$DAY,"/",data_new$MONTH,"/",data_new$YEAR),paste0(data_new$HOUR,":",data_new$MINUTE,":",data_new$SECOND)), format = "%d/%m/%Y %H:%M:%S")),
                     Cond = data_new$COND/1000, Tur = data_new$TURB, #Sal = salinity(data_new$Cond, data_new$Tem, data_new$Pres), 
                     logCond = log(data_new$COND/1000), logTur = log(data_new$TURB))
  
  if (site == "adour") {
    data <- data.frame(data,
                       label_Cond = data_new$SENSOR_COND, label_Tur = data_new$SENSOR_TURB,
                       type_Cond = ifelse(as.character(data_new$LABEL_COND) == "N1", "0", as.character(data_new$LABEL_COND)), type_Tur = data_new$LABEL_TURB)
    
  } else if (site == "adourEnv") {
    data <- data.frame(data,
                       label_Cond = data_new$NATURE_COND, label_Tur = data_new$NATURE_TURB,
                       type_Cond = data_new$NATURE_LABEL_COND, type_Tur = data_new$NATURE_LABEL_TURB)
  }  
  
  n_tlag <- as.numeric(diff(zoo(data$Timestamp)))
  cases_lagCond <- na.action(na.omit(data$logCond))
  cases_lagTur <- na.action(na.omit(data$logTur))
  match(cases_lagTur,cases_lagCond)
  cases_lag <- unique(cases_lagTur,cases_lagCond)
  
  # Group anomalies types into classes --------------------------------------
  # NOTE: make sure you run the data reading section again if you changed the PerfClass (on line 14)
  # Those anomaly classes depend on definitions given in Leight (2019)
  
  Class1 <- c("A", "D", "I", "J")
  Class2 <- c("F", "G", "K")
  Class3 <- c("B", "H", "E", "L", "C")
  ClassAll <- union_all(Class1, Class2, Class3)
  PerfClass = 5 # 1 for class1, 2 for class2, 3 for class3, other for combined classes
  
  sort_cases <- function(x){
    
    Class1 <- c("A", "D", "I", "J")
    Class2 <- c("F", "G", "K")
    Class3 <- c("B", "H", "E", "L", "C")
    ClassAll <- union_all(Class1, Class2, Class3)
    
    if (!(x %in% 1:3)) {
      return(ClassAll)
    }
    if (x == 1) {
      return(Class1)
    }
    if (x == 2) {
      return(Class2)
    }
    if (x == 3) {
      return(Class3)
    }
    
  }
  
  Class <- sort_cases(PerfClass)
  
  
  cases_lag2 <- NULL
  for (i in 1:length(cases_lag) - 1) cases_lag2 <- c(cases_lag2, n_tlag[cases_lag[i + 1]] - n_tlag[cases_lag[i]])
  cases_lag3 <- c(cases_lag[1], cases_lag[cases_lag2 > 1])
  
  if (site == "adour") {
    data$label_Tur[cases_lagTur[is.na(data$logTur[cases_lagTur])]] <- 1
    data$type_Tur[cases_lagTur[is.na(data$logTur[cases_lagTur])]] <- "K"
    data$type_Tur[cases_lagTur[is.finite(cases_lagTur[data$Tur[cases_lagTur] < 0])]] <- "F"
    data$type_Tur <- recode(data$type_Tur, 'E – H' = "H", .missing = NULL)
    
    data$label_Cond[cases_lagCond[is.na(data$logCond[cases_lagCond])]] <- 1
    data$type_Cond[cases_lagCond[is.na(data$logCond[cases_lagCond])]] <- "K"
    data$type_Cond[cases_lagCond[is.finite(cases_lagCond[data$Cond[cases_lagCond] < 0])]] <- "F"
    
  } else if (site == "adourEnv") {
    data <- na.omit(data)
  }
  
  data <- data[which((!is.na(data$logTur) | !is.nan(data$logTur)) & (!is.na(data$logCond) | !is.nan(data$logCond))),]
  data <- data[which(!is.na(data$logTur) & !is.na(data$logCond) & is.finite(data$logTur) & is.finite(data$logCond)),]
  
  table(data$type_Cond) / length(data$type_Cond); table(data$type_Tur) / length(data$type_Tur)
  table(data$type_Cond); table(data$type_Tur)
  
  # Check everything is ok
  data[which(is.na(data$logCond)),]; data[which(is.na(data$logTur)),]
  data[which(is.nan(data$logCond)),]; data[which(is.nan(data$logTur)),]
  
  RatiosClass <- setNames(data.frame(matrix(ncol = length(unlist(ClassAll)), nrow = 0)), unlist(ClassAll))
  
  ChosenVars <- c("Timestamp", "Tur", "Cond") 
  
  data$label_Cond[!data$type_Cond %in% unlist(ClassAll)] <- 0
  data$type_Cond[!data$type_Cond %in% unlist(ClassAll)] <- 0
  data$label_Tur[!data$type_Tur %in% unlist(ClassAll)] <- 0
  data$type_Tur[!data$type_Tur %in% unlist(ClassAll)] <- 0
  
  # To select only those anomalies of Class 3
  data$label_Cond3 <- data$label_Cond
  data$label_Tur3 <- data$label_Tur
  data$label_Cond3[!data$type_Tur %in% unlist(Class3)] <- 0
  data$label_Tur3[!data$type_Tur %in% unlist(Class3)] <- 0
  
  data_training <- data %>% 
    dplyr::filter(label_Cond == 0 & label_Tur == 0)
  
  # remove last drifting values
  #data <- (data[data$Timestamp <= as.POSIXct("2019-07-11 10:45:32 CET"),])
  
  data_test <- data
  
  RNN_trainingdata <- data_training %>% subset(select = c(Timestamp, logTur, logCond, label_Tur, label_Cond, label_Tur3, label_Cond3, type_Tur, type_Cond))
  
  RNN_testdata <- data %>% subset(select = c(Timestamp, logTur, logCond, label_Tur, label_Cond, label_Tur3, label_Cond3, type_Tur, type_Cond))
  
}



# Read Garonne ------------------------------------------------------------

if (site == "garonne") {
  setwd(paste0(dirname,"PROJECT_POSTDOC_KERRIE/DATA_France"))
  data <- data.frame(read.table(file = "dataVerdonTagged.txt", header = F, sep = ";" ,dec = "."))
  data <- data.frame(Timestamp = with(data, as.POSIXct(data$V1, format = "%d/%m/%y %H:%M")),
                     Level = data$V2, 
                     logLevel = log(data$V2 + 1),
                     label_Level = data$V3,
                     type_Level = 0)
  data$type_Level <- as.character(data$type_Level)
  
  LastYr <- function(x) seq(x, length = 2, by = "+1 year")[2]
  toPOSIXct <- function(x) as.POSIXct(x, origin = "1970-01-01")
  
  # retain the first year
  #data <- data[which(data$Timestam == min(data$Timestamp)) : which(data$Timestam == toPOSIXct(sapply(min(data$Timestamp), LastYr))), ]
  
  # Group anomalies types into classes --------------------------------------
  # NOTE: make sure you run the data reading section again if you changed the PerfClass (on line 14)
  # Those anomaly classes depend on definitions given in Leight (2019)
  
  Class1 <- c("A", "D", "I", "J")
  Class2 <- c("F", "G", "K")
  Class3 <- c("B", "H", "E", "L", "C")
  ClassAll <- union_all(Class1, Class2, Class3)
  PerfClass = 5 # 1 for class1, 2 for class2, 3 for class3, other for combined classes
  
  sort_cases <- function(x){
    
    Class1 <- c("A", "D", "I", "J")
    Class2 <- c("F", "G", "K")
    Class3 <- c("B", "H", "E", "L", "C")
    ClassAll <- union_all(Class1, Class2, Class3)
    
    if (!(x %in% 1:3)) {
      return(ClassAll)
    }
    if (x == 1) {
      return(Class1)
    }
    if (x == 2) {
      return(Class2)
    }
    if (x == 3) {
      return(Class3)
    }
    
  }
  
  Class <- sort_cases(PerfClass)
  
  
  n_tlag <- as.numeric(diff(zoo(data$Timestamp)))
  cases_lagLevel <- na.action(na.omit(data$logLevel))
  
  data[which(is.na(data$logLevel)),]; data[which(data$label_Level == 1),]
  
  data$type_Level[which(data$label_Level == 1)] <- "D"
  data$label_Level[cases_lagLevel[is.na(data$logLevel[cases_lagLevel])]] <- 1
  data$type_Level[cases_lagLevel[is.na(data$logLevel[cases_lagLevel])]] <- "K"
  data$label_Level[which(data$Level > 8 | data$Level < -1)] <- 1
  data$type_Level[which(data$Level > 8 | data$Level < -1)] <- "A"
  
  data$logLevel[is.na(data$logLevel)] <- -5
  data$logLevel[is.infinite(data$logLevel)] <- -5
  
  table(data$label_Level) / length(data$label_Level)
  table(data$type_Level)
  
  # check everything is ok
  data[which(is.na(data$logLevel)),]
  data[which(is.nan(data$logLevel)),]
  
  RatiosClass <- setNames(data.frame(matrix(ncol = length(unlist(ClassAll)), nrow = 0)), unlist(ClassAll))
  
  ChosenVars <- c("Timestamp", "Level")
  
  data$label_Level[!data$type_Level %in% unlist(ClassAll)] <- 0
  data$type_Level[!data$type_Level %in% unlist(ClassAll)] <- 0
  
  # Only for six-month period (with enought anomalies)
  data <- (data[data$Timestamp >= as.POSIXct("2016-01-01 00:00:00 CET"),])
  data <- (data[data$Timestamp <= as.POSIXct("2016-07-01 00:00:00 CET"),])
  
  # To select only those anomalies of Class 3
  data$label_Level3 <- data$label_Level
  data$label_Level3[!data$type_Level %in% unlist(Class3)] <- 0
  
  data_training <- data %>% filter(label_Level == 0)
  data_test <- data
  
  RNN_trainingdata <- data_training %>% subset(select = c(Timestamp, logLevel, label_Level, label_Level3, type_Level))
  
  RNN_testdata <- data %>% subset(select = c(Timestamp, logLevel, label_Level, label_Level3, type_Level))
  
}



# Read Marseille ----------------------------------------------------------

if (site == "marseille") {
  setwd(paste0(dirname,"PROJECT_POSTDOC_KERRIE/DATA_France"))
  data_new <- data.frame(read.table(file = "190627ALL_Vfinal.csv", header = F, skip = 1, sep = ";" ,dec = ","))
  data <- data.frame(Timestamp = with(data_new, as.POSIXct(data_new$V1, format = "%d/%m/%Y %H:%M")),
                     Tur = data_new$V2, Ammo = data_new$V3, Oxy = data_new$V4, Cond = data_new$V5, pH = data_new$V6,
                     logTur = log(data_new$V2), logCond = log(data_new$V5), logAmmo = log(data_new$V3), logOxy = log(data_new$V4), logpH = log(data_new$V6),
                     label_Tur = ifelse(grepl("Turb", data_new$V9, ignore.case = TRUE, perl = TRUE) | (!grepl(" ", data_new$V9, ignore.case = TRUE, perl = TRUE) & !is.na(data_new$V7)), 1, 0),
                     label_Cond = ifelse(grepl("Cond", data_new$V9, ignore.case = TRUE, perl = TRUE) | (!grepl(" ", data_new$V9, ignore.case = TRUE, perl = TRUE) & !is.na(data_new$V7)), 1, 0),
                     label_Oxy = ifelse(grepl("O2", data_new$V9, ignore.case = TRUE, perl = TRUE) | (!grepl(" ", data_new$V9, ignore.case = TRUE, perl = TRUE) & !is.na(data_new$V7)), 1, 0),
                     label_Amm = ifelse(grepl("Amm", data_new$V9, ignore.case = TRUE, perl = TRUE) | (!grepl(" ", data_new$V9, ignore.case = TRUE, perl = TRUE) & !is.na(data_new$V7)), 1, 0),
                     label_pH = ifelse(grepl("ph", data_new$V9, ignore.case = TRUE, perl = TRUE) | (!grepl(" ", data_new$V9, ignore.case = TRUE, perl = TRUE) & !is.na(data_new$V7)), 1, 0))
  
  
  # Group anomalies types into classes --------------------------------------
  # Here there are defined another set of anomaly types; better have a look when plotting data
  # I think those anomalies are more related to environmental events 
  # D ==> Sensor drift
  # DC ==> Sensor failure
  # E ==> Quality degradation event
  # M ==> Maintenance
  # RP ==> Pump restart
  
  ClassAll <- union_all("D", "DC", "E", "M", "RP")
  RatiosClass <- setNames(data.frame(matrix(ncol = length(unlist(ClassAll)), nrow = 0)), unlist(ClassAll))
  
  data$type_Tur <- as.factor(as.character(ifelse(grepl("RP", data_new$V8, ignore.case = FALSE), "RP", 
                                                 ifelse(grepl("M", data_new$V8, ignore.case = FALSE), "M",
                                                        ifelse(grepl("E", data_new$V8, ignore.case = FALSE), "E",
                                                               ifelse(grepl("DC", data_new$V8, ignore.case = FALSE), "DC",
                                                                      ifelse(grepl("D", data_new$V8, ignore.case = FALSE), "D", 0)))))))
  data$type_Cond <- data$type_Tur #copy the same type of anomalies
  
  n_tlag <- as.numeric(diff(zoo(data$Timestamp)))
  cases_lagTur <- na.action(na.omit(data$logTur))
  
  data[which(is.na(data$logTur)),]; data[which(data$label_Tur == 1),]
  
  data <- data[which((!is.na(data$logTur) & !is.nan(data$logTur))),]
  
  # To select only those anomalies of Class 3????
  data$label_Cond3 <- data$label_Cond
  data$label_Tur3 <- data$label_Tur
  data$label_Cond3[!data$type_Tur %in% unlist(Class3)] <- 0
  data$label_Tur3[!data$type_Tur %in% unlist(Class3)] <- 0
  
  # Check everything is ok
  data[which(is.na(data$logTur)),]
  
  ChosenVars <- c("Timestamp", "Tur", "Cond")
  
  #plot(data$Timestamp, data$logTur, col = ifelse(data$label_Tur == 1, "red", "black"))
  #plot(data$Timestamp, data$logCond, col = ifelse(data$label_Cond == 1, "red", "black"))
  
  data_training <- data %>% 
    dplyr::filter(label_Cond == 0 & label_Tur == 0)
  
  # Only for one-year period (with enought anomalies)
  data <- (data[data$Timestamp >= as.POSIXct("2018-01-01 00:00:00 CET"),])
  data <- (data[data$Timestamp <= as.POSIXct("2019-01-01 00:00:00 CET"),])
  
  data_test <- data
  
  RNN_trainingdata <- data_training %>% subset(select = c(Timestamp, logTur, logCond, label_Tur, label_Cond,  label_Tur3, label_Cond3, type_Tur, type_Cond))
  
  RNN_testdata <- data %>% subset(select = c(Timestamp, logTur, logCond, label_Tur, label_Cond,  label_Tur3, label_Cond3, type_Cond))
}

# Convert a sequence of days to number of colums, based on the time-spam [median(n_tlag)] of each dataframe

sld_window.seq <- round(sld_window.seq * (24 / (median(n_tlag, na.rm = T) / 3600)))



# Save data ---------------------------------------------------------------

data_vars <- RNN_testdata[paste0("log", selectVars)][[1]]
data_label <- RNN_testdata[paste0("label_",selectVars)][[1]]
data_type <- RNN_testdata[paste0("type_",selectVars)][[1]]

setwd(paste0(dirname,"PROJECT_POSTDOC_KERRIE/AD_ANN"))
save(data_vars, data_label, data_type, sld_window.seq, 
     RatiosClass, ClassAll, 
     file = paste0(site, selectVars, "_data.RData"), version = 2)