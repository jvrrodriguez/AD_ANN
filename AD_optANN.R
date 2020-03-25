# Install packages via CRAN first if required before running the code below
# Load libraries --------------------------------------------------------

library('tfruns')
library('tfestimators')
library('mlrMBO')


# Load data for each locality ---------------------------------------------

require("sandyTur_data.RData")
data_name <- "sandyTur"



# Load user-defined functions for AD in time-series data ------------------

# functions for AD using ANN, bot for supervised (sRNN_fit) and semi-supervised classification (uRNN_fit)

source("AD_funsANN.R")
source("AD_performance.R")


# Example of semi-supervised classification -------------------------------

data_flags <- list(n_features = 1, n_split = 2, p_split = 0.5)


# Generate a design of the variability of starting values --------------------------

set.seed(123)

ParamSet <- makeParamSet(
  makeIntegerParam("n_layers", lower = 1, upper = 3),
  makeDiscreteParam("n_units", values = 2^(1:6)),
  makeDiscreteParam("dropout", values = c(0.001, seq(0.1, 0.9, 0.2))),
  makeDiscreteParam("activation", values = c("relu","tanh","sigmoid", "hard_sigmoid", "linear")),
  makeIntegerParam("weight_constraint", lower = 1, upper = 5),
  
  makeDiscreteParam("optimizer_type", values = c("sgd", "RMSProp", "adagrad", "adam", "adamax")),
  makeDiscreteParam("momentum", values = c(0.001, seq(0.1, 0.9, 0.2))),
  makeDiscreteParam("learning_rate", values = c(0.001, 0.01, 0.1, 0.2, 0.3)),
  makeDiscreteParam("kernel_initializer", values = c("uniform", "normal", "zero", "lecun_uniform", "lecun_normal")),
  makeDiscreteParam("batch_size", values = 4^(1:5)), 
  
  makeDiscreteParam("sld_window", values = sld_window.seq),
  makeDiscreteParam("thrsld_class", values = c(seq(0.4, 0.9, 0.1), 0.95, 0.999)))

# Proceed with a sufficient number of repetitions (otherwise errors will come out)

RNN_reps <- list(init = 250, opt = 250)

fit_scores <- c("acc", "b_acc", "MCC")[2]


# Start with random search of hyperparameter values -----------------------

uRNN_rnd.reps <- RNN_reps$init 
ParamDesign <- generateDesign(n = uRNN_rnd.reps, par.set = ParamSet)
uRNN_scores <- data.frame()

time.taken <- array(NA, dim = uRNN_rnd.reps)
start.time <- Sys.time()

for (j in 1:uRNN_rnd.reps) {
  cat("Running starting values // Model: uRNN // Data:", data_name, "// Complete:", j/dim(ParamDesign)[1]*100, "%"); print(j)
  
  RNN_flags <- ParamDesign[j,]
  uRNN_rnd <- uRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste0(data_name, "uRnd"))
  uRNN_scores <- rbind(uRNN_scores, uRNN_rnd$scores)
  ParamDesign$y[[j]] <- uRNN_rnd$scores[[fit_scores]]
  
  write.csv(ParamDesign, file = paste0(data_name, "_uBRndPoints.csv"))
  write.csv(uRNN_scores, file = paste0(data_name, "_uBRndscores.csv"))
}

time.taken <- data.frame(RndPoints = Sys.time() - start.time)


# Now proceed optimization of hyperparameters ----------------------------------

# Generate an object to save iterations of hyperparameter optimization

uRNN_ctrl = makeMBOControl()
uRNN_ctrl = setMBOControlInfill(uRNN_ctrl, crit = crit.ei)

uRNN_opt.state = initSMBO(
  par.set = ParamSet,
  design = ParamDesign,
  control = uRNN_ctrl,
  minimize = TRUE,
  noisy = FALSE)

uRNN_opt.reps <- RNN_reps$opt
uRNN_proposePoints <- NULL
uRNN_scores <- data.frame()

start.time <- Sys.time()

for (j in 1:uRNN_opt.reps) {
  cat("Optimizing values // Model: uRNN // Data:", data_name, "// Complete:", j/uRNN_opt.reps*100, "%"); print(j)
  
  RNN_flags = proposePoints(uRNN_opt.state)$prop.points
  uRNN_opt = uRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste0(data_name, "_uBOpt"))
  uRNN_scores <- rbind(uRNN_scores, uRNN_opt$scores)
  y <- uRNN_opt$scores[[fit_scores]]
  uRNN_proposePoints <- rbind(uRNN_proposePoints, cbind(j, RNN_flags, y))
  
  write.csv(uRNN_scores, file = paste0(data_name, "_uBOptscores.csv"))
  write.csv(uRNN_proposePoints, file = paste0(data_name, "_uBOptproposePoints.csv"))
  
  updateSMBO(uRNN_opt.state, 
             x = RNN_flags,
             y = y)
}

time.taken <- data.frame(time.taken, OptPoints = Sys.time() - start.time)

proposePoints(uRNN_opt.state)
finalizeSMBO(uRNN_opt.state)
uRNN_opt.state$opt.result$mbo.result$x
uRNN_opt.state$opt.result$mbo.result$y
plot(finalizeSMBO(uRNN_opt.state))

uRNN_proposePoints <- read.csv(file = paste0(data_name, "_uBOptproposePoints.csv"))
ParamBest <- uRNN_proposePoints[order(uRNN_proposePoints$y, decreasing = T),][1,][3:14]

modelBest <- load_model_hdf5(paste0(data_name, "_uBOptBest.h5"))

for (cc in unlist(ClassAll)) {
  RatiosClass[1,cc] <- round(sum(data_label[data_type == cc]))
  RatiosClass[2,cc] <- round(sum(uRNN_best$classify[data_type == cc], na.rm = T))
  RatiosClass[3,cc] <- RatiosClass[2,cc] / RatiosClass[1,cc]
}

save(ParamBest, modelBest, uRNN_opt.state, uRNN_best, uRNN_proposePoints, uRNN_scores,
     RNN_trainingdata, RNN_testdata,
     RatiosClass, time.taken,
     file = paste0(data_name, "_uBOptBest.RData"), version = 2)



# Example of supervised classification ------------------------------------

data_flags <-  list(n_features = 1, n_split = 1, p_split = 0.5)

fit_scores <- c("acc", "b_acc", "MCC")[2]


# Start with random search of hyperparameter values -----------------------

sRNN_rnd.reps <-  RNN_reps$init
ParamDesign <- generateDesign(n = sRNN_rnd.reps, par.set = ParamSet)
sRNN_scores <- data.frame()

time.taken <- array(NA, dim = sRNN_rnd.reps)
start.time <- Sys.time()

for (j in 1:sRNN_rnd.reps) {
  cat("Running starting values // Model: sRNN // Data:", data_name, "// Complete:", j/dim(ParamDesign)[1]*100, "%"); print(j)
  
  RNN_flags <- ParamDesign[j,]
  sRNN_rnd <- sRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste0(data_name, "_s1Rnd"))
  sRNN_scores <- rbind(sRNN_scores, sRNN_rnd$scores)
  ParamDesign$y[[j]] <- sRNN_rnd$scores[[fit_scores]]
  
  write.csv(ParamDesign, file = paste0(data_name, "_sRndPoints.csv"))
  write.csv(sRNN_scores, file = paste0(data_name, "_sRndscores.csv"))
}

time.taken <- data.frame(RndPoints = Sys.time() - start.time)


# Now proceed optimization of hyperparameters ----------------------------------

# Generate an object to save iterations of hyperparameter optimization

sRNN_ctrl = makeMBOControl()
sRNN_ctrl = setMBOControlInfill(sRNN_ctrl, crit = crit.ei)
                                
sRNN_opt.state = initSMBO(
  par.set = ParamSet,
  design = ParamDesign,
  control = sRNN_ctrl,
  minimize = FALSE,
  noisy = FALSE)

sRNN_reps <- RNN_reps$opt
sRNN_proposePoints <- NULL
sRNN_scores <- data.frame()

start.time <- Sys.time()

for (j in 1:sRNN_reps) {
  cat("Optimizing values // Model: sRNN // Data:", data_name, "// Complete:", j/sRNN_reps*100, "%"); print(j)
  RNN_flags = proposePoints(sRNN_opt.state)$prop.points
  sRNN_opt = sRNN_fit(data_vars, data_label, data_flags, RNN_flags, paste0(data_name, "_sBOpt"))
  sRNN_scores <- rbind(sRNN_scores, sRNN_opt$scores)
  y <- sRNN_opt$scores[[fit_scores]]
  time.taken[[j + sRNN_rnd.reps]] <- sRNN_opt$scores["Training_time"]
  
  sRNN_proposePoints <- rbind(sRNN_proposePoints, cbind(j, RNN_flags, y))
  write.csv(sRNN_scores, file = paste0(data_name, "_uBOptscores.csv"))
  write.csv(sRNN_proposePoints, file = paste0(data_name, "_sBOptproposePoints.csv"))
  
  updateSMBO(sRNN_opt.state,
             x = RNN_flags,
             y = y)
}

time.taken <- data.frame(time.taken, OptPoints = Sys.time() - start.time)
       
proposePoints(sRNN_opt.state)
finalizeSMBO(sRNN_opt.state)
sRNN_opt.state$opt.result$mbo.result$x
sRNN_opt.state$opt.result$mbo.result$y
plot(finalizeSMBO(sRNN_opt.state))

sRNN_proposePoints <- read.csv(file = paste0(data_name, "_sBOptproposePoints.csv"))
ParamBest <- sRNN_proposePoints[order(sRNN_proposePoints$y, decreasing = T),][1,][3:14]

data_type <- RNN_testdata[paste0("type_",selectVars)][[1]]

modelBest <- load_model_hdf5(paste0(data_name, "_sBOptBest.h5"))

for (cc in unlist(ClassAll)) {
  RatiosClass[1,cc] <- round(sum(data_label[data_type == cc]))
  RatiosClass[2,cc] <- round(sum(sRNN_best$classify[data_type == cc], na.rm = T))
  RatiosClass[3,cc] <- RatiosClass[2,cc] / RatiosClass[1,cc]
}

save(ParamBest, modelBest, sRNN_opt.state, sRNN_best, sRNN_proposePoints, sRNN_scores,
     RNN_trainingdata, RNN_testdata,
     RatiosClass, time.taken,
     file = paste0(data_name, "_sBOptBest.RData"), version = 2)                         