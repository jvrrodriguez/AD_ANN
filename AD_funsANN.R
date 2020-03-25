# Functions for Anomaly detection in time-series data
# To test model performance, it is necessary to add and additional array of labelled time-series including anomalies  
# Parameters to run models are detailed below:

# data_vars. Array of time-series data
# data_label. Array of labelled time-series data which includes normal (0) and anomalous cases (1)
# data_flags. Parameters to split data. It includes the number fitted variables (n_features), the number of splits (n_split) and the split location relative of the whole time-series (p_split). By default we use n_features=1, n_split=2 for semi-supervised and n_split=1 for supervised classification and p_split = 0.5. 
# RNN_flags. Hyperparameters to fit RNN models. It included a large number of hyperparameters related to (i) the network structure and (ii) the training algorithm. The definition of the whole suited of hyperparameters is required for model running.
# RNN_name. Name to save output keras models


# Install packages via CRAN first if required before running the code below
# Load libraries --------------------------------------------------------

library('MASS')
library('keras')
library('kerasR')


# Fit Semi-Supervised classification --------------------------------------

uRNN_fit <- function(data_vars, data_label, data_flags, RNN_flags, RNN_name) {
  
  # Building the LSTM model
  
  flags <- flags(
    flag_boolean("stateful", FALSE),
    # Should we use several layers str8of LSTM?
    # Again, just included for completeness, it did not yield any superior 
    # performance on this task.
    # This will actually stack exactly one additional layer of LSTM units.
    flag_boolean("stack_layers", FALSE),
    # number of samples fed to the model in one go
    flag_integer("batch_size", as.integer(as.character(RNN_flags["batch_size"][[1]]))),
    # size of the hidden state, equals size of predictions
    flag_integer("n_timesteps", as.integer(as.character(RNN_flags["sld_window"][[1]]))),
    #classification threshold for binary classifaction 
    flag_integer("thrsld_class", as.numeric(as.character(RNN_flags["thrsld_class"][[1]]))),
    # how many epochs to train for
    flag_integer("n_epochs", 100),
    # fraction of the units to drop for the linear transformation of the inputs
    flag_numeric("dropout", as.numeric(as.character(RNN_flags["dropout"][[1]]))),
    # fraction of the units to drop for the linear transformation of the 
    # recurrent state
    flag_numeric("recurrent_dropout", as.numeric(as.character(RNN_flags["dropout"][[1]]))),
    # loss function. Found to work better for this specific case than mean
    # squared error
    flag_string("loss", "mae"),
    # optimizer = stochastic gradient descent. Seemed to work better than adam 
    # or rmsprop here (as indicated by limited testing)
    flag_string("optimizer_type", as.character(RNN_flags["optimizer_type"][[1]])),
    # size of the LSTM layer
    flag_integer("n_units", as.integer(as.character(RNN_flags["n_units"][[1]]))),
    # learning rate
    flag_numeric("learning_rate", as.numeric(as.character(RNN_flags["learning_rate"][[1]]))),
    # momentum, an additional parameter to the optimizer
    flag_numeric("momentum", as.numeric(as.character(RNN_flags["momentum"][[1]]))),
    # parameter to the early stopping callback
    flag_integer("patience", 25), #number of epochs with no improvement after which training will be stopped
    # activation function
    flag_string("activation", as.character(RNN_flags["activation"][[1]])),
    # 
    flag_integer("weight_constraint", as.integer(as.character(RNN_flags["weight_constraint"][[1]]))),
    # number of hidden layers of the LSTM
    flag_integer("n_layers", as.integer(as.character(RNN_flags["n_layers"][[1]])))
  )
  
  if (is.null(RNN_name)) 
    RNN_name <- "model_uRNN"
    
  if (is.null(data_flags)) {
    data_flags <- list(n_features = 1, n_split = 2, p_split = 0.5)
  }
  
  if (is.null(RNN_flags["sld_window"])) {
    RNN_flags$sld_window <- 1
  }
  
  if (is.null(RNN_flags["thrsld_class"])) {
    RNN_flags$thrsld_class <- 0.5
  }
  
  # Set hyperparameters for training
  
  n_split <- data_flags["n_split"][[1]]
  p_split <- data_flags["p_split"][[1]]
  n_features <- data_flags["n_features"][[1]]
  
  batch_size <- as.integer(as.character(RNN_flags["batch_size"][[1]]))
  n_timesteps <- as.integer(as.character(RNN_flags["sld_window"][[1]]))
  thrsld_class <- as.numeric(as.character(RNN_flags["thrsld_class"][[1]]))
  
  # Setup data
  # Here I retain only the "normal" data: select only the "normal" values tagged as 0
  
  RNNdata_vars <- series_to_RNN(data_vars[data_label == 0], n_timesteps, batch_size, n_split, p_split, scale = T, scale.center = NA, scale.scale = NA, thirdDim = T) 
  
  data_train <- list(RNNdata_vars[[1]][[1]], RNNdata_vars[[1]][[2]])
  data_validation <- list(RNNdata_vars[[2]][[1]], RNNdata_vars[[2]][[2]])
  
  # Clear keras cache
  
  k_clear_session()
  
  # Callbacks to be passed to the fit() function
  
  callbacks <- list(
    callback_early_stopping(monitor = "val_loss",
                            patience = flags$patience),
    callback_model_checkpoint(filepath = paste0(RNN_name,".h5"),
                              monitor = "val_loss",
                              save_best_only = TRUE
    ))
  
  # Build model
  
  model <- keras_model_sequential() 
  
  if (flags$n_layers == 1) {
    model %>% 
      layer_lstm(units = flags$n_units, 
                 dropout = flags$dropout,
                 recurrent_dropout = flags$recurrent_dropout,
                 recurrent_constraint = max_norm(flags$weight_constraint),
                 return_sequences = TRUE,
                 stateful = TRUE,
                 batch_input_shape = c(flags$batch_size, flags$n_timesteps, n_features)) %>% 
      time_distributed(layer_dense(units = n_features, activation = flags$activation))
    
  } else if (flags$n_layers == 2) {
    model %>% 
      layer_lstm(units = flags$n_units,
                 dropout = flags$dropout,
                 recurrent_dropout = flags$recurrent_dropout,
                 recurrent_constraint = max_norm(flags$weight_constraint),
                 return_sequences = TRUE,
                 stateful = TRUE,
                 batch_input_shape = c(flags$batch_size, flags$n_timesteps, n_features)) %>%
      layer_lstm(units = flags$n_units,
                 return_sequences = TRUE,
                 stateful = TRUE) %>%
      time_distributed(layer_dense(units = n_features, activation = flags$activation))
    
  } else if (flags$n_layers == 3) {
    model %>% 
      layer_lstm(units = flags$n_units,
                 dropout = flags$dropout,
                 recurrent_dropout = flags$recurrent_dropout,
                 recurrent_constraint = max_norm(flags$weight_constraint),
                 return_sequences = TRUE,
                 stateful = TRUE,
                 batch_input_shape = c(flags$batch_size, flags$n_timesteps, n_features)) %>%
      layer_lstm(units = flags$n_units,
                 return_sequences = TRUE,
                 stateful = TRUE) %>%
      layer_lstm(units = flags$n_units,
                 return_sequences = TRUE,
                 stateful = TRUE) %>%
      time_distributed(layer_dense(units = n_features, activation = flags$activation))
  } 
  
  # Compile model
 
   model %>%
    compile(
      loss = flags$loss,
      optimizer = flags$optimizer_type,
      # in addition to the loss, Keras will inform us about current 
      # MSE while training
      metrics = list("mean_squared_error")
    )
  
  # Train the model with the train dataset
  
  traintime <- system.time(
    history <- model %>% fit(
      x          = data_train[[1]],
      y          = data_train[[2]], 
      validation_data = data_validation,
      batch_size = flags$batch_size,
      epochs     = flags$n_epochs,
      callbacks = callbacks, verbose = 2
    ))
  
  # Load best model
  
  model <- load_model_hdf5(paste0(RNN_name,".h5"))
  
  # Evaluate the model with the eval dataset
  
  score <- model %>%
    evaluate(data_validation[[1]], data_validation[[2]], batch_size = flags$batch_size, verbose = 0)
  
  RNN_datatest <- series_to_RNN(data_vars, n_timesteps, batch_size, n_split, p_split, scale = T, scale.center = RNNdata_vars[[4]][1, ], scale.scale = RNNdata_vars[[4]][2, ], thirdDim = T) 
  
  # Predict test set
  
  pred_test <- model %>%
    predict(RNN_datatest[[3]], batch_size, verbose = 1)
  
  # Retransform values to original (log) scale
  
  pred_test <- (pred_test * RNN_datatest[[3 + 1]][2,] + RNN_datatest[[3 + 1]][1,])
  
  # Build a dataframe that has both actual and predicted values
  
  RNN_testpred <- c(rep(NA, nrow(pred_test)), rep(NA, length(data_vars) - nrow(pred_test)))
  
  for (i in n_timesteps:nrow(pred_test)) {
    inipos <- n_timesteps + i - 1
    endpos <- inipos + length(pred_test[i,,1])
    RNN_testpred[inipos:endpos] <- pred_test[i,,1]
  }
  
  RNN_testres <- abs(data_vars - RNN_testpred)
  RNN_testthld <- fitdistr(RNN_testres[is.finite(RNN_testres)], "normal", na.rm = T)$estimate
  RNN_testcumprob <- pnorm(RNN_testres, mean = RNN_testthld[1], sd = RNN_testthld[2])
  
  Output <- data.frame(x = 0, y = ifelse(RNN_testcumprob > thrsld_class, 1, 0))
  Truth <- data.frame(x = 0, y = data_label)
  
  RNN_performance <- ADPerformance2(Output, Truth, print_out = F) # IMPORTANT: variables in columns need to have same name 
  
  # Add MSE RMSE and Training and Testing + classification time to the performance dataframe
  
  MSE <- sum((data_vars - Output$y)^2, na.rm = T) / length(Output$y)
  RMSE <- sqrt(MSE)
  RNN_performance <- data.frame(RNN_performance, MSE = MSE, RMSE = RMSE, Training_time = traintime[[3]])
  
  return(list(hyperpars = flags, scores = RNN_performance, pred = RNN_testpred, prob = RNN_testcumprob, classify = Output$y))
}



# Fit Supervised classification -------------------------------------------

sRNN_fit <- function(data_vars, data_label, data_flags, RNN_flags, RNN_name) {
  
  # Building the LSTM model
  
  flags <- flags(
    flag_boolean("stateful", FALSE),
    # Should we use several layers str8of LSTM?
    # Again, just included for completeness, it did not yield any superior 
    # performance on this task.
    # This will actually stack exactly one additional layer of LSTM units.
    flag_boolean("stack_layers", FALSE),
    # number of samples fed to the model in one go
    flag_integer("batch_size", as.integer(as.character(RNN_flags["batch_size"][[1]]))),
    # size of the hidden state, equals size of predictions
    flag_integer("n_timesteps", as.integer(as.character(RNN_flags["sld_window"][[1]]))),
    #classification threshold for binary classifaction 
    flag_integer("thrsld_class", as.numeric(as.character(RNN_flags["thrsld_class"][[1]]))),
    # how many epochs to train for
    flag_integer("n_epochs", 100),
    # fraction of the units to drop for the linear transformation of the inputs
    flag_numeric("dropout", as.numeric(as.character(RNN_flags["dropout"][[1]]))),
    # fraction of the units to drop for the linear transformation of the 
    # recurrent state
    flag_numeric("recurrent_dropout", as.numeric(as.character(RNN_flags["dropout"][[1]]))),
    # loss function. Found to work better for this specific case than mean
    # squared error
    flag_string("loss", "mae"),
    # optimizer = stochastic gradient descent. Seemed to work better than adam 
    # or rmsprop here (as indicated by limited testing)
    flag_string("optimizer_type", as.character(RNN_flags["optimizer_type"][[1]])),
    # size of the LSTM layer
    flag_integer("n_units", as.integer(as.character(RNN_flags["n_units"][[1]]))),
    # learning rate
    flag_numeric("learning_rate", as.numeric(as.character(RNN_flags["learning_rate"][[1]]))),
    # momentum, an additional parameter to the optimizer
    flag_numeric("momentum", as.numeric(as.character(RNN_flags["momentum"][[1]]))),
    # parameter to the early stopping callback
    flag_integer("patience", 25), #number of epochs with no improvement after which training will be stopped
    # activation function
    flag_string("activation", as.character(RNN_flags["activation"][[1]])),
    # 
    flag_integer("weight_constraint", as.integer(as.character(RNN_flags["weight_constraint"][[1]]))),
    # number of hidden layers of the LSTM
    flag_integer("n_layers", as.integer(as.character(RNN_flags["n_layers"][[1]])))
  )
  
  if (is.null(RNN_name)) 
    RNN_name <- "model_sRNN"
  
  if (is.null(data_flags)) {
    data_flags <- list(n_features = 1, n_split = 1, p_split = 0.5)
  }
  
  if (is.null(RNN_flags["sld_window"])) {
    RNN_flags$sld_window <- 1
  }
  
  if (is.null(RNN_flags["thrsld_class"])) {
    RNN_flags$thrsld_class <- 0.5
  }
  
  
  # Set hyperparameters for training
  
  n_split <- data_flags["n_split"][[1]]
  p_split <- data_flags["p_split"][[1]]
  n_features <- data_flags["n_features"][[1]]
  
  batch_size <- as.integer(as.character(RNN_flags["batch_size"][[1]]))
  n_timesteps <- as.integer(as.character(RNN_flags["sld_window"][[1]]))
  thrsld_class <- as.numeric(as.character(RNN_flags["thrsld_class"][[1]]))
  
  # Setup data
  
  RNNdata_vars <- series_to_RNN(data_vars, n_timesteps, batch_size, n_split, p_split, scale = T, scale.center = NA, scale.scale = NA, thirdDim = T) 
  RNNdata_label <- series_to_RNN(data_label, n_timesteps, batch_size, n_split, p_split, scale = F, scale.center = NA, scale.scale = NA, thirdDim = F)
  
  if (n_timesteps == 1) {
    RNNdata_label <- list(RNNdata_label[[1]][[1]],
                          RNNdata_label[[2]][[1]],
                          RNNdata_label[[3]][[1]])
    
  } else if  (n_timesteps > 1) {
    RNNdata_label <- list(apply(RNNdata_label[[1]][[1]], 1, series_1d),
                          apply(RNNdata_label[[2]][[1]], 1, series_1d),
                          apply(RNNdata_label[[3]][[1]], 1, series_1d))
  }
  
  data_train <- list(RNNdata_vars[[1]][[1]], RNNdata_label[[1]])
  data_validation <- list(RNNdata_vars[[2]][[1]], RNNdata_label[[2]])
  
  
  # Clear keras cache
  
  k_clear_session()
  
  # Callbacks to be passed to the fit() function
  
  callbacks <- list(
    callback_early_stopping(monitor = "val_loss",
                            patience = flags$patience),
    callback_model_checkpoint(filepath = paste0(RNN_name,".h5"),
                              monitor = "val_loss",
                              save_best_only = TRUE
    ))
  
  # Build model
  
  model <- keras_model_sequential() 
  
  if (flags$n_layers == 1) {
    model %>% 
      layer_lstm(units = flags$n_units, 
                 dropout = flags$dropout,
                 recurrent_dropout = flags$recurrent_dropout,
                 recurrent_constraint = max_norm(flags$weight_constraint),
                 stateful = TRUE,
                 batch_input_shape = c(flags$batch_size, flags$n_timesteps, n_features)) %>% 
      layer_dense(units = n_features, activation = flags$activation)
    
  } else if (flags$n_layers == 2) {
    model %>% 
      layer_lstm(units = flags$n_units,
                 dropout = flags$dropout,
                 recurrent_dropout = flags$recurrent_dropout,
                 recurrent_constraint = max_norm(flags$weight_constraint),
                 return_sequences = TRUE,
                 stateful = TRUE,
                 batch_input_shape = c(flags$batch_size, flags$n_timesteps, n_features)) %>%
      layer_lstm(units = flags$n_units,
                 stateful = TRUE) %>%
      layer_dense(units = n_features, activation = flags$activation)
    
  } else if (flags$n_layers == 3) {
    model %>% 
      layer_lstm(units = flags$n_units,
                 dropout = flags$dropout,
                 recurrent_dropout = flags$recurrent_dropout,
                 recurrent_constraint = max_norm(flags$weight_constraint),
                 return_sequences = TRUE,
                 stateful = TRUE,
                 batch_input_shape = c(flags$batch_size, flags$n_timesteps, n_features)) %>%
      layer_lstm(units = flags$n_units,
                 return_sequences = TRUE,
                 stateful = TRUE) %>%
      layer_lstm(units = flags$n_units,
                 stateful = TRUE) %>%
      layer_dense(units = n_features, activation = flags$activation)
  } 
  
  # Compile model
  
  model %>%
    compile(
      loss = flags$loss,
      optimizer = flags$optimizer_type,
      # in addition to the loss, Keras will inform us about current 
      # MSE while training
      metrics = "accuracy"
    )
  
  # Train the model with the train dataset
  
  traintime <- system.time(
    history <- model %>% fit(
      x          = data_train[[1]],
      y          = data_train[[2]], 
      validation_data = data_validation,
      batch_size = flags$batch_size,
      epochs     = flags$n_epochs,
      callbacks = callbacks, verbose = 2
    ))
  
  # Load best model
  
  model <- load_model_hdf5(paste0(RNN_name,".h5"))
  
  # Predict test set
  
  pred_test <- model %>%
    predict(RNNdata_vars[[3]], batch_size, verbose = 1)
  
  # In case you generate predictions based on the second set of data (see above) 
  
  if (length(data_vars) == nrow(pred_test)) {
    RNN_testpred <- array(pred_test)
    
  } else if (length(data_vars) != nrow(pred_test)) {
    RNN_testpred <- c(array(pred_test), rep(NA, length(data_vars) - nrow(pred_test)))
  }
  
  AsThld <- thrsld_class
  if (flags$activation == "tanh") {
    RNN_testpred <- range01(tanh(RNN_testpred)^2)
  } else if (flags$activation != "tanh" & mean(data_vars) < 0.5) {
    AsThld <- 0.5 - mean(data_label)
  }
  
  Output <- data.frame(x = 0, y = ifelse(RNN_testpred > AsThld, 1, 0))
  Truth <- data.frame(x = 0, y = data_label)
  
  RNN_performance <- ADPerformance2(Output, Truth, print_out = F) # IMPORTANT: variables in columns need to have same name 
  
  # Add MSE RMSE and Training and Testing + classification time to the performance dataframe
  
  MSE <- sum((data_vars - Output$y)^2, na.rm = T) / length(Output$y)
  RMSE <- sqrt(MSE)
  RNN_performance <- data.frame(RNN_performance, MSE = MSE, RMSE = RMSE, Training_time = traintime[[3]])
  
  return(list(hyperpars = RNN_flags, scores = RNN_performance, pred = NA, prob = RNN_testpred, classify = Output$y))
}



# Additional functions to re-shape data -----------------------------------


series_to_RNN <- function(data, sld_window, batch_size, n_split, p_split, scale, scale.center, scale.scale, thirdDim){
  
  # Reset initial values
  
  data[!is.finite(data)] <- NA
  scale[!is.finite(scale)] <- T
  scale.center[!is.finite(scale.center)] <- NA
  scale.scale[!is.finite(scale.scale)] <- NA
  thirdDim[!is.finite(thirdDim)] <- F
  
  # Scale training data with their own information
  
  if (scale == T & (is.na(scale.center) | is.na(scale.scale))) {
    data_sc <- scale(data)
    center_history_data <- attr(data_sc, "scaled:center") 
    scale_history_data <- attr(data_sc, "scaled:scale")
    data_sc <- scale(data, center = center_history_data, scale = scale_history_data)
  }
  
  # Scale training data with added information
  
  if (scale == T & (!is.na(scale.center) & !is.na(scale.scale))) {
    data_sc <- scale(data, center = scale.center, scale = scale.scale)
  }
  
  # Do not scale data (if necessary)
  
  if (scale == F & (is.na(scale.center) & is.na(scale.scale))) {
    data_sc <- data 
  }
  
  build_matrix <- function(data, overall_timesteps) {
    t(sapply(1:(length(data) - overall_timesteps + 1), function(x) 
      data[x:(x + overall_timesteps - 1)]))
  }
  
  reshape_X_3d <- function(X) {
    dim(X) <- c(dim(X)[1], dim(X)[2], 1)
    return(X)
  }
  
  n_predictions <- sld_window
  
  # Give some values
  if (is.na(n_split)) n_split <- 1
  if (is.na(sld_window)) sld_window <- 1
  
  # Build the windowed matrices
  
  train_sc <- data_sc[ 1:(length(data_sc) * p_split) ]
  valid_sc <- data_sc[ (length(data_sc) * p_split + 1):length(data_sc)]
  test_sc <- data_sc
  
  if (sld_window * n_split == 1) {
    train_mt <-  train_sc
    valid_mt <-  valid_sc
    test_mt <-  test_sc
    
  } else if (sld_window * n_split > 1) {
    train_mt <-  build_matrix(train_sc, sld_window * n_split)
    valid_mt <- build_matrix(valid_sc, sld_window * n_split)
    test_mt <- build_matrix(test_sc, sld_window * n_split)
  }
  
  if (scale == T) {
    data_list <-  list( list(1:sld_window), list(1:sld_window),  list(1:sld_window))
    data_list[[(3 + 1)]] <- as.matrix(c(attr(data_sc, "scaled:center") , attr(data_sc, "scaled:scale")))
    rownames(data_list[[(3 + 1)]]) <- c("center","scale")
    
  } else if (scale == F)
    data_list <- list( list(1:sld_window), list(1:sld_window),  list(1:sld_window))
  
  x <- seq(1:(sld_window * n_split))
  xsplit <- split(x, ceiling(x/sld_window))
  #even <- seq_len(nrow(Xytrain_Level[[1]])) %% 2 ## Select even rows
  
  if (sld_window * n_split == 1) {
    data_train <- train_mt[1:(length(train_mt) %/% batch_size * batch_size)]
    data_valid <- valid_mt[1:(length(valid_mt) %/% batch_size * batch_size)]
    data_test <- test_mt[1:(length(test_mt) %/% batch_size * batch_size)]
    
    if (thirdDim == T) {
      dim(data_train ) <- c(length(data_train), 1, 1)
      dim(data_valid ) <- c(length(data_valid), 1, 1)
      dim(data_test ) <- c(length(data_test), 1, 1)
    }
    
    data_list[[1]][[1]] <- data_train
    data_list[[2]][[1]] <- data_valid
    data_list[[3]][[1]] <- data_test
    
  } else if (sld_window * n_split > 1) {
    
    for (i in 1:n_split) {
      data_train <- matrix(train_mt[,xsplit[[i]]], ncol = length(xsplit[[i]]))
      data_train <- matrix(data_train[1:(nrow(data_train) %/% batch_size * batch_size), ], ncol = length(xsplit[[i]]))
      data_valid <- matrix(valid_mt[,xsplit[[i]]], ncol = length(xsplit[[i]]))
      data_valid <- matrix(data_valid[1:(nrow(data_valid) %/% batch_size * batch_size), ], ncol = length(xsplit[[i]]))
      data_test <- matrix(test_mt[,xsplit[[i]]], ncol = length(xsplit[[i]]))
      data_test <- matrix(data_test[1:(nrow(data_test) %/% batch_size * batch_size), ], ncol = length(xsplit[[i]]))
      
      # Add on the required third axis
      
      if (thirdDim == T & sld_window == 1) {
        dim(data_train) <- c(length(data_train), 1, 1)
        dim(data_valid) <- c(length(data_valid), 1, 1)
        dim(data_test) <- c(length(data_test), 1, 1)
        
      } else if (thirdDim == T & sld_window > 1) {
        data_train <- reshape_X_3d(data_train)
        data_valid <- reshape_X_3d(data_valid)
        data_test <- reshape_X_3d(data_test)
      }
      
      data_list[[1]][[i]] <- data_train
      data_list[[2]][[i]] <- data_valid
      data_list[[3]][[i]] <- data_test
    }
  }
  
  return(data_list)
}

# Convert to 1D arrays

series_1d <- function(x)  ifelse(sum(x) == 0, 0, 1)

# Standardiwe range

range01 <- function(x){(x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T))}