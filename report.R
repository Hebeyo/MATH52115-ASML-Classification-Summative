'
Summative assignment code
Z code: Z0178393
CIS num: rntn68
'

install.packages("rsample")
install.packages("keras")
install.packages("recipes")
install.packages("data.table")
install.packages("mlr3verse")
install.packages("mlr3")
install.packages("mlr3proba")
install.packages("mlr3learners")
install.packages("fstcore")
install.packages("ggforce")
install.packages("rgeos")
install.packages("sp")
install.packages("GGally")
install.packages("forcats")
install.packages("stringr")
install.packages("dplyr")
install.packages("purrr")
install.packages("tidyr")
install.packages("tibble")
install.packages("ggplot2")
install.packages("stats")
install.packages("skimr")
install.packages("Matrix")
install.packages("rsample")
install.packages("tidyverse")
install.packages("graphics")
install.packages("grDevices")
install.packages("utils")
install.packages("datasets")
install.packages("methods")
install.packages("readr")

Heart_failure <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
View(Heart_failure)

library("skimr")
skim(Heart_failure)

DataExplorer::plot_bar(Heart_failure, ncol = 3)
DataExplorer::plot_histogram(Heart_failure, ncol = 3)
Heart_failure$fatal_mi<-as.factor(Heart_failure$fatal_mi)
DataExplorer::plot_boxplot(Heart_failure, by = "fatal_mi", ncol = 3)

library("tidyverse")
library("ggplot2")
ggplot(Heart_failure,
       aes(x = age, y = fatal_mi) )+
  geom_point()
ggplot(Heart_failure,
       aes(x = time, y = fatal_mi) )+
  geom_point()

library("GGally")
ggpairs(Heart_failure %>% select(fatal_mi, age, anaemia, creatinine_phosphokinase, diabetes),
        aes(color = fatal_mi))
ggpairs(Heart_failure %>% select(fatal_mi, ejection_fraction, high_blood_pressure, platelets, serum_creatinine),
        aes(color = fatal_mi))
ggpairs(Heart_failure %>% select(fatal_mi, serum_sodium, sex, smoking, time),
        aes(color = fatal_mi))

Heart_failure$fatal_mi<-as.factor(Heart_failure$fatal_mi)
library("data.table")
library("mlr3verse")
library("mlr3")
set.seed(212) # set seed for reproducibility
Heart_task <- TaskClassif$new(id = "fatal_mi_new",
                              backend = Heart_failure, 
                              target = "fatal_mi"
                              #positive = "bad"
)
Heart_task

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(Heart_task)

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
lrn_lda <- lrn("classif.lda", predict_type = "prob")

res_baseline <- resample(Heart_task, lrn_baseline, cv5, store_models = TRUE)
res_cart <- resample(Heart_task, lrn_cart, cv5, store_models = TRUE)
res_log_reg <- resample(Heart_task, lrn_log_reg, cv5, store_models = TRUE)
res_lda <- resample(Heart_task, lrn_lda, cv5, store_models = TRUE)

res <- benchmark(data.table(
  task       = list(Heart_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_log_reg,
                    lrn_lda),
  resampling = list(cv5)
), store_models = TRUE)

res

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

trees <- res$resample_result(2)
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.3)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

plot(res$resample_result(2)$learners[[5]]$model, compress = TRUE, margin = 0.3)
text(res$resample_result(2)$learners[[5]]$model, use.n = TRUE, cex = 0.8)

lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
res_cart_cv <- resample(Heart_task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.016, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")
lrn_lda <- lrn("classif.lda", predict_type = "prob")
lrnsp_lda <- lrn("classif.lda", predict_type = "prob",id="super")

pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

pl_factor <- po("encode")

spr_lrn <- gunion(list(
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("learner_cv", lrn_lda),
      po("nop") 
    )),
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)
  po(lrnsp_lda)

spr_lrn$plot()

res_spr <- resample(Heart_task, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

