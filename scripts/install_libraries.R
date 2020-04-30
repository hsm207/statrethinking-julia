install.packages("tidyverse", repos="https://cloud.r-project.org/", Ncpus=8)

install.packages(c("coda","mvtnorm","devtools","loo","dagitty"), repos="https://cloud.r-project.org/", Ncpus=8)
library(devtools)
devtools::install_github("rmcelreath/rethinking", repos="https://cloud.r-project.org/")

install.packages("GGally", repos="https://cloud.r-project.org/")