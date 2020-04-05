install.packages("tidyverse", repos="https://cloud.r-project.org/", Ncpus=8)

install.packages(c("devtools","mvtnorm","loo","coda"), repos="https://cloud.r-project.org/", dependencies=TRUE, Ncpus=8)
library(devtools)
install_github("rmcelreath/rethinking",ref="Experimental")

install.packages("GGally")