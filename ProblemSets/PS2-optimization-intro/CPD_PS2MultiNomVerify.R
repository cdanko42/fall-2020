df <- read.csv("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv", stringsAsFactors=FALSE, header=T)
df <- df[complete.cases(df$occupation),]
df$white <- df$race
df$white[df$white!= 1] <- 0
df$occupation[df$occupation ==8] = 7
df$occupation[df$occupation ==9] = 7
df$occupation[df$occupation ==10] = 7
df$occupation[df$occupation ==11] = 7
df$occupation[df$occupation ==12] = 7
df$occupation[df$occupation ==13] = 7
library(nnet)
multinomModel <- multinom(occupation ~ age + white + collgrad, data=df)
multinomModel
