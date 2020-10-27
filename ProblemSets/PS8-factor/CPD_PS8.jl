using MultivariateStats
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS8-factor/nlsy.csv"
df = CSV.read(HTTP.get(url).body)

## Question 1

ols = glm(@formula(logwage ~ black + hispanic+female+schoolt + gradHS+grad4yr), df, Normal())

## Question 2

vcov(ols)

## Question 3
ols = glm(@formula(logwage ~ black + hispanic+female+schoolt + gradHS+grad4yr+asvabAR+asvabCS+asvabMK+asvabNO+asvabPC+asvabWK), df, Normal())

##Yes. One small part is that the returns to schooling on wage become negative once including all of these variables. In the first regression, the coefficient was positive
##Which we should expect. There is likely some simultaneity in this equation

## Question 4

asvabMat = zeros(size(df, 1), 6)
asvabMat[:,1] = df.asvabAR
asvabMat[:,2] = df.asvabCS
asvabMat[:,3] = df.asvabMK
asvabMat[:,4] = df.asvabNO
asvabMat[:,5] = df.asvabPC
asvabMat[:,6] = df.asvabWK
asvabMat = asvabMat'


M = fit(PCA, asvabMat; maxoutdim=1)
asvabPCA = MultivariateStats.transform(M, asvabMat)
asvabPCA  = asvabPCA'

##Question 5
M = fit(FactorAnalysis, asvabMat; maxoutdim=1)
asvabfac = MultivariateStats.transform(M, asvabMat)
asvabfac  = asvabfac'

## Question 6

Xm = zeros(size(df, 1), 4)
Xm[:,1] .= 1
Xm[:,2] = df.black
Xm[:,3] = df.hispanic
Xm[:,4] = df.female
 
function factor(theta, M, Xm, X, y) 
## Standardnormal
d= Normal(0,1)

asvabMat = asvabMat'

alpha = theta[1:24]
bigalpha = reshape(alpha, (4, 6))
gamma = theta[25:30]
beta = theta[31:37]
delta = theta[38]
sigma = theta[39:45]
Li3 = zeros(size(M,1))

function integrallike(x)
for i in size(M,1)
for j in size(M,2)
Li[i,j] = (1\sigma[j + 1]).*pdf(d, ((M[i,j] - Xm[i,:]*bigalpha[:,j] - gamma[j]*x)/sigma[j+1]))
end
end

Li2 = Li[:, 1]

for j in (size(M,2)-1)
Li2 = Li2.*Li[:,j+1]
end

for i in size(M,1)
Li2[i] = Li2*((1\sigma[1])).*pdf(d, ((y[i] - Xm[i,:]*beta - delta*x)/sigma[1]))
end

return(Li2)
end

Li3= integrallike(nodes)

for i in size(M,1)
Li3[i] = log(sum(weights.*Li3[i]))
end

return(-sum(Li3))
end


