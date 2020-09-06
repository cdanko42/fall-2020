using GLM
using Optim
using DataFrames
using CSV
using HTTP
using FreqTables

function PS2()
## Problem 1
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
negf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)  # random number as starting value
result = optimize(negf, startval, LBFGS())

## Problem 2
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

## Solve objective function
function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), 
                        Optim.Options(g_tol=1e-6, iterations=100_000, 
                        show_trace=true))
println(beta_hat_ols.minimizer)

## Check work
bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
println(bols_lm)

## Problem 3
## Create objective function for logit likelihood

function logitregress(beta, X, y)
	negloglike = -sum(y.*X*beta - log.(1 .+exp.(X*beta)))
	return negloglike
end

## Optimize!

beta_tilde_loglike = optimize(b -> logitregress(b, X, y), rand(size(X,2)), LBFGS(), 
                        Optim.Options(g_tol=1e-6, iterations=100_000, 
                        show_trace=true))
println(beta_tilde_loglike.minimizer)

## Problem 4

## Check work using glm
loglink = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println(loglink)
##Hurray

## Problem 5

## Clean dataframe
dropmissing(df, :occupation)
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]

b= zeros(4, 6)

## we normalize with respect to 1, to match r code 
function multinomregress(b, X, y)
	firsterm = zeros(length(df.occupation), 6)
	secondterm = zeros(length(df.occupation), 6)
	for j in 1:6
		y = deepcopy(df.occupation)
		y[y.!= j+1] .= 0
		y[y.== j+1] .= 1
		firsterm[:, j] = (y.*X*b[:, j])
		secondterm[:, j] = exp.(X*b[:, j])
	end
	firsterm = sum(firsterm, dims=2)
	secondterm = sum(secondterm, dims= 2)
	multinom = -sum((firsterm - log.(1 .+ secondterm)))
	return multinom
end

x0 = zeros(4, 6)
multinom_please_work = optimize(b -> multinomregress(b, X, y), x0, LBFGS(), 
                        Optim.Options(g_tol=1e-5, iterations=100_000, 
                        show_trace=true))
println(multinom_please_work.minimizer)
end