using SMM
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV
using FreqTables
using Distributions

## Problem 1
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1


function ols_gmm(alpha, X, y)
	g = y .- X*alpha 
	J = g'*I*g
	return J
end
gmm_optim = optimize(a -> ols_gmm(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000))
println(gmm_optim)
## Check work
bols = inv(X'*X)*X'*y

## Question 2a
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit(alpha, X, y)
        
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

    alpha_zero = zeros(6*size(X,2))
    alpha_rand = rand(6*size(X,2))
    alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
    alpha_start = alpha_true.*rand(size(alpha_true))
    println(size(alpha_true))
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)

## 2b
function logit_gmm(alpha, X, y)
    K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
    g = bigY .- P
	g = reshape(g, (J*N, 1))
    Jf = g'*I*g
    return Jf[1,1]
end

alpha_hat_optim = optimize(a -> logit_gmm(a, X, y), alpha_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(alpha_hat_optim)

alpha_hat_optim = optimize(a -> logit_gmm(a, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(alpha_hat_optim)

Xrand = rand(1000, 2)
betatrue = [[(.25),	(.1),(.05)] [(.05),(.25),(.4)]]
p = Xrand*betatrue'
epsilon = rand(GeneralizedExtremeValue(0, 1, 0),1000,3)

Ychoice = zeros(1000,1)
for i in 1:size((Xrand),1)
Ychoice[i,1] = argmax(p[i,:] + epsilon[i,:])
end

alpha_new = [.25, .1, .05, .25]
alpha_hat_optim = optimize(a -> logit_gmm(a, Xrand, Ychoice), alpha_new, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(alpha_hat_optim)

## Q4

MA = SMM.parallelNormal() # Note: this line may take up to 5 minutes to execute
dc = SMM.history(MA.chains[1])
dc = dc[dc[:accepted].==true, :]
println(describe(dc))

## Q5 

function gmm_smm(θ, X, y, D)
    K = size(X,2)
    N = size(y,1)
	J = length(unique(y))
	β = [reshape(θ,K,J-1) zeros(K)]
    # N+1 moments in both model and data
    gmodel = zeros(N,D)
    gdata= y
    #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ####
    # This is critical!                   #
    Random.seed!(1234)                    #
    # You must always use the same ε draw #
    # for every guess of θ!               #
    #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
    # simulated model moments
    for d=1:D
		p = zeros(1000,J)
		Ychoice = zeros(N,1)
        ε = rand(GeneralizedExtremeValue(0, 1, 0),N,J)
        p = X*β
        for i in 1:N
			Ychoice[i,1] = argmax(p[i,:] + ε[i,:])
		end
		gmodel[1:end,d] = Ychoice
	end
    # criterion function
    err = vec(gdata .- mean(gmodel; dims=2))
    # weighting matrix is the identity matrix
    # minimize weighted difference between data and moments
    Jf = err'*I*err
    return Jf
end

alpha_hat_smm = optimize(a -> gmm_smm(a, X, y, 1000), alpha_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(alpha_hat_smm)

	
	