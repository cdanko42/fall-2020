using Distributions
using Optim
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using HTTP

function ps4()
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

function mlogit_with_Z(theta, X, Z, y)
        
        alpha = theta[1:end-1]
        gamma = theta[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end
    startvals = [2*rand(7*size(X,2)).-1; .1]
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println([theta_hat_mle_ad theta_hat_mle_ad_se])

cd("C:/Users/chris/Documents/fall-2020/ProblemSets/PS4-mixture")

include("lgwt.jl") 

d = Normal(0,1)

nodes, weights = lgwt(7,-4,4)
    
# now compute the integral over the density and verify it's 1
sum(weights.*pdf.(d,nodes))
    
# now compute the expectation and verify it's 0
sum(weights.*nodes.*pdf.(d,nodes))

##b) 
## Now estimate the variance of the distribution
nodes, weights = lgwt(7, -10, 10)
##Compute integral
sum(weights.*(nodes.^2).*pdf.(d,nodes))
## Now do the same thing with 10 nodes
nodes, weights = lgwt(10, -10, 10)
##Compute integral
sum(weights.*(nodes^2).*pdf.(d,nodes))
## The second attempt actually estimates it reasonably well

##c)
rng = MersenneTwister(8675309)
rdraw = (rand!(rng, zeros(1000000)).-(.5)).*20
d = Normal(0,2)
20*mean(rdraw.^2 .*pdf.(d,rdraw))
## This is indeed 4
20*mean(rdraw .*pdf.(d,rdraw))
## This is very close to 0 
20*mean(pdf.(d,rdraw))
##This is indeed 1

## Now try the same thing with 1000 draws
rdraw = (rand!(rng, zeros(1000)).-(.5)).*20
d = Normal(0,2)
20*mean(rdraw.^2 .*pdf.(d,rdraw))
## This is pretty close
20*mean(rdraw .*pdf.(d,rdraw))
## This is decently far from 0 
20*mean(pdf.(d,rdraw))
##This is pretty close to 1

X = [df.age df.white df.collgrad] 
N = length(y)
K = size(X, 2)


function mlogit_with_Z_quad(theta, X, Z, y)
        
        alpha = theta[1:end-2]
        gamma = theta[end-1]
		gammavar =exp(theta[end])
        
		K = size(X,2)
        J = length(unique(y))
        N = length(y)
		Time = unique(df.year)
		R = length(unique(df.year))
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end
        
		B = zeros(N) .= 1 
        P = num./repeat(dem,1,J)
		for j=1:J
			B = B.*P[,j]
		end
		
		nodes, weights = lgwt(7, -4*gammavar, 4*gammavar)
		d = Normal(gamma,gammavar)
		t = zeros(N, R)
		for r=1:R
			t[:,r] = df.year .= Time[r]
			t[:,r] = t[:,r].*B
			for n=1:N
				t[n,r] =  sum(weights[r].*t[n,r].*pdf.(d,nodes[r]))
			end
		end
        t .= log.(t)
		t .= sum(t, 2)
		loglike = sum(t)
        return loglike
    end
    startvals = [2*rand(7*size(X,2)).-1; .1;1]
    td = TwiceDifferentiable(theta -> mlogit_with_Z_quad(theta, X, Z, y), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println([theta_hat_mle_ad theta_hat_mle_ad_se])
    startvals = [2*rand(7*size(X,2)).-1; .1]
    td = TwiceDifferentiable(theta -> mlogit_with_Z_quad(theta, X, Z, y), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println([theta_hat_mle_ad theta_hat_mle_ad_se])
	
function mlogit_with_Z_mcmc(theta, X, Z, y)
        
        alpha = theta[1:end-2]
        gamma = theta[end-1]
		gammavar =theta[end]
        
		K = size(X,2)
        J = length(unique(y))
        N = length(y)
		Time = unique(df.year)
		R = length(unique(df.year))
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end
        
		B = zeros(N) .= 1 
        P = num./repeat(dem,1,J)
		for j=1:J
			B = B.*P[,j]
		end
		
		rng = MersenneTwister(8675309)
		rdraw = (rand!(rng, zeros(1000000)).-(.5)).*(10*gammavar)
		d = Normal(gamma,gammavar)
		20*mean(rdraw.^2 .*pdf.(d,rdraw))
		d = Normal(gamma,gammavar)
		t = zeros(N, R)
		for r=1:R
			t[:,r] = df.year .= Time[r]
			t[:,r] = t[:,r].*B
			for n=1:N
				t[n,r] =  (10*gammavar)*sum(t[n,r].*pdf.(d,rdraw))
			end
		end
        t .= log.(t)
		t .= sum(t, 2)
		loglike = sum(t)
        return loglike
    end
    startvals = [2*rand(7*size(X,2)).-1; .1;1]
    td = TwiceDifferentiable(theta -> mlogit_with_Z_mcmc(theta, X, Z, y), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println([theta_hat_mle_ad theta_hat_mle_ad_se])
    startvals = [2*rand(7*size(X,2)).-1; .1]
    td = TwiceDifferentiable(theta -> mlogit_with_Z_mcmc(theta, X, Z, y), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println([theta_hat_mle_ad theta_hat_mle_ad_se])
end
	