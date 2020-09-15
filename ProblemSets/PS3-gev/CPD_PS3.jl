using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables

function ps3()
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

function mlogit(alpha, X, y, Z)
        
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
		gamma = alpha[(K*(J-1))+1]
        bigAlpha = [reshape(alpha[1:(K*(J-1))],K,J-1) zeros(K)]
        
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

K = size(X,2)
 J = length(unique(y))
x0 = rand(K*(J-1)+1)

multinom_please_work = optimize(b -> mlogit(b, X, y, Z), x0, LBFGS(), 
                        Optim.Options(g_tol=1e-5, iterations=100_000, 
                        show_trace=true, allow_f_increases=true))
println(multinom_please_work)
x0

## Question 2
## For a unit change in elenwage, the utility will change with respect to gamma. Since this is a log, gamma/100 should give us the change in utility
## So as wage increases, the probabilty of taking that job increases, which makes perfect sense

function nestlogit(alpha, X, y, Z)
		K = size(X,2)
        J = length(unique(y))
        N = length(y)
		Y = zeros(N,J)
		lambda = zeros(3)
		lambda[1:2] = alpha[(K*2+1):(K*2+2)]
		gamma = alpha[K*2+3]
		bigY = zeros(N,J)
		for j=1:J
            bigY[:,j] = y.==j
        end
		bigAlpha = [reshape(alpha[1:(K*2)],K,2) zeros(K)]
		firstnest = zeros(N, 3)
		secondnest = zeros(N, 4)
		num = zeros(N,J)
        dem = zeros(N)
		secondterm = zeros(N, J)
		m = zeros(N)
		n = zeros(N)
		for j=1:J
			if j <= 3
				num[:,j] = exp.(((X*bigAlpha[:,1] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[1]))
				firstnest[:,j] =  exp.(((X*bigAlpha[:,1] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[1]))
			elseif (j >= 4 && j <= 7)
				num[:,j] = exp.(((X*bigAlpha[:,2] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[2]))
				secondnest[:,(j-3)] = exp.(((X*bigAlpha[:,2] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[2]))
			else 
				num[:,j] .= 1
			end
        end
        m = sum(firstnest, dims=2)
		n = sum(secondnest, dims=2)
		secondterm[:, 1:3] .= (m).^(lambda[1]-1)
		secondterm[:, 4:7] .= (n).^(lambda[2]-1)
		secondterm[:, 8] .= 1
		num = num.*secondterm
		dem = 1 .+ m + n
        P = num./repeat(dem,1,J)

        loglike = -sum(bigY.*log.(P) )
        
        return loglike
	end
	
x1 = rand(K*2+3)

nestlogit_results = optimize(b -> nestlogit(b, X, y, Z), x1, LBFGS(), 
                        Optim.Options(g_tol=1e-5, iterations=100_000, 
                        show_trace=true, allow_f_increases=true))
println(nestlogit_results)
x1
end
ps3()