using DataFramesMeta
using HTTP
using Optim
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV

function ps5()
## Question 1
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body)

# create bus id variable
df = @transform(df, bus_id = 1:size(df,1))

#---------------------------------------------------
# reshape from wide to long (must do this twice be-
# cause DataFrames.stack() requires doing it one 
# variable at a time)
#---------------------------------------------------
# first reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])

## Estimate using GLM
myopic = glm(@formula(Y ~ Branded + Odometer), df_long, Binomial(), LogitLink())
println(myopic)

## Question 3
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body)
Y = df[1:20]
Odo = df[21:40]
Xst = df[43:62]

#b)
@views @inbounds function dynamic(theta)
include("C:/Users/chris/Documents/fall-2020/ProblemSets/PS5-ddc/create_grids.jl")
zval,zbin,xval,xbin,xtran= create_grids()
FV = zeros(20301,2,21)
beta= .9
for t in 0:20
for b in 1:2
for z in 1:101
for x in 1:201
row = x+ (z-1)*xbin
v = theta[0] + theta[1].*xval[x]+theta[2].*zval[z] + beta.*xtran[row,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
condv = beta.*xtran[1+(z-1)*xbin,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
FV[t] = beta.*log(exp(condv)+ exp(v))
end
end
end
end

#d)
loglike = 0
for i in 1:size(Y, 1)
for t in 1:20
N = size(Y, 1)
P = zeros(N, 20)
P1 = zeros(N, 20)
P0 = zeros(N, 20)
rep =1+ (df.Zst[i]-1)*xbin
notrep = Xst[i ,t] + (df.Zst[i]-1)*xbin
P[i,t] =theta[0] + theta[1]*Xst[i,t] + theta[2]*df.Branded[i] +(xtran[rep,:].-xtran[notrep, :])'*FV[notrep:notrep+xbin-1, df.Branded[i], t+1]
end
end
P1 = exp.(P)./(1 .+exp.(P))
P0 = 1 .- P1
loglike = log(sum(P1))+log(sum(P0))
return loglike
end

x0 = [0,0,0]
dynam_please_work = optimize(b -> dyanamic(b), x0, LBFGS(), 
                        Optim.Options(g_tol=1e-5, iterations=100_000, 
                        show_trace=true, allow_f_increases=true))
println(dynam_please_work)
end