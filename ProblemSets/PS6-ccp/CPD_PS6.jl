using DataFramesMeta
using HTTP
using Optim
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV

include("C:/Users/chris/Documents/fall-2020/ProblemSets/PS6-ccp/create_grids.jl")

function ps6()
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

## Question 2
flexlog = glm(@formula(Y ~ Odometer*(Odometer)^2*RouteUsage*(RouteUsage)^2*Branded*time*(time)^2), df_long, Binomial(), LogitLink())
println(flexlog)

Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
X = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
Z = Vector(df[:,:RouteUsage])
B = Vector(df[:,:Branded])
N = size(Y,1)
T = size(Y,2)
Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
Zstate = Vector(df[:,:Zst])
    
##Question 3a
zval,zbin,xval,xbin,xtran = create_grids()

E = DataFrame()
		E.Odometer = kron(ones(zbin), xval)
		E.RouteUsage = kron(ones(xbin),zval)
		E.Branded = zeros(size(xtran,1))
		E.time= zeros(size(xtran,1))
		
@views @inbounds function futval(Zstate,Xstate,xtran,zbin,xbin,E)
		
        FV=zeros(zbin*xbin,2,T+1)

        # First loop: solve the backward recursion problem given the values of α
        # This will give the future value for *all* possible states that we might visit
        # This is why the FV array does not have an individual dimension
        for t=2:T
            for b=0:1
				E.time .= t
				E.Branded .= b 
				p0 = predict(flexlog, E)
                FV[:, b+1, t] = - .9 .*log.(p0)
            end
        end
		
		FVT1 = zeros(size(Y,1), T)
		for i=1:size(df,1)
		notrep = Int((Zstate[i]-1)*xbin+1)
		for t=1:20
		rep  = Int(Xstate[i,t] + (Zstate[i]-1)*xbin)    																		
		FVT1[i,t] = (xtran[rep,:].-xtran[notrep,:])'*FV[notrep:notrep+xbin-1,B[i]+1, t+1]
		end
		end
	mat = FVT1'[:]
	return mat
end	

ftv1 = futval(Zstate,Xstate,xtran,zbin,xbin,E)
df_long = @transform(df_long, fv=ftv1)

theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded), 
                    df_long, Binomial(), LogitLink(), 
                    offset=df_long.fv)
println(theta_ccp)
end

@time ps6()