using Random
using JLD2 
using FileIO
using DataFrames
using CSV
using Distributions

function q1()
## i) 

Random.seed!(1234)
x = 15
y = -5
A = zeros(10, 7)
rand!(A)
A = A .* x
A = A .+ y

## ii)

x = 15
y=-2 
B = zeros(10, 7)
randn!(B)
B = B .* x
B = B .+ y 

## iii) 
C = A[1:5,1:5]
E = B[1:5, 6:7]
C = transpose(C)
E = transpose(E)
C = [C; E]
C= transpose(C)
C = convert(Matrix, C)

## iv)
x = zeros(0)

for i in 1:10, j= 1:7 
	if A[i,j] > 0
		append!(x,0)
	else 
		append!(x, A[i,j])
	end
end

D = reshape(x, 10, 7)

## b)
length(A)

## c)
length(unique(D))

## d)

E = reshape(B, 70, 1)

## The easier way is not vectorizing at all and using .* and .-, ect, as operators

## e) 

F = zeros(10,7,2)
F[:, :, 1]= A
F[:, :, 2]=B
F

## f)

F = permutedims(F, [3, 1, 2])

## g) 

G= kron(B, C)
## not run in function, causes error kron(C, F)

##Nothing doing for the Kronecker product (C,F). 

## h)
file = File(format"JLD2", "matrixpractice.jld2")
save(file, "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

## i)

file = File(format"JLD2", "firstmatrix.jld2")
save(file, "A", A, "B", B, "C", C, "D", D)

## j)

C = convert(DataFrame, C)

CSV.write("C:/Users/chris/Documents/fall-2020/ProblemSets/Cmatrix.csv", C)

## k) 

D = convert(DataFrame, D)
CSV.write("C:/Users/chris/Documents/fall-2020/ProblemSets/Dmatrix.dat", D, delim=" ")

C= convert(Array,C)
D= convert(Array, D)

## l)
return[A,B,C,D]
end

## 2)

## a)
function q2(L...)
	AB = zeros(10, 7)
	for i in 1:10, j= 1:7
		AB[i,j] = A[i,j]*B[i,j]
	end
	AB2 = A .* B
	## b)
	x = zeros(0)
	for i in 1:5, j=1:7
		if abs(C[i,j]) > 5  
			append!(x, 0)
		else
			append!(x, C[i,j])
		end
	end
	Cprime = reshape(x, 5, 7)
	Cprime2 = C
	Cprime2[Cprime2 .> 5] .= 0
	Cprime2[Cprime2 .< -5] .= 0
	## c)
	a = zeros(15169)
	b = zeros(15169)
	rand!(Binomial(20, 0.6),a)
	rand!(Binomial(20, 0.5),b)
	X = zeros(15169,6,5)
	for t in 1:5
		x = zeros(15169)
		y= zeros(15169)
		z = zeros(15169)
		rand!(x)
		randn!(y)
		randn!(z)
		X[:, 1,t] .= 1
		X[findall(h->h > .75*(6-t)/5, x) , 2, t] .= 1
		for i in 1:15169
			X[i, 3, t] = (y[i]*(5*(t-1))+(15+t-1))
			X[i, 4, t] = (z[i]*(1/ℯ))+ (π*(6-t)/3)
			X[i, 5, t] = a[i]
			X[i, 6, t] = b[i]
		end
	end
	## d) 
	p = [.75+ .25i for i=1:5]
	p1 = [log(i) for i=1:5]
	p2 = [-sqrt(i) for i=1:5]
	p3 = [ℯ^i - ℯ^(i+1) for i=1:5]
	p4 = [i for i=1:5]
	p5 = [i/3 for i=1:5]
	append!(p, p1)	
	append!(p, p2)
	append!(p, p3)
	append!(p, p4)
	append!(p, p5)
	p =reshape(p, 5, 6)
	β = transpose(p)
	## e) 
	Y=[]
	for t=1:5
		e= zeros(15169)
		randn!(e).*.36
		append!(Y, X[:,:,t]*β[:,t] .+ e)
	end
	Y = reshape(Y, 15169, 5)
	return
end

## 3
function q3()	
	##a)
	nlsw88 = CSV.read("C:/Users/chris/Documents/fall-2020/ProblemSets/PS1-julia-intro/nlsw88.csv")
	file = File(format"JLD2", "nslw88.jld2")
	save(file, "nlsw88", nlsw88)
	##b)
	by(nlsw88, :never_married, nrow)
	## about 10.42% have never been married
	by(nlsw88, :never_married, nrow)
	## about 23.7% are college graduates
	##c) 
	by(nlsw88, :race, nrow)
	## 26% are in race group 2
	## 72.9% are in race group 1
	## 1.1% are in race group 3
	##d) 
	describe(nlsw88)
	## There are 2 observations 
	##e)
	h = vcat(:industry, :occupation)
	by(nlsw88, h, nrow)
	##f) 
	sub1 = nlsw88[:, [:industry, :occupation, :wage]]
	by(sub1, h, mean = :wage => mean)
	return
end

function q4()
## 4
#a) 

load("firstmatrix.jld2")

#b, c)
function matrixops(g::Array, h::Array)
	##Given two matrices of the same size, returns the element by element product in a new matrix x1
	##Also returns the product of matrices transpose(g)*h as x2
	##Returns the sum of all elements of both matrices as x3
	x1 = g .* h
	x2 = transpose(g)*h
	x3 = sum(g + h)
	return [x1, x2, x3]
end

# d)

a=matrixops(A,B)

# e)
function matrixops(g::Array, h::Array)
	##Given two matrices of the same size, returns the element by element product in a new matrix x1
	##Also returns the product of matrices transpose(g)*h as x2
	##Returns the sum of all elements of both matrices as x3
	if size(g) == size(h)
		x1 = g .* h
		x2 = transpose(g)*h
		x3 = sum(g + h)
		return [x1, x2, x3]
	else
		print("Slow down there, cowfolk! You need two matrices to be the same size if you want to you this here function")
	end	
end

## f) 
matrixops(C,D)

##I get an error message :(

#g) 
nlsw88 = CSV.read("C:/Users/chris/Documents/fall-2020/ProblemSets/PS1-julia-intro/nlsw88.csv")
exp = convert(Array, nlsw88.ttl_exp)
wage = convert(Array, nlsw88.wage)	

matrixops(exp, wage)
return
end

A,B, C, D = q1()
q2(A,B,C,D)
q3()
q4()