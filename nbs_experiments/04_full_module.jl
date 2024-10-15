using DifferentialEquations;
using Polynomials;
using LinearAlgebra;
using MAT
using NPZ

"""
# DESCRIPTION
Given a matrix [n,m] as an input, the function transforms it into a Vector{n}, for which each element is of dimension m. 
# PROTOTYPE
 matrix2Vec(M::Matrix{T})
# INPUT
Matrix[n,m]
# OUTPUT
Vector{n}
# DEPENDENCIES

# NOTES

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 06-2024
"""
function matrix2Vec(M) 
    out = []
    for i in axes(M,1)
        push!(out,Vector(M[i,:]))
    end
    return out;
end 

"""
# DESCRIPTION
Given a Vector{n}, for which each element is of dimension m, the function transforms it into a  matrix [n,m]
# PROTOTYPE
 vec2Matrix(vec::Vector{Vector{T}})
# INPUT
Vector{n}
# OUTPUT
Matrix[n,m]
# DEPENDENCIES

# NOTES

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 06-2024
"""
function vec2Matrix(vec) 
    return reduce(vcat,transpose.(vec)); 
end


"""
# DESCRIPTION
 Equations of motion of the Circular Restricted 3 Body problem in normalized units 
 (the distance between the two primaries is 1, relative angular velocity is 1). 
 The function accepts the state vector expressed in Cartesian components. 
    The function does not provide any output and it is written in the form accepted by numerical solvers.

# PROTOTYPE
eom_crtbp!(dX,X,p,t)
# INPUT
 dX [6,1]   derivative of the state     d[x,y,z,vx,vy,vz]/dt\\
 X [6,1]    state vector in Cartesian components [x,y,z,vx,vy,vz]\\
 p [1,1]    parameter vector μ [#]\\
 t [1,1]    time instant [#]
# OUTPUT

# DEPENDENCIES

# NOTES
 The system is time-independent i.e. EOM do not depend explicitly on time.

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 04-2024
"""
function eom_crtbp!(dX,X,p,t)
    μ = p[1];

    r1 = sqrt((X[1]+μ)^2 + X[2]^2 + X[3]^2);
    r2 = sqrt((X[1]-(1-μ))^2 + X[2]^2 + X[3]^2)
    
    dX[1] = X[4];
    dX[2] = X[5];
    dX[3] = X[6];
    dX[4] = X[1] + 2*X[5] - (1-μ)*(X[1]+μ)/r1^3 - μ*(X[1]-(1-μ))/r2^3;
    dX[5] =  X[2] - 2*X[4] - X[2]*((1-μ)/r1^3 + μ/r2^3);
    dX[6] = - X[3]*((1-μ)/r1^3 + μ/r2^3);

end

"""
# DESCRIPTION
 Equations of motion of the Circular Restricted 3 Body problem in normalized units (the distance between the two primaries is 1, relative angular velocity is 1). 
 The function accepts the state vector expressed in Cartesian components. It returns the derivative of the state vector.
# PROTOTYPE
dynamics_crtbp(X, μ)

# INPUT
 X [6,1]    state vector in Cartesian components [x,y,z,vx,vy,vz]\\
 μ [1,1]    Gravitational parameter [#]

# OUTPUT
 dX [6,1]   derivative of the state     d[x,y,z,vx,vy,vz]/dt

# DEPENDENCIES

# NOTES

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 04-2024
"""
function dynamics_crtbp(X, μ)

    r1 = sqrt((X[1]+μ)^2 + X[2]^2 + X[3]^2);
    r2 = sqrt((X[1]-(1-μ))^2 + X[2]^2 + X[3]^2)
    
    dX = zeros(6);
    dX[1] = X[4];
    dX[2] = X[5];
    dX[3] = X[6];
    dX[4] = X[1] + 2*X[5] - (1-μ)*(X[1]+μ)/r1^3 - μ*(X[1]-(1-μ))/r2^3;
    dX[5] =  X[2] - 2*X[4] - X[2]*((1-μ)/r1^3 + μ/r2^3);
    dX[6] = - X[3]*((1-μ)/r1^3 + μ/r2^3);

    return dX

end

"""
# DESCRIPTION
 This function outputs the Jacobian of the dynamical equations relative to the state vector. The dimension of the output is thus 6x6
# PROTOTYPE
jacobian_crtbp(X,μ)

# INPUT
 X [6,1]    state vector in Cartesian components [x,y,z,vx,vy,vz]\\
 μ [1,1]    Gravitational parameter [#]

# OUTPUT
DF [6,6]    Jacobian of the EOM (components by row) to the state vector elements (components by column)

# DEPENDENCIES

# NOTES

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 05-2024
"""
function jacobian_crtbp(X,μ)
    x, y, z, xp, yp, zp = X
    
    Z = zeros(3, 3);
    I = Diagonal(ones(3));
    OM = [0 1 0; -1 0 0; 0 0 0];

    r1 = sqrt( (x + μ)^2 + y^2 + z^2 );
    r2 = sqrt( (x - (1 - μ))^2 + y^2 + z^2 );

    Upq = zeros(3,3);
    Upq[1,1] = (μ - 1)/r1^3 - μ/r2^3 + (3*μ*(μ + x - 1)^2)/r2^5 - (3*(μ + x)^2*(μ - 1))/r1^5 + 1;
    Upq[1,2] = (3*μ*y*(2*μ + 2*x - 2))/(2*r2^5) - (3*y*(2*μ + 2*x)*(μ - 1))/(2*r1^5);
    Upq[1,3] = (3*μ*z*(2*μ + 2*x - 2))/(2*r2^5) - (3*z*(2*μ + 2*x)*(μ - 1))/(2*r1^5);
    
    Upq[2,1] =  Upq[1,2];
    Upq[2,2] = (μ - 1)/r1^3 - μ/r2^3 + (3*μ*y^2)/r2^5 - (3*y^2*(μ - 1))/r1^5 + 1;
    Upq[2,3] = (3*μ*y*z)/r2^5 - (3*y*z*(μ - 1))/r1^5;
    
    Upq[3,1] =  Upq[1,3];
    Upq[3,2] =  Upq[2,3];
    Upq[3,3] = (μ - 1)/r1^3 - μ/r2^3 - (3*z^2*(μ - 1))/r1^5 + (3*μ*z^2)/r2^5;
    
    # A matrix expression
    return [Z I; Upq 2*OM];

end

"""
# DESCRIPTION
 Equations of motion of the State transition Matrix of the CR3BP. 
 If Phi is the STM, then its EOM are Phi_dot = A*Phi, where A is the Jacobian of the EOM relative to the state vector 
# PROTOTYPE
 eom_stm_crtbp!(dX, X, p, t)
# INPUT
 dX [42,1]   Derivative of the state (1:6) and STM (7:42)     d[x,y,z,vx,vy,vz, Phi...]/dt\\
 X [6,1]    State vector in Cartesian components and STM components [x,y,z,vx,vy,vz, Phi...]\\
 p [1,1]    parameter vector μ [#]\\
 t [1,1]    time instant [#]
# OUTPUT

# DEPENDENCIES

# NOTES
 The system is time-independent i.e. EOM do not depend explicitly on time.

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 05-2024
"""
function eom_stm_crtbp!(dX, X, p, t)

    μ = p[1];

    r1 = sqrt((X[1]+μ)^2 + X[2]^2 + X[3]^2);
    r2 = sqrt((X[1]-(1-μ))^2 + X[2]^2 + X[3]^2);

    Phiv = X[7:end];
    Phi = reshape(Phiv,(6,6));
    Xv = X[1:6];
    A = jacobian_crtbp(Xv,μ)
    dPhi = A*Phi;
    
    dX[1] = X[4];
    dX[2] = X[5];
    dX[3] = X[6];
    dX[4] = X[1] + 2*X[5] - (1-μ)*(X[1]+μ)/r1^3 - μ*(X[1]-(1-μ))/r2^3;
    dX[5] =  X[2] - 2*X[4] - X[2]*((1-μ)/r1^3 + μ/r2^3);
    dX[6] = - X[3]*((1-μ)/r1^3 + μ/r2^3);

    dX[7:end] = reshape(dPhi,(36,1));
end 

"""
# DESCRIPTION
 The trajectory of the state vector is provided at imposed time-nodes integrating the EOM of the CR3BP

# PROTOTYPE
 propagate_trajectory(X0, tvec, μ, reltol = 1e-16, abstol = 1e-16, solver = Vern7())

# INPUT
 X0 [6,1] Initial state vector [x, y, z, vx, vy, vz] \\
 tvec time vector expressed in 2 ways: 
  +  [1,3] (t_0 dt t_end)    Uniform spacing time vector of step size dt, from start t_0 to end t_end 
  +  [1,N]  user-defined time nodes of length N \\ 

μ   [1,1] Gravitational constant
# OPTIONAL INPUT 
reltol [1,1] relative tolerance of the solver [default: 1e-16] \\
abstol [1,1] absolute tolerance of the solver [default: 1e-16] \\
solver [@function] solver used by ODE solver [default: Vern7]

# OUTPUT
 X [Vector{N}] Vector of dimension N for each node. Each node has dimension 6 and contains the state [6,1] \\
 t_vec [N,1] time vector used for the integration

# DEPENDENCIES

# NOTES
 The system is time-independent i.e. EOM do not depend explicitly on time.

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 05-2024
"""
function propagate_trajectory(X0, tvec, μ; reltol = 1e-16, abstol = 1e-16, solver = Vern7())
    size(X0,2) > 1 && error("Initial condition shall be a column vector")
    tspan = (tvec[1],tvec[end]);
    if length(tvec) ==3 # case when only the uniform spacing is defined
        out_time = tvec[2]; 
    elseif length(tvec) > 3 # case in which the solution is requested for a generic time vector
        out_time = tvec;
    else
        error("specify the time information in a vector of length at least 3.")
    end
    prob = ODEProblem(eom_crtbp!, X0,tspan,μ);
    sol = solve(prob, solver ,reltol = reltol, abstol = abstol, saveat=out_time);

    return sol.u, sol.t
end

"""
# DESCRIPTION
 Given an initial state X0, outputs the state vector X after a given time step dt, also providing the State Transition Matrix

# PROTOTYPE
 get_state(X0, dt, μ; reltol = 1e-10, abstol = 1e-10, solver = Vern7())

# INPUT
 X0 [6,1] Initial state vector [x, y, z, vx, vy, vz] \\
 dt [1,1] Time interval for the integration of initial state X0 \\
μ   [1,1] Gravitational constant
# OPTIONAL INPUT 
reltol [1,1] relative tolerance of the solver [default: 1e-10] \\
abstol [1,1] absolute tolerance of the solver [default: 1e-10] \\
solver [@function] solver used by ODE solver [default: Vern7]

# OUTPUT
 X [6,1] state vector after integration \\
 Phi [6,6] State Transition matrix: dX = Phi*dX0

# DEPENDENCIES

# NOTES
 The system is time-independent i.e. EOM do not depend explicitly on time.

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 05-2024
"""
function get_state(X0, dt, μ; reltol = 1e-10, abstol = 1e-10, solver = Vern7())
    size(X0,2) > 1 && error("Initial condition shall be a column vector")
    Phi0 = Diagonal(ones(6));
    tspan = [0,dt];

    X0_aug = vcat(X0, reshape(Phi0,(36,1)));
    prob = ODEProblem(eom_stm_crtbp!, X0_aug,tspan,μ);
    sol = solve(prob, solver ,reltol = reltol, abstol = abstol, saveat=dt);

    Xfin = sol[end];
    X_end = Xfin[1:6];
    Phi = reshape(Xfin[7:end],(6,6));

    if norm(Phi - Phi0) == 0
        print("reducing tolerance \n")
        sol = solve(prob, solver ,reltol = reltol*1e5, abstol = abstol*1e5, saveat=dt);
        Xfin = sol[end];
        X_end = Xfin[1:6];
        Phi = reshape(Xfin[7:end],(6,6));

    end
    
    return X_end, Phi

end

"""
# DESCRIPTION 
 This funcition computes the Jacobi constant associated to the CR3BP\\
# PROTOTYPE
E,J = jacobi(X,μ)

# INPUT
 X     [6,1]   State vector in the barycentric RF      [x;y;z;vx;vy;vz]\\
 μ     [1,1]   Gravitational parameter m2/(m1 + m2)    [#]\\
# OUTPUT
 E      [1,1]   Energy of the orbit                     [LU^2/TU^2]\\
 J      [1,1]   Jacobi constant J = -2*E                [LU^2/TU^2]\\
 DJ     [1,6]   Jacobian of the Jacobi Constant to state X         

# DEPENDENCIES

# NOTES

# AUTHOR AND VERSION\\
	Ver. 1 - W. Litteri - 04/2024
"""
function jacobi(X,μ)

    length(X) > 6 && error("state vector shall be of size 6.")
     (x,y,z, xp, yp, zp) = X


    μ1 = 1-μ;
    μ2 = μ;

    r1 = sqrt((x+μ2)^2 + y^2 + z^2);
    r2 = sqrt((x-μ1)^2 + y^2 + z^2);

    K = 0.5*(xp^2 + yp^2 + zp^2); #kinetic energy 
    Ubar = -0.5*(x^2 + y^2) -μ1/r1 - μ2/r2 -0.5*μ1*μ2;

    E = K + Ubar;
    J = -2*E;

    # Jacobian of the Jacobian constant relative to the state
    dJx = 2*x - 2*μ1*(x+μ2)/r1^3 - 2*μ2*(x-μ1)/r2^3;
    dJy = 2*y - 2*μ1*y/r1^3 - 2*μ2*y/r2^3;
    dJz = -2*(μ1/r1^3 + μ2/r2^3)*z;

    
    DJ = [dJx dJy dJz -2*xp -2*yp -2*zp]; 

    return E, J, DJ;
end

"""
# DESCRIPTION
This function computes the constant of the CR3BP (Circular Restricted 3 Body problem) 
    providing the normalized units, gravitational parameter for the system under study.

# PROTOTYPE
 μ, LU,TU,VU,LPs = model_constants(type)

# INPUT
 type: a string specifying the system CR3BP. The biggest primary is the first in the list.\\
    "Earth_Moon" - "Sun_Earth" - "Sun_Mars" - "Jupiter-Europa"

# OUTPUT
 μ      [1,1] gravitational parameter (0,1/2)   [#]\\
 LU     [1,1] distance unit                     [km]\\
 TU     [1,1] time unit                         [s]\\
 VU     [1,1] velocity unit                     [km/s]\\
 LPs    [6,5] Lagrange points states in         [LU...; LU/TU...]\\

# DEPENDENCIES
 lagrange_points(μ)
# NOTES

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 04-2024
"""
function model_constants(type)
    if type == "Earth_Moon"
        μ = 1.215058560962404e-2;
        LU = 389703; #km
        TU = 382981; #s
        
    elseif type == "Sun_Earth"
        μ = 3.054200000000000e-6;
        LU = 149597871; #km
        TU = 5022635; #s

    elseif type == "Sun_Mars"
        μ = 3.227154996101724e-7;
        LU = 208321282; #km
        TU = 8253622; #s

    elseif type == "Jupiter_Europa"
        μ = 2.528017528540000e-5;
        LU = 668519; #km
        TU = 48562; #s

    elseif type == "Mars_Phobos"
        μ = 1.611081404409632e-8;
        LU = 9468; #km
        TU = 4452; #s

    elseif type == "Saturn_Enceladus"
        μ = 1.901109735892602e-7;
        LU = 238529; #km
        TU = 18913; #s

    elseif type == "Saturn_Titan"
        μ = 2.366393158331484e-4;
        LU = 1195677; #km
        TU = 212238; #s

    else
        error("Invalid syetem. Check for spelling, or add features.")
    end

    VU = LU/TU;
    LPs = lagrange_points(μ);
    
    return μ, LU, TU,VU, LPs;
end 

"""
# DESCRIPTION
 The function computes the location of the Lagrange Points (LP) in the normalized space, 
    once known the gravitational parameter μ. The result is given in the form of a state space vector
    for which the velocity components are zero.
# PROTOTYPE
LP_matrix = lagrange_points(μ)
# INPUT
μ           [1,1] gravitational parameter (0,1/2)   [#]
# OUTPUT
LP_matrix   [6,5] Matrix containing the states, by column of the LPs [LP1, LP2, LP3, LP4, LP5] [LU...,LU/TU...]
# DEPENDENCIES

# NOTES

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 04-2024
"""
function lagrange_points(μ)
    X = zeros(6,5);

    # Lagrange point L1

        poly = reverse([1, -(3-μ), (3-2*μ), -μ, 2*μ, -μ]);
        rt = roots(Polynomial(poly));
        k = findnext(isreal, rt,1);
        d = rt[k]
        X[1,1] = 1 - μ - d;

        
    # Lagrange point L2    
       
        poly = reverse([1, (3-μ), (3-2*μ), -μ, -2*μ, -μ ]);
        rt = roots(Polynomial(poly));
        k = findnext(isreal, rt,1);
        d = rt[k]
        X[1,2] = 1 - μ + d;

    
    # Lagrange point L3    

        poly = reverse([1, (2+μ), (1+2*μ), -(1-μ), -2*(1-μ), -(1-μ)]);
        rt = roots(Polynomial(poly));
        k = findnext(isreal, rt,1);
        d = rt[k]
        X[1,3] = - μ - d;
     
    # Lagrange point L4    

        X[1,4] = 0.5 - μ;
        X[2,4] = sqrt(3)/2;
    
    
    # Lagrange point L5    

        X[1,5] = 0.5 - μ;
        X[2,5] = -sqrt(3)/2; 
    
    return X
end

dim = 6

"""
# DESCRIPTION
Set the constraints to implement the differential correction scheme. Given a trajectory X and the time vector X, it outputs the constraint vector, 
its Jacobian to be used in a differential correction algorithm.
# PROTOTYPE
constraints(X, t_vec, μ; X_end = [], time_flight = [], jacobi_constant = [], variable_time = true)\\
# INPUT
X [Vector{N}] Vector of dimension N for each node. Each node has dimension 6 and contains the state [6,1] \\
tvec [Vector{N}] time vector containing the time instants for the state vector\\
μ [1,1] gravitational parameter m2/(m1 + m2) [#] 
# OPTIONAL INPUT\\
X_end [Vector{6}] Specify the final state of the trajectory. If not specified, the X_end = X_0 (periodic orbit constraint). Default: [] \\
time_flight [1,1] Time of flight. If specified, the trajectory will try to end at this time of flight. Default: []\\
jacobi_constant [1,1] The orbit is modified to meet a given Jacobi constant. Default: []\\
variable_time [bool] if true, the intervals duration T_1, T_2, ... T_(N-1) is a variable. Note: variable time is required when a given time_flight is set. Default: true

# OUTPUT
X_big [Vector{dim_X_big}] Vector of Variables of dimension "dim_X_big" which depend on the imposed constraints. 
  + if variable_time = true: dim_X_big = Nx6 + (N-1) i.e. the N nodes (first and last nodes are free to change), and the N-1 time intervals
  + if variable_time = false: dim_X_big = Nx6 

F [Vector{dim_F}] Vector of constraints for the differential correction algorithm, of dimension "dim_F"
  + if time_flight and jacobi_constant not specified: dim_F = Nx6 (only dynamical constraint)
  + if time_flight is specified: dim_F = Nx6 + 1 i.e. dynamical constraint + Time of Flight (TOF): sum(time intervals) = desired TOF
  + if jacobi_constant is specified: dim_F = Nx6 + 1 i.e. dynamical constraint + jacobi constant: sum(jacobi constant of each node) = Nx(desired jacobi constant)

DF [dim_F, dim_X_big] Jacobian of the constraint matrix

# DEPENDENCIES

# NOTES

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 06-2024
"""
function constraints(X, t_vec, μ; X_end = [], time_flight = [], jacobi_constant = [], variable_time = true)
    n = length(t_vec);
    T_vec = t_vec[2:end] - t_vec[1:end-1]

    if isempty(time_flight) && isempty(jacobi_constant)
        dim_F = n*6;
    elseif !isempty(time_flight)
        dim_F = n*6 + 1;
    elseif !isempty(jacobi_constant)
        #dim_F = n*6 + n;
        dim_F = n*6 + 1;

    end

    if variable_time 
        X_big = vcat(X..., T_vec);
    else
        X_big = vcat(X...);
    end

    F = zeros(dim_F); 
    DF = zeros(dim_F,length(X_big))

        # building the F and DF matrices
        #Jt = zeros(n-1)
    for i in 1:(n-1)
        #print(i)
        ind_x = (i-1)*6 .+ (1:6);
        ind_y = (i-1)*6 .+ (1:12);
        Xf_i,Phi = get_state(X[i], T_vec[i], μ);
        
        F[ind_x] = Xf_i - X[i+1];
        DF[ind_x, ind_y] = hcat(Phi,-Diagonal(ones(6)));
        if variable_time
            DF[ind_x, 6*n + i] = dynamics_crtbp(Xf_i, μ);
        end
    end
        
    if isempty(X_end) #periodic orbit enforcing 
        F[n*6 .+ (-5:0)] = X[end] - X[1]
        DF[n*6 .+ (-5:0),n*6 .+ (-5:0)] = Diagonal(ones(6))
        DF[n*6 .+ (-5:0),1:6] = -Diagonal(ones(6))
    else # impose the final state to be equal to the imposed one 
        F[n*6 .+ (-5:0)] = X[end] - X
        DF[n*6 .+ (-5:0),n*6 .+ (-5:0)] = Diagonal(ones(6))
    end
        
    if !isempty(time_flight)  #enforce the time of flight
         F[end] = sum(T_vec) - time_flight;
        DF[end,end .+ (-(n-2) : 0)] = transpose(ones(n-1));
    
    elseif !isempty(jacobi_constant) # enforce the jacobi constant as sum of all the jacobi constants
        J = 0
        kk = n
        for i in 1:kk
            _,J_i,DJ_i = jacobi(X[i],μ);
            J += J_i;
            DF[end,(i-1)*6 .+ (1:6)] = DJ_i;
        end
            
        F[end] = J - kk*jacobi_constant;
            
        
    end
 return X_big, F, DF
end #constraints



"""
# DESCRIPTION
This function performs the differential correction scheme, based on the definition of the constraint. Given an initial guess (X_old, t_vec_old), it corrects the 
    nodes in a multiple shooting approach, until when convergence is reached to the desired features, or a maximum number of iterations is exceeded.
# PROTOTYPE
differential_correction(X_old,t_vec_old,μ; variable_time = true, time_flight=[],jacobi_constant = [],  X_end=[], tol = 1e-9, max_iter = 20, printout = false, DF_0 = [], DX_0 = [], X_big_0 = [], δ = [])
# INPUT
X [Vector{N}] Vector of dimension N for each node. Each node has dimension 6 and contains the state [6,1] \\
tvec [Vector{N}] time vector containing the time instants for the state vector\\
μ [1,1] gravitational parameter m2/(m1 + m2) [#] 
# OPTIONAL INPUT
variable_time [bool] [default: true] Set the time intervals as variables of the problem. This shall be true when time of flight is imposed.\\
time_flight [1,1] [default: none] Impose the time of flight on the trajectory (i.e. the period of the orbit)\\
jacobi_constant [1,1] [default: none] Impose the Jacobi constant of the orbit\\
X_end [6,1] [default: none] Impose the final state of the trajectory. If not present, then X_end=X_0 i.e. periodic orbit constraint\\
tol [1,1] [default: 1e-9] Tolerance on the constraint vector for exiting the loop \\
max_iter [1,1] [default: 20] maximum number of iterations for the correction loop. Avoid using a large max_iter\\
printout [bool] [default: false] visulize the result of the iterations, iteration number | error, and convergence
## Pseudo-arclength implementation functinality
These following input are provided to perform the PA continuation scheme, enlarging the dimenion of the problem including also the tangent direction to the family\\
For more info on the procedure, check Spreen, 2021, pag 63.\\
DX_0 [Vector{dim_X_big}] [default: none] Vector of the null-space of the Jacobian of the constraint vectors, computed at the previously converged solution \\
X_big_0 [Vector{dim_X_big}] [default: none] Vector of the problem variables of dimension imposed by function constraints()
δ [1,1] [default: none] step of the pseudo arclength scheme: (X_big_i - X_big_zero)^T*DX_0 = δ [Spreen, 2021, eq. 3.29]

# OUTPUT
X_new [Vector{N}] Vector of dimension N for the new trajectory. Each node has dimension 6 and contains the state [6,1] \\
tvec [Vector{N}] time vector containing the time instants for the state vector\\
error [1,1] residual error on the constraint at return point\\
k [1,1] number of iterations required to converge
success [1,1] Convergence of the algorithm. +1 is converged, -1 if not converged. 
# DEPENDENCIES

# NOTES

# AUTHOR AND VERSION
	Ver. 1 - W. Litteri - 06-2024
"""
function differential_correction(X_old,t_vec_old,μ; variable_time = true, time_flight=[],jacobi_constant = [],  X_end=[], tol = 1e-9, max_iter = 20, printout = false, DX_0 = [], X_big_0 = [], δ = [])

    k = 0; 
    n = length(t_vec_old); 

    
    #if the time is not a variable, then the time of flight cannot be imposed
    if !variable_time && !isempty(time_flight) 
        error("Set the time nodes as variables to specify the time of flight.")
    end

    while k < max_iter
        k += 1;
        
        global X_big, F, DF = constraints(X_old, t_vec_old, μ; X_end = X_end, time_flight = time_flight, jacobi_constant = jacobi_constant, variable_time = variable_time)

        if isempty(DX_0)
            # The criterion chosen for convergence is the norm of the constraint vector. One alternative could also be the norm of the state correction δX = (X_big_new - X_big_old). 
            # They are in the same order of magnitude, and the norm(F) (or G) approach seems to be more conservative than the δX one.

            if norm(F) <= tol && t_vec_old[end] > 1e-6
                if printout
                    print("converged in ", k, " iterations \n")
                end
            
                success = 1 # operation succeded: return 1
                return X_old, t_vec_old, norm(F), k, success 
                break

            elseif norm(F) >= 10 && k>1
                if printout
                    print("solution diverged after ", k, " iterations \n")
                end
            
                success = -1 # operation failed: return -1
                return X_old, t_vec_old, norm(F), k, success 
                break


            end
        # creation of the new states and times
            
            X_big_new = X_big - pinv(DF)*F;
        else
            arc_constr = dot(X_big - X_big_0, DX_0) - δ
            global G = vcat(F, arc_constr)
            DG = vcat(DF, reduce(hcat, DX_0))
            
            if norm(G) <= tol && t_vec_old[end] > 1e-6
                if printout
                    print("converged in ", k, " iterations \n")
                end
            
                success = 1 # operation succeded: return 1
                return X_old, t_vec_old, norm(G), k, success 
                break

            elseif norm(G) >= 10 && k>1
                if printout
                    print("solution diverged after ", k, " iterations \n")
                end
            
                success = -1 # operation failed: return -1
                return X_old, t_vec_old, norm(G), k, success 
                break

            end

        X_big_new = X_big - pinv(DG)*G;
        end
    
        X_new = []; 
        
        for i = 1:n
            push!(X_new,X_big_new[(i-1)*6 .+ (1:6)]);
        end
        
        if variable_time
            t_vec_new = zeros(1);
            for i=1:n-1
                push!(t_vec_new, t_vec_new[end] + X_big_new[6*n + i]);
            end
            t_vec_old = copy(t_vec_new);

        end
        # end of creation of new states and times 

        # update of states and times to subsequent iteration
        X_old = copy(X_new); 
        

        if isempty(DX_0)
            printout && print(k, "   |   ", norm(F), "\n")
        else
            printout && print(k, "   |   ", norm(G),"\n")
        end

       
        
    end #while

    
    success = -1 # operation failed: return -1
    if isempty(DX_0)
        return X_old, t_vec_old, norm(F), k, success 
    else
        return X_old, t_vec_old, norm(G), k, success 
    end
    
end # differential_correction


experiment_number = "1"
folder_address = "C:/Users/walth/University of Strathclyde/MAE_OrbitGPT - General/Experiments/experiment_"*experiment_number*"/"

#orbits_address = folder_address*"exp1_generated_data.npy"
family = "family_10/"
type = "orto"
#orbits_address = folder_address*"continuations/"*family*family[1:end-1]*"_continuation_"*type*"_sorted_end.npy"
#orbit_address_not_sorted = folder_address*"continuations/"*family*family[1:end-1]*"_continuation_"*type*"_end.npy"
orbits_address = folder_address*"continuations/"*"family_10_12/sampling_12_sorted.npy"
orbits_address_not_sorted = folder_address*"continuations/"*"family_10_12/sampling_12.npy"
orbit_array = npzread(orbits_address)
orbit_array_not_sorted = npzread(orbits_address_not_sorted)

m,n,l = size(orbit_array)

μ, LU, TU,VU, LPs = model_constants("Earth_Moon");
 
ICs_corrected = zeros(1,11) # kk
print("ciao")
save_result = false
##
for i in 1:m
    print("\n \n ", i, "\n")
    t_vec_old_i_pre = orbit_array[i,:,1]
    X_old_i_pre = matrix2Vec(orbit_array[i,:,2:7])

    t_vec_old_i_post = [t_vec_old_i_pre[1]; t_vec_old_i_pre[30:15:300]] #t_vec_old_i_pre #
    X_old_i_post = [[X_old_i_pre[1]];X_old_i_pre[30:15:300] ] #X_old_i_pre #

    #global X_new, t_vec_new, err_i, k, state = differential_correction(X_old_i_post,t_vec_old_i_post,μ, time_flight = t_vec_old_i_post[end], max_iter = 20, printout = true, tol = 1e-9)
    global X_new, t_vec_new, err_i, k, state = differential_correction(X_old_i_post,t_vec_old_i_post,μ, max_iter = 20, printout = false, tol = 1e-9)
    #print(k, "\n")
    if state == 1
        print("success! Periodic orbit converged \n")

        global ICs_corrected = [ICs_corrected;[Int(i) t_vec_old_i_post[end] t_vec_new[end] k err_i X_new[1]... ]]
        
    end
end
ICs_corrected = ICs_corrected[2:end,:]