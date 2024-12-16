function [A,B,C,D,x,x_CD] =  construct_LPV(x0,x0_CD,n_var,ns,nc,ny,inputs,state_dx,outputs,gaopt,fminconopt)
    
    rng(34535)

    % set options
    options_ga = optimoptions('ga','UseParallel',true,'Display','iter','MaxGenerations',n_var*2);
    
    m = mean(inputs,1);
    m(2) = m(2)/1e4;
    disp(m)
    
    % solve for C and D matrices
    CD = linsolve(inputs,outputs);
    CD = CD';

    C = CD(:,nc+1:end);
    D = CD(:,1:nc);

    x0_CD = CD(:);

    % set fmincon options
    options_fmincon = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxIterations',500,'MaxFunctionEvaluations',20000,'UseParallel',true,"EnableFeasibilityMode",true,'StepTolerance',1e-14,'ConstraintTolerance',1e-3);
    options_fmincon = optimoptions('fmincon','Display','iter','Algorithm','interior-point','MaxIterations',500,'MaxFunctionEvaluations',20000,'UseParallel',true,"EnableFeasibilityMode",true,'StepTolerance',1e-8,'ConstraintTolerance',1e-4);
     n_CD = length(x0_CD);
  
    if isempty(x0)

        % if x0 is empty, then it implies that this is the first call, and
        % we should use ga to find a good starting point

        x = ga(@(x) objective(x,ns,nc,ny,inputs,state_dx,outputs,'deriv'),n_var,[],[],[],[],[],[],@(x)constraints(x,ns,nc),[],options_ga) ;

        [x,F] = fmincon(@(x) objective(x,ns,nc,ny,inputs,state_dx,outputs,'deriv'),x,[],[],[],[],[],[],@(x)constraints(x,ns,nc),options_fmincon);
        
        %disp('Estimating C and D matrices')
        % x_CD = ga(@(x) objective(x,ns,nc,ny,inputs,state_dx,outputs,'output'),n_CD,[],[],[],[],[],[],[],[],options_ga);
        % 
        % x_CD = fmincon(@(x) objective(x,ns,nc,ny,inputs,state_dx,outputs,'output'),x_CD,[],[],[],[],[],[],[],options_fmincon);

    else
        
        % else, use fmincon to find the model parameters
        [x,F] = fmincon(@(x) objective(x,ns,nc,ny,inputs,state_dx,outputs,'deriv'),x0,[],[],[],[],[],[],@(x)constraints(x,ns,nc),options_fmincon);
        %disp('Estimating C and D matrices')
        %x_CD = ga(@(x) objective(x,ns,nc,ny,inputs,state_dx,outputs,'output'),n_CD,[],[],[],[],[],[],[],[],options_ga);

        %x_CD = fmincon(@(x) objective(x,ns,nc,ny,inputs,state_dx,outputs,'output'),x0_CD,[],[],[],[],[],[],[],options_fmincon);
    end


     % get linear model
    [A,B] = LTI_function(x,ns,nc);
    x_CD = [];

    % estimate CD

    

   %[C,D] = output_function(x_CD,ns,nc,ny);


end


function V = objective(x,ns,nc,ny,inputs,dx_act,outputs,func)

    if strcmpi(func,'deriv')

        % get linear model
        [A,B] = LTI_function(x,ns,nc);
        
        % evaluate state derivatives
        dx_predicted = inputs*[B,A]';

    elseif strcmpi(func,'output')

        [C,D] = output_function(x,ns,nc,ny);

        dx_predicted = inputs*[D,C]';

        dx_act = outputs;
    end

    % number of data samples
    N = length(inputs);

    % calculate error
    error = dx_act - dx_predicted;
  
    % calculate loss
    V = 1/N*(trace(error'*error));

end

function [C,D] = output_function(x,ns,nc,ny)

CD = reshape(x,[ny,ns+nc]);

C = CD(:,nc+1:end);
D = CD(:,1:nc);


end



function [A,B] = LTI_function(x,ns,nc)


    % reshape
    x = reshape(x,[ns/2,ns+nc]);
    
    % extract right elements
    B_par = x(:,1:nc);
    A_par = x(:,nc+1:nc+ns);
    
    A = [zeros(ns/2),eye(ns/2);A_par];
    B = [zeros(ns/2,nc);B_par];

end

function [c,ceq] = constraints(x,ns,nc)

    % get the linear model
    [A,~] = LTI_function(x,ns,nc);
    
    % evaluate the eigen values
    eigA = eig(A);
    
    % real values
    c = max(real(eigA))+0.0;
    
    % no equality constraints
    ceq = [];

end