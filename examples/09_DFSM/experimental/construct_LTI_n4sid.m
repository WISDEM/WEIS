function [A,B,C,x0] = construct_LTI_n4sid(inputs,outputs,n_inputs,n_outputs,outputs_max,n_tests,nx,nt,dt)
    
    % scale outputs
    outputs = outputs./outputs_max;

    % organize the input/output data into the iddata type
    if n_tests == 1

        data = iddata(outputs,inputs,dt);

    elseif n_tests > 1
        
        % initialize storage cell
        data = cell(n_tests,1);

        % reshape inputs and outputs
        
        inputs = reshape(inputs,n_tests,nt,n_inputs);
        outputs = reshape(outputs,n_tests,nt,n_outputs);
    
        % loop through and organize and store data
        for i = 1:n_tests
            data{i} = iddata(squeeze(outputs(i,:,:)),squeeze(inputs(i,:,:)),dt);
        end
        
        % merge cells
        data = merge(data{:});
    
    end

    % options
    options = n4sidOptions('InitialState','estimate','N4Weight','SSARX','Focus',...
                'simulation','EnforceStability',true,'Display','off');
    
    % construct model
    [sys,x0] = n4sid(data,'nx',nx,'Ts',dt,options,'DisturbanceModel','none');
    
    % extract system matrices
    A = sys.A;
    B = sys.B;
    C = sys.C;


end
