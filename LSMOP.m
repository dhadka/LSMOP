function [Output,Boundary] = LSMOP(Operation,Problem,M,Input)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  The source code of the test suite of large-scale multi- and many-objective optimization problems (LSMOP)
%%
%%  See the details of LSMOP in the following paper
%%
%%  R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, 
%%  Test Problems for Large-scale Multi- and Many-objective Optimization,
%%  IEEE Transactions on Cybernetics, 2016
%%
%%  The source code of LSMOP is implemented by Ran Cheng 
%%
%%  If you have any questions about the code, please contact: 
%%  Ran Cheng at ranchengcn@gmail.com
%%  Prof. Yaochu Jin at yaochu.jin@surrey.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Operation: 'init' - population initialization
%%            'fitness' - fitness evaluation
%%            'PF' - sample true Pareto front
%%            'PS' - sample true Pareto set
%% Problem: 'LSMOP1' to 'LSMOP9'
%% M: number of objectives
%% Input: if Operation is 'init', Input is the population size
%%        if Operation is 'fitness', Input is the Population to be evaluated
%%        if Operation is 'PF' or 'PS', Input is the sample size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

format long

% global variables
persistent N_k N_ns La Lb Aa Ab Ac NNg D;
% parameters for the test functions
N_k = 5; % number of subcomponents in each decision variable group
N_ns = M*100; % number of decision variables in X^s

Boundary = NaN;
switch Operation
    % population initialization
    case 'init'
        La = []; Lb = []; NNg = {};
        
        % chaos based random number generator
        a = 3.8; c0 = 0.1;
        chaos = @(c) a*c*(1 - c);
        c = chaos(c0); C = [c];
        for i = 1:M - 1
            c = chaos(c);
            C = [C,c];
        end
        
        % non-uniform subcomponent sizes (N_k subcomponents in each decision variable group)
        NNg = round(C(1:M)/sum(C(1:M))*N_ns);
        NNg = ceil(NNg/N_k);
        
        % number of decision variables
        N_ns = sum(NNg)*N_k;
        D = (M - 1) + N_ns;
        
        % boundaries of decision variables
        lu = [zeros(1,M - 1), 0*ones(1,D - M + 1); ones(1,M - 1), 10*ones(1,D - M + 1)];
        Boundary = [lu(2,:); lu(1,:)];

        % variable linkage function
        La = 1 + [M:D]/D;
        Lb = 1 + cos(0.5*pi*[M:D]/D);

        % correlation matrix
        Aa = eye(M); %indepdent correlation
        Ab = eye(M);
        for i = 1:M - 1;
            Ab(i,i+1) = 1; % overlapped correlation
        end;
        Ac = ones(M,M); % full correlation
        
        % output
        Population = rand(Input,D);
        Population = Population.*repmat(Boundary(2,:),Input,1)+(1-Population).*repmat(Boundary(1,:),Input,1);
        Output   = Population;
        
        % fitness evaluation
    case 'fitness'
        %Population = Input;
        %[ps,D] = size(Population);
        
        Population = [ones(1,M - 1), 10*ones(1,D - M + 1)];
        [ps,D] = size(Population);

        % variable linkages
        switch Problem
            case {'LSMOP1','LSMOP2', 'LSMOP3','LSMOP4'}
                % linear linkage
                Population(:,M:end) = Population(:,M:end).*(repmat(La,ps,1)) - 10*repmat(Population(:,1), [1 D - M + 1]);
            case {'LSMOP5','LSMOP6', 'LSMOP7','LSMOP8', 'LSMOP9'}
                % non-linear linkage
                Population(:,M:end) = Population(:,M:end).*(repmat(Lb,ps,1)) - 10*repmat(Population(:,1), [1 D - M + 1]);
        end;

        % non-uniform decision vairable groups
        Xf = Population(:,1:M - 1); % decision variables defining H(x)
        Xs = {}; % decision variables defining G(x)
        for i = 1:M
            if( i > 1 )
                idx_xs1 = M + N_k*sum(NNg(1:i - 1));
            else
                idx_xs1 = M;
            end;
            idx_xs2 = idx_xs1 + N_k*NNg(i) - 1;
            Xs{i} = Population(:, idx_xs1 : idx_xs2);
        end;
        
        % basic single-objective functions
        switch Problem
            case 'LSMOP1'
                g1_func = @(x) sphere_func(x);   % unimodal, separable
                g2_func = @(x) sphere_func(x);   % unimodal, separable
            case 'LSMOP2'
                g1_func = @(x) griewank_func(x);  % multimodal, non-separable
                g2_func = @(x) schwefel_func(x);   % unimodal, non-separable
            case 'LSMOP3'
                g1_func = @(x) rastrigin_func(x);     % multimodal, separable
                g2_func = @(x) rosenbrock_func(x);  % multimodal, non-separable
            case 'LSMOP4'
                g1_func = @(x) ackley_func(x);     % multimodal, separable
                g2_func = @(x) griewank_func(x);  % multimodal, non-separable
            case 'LSMOP5'
                g1_func = @(x) sphere_func(x);   % unimodal, separable
                g2_func = @(x) sphere_func(x);   % unimodal, separable
            case 'LSMOP6'
                g1_func = @(x) rosenbrock_func(x); % multimodal, non-separable
                g2_func = @(x) schwefel_func(x);   % unimodual, non-separable
            case 'LSMOP7'
                g1_func = @(x) ackley_func(x);     % multimodal, separable
                g2_func = @(x) rosenbrock_func(x);   % multimodal, non-separable
            case 'LSMOP8'
                g1_func = @(x) griewank_func(x);     % multimodal, non-separable
                g2_func = @(x) sphere_func(x); % unimodal, separable
            case 'LSMOP9'
                g1_func = @(x) sphere_func(x);  % unimodal, separable
                g2_func = @(x) ackley_func(x);     % multimodal, non-separable
        end
        
        % basic function values
        G = zeros(ps,M);
        for i = 1 : M
            g1 = 0; g2 = 0; %objective function value for g^I and g^II
            % decision variable group correlated with objective i
            nss = NNg(i); % decision variable subcomponent size
            segXss = cell2mat(Xs(i)); % divide decision variable group into subcomponents
            % objective value calculation
            for j = 1 : N_k
                Xss = segXss(:, (j - 1)*nss + 1:j*nss);
                g1 = g1 + g1_func(Xss)/nss;
                g2 = g2 + g2_func(Xss)/nss;
            end;
            if (mod(i,2) == 1)
                G(:,i) = g1;
            else
                G(:,i) = g2;
            end;
        end;
        G = G/N_k;

        disp('G');
        disp(G);

        %objective values
        F = zeros(ps, M);
        for i = 1 : M
            switch Problem
                case {'LSMOP1','LSMOP2', 'LSMOP3', 'LSMOP4'}
                    Gi = G*Aa(i,:)';
                    F(:,i) = (1 + Gi).*prod(Xf(:,1:M-i),2);
                    if i > 1
                        F(:,i) = F(:,i).*(1-Xf(:,M-i+1));
                    end
                case {'LSMOP5','LSMOP6', 'LSMOP7', 'LSMOP8'}
                    Gi = G*Ab(i,:)';
                    F(:,i) = (1+ Gi).*prod(cos(0.5.*pi.*Xf(:,1:M-i)),2);
                    if i > 1
                        F(:,i) = F(:,i).*sin(0.5.*pi.*Xf(:,M-i+1));
                    end
                case {'LSMOP9'}
                    Gi = 1 + G*Ac(i,:)';
                    if i < M
                        F(:,i) = Xf(:,i);
                    else
                        Temp = repmat(Gi,1,M-1);
                        h    = M-sum(F(:,1:M-1)./(1+Temp).*(1+sin(3*pi.*F(:,1:M-1))),2);
                        F(:,M) = (1 + Gi).*h;
                    end;
            end
        end

        disp('F');
        disp(F);
        
        Output = F;
        
        % true Pareto front (f)
    case 'PF'
        switch Problem
            case {'LSMOP1','LSMOP2', 'LSMOP3', 'LSMOP4'}
                PF = T_uniform(Input,M);
            case {'LSMOP5','LSMOP6', 'LSMOP7', 'LSMOP8'}
                PF = T_uniform(Input,M);
                for i = 1 : size(PF,1)
                    PF(i,:) = PF(i,:)./norm(PF(i,:));
                end
            case {'LSMOP9'}
                Temp = T_repeat(Input,M-1);
                PF = zeros(size(Temp,1),M);
                PF(:,1:M-1) = Temp;
                PF(:,M)     = 2*(M-sum(PF(:,1:M-1)/2.*(1+sin(3*pi.*PF(:,1:M-1))),2));
                PF = T_sort(PF);
        end
        Output = PF;
        % true Pareto set (x) 
        % ONLY for 2-objctive problems !!!
    case '2objPS'
       if(M > 2)
            error('PS can only be sampled for 2-objctive problems !!!');
        end;
        PS = zeros(Input,D);
        PS(:,1) = linspace(0, 1, Input);
        switch Problem
            case {'LSMOP1','LSMOP2', 'LSMOP3', 'LSMOP4'}
                PS(:,M:end) = 10*repmat(PS(:,1), [1 D - M + 1])./(repmat(La,Input,1));
            case {'LSMOP5','LSMOP6', 'LSMOP7', 'LSMOP8', 'LSMOP9'}
                PS(:,M:end) = 10*repmat(PS(:,1), [1 D - M + 1])./(repmat(Lb,Input,1));
        end
        % for Rosenbrock's function whose optimum is [1, 1, ..., 1]
        for i = 1 : M
            if( i > 1 )
                idx_xs1 = M + N_k*sum(NNg(1:i - 1));
            else
                idx_xs1 = M;
            end;
            idx_xs2 = idx_xs1 + N_k*NNg(i) - 1;
            mask = zeros(Input, D); mask(:, idx_xs1:idx_xs2) = 1;
            if ( strcmp(Problem,'LSMOP3') == 1 && mod(i, 2) == 0)
                PS(:, M:end)  = (10*repmat(PS(:,1), [1 D - M + 1]) + 1)./(repmat(La,Input,1)).*mask(:, M:end) + PS(:, M:end).*~mask(:, M:end);
            end;
            if ( strcmp(Problem,'LSMOP6') == 1 && mod(i, 2) == 1)
                PS(:, M:end)  = (10*repmat(PS(:,1), [1 D - M + 1]) + 1)./(repmat(Lb,Input,1)).*mask(:, M:end)  + PS(:, M:end).*~mask(:, M:end);
            end;
            if ( strcmp(Problem,'LSMOP7') == 1 && mod(i, 2) == 0)
                PS(:, M:end) = (10*repmat(PS(:,1), [1 D - M + 1]) + 1)./(repmat(Lb,Input,1)).*mask(:, M:end)  + PS(:, M:end).*~mask(:, M:end);
            end;
        end;
        Output = PS;
end


end

%%%%%%%%%%%%%% Basic Single-objctive Functions %%%%%%%%%%%%%%%%%%

% 	Sphere Function
%   Separable
function fit = sphere_func(x)
fit=sum(x.^2,2);
end

%   Schwefel's Problem 2.21
%   Non-separable
function fit = schwefel_func(x)
fit = max(abs(x), [], 2);
end

% 	Rosenbrock's Function
%   Non-separable
function fit = rosenbrock_func(x)
D = size(x,2);
fit = sum(100.*(x(:,1:D-1).^2-x(:,2:D)).^2+(x(:,1:D-1)-1).^2,2);
end

%   Rastrign's Function
%   Separable
function fit = rastrigin_func(x)
fit = sum(x.^2-10.*cos(2.*pi.*x)+10,2);
end

%   Griewank's Function
%   Non-separable
function fit = griewank_func(x)
fit = 1;
D = size(x,2);
for i=1:D
    fit = fit.*cos(x(:,i)./sqrt(i));
end
fit =sum(x.^2,2)./4000-fit+1;
end

%   Ackley's Function
%   Separable
function fit = ackley_func(x)
fit = sum(x.^2,2);
D = size(x,2);
fit = 20-20.*exp(-0.2.*sqrt(fit./D))-exp(sum(cos(2.*pi.*x),2)./D)+exp(1);
end


%%%%%%%%%% accessory functions for sampling true PF and PS %%%%%%%%%%

function W = T_uniform(k,M)
H = floor((k*prod(1:M-1))^(1/(M-1)));
while nchoosek(H+M-1,M-1) >= k && H > 0
    H = H-1;
end
if nchoosek(H+M,M-1) <= 2*k || H == 0
    H = H+1;
end
k = nchoosek(H+M-1,M-1);
Temp = nchoosek(1:H+M-1,M-1)-repmat(0:M-2,nchoosek(H+M-1,M-1),1)-1;
W = zeros(k,M);
W(:,1) = Temp(:,1)-0;
for i = 2 : M-1
    W(:,i) = Temp(:,i)-Temp(:,i-1);
end
W(:,end) = H-Temp(:,end);
W = W/H;
end

function W = T_repeat(k,M)
if M > 1
    k = (ceil(k^(1/M)))^M;
    Temp = 0:1/(k^(1/M)-1):1;
    code = '[c1';
    for i = 2 : M
        code = [code,',c',num2str(i)];
    end
    code = [code,']=ndgrid(Temp);'];
    eval(code);
    code = 'W=[c1(:)';
    for i = 2 : M
        code = [code,',c',num2str(i),'(:)'];
    end
    code = [code,'];'];
    eval(code);
else
    W = [0:1/(k-1):1]';
end
end

function FunctionValue = T_sort(FunctionValue)
Choose = true(1,size(FunctionValue,1));
[~,rank] = sortrows(FunctionValue);
for i = rank'
    for j = rank(find(rank==i)+1:end)'
        if Choose(j)
            k = 1;
            for m = 2 : size(FunctionValue,2)
                if FunctionValue(i,m) > FunctionValue(j,m)
                    k = 0;
                    break;
                end
            end
            if k == 1
                Choose(j) = false;
            end
        end
    end
end
FunctionValue = FunctionValue(Choose,:);
end
