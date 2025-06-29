    
    UseGpu = 1; % if zero, uses CPU
    
    if UseGpu 
        if canUseGPU
        gpu = gpuDevice();
        reset(gpu)
        
        gpu = gpuDevice();
        disp(gpu)
        wait(gpu)
        
        clear all; clc;
        UseGpu = 1;
        disp("GPU Selected")
        else 
            clear; clc;
            UseGpu = 0;
            disp("GPU cannot be used, selected CPU")
        end
    else 
        clear all; clc;
        UseGpu = 0;
        disp("CPU Selected")
    end
    
    
    %% Data preparation
    
    % This section should be customised for your input(s)
    % I prepared a function which simulates very simple profiles
    % you can put here profiles from experiments or numerical datasets
    % a dataset used for testing the algorithm has been developed using 
    % TokaLab - https://tokalab.github.io/ 
    
    % Synthetic dataset (no physics-realted, just to test)
    
    [n,T,r] = generate_kinetic_profiles(63,1000,0.1);

    % data should be placed in this strcture

    Data.N = n; % density
    Data.T = T; % temperature
      
    %% Preprocessing and Normalisation
    
    % Normalisation Type I - Shape
    
    Norm.N_mean = max(median(Data.N,1),5e18);
    Norm.T_mean = max(median(Data.T,1),300);
    
    Data.N_norm = Data.N./Norm.N_mean;
    Data.T_norm = Data.T./Norm.T_mean;
    
    %% dl array preparation
    
    X = [Data.N_norm; Data.T_norm];
    dX = dlarray(X,'CB');
    
    if canUseGPU
        dX = gpuArray(dX);
    end
    
    %% PCA based sampling

    Config.Sampling_PCA = 1;
    
    if Config.Sampling_PCA == 1
        [~,PC,~,~] = pca(X');
        PC = PC(:,1:2);
    
        [PC_1g,PC_2g] = meshgrid(linspace(min(PC(:,1)),max(PC(:,1)),10),...
            linspace(min(PC(:,2)),max(PC(:,2)),11));
        
        PC_1d = mean(diff(PC_1g'),'all');
        PC_2d = mean(diff(PC_2g),'all');
    
        PC_1g = PC_1g(:)';
        PC_2g = PC_2g(:)';
    
        PC_dist = (PC(:,1)-PC_1g).^2 + (PC(:,2)-PC_2g).^2;
        [~,PC_ind_sort] = sort(PC_dist,1);
        
    end
    
    %% AutoEncoder Initialisation
    
    VAE.CodeSize = 8;
    VAE.Encoder = [50 20 20 20 10];
    VAE.Decoder = flip(VAE.Encoder);
    
    [~,~,parameters] = VAE_Network(dX(:,1),VAE,[],0,0);
    [dXp,dCode] = VAE_Network(dX(:,1:10),VAE,parameters,1,0);
    
    %% Model Gradient
    
    accfun = dlaccelerate(@MG_AutoEncoder_v1);
    
    %% Training Options and Initialisation
    
    % Initialisation
    iteration = 0;
    averageGrad = [];
    averageSqGrad = [];
    
    % Learning rate
    VAE.LearningRate0 = 1e-3;
    VAE.DecayRate = 1e-4;
    VAE.MiniBatchSize = 10000;
    VAE.alpha = 1;
    VAE.Sigma = 0.01;
    
    %% Training
    
    figure(1)
    clf
    
    N = size(dX,2);
    
    for epoch = 1 : 1000
    
        for i = 1 : 100
    
            % take mini batch
            if Config.Sampling_PCA == 1
                ind_for_sort = randsample(size(PC_ind_sort,1),floor(VAE.MiniBatchSize/size(PC_ind_sort,2)));
                ind_batch = PC_ind_sort(ind_for_sort,:);
                ind_batch = ind_batch(:); 
            else
                ind_batch = randsample(N,VAE.MiniBatchSize);
            end
    
            dX_batch = dX(:,ind_batch);
    
            % Evaluate gradients
            [gradients,MSE,Reg] = dlfeval(accfun,dX_batch,VAE,parameters);
    
            % calcolo le iterazioni
            iteration = iteration + 1;
    
            % Learning rate update
            LearningRate = VAE.LearningRate0./(1+VAE.DecayRate*iteration);
    
            % adam update
            [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
                averageSqGrad,iteration,LearningRate);
    
        end
    
    
        %% Plotting Data
    
        ind_plot = randsample(size(X,2),100);
    
        X_plot = X(:,ind_plot);
        dX_plot = dX(:,ind_plot);
    
        MSE = double(extractdata(gather(MSE)));
        Reg = double(extractdata(gather(Reg)));
    
        [dXp,Codep] = VAE_Network(dX_plot,VAE,parameters,1,VAE.Sigma);
    
        Xp = double(extractdata(gather(dXp)));
        
        Codep = double(extractdata(gather(Codep)));
    
        figure(1)
        subplot(2,3,1)
        plot(epoch,MSE,'.b','markersize',16)
        hold on
        plot(epoch,Reg,'.r','markersize',16) 
        ylim([max(MSE,Reg)*0.1 inf])
        set(gca,'yscale','log')
        grid on
        grid minor
        xlabel("epoch")
        ylabel("Loss terms")
        legend("MSE","Reg",'Location','southwest')
    
        subplot(2,3,4)
        hold off
        plot(X_plot(:),Xp(:),'.b')
        hold on
        plot([0 5],[0 5],...
            '-.r','LineWidth',1.2)
        grid on
        grid minor
        xlabel("Target (normalised)")
        ylabel("Predicted (normalised)")
        ylim([0 3])
        xlim([0 3])

        subplot(2,3,2)
        hold off
        plot(X_plot(1:63,1),'.-b')
        hold on
        plot(Xp(1:63,1),'.-r')
        grid on
        grid minor
        xlabel("ch")
        ylabel("Density")
    
        subplot(2,3,3)
        hold off
        plot(X_plot(1:63,30),'.-b')
        hold on
        plot(Xp(1:63,30),'.-r')
        grid on
        grid minor
        xlabel("ch")
        ylabel("Density")
    
        subplot(2,3,5)
        hold off
        plot(X_plot(63+(1:63),1),'.-b')
        hold on
        plot(Xp(63+(1:63),1),'.-r')
        grid on
        grid minor
        xlabel("ch")
        ylabel("Temperature")
        try % works only if code size larger than one
            subplot(2,3,6)
            plot(Codep(1,:),Codep(2,:),'.b')
            grid on
            grid minor
            xlabel("Code 1")
            ylabel("Code 2")

        end
    
        drawnow
    
    end

    
    
    
    

