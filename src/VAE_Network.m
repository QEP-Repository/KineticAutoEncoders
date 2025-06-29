
function [X,Code,parameters] = VAE_Network(X,VAE,parameters,Predict,Sigma)

if Predict == 0

    %% Initialisation

    OutputSize = size(X,1);

    % Encoder

    for i = 1 : length(VAE.Encoder)

        parameters.("en"+i).weights = dlarray(normrnd(0,1,VAE.Encoder(i),size(X,1)))/5;
        parameters.("en"+i).bias = dlarray(zeros(VAE.Encoder(i),1));

        X = fullyconnect(X,parameters.("en"+i).weights, parameters.("en"+i).bias);
        X = tanh(X);

    end

    % Code
    parameters.("code").weights = dlarray(normrnd(0,1,VAE.CodeSize,size(X,1)));
    parameters.("code").bias = dlarray(zeros(VAE.CodeSize,1));

    X = fullyconnect(X,parameters.("code").weights, parameters.("code").bias);
    % Code = tanh(X);
    Code = X;

    X = normrnd(Code,Sigma);

    % Decoder

    for i = 1 : length(VAE.Decoder)

        parameters.("de"+i).weights = dlarray(normrnd(0,1,VAE.Decoder(i),size(X,1)))/10;
        parameters.("de"+i).bias = dlarray(zeros(VAE.Decoder(i),1));

        X = fullyconnect(X,parameters.("de"+i).weights, parameters.("de"+i).bias);
        X = tanh(X);

    end

    parameters.("output").weights = dlarray(normrnd(0,1,OutputSize,size(X,1)));
    parameters.("output").bias = dlarray(zeros(OutputSize,1));

    X = fullyconnect(X,parameters.("output").weights, parameters.("output").bias);

elseif Predict == 1 % Encoder + Decoder

    % Encoder

    for i = 1 : length(VAE.Encoder)

        X = fullyconnect(X,parameters.("en"+i).weights, parameters.("en"+i).bias);
        X = tanh(X);

    end

    % Code

    X = fullyconnect(X,parameters.("code").weights, parameters.("code").bias);
    % Code = tanh(X);
    Code = X;
    X = normrnd(Code,Sigma);

    % Decoder

    for i = 1 : length(VAE.Decoder)

        X = fullyconnect(X,parameters.("de"+i).weights, parameters.("de"+i).bias);
        X = tanh(X);

    end

    X = fullyconnect(X,parameters.("output").weights, parameters.("output").bias);

elseif Predict == 2 % Decoder (Generative)

    X = normrnd(X,Sigma);

    for i = 1 : length(VAE.Decoder)

        X = fullyconnect(X,parameters.("de"+i).weights, parameters.("de"+i).bias);
        X = tanh(X);

    end

    X = fullyconnect(X,parameters.("output").weights, parameters.("output").bias);

elseif Predict == 3 % Decoder (Generative)

    % Encoder

    for i = 1 : length(VAE.Encoder)

        X = fullyconnect(X,parameters.("en"+i).weights, parameters.("en"+i).bias);
        X = tanh(X);

    end

    % Code

    Code = fullyconnect(X,parameters.("code").weights, parameters.("code").bias);
    % Code = tanh(X);
end

end




