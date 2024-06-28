% Cargar los datos de entrenamiento desde el archivo CSV
trainData = readtable('Train/C1_vl_train.csv');
trainInputData = trainData{:, 1:4};  % Las 4 primeras columnas son las entradas
trainOutputData = trainData{:, 5};   % La última columna es la salida

% Cargar los datos de validación desde el archivo CSV
valData = readtable('Validation/C1_vl_val.csv');
valInputData = valData{:, 1:4};  % Las 4 primeras columnas son las entradas
valOutputData = valData{:, 5};   % La última columna es la salida

% Combinar datos de entrada y salida en matrices
trainingData = [trainInputData trainOutputData];
validationData = [valInputData valOutputData];

% Crear el sistema FIS inicial con funciones de membresía gaussmf
fis = sugfis()

% Añadir variables de entrada con 4 funciones de membresía gaussmf para las 3 primeras entradas
for i = 1:3
    fis = addInput(fis, 'input', ['input' num2str(i)], [min(trainInputData(:,i)) max(trainInputData(:,i))]);
    for j = 1:4
        fis = addMF(fis, 'input', i, ['gaussmf' num2str(j)], 'gaussmf', [0.1 rand]);
    end
end

% Añadir la cuarta variable de entrada con 3 funciones de membresía gaussmf
fis = addInput(fis, 'input', 'input4', [min(trainInputData(:,4)) max(trainInputData(:,4))]);
for j = 1:3
    fis = addMF(fis, 'input', 4, ['gaussmf' num2str(j)], 'gaussmf', [0.1 rand]);
end

% Añadir la variable de salida
fis = addOutput(fis, 'output', 'output1', [min(trainOutputData) max(trainOutputData)]);
for j = 1:(4*4*3)  % El número de funciones de membresía de salida depende del número de reglas
    fis = addMF(fis, 'output', 1, ['constant' num2str(j)], 'constant', rand);
end

% Generar las reglas (esto puede necesitar ajustes dependiendo de la estructura específica del problema)
ruleList = genFis(fis);  % Generar reglas automáticamente
fis = addrule(fis, ruleList);

% Opciones de entrenamiento
numEpochs = 100;
errorGoal = 0;
initialStepSize = 0.01;  % Learning rate inicial
stepSizeDecreaseRate = 0.9;  % Tasa de decremento del learning rate
stepSizeIncreaseRate = 1.1;  % Tasa de incremento del learning rate

% Entrenar el modelo ANFIS con datos de validación
[trainedFIS, trainError, stepSize, valFIS, valError] = anfis(trainingData, fis, ...
    [numEpochs, errorGoal, initialStepSize, stepSizeDecreaseRate, stepSizeIncreaseRate], [], validationData);

% Evaluar el modelo entrenado (opcional)
inputTestData = valInputData;  % Utilizar los datos de validación para evaluar el rendimiento
outputTestData = evalfis(inputTestData, trainedFIS);

% Mostrar resultados
disp('Datos de prueba:')
disp(inputTestData)
disp('Resultados del modelo ANFIS:')
disp(outputTestData)

% Graficar los errores de entrenamiento y validación
figure;
plot(1:numEpochs, trainError, 'b', 'LineWidth', 1.5);
hold on;
plot(1:numEpochs, valError, 'r', 'LineWidth', 1.5);
xlabel('Épocas');
ylabel('Error');
legend('Error de entrenamiento', 'Error de validación');
title('Error de entrenamiento y validación durante el entrenamiento ANFIS');
grid on;
