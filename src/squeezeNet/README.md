# _CNN SqueezeNet_
Esta pasta contém todos os scripts utilizados para treinar, validar e testar a rede neural SqueezeNet, bem como os arquivos auxiliares destes processos. 

*Acesso rápido*
- [Treinamento e Validação](https://github.com/LCAD-UFES/breast_cancer_analyzer_LCAD/tree/master/src/squeezeNet#treinando-a-squeezenet)
- [Teste](https://github.com/LCAD-UFES/breast_cancer_analyzer_LCAD/tree/master/src/squeezeNet#teste)
- [Estrutura de arquivos](https://github.com/LCAD-UFES/breast_cancer_analyzer_LCAD/tree/master/src/squeezeNet#estrutura-de-arquivos)

--- 

### Estrutura de arquivos
#### Pasta *aux_files*
- Contém todos os arquivos auxiliares utilizados como inputs para a rede.
- Estes arquivos representam os conjuntos de treino, validação e teste.
- O nome dos arquivos seguem as lógicas abaixo:
  - *cbisddsm_OF10_automatic_cropped_dataset.txt* => contém o nome das imagens que foram aleatoriamente escolhidas para compor o conjunto de treino e validação para o teste de OverFitting (OF). 
    - *cbisddsm* - nome da base originária
    - *OF* - objetivo do conjunto 
    - *10* - quantidade de imagens de cada classe (OBS: Este número consta apenas nos arquivos de OF)
    - *automatic_cropped_dataset* - nome da base de recortes
  - *cbisddsm_train_2020_02_13.txt*, *cbisddsm_val_2020_02_13.txt*, *cbisddsm_test_2020_02_13.txt* => contém o nome das imagens que foram aleatoriamente escolhidas para compor o conjunto de treino, validação e teste que serão analisadas pela rede. 
    - *cbisddsm* - nome da base originária
    - *train* ou *test* ou *val* - objetivo do conjunto 
    - *2020_02_13* - data que o arquivo foi criado
    - OBS: Alguns arquivos podem ter observações após a data.
- O arquivo *training_dataset_ijcnn.txt* contém a lista de imagens na ordem que as mesmas foram lidas pela rede para gerar os resultados descritos no artigo submetido ao IJCNN.

#### Pasta *metrics*
- Contém os scripts escritos pelo Me. [Raphael Carneiro](carneiro.raphael@lcad.inf.ufes.br), sob a orientação do Prof. Dr. [Alberto F. De Souza](alberto@lcad.inf.ufes.br), para filtrar as inferência feitas pela rede e indicar um diagnóstico de "câncer" ou "não câncer" para cada "Paciente" que está no conjunto de teste.

*Obs: Descrição em construção*

#### Pastas *runs_.../*
1. Todo treinamento cria uma pasta *squeezenet1_1/{numero}* dentro da caminho indicado *na variável global RUNS_FOLDER* que está no script de treino.
2. Dentro da pasta *squeezenet1_1*, são criadas pasta para cada treino realizado. Essas pastas são criadas em ordem numérica e crescente. Logo, caso você apague algum pasta e esta não seja a última, a próxima pasta receberá o número da que você apagou.
3. A pasta de cada treino contém:
   - **pasta *models*** = contém todos os pesos salvos durante o treinamento. Subir para o git o peso que tiver a maior acurácia média no conjunto de validação. Este será o peso utilizado para fazer as inferências no conjunto de teste.
   - **classification_error.txt** = Este arquivo contém o nome de todas as imagens que não foram classificadas corretamente pela rede durante o cálculo das matrizes de confusão. *É um arquivo pesado, portanto, não sincronizado para o git.*
   - **loss_log.txt** - Contém a informação, da "loss do batch" e o "step" a que ela se refere. Ao final de cada época está o valor da *Learning_Rate* impresso.
   - **results.txt** = Contém todas as matrizes de confusão calculas sobre o conjunto de validação completo. As matrizes de confusão calculas ao salvar cada modelo da rede, ou seja, a cada check point e ao final de cada época. 
   - **training_dataset.txt** - Contém o caminho absoluto das imagens na ordem em que elas foram inseridas na rede. 
   - **training_log.txt** - Contém todas as informações de acerto ou erro para cada imagem que compõe o batch. Essa informação é calculada com base na classe de cada imagem e na inferência a rede fez sobre cada imagem.
   - **treinamento_... .py** = É a cópia fiel do script que foi executado para realizar o treinamento. Dessa forma, os valores dos hiperparâmetros e demais configurações ficam salvas.
   

---

## Treinando a SqueezeNet

### Para utilizar o script é necessário seguir todas as orientações contidas no ReadMe principal do repositório [Acesse aqui](https://github.com/LCAD-UFES/breast_cancer_analyzer_LCAD#requisitos)

--- 

### Os arquivos .py iniciados com *treinamento_* são os scripts para treinamento da rede.

1. Para treinar a SqueezeNet com o manual_cropped_dataset, acesse a pasta: 
   ```bash
   $ cd breast_cancer_analyzer_LCAD/src/squeezeNet
   ```
2. Com seu editor de preferência, abra o script ***treinamento_cancer_tissue.py*** e altere as seguintes variáveis globais:

```
RUNS_FOLDER = 'colocar, entre aspas simples, o caminho absoluto da pasta onde
você quer salvar a pasta do treino'
```

```
Exemplo: 
RUNS_FOLDER = '/home/breast_cancer_analyzer_LCAD/src/squeezeNet/runs_manual_cropped_dataset'
```
```
TRAINING = (
	'colocar, entre aspas simples, o caminho absoluto do arquivo com o 
	nome das imagens para treino. não esquecer a vírgula após a aspas',
	)
```

```
Exemplo: 
TRAINING = (
        '/home/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/cbisddsm_train_2019_10_15.txt',
)
```
```
TRAINING_DIR = (
        'colocar, entre aspas simples, o complemento do caminho das imagens
         para treino. não esquecer a vírgula após a aspas',
)
```

```
Exemplo:
TRAINING_DIR = (
        '/home/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/manual_cropped_dataset',
)
```
- Cada linha do arquivo *cbisddsm_train_2019_10_15.txt* contém parte do caminho para as imagens que serão utilizadas no treino separadas por um espaço da classe daquela imagem. 
  - Exemplo: ``` augmented_malignant/Calc-Test_P_01471_RIGHT_CC_MALIGNANT_Crop_0_180D.png 1 ```
  - Atente-se que o valor de TRAINING_DIR + a parte do caminho da imagem representam o caminho absoluto para cada imagem do conjunto de teste. 
  - Para testar, tente abrir a imagem via terminal utilizando o *eog*.
  ```bash
  $ eog /home/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/manual_cropped_dataset/augmented_malignant/Calc-Test_P_01471_RIGHT_CC_MALIGNANT_Crop_0_180D.png
  ```
  - Caso o caminho seja inválido, ajuste o valor de TRAINING_DIR.
     - Não é necessário colocar a / no final do caminho indicado em TRAINING_DIR. 

```
TEST = (
         'Colocar, entre aspas simples, o caminho absoluto do arquivo com o
         nome das imagens para validação. Não esquecer a vírgula após a aspas.',
)
```

```
Exemplo:
TEST = (
         '/home/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/cbisddsm_val_2019_10_15.txt',
)
```
```
TEST_DIR = (
        'Colocar, entre aspas simples, o complemento do caminho das imagens
         para validação. Não esquecer a vírgula após a aspas',
)
```

```
Exemplo:
TEST_DIR = (
         '/home/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/manual_cropped_dataset',
)
```

- Cada linha do arquivo *cbisddsm_val_2019_10_15.txt* contém parte do caminho para as imagens que serão utilizadas para validação separadas por um espaço da classe daquela imagem. 
  - Exemplo: ``` good/Calc-Training_P_00991_LEFT_CC_BENIGN_WITHOUT_CALLBACK_Crop_0.png 0 ```
  - Atente-se que o valor de TEST_DIR + a parte do caminho da imagem representam o caminho absoluto para cada imagem do conjunto de validação. 
  - Para testar, tente abrir a imagem via terminal utilizando o *eog*.
  ```bash
  $ eog /home/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/manual_cropped_dataset/good/Calc-Training_P_00991_LEFT_CC_BENIGN_WITHOUT_CALLBACK_Crop_0.png
  ```
  - Caso o caminho seja inválido, ajuste o valor de TEST_DIR.
     - Não é necessário colocar a / no final do caminho indicado em TEST_DIR. 

- **As variáveis globais abaixo, representam alguns dos hiperparâmetros da rede**

  - BATCH_SIZE representa a quantidade de imagens que serão analisadas juntas. 
  - ACCUMULATE representa quantas vezes você deseja acumular os valores resultantes da análise do BATCH_SIZE. Se ACCUMULATE = 2, significa que o valor resultante da análise será o equivalente a BATCH_SIZE x 2.  
    ```
    BATCH_SIZE, ACCUMULATE = 128, 1
    ``` 

  - EPOCHS representa a quantidade de épocas para análise do conjunto de treino. 
    ```
    EPOCHS = 100
    ```

  - SAVES_PER_EPOCH representa a quantidade de check points que você deseja salvar a cada época. 
    ```
    SAVES_PER_EPOCH = 10
    ```

  - INITIAL_LEARNING_RATE representa a taxa de aprendizado inicial da rede. Este valor é utilizado para atualizar o valor das sinapses e dos neurônios da rede.
    ```
    INITIAL_LEARNING_RATE = 0.0003
    ```

  - LAST_EPOCH_FOR_LEARNING_RATE_DECAY representa em qual época será finalizada a diminuição do valor da taxa de aprendizado
    ```
    LAST_EPOCH_FOR_LEARNING_RATE_DECAY = 80
    ```

  - DECAY_RATE representa o valor pelo qual a taxa de aprendizado será dividida a cada decaimento. 
    ```
    DECAY_RATE = 2
    ```

  - DECAY_STEP_SIZE representa a quantidade de épocas necessárias para fazer a atualização da taxa de aprendizado
    ```
    DECAY_STEP_SIZE = 11
    ```

  - O ajuste destes hiperparâmetros são essenciais para melhorar o aprendizado da rede.

3. Salve o arquivo. É uma boa prática salvar o arquivo sempre que fizer alguma alteração. =D

4. Considerando que o ambiente virtual já está ativado, basta digitar o comando:
   ```bash
   $ python treinamento_cancer_tissue.py
   ```
   - Todas as informações que serão apresentadas em sua tela durante o treinamento estarão salvas nos arquivos da pasta do treino. ([Leia aqui](https://github.com/LCAD-UFES/breast_cancer_analyzer_LCAD/tree/master/src/squeezeNet#pastas-runs_))

5. Os scripts de treinamento tem vários comentários, veja-os caso deseje fazer um uso diferenciado do mesmo.

## Teste

**Os arquivos .py iniciados com *test_* são os scripts para teste de todos os pesos salvos durante o treinamento.**

**Os arquivos .py iniciados com *test_prob* são os scripts para teste apenas do peso que obteve a melhor acurácia no conjunto de validação.**

---

### Utilizando o script *test_cancer_tissue.py*
1. Acesse a pasta: 
     ```bash
     $ cd breast_cancer_analyzer_LCAD/src/squeezeNet
     ```
2. Com seu editor de preferência, abra o script ***test_cancer_tissue.py*** e altere as seguintes variáveis globais:

   ```
   TEST = (
   		 'Colocar, entre aspas simples, o caminho absoluto do arquivo com o
         nome das imagens para teste. Não esquecer a vírgula após a aspas',
   	)
   ```
   ```
   Exemplo:
   TEST = (
   	'   /home/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/cbisddsm_test_2019_10_15.txt',
   	)
   ```
   ```   
   TEST_DIR = (
        'Colocar, entre aspas simples, o complemento do caminho das imagens
         para teste. Não esquecer a vírgula após a aspas',
   )
   ```
   ```   
   Exemplo:
   TEST_DIR = (
        '/home/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/manual_cropped_dataset',
   )
   ```
   - Cada linha do arquivo *cbisddsm_test_2019_10_15.txt* contém parte do caminho para as imagens que serão utilizadas para o teste separadas por um espaço da classe daquela imagem. 
     - Exemplo: ``` good/Calc-Training_P_00937_RIGHT_CC_BENIGN_Crop_0.png 0 ```
     - Atente-se que o valor de TEST_DIR + a parte do caminho da imagem representam o caminho absoluto para cada imagem do conjunto de teste.
     - Verifique se o caminho está correto, tente abrir a imagem via terminal utilizando o *eog*.
     ```bash
     $ eog /home/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/manual_cropped_dataset/good/Calc-Training_P_00937_RIGHT_CC_BENIGN_Crop_0.png
     ```
     - Caso o caminho seja inválido, ajuste o valor de TEST_DIR.
       - Não é necessário colocar a / no final do caminho indicado em TEST_DIR. 

3. Salve o arquivo. É uma boa prática salvar o arquivo sempre que fizer alguma alteração. =D

4. Considerando que o ambiente virtual já está ativado, faça as adaptações necessárias no comando abaixo:
   ```bash
   $ python test_cancer_tissue.py caminho/para/pasta/do/treino/models/ caminho/para/pasta/do/treino/all_confusion_matrix_with_testSet.txt
   ```
   - Considerando que você está na pasta src/squeezeNet
     - O primeiro argumento refere-se a parte restante do caminho até a pasta models onde estão os pesos salvos durante o treinamento.
     - O segundo argumento refere-se ao caminho e nome do arquivo que será criado para armazenar as matrizes de confusão do conjunto de teste.
       - Se você tiver acesso a este projeto, favor utilizar o nome "all_confusion_matrix_with_testSet.txt". Assim, mantemos um padrão para todas as pastas com todos os treinos. =)

---

### Utilizando o script *test_prob_cancer_tissue.py*

1. Acesse a pasta: 
     ```bash
     $ cd breast_cancer_analyzer_LCAD/src/squeezeNet
     ```
2. Com seu editor de preferência, abra o script ***test_prob_cancer_tissue.py*** e altere as seguintes variáveis globais:

   ```
   INITIAL_MODEL = 'colocar o caminho absoluto para o peso que obteve 
   					a maior acurácia no conjunto de validação.'
   ```
   ```
   Exemplo: 
   INITIAL_MODEL = '/home/breast_cancer_analyzer_LCAD/src/squeezeNet/runs_manual_cropped_dataset/squeezenet1_1/03/models/squeezenet1_1_94_9.pth'
   ```
   ```
   TEST = (
   		 'Colocar, entre aspas simples, o caminho absoluto do arquivo com o
         nome das imagens para teste. Não esquecer a vírgula após a aspas',
   	)
   ```
   ```
   Exemplo:
   TEST = (
   	'   /home/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/cbisddsm_test_2019_10_15.txt',
   	)
   ```
   ```   
   TEST_DIR = (
        'Colocar, entre aspas simples, o complemento do caminho das imagens
         para teste. Não esquecer a vírgula após a aspas',
   )
   ```
   ```   
   Exemplo:
   TEST_DIR = (
        '/home/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/manual_cropped_dataset',
   )
   ```
   - Cada linha do arquivo *cbisddsm_test_2019_10_15.txt* contém parte do caminho para as imagens que serão utilizadas para o teste separadas por um espaço da classe daquela imagem. 
     - Exemplo: ``` good/Calc-Training_P_00937_RIGHT_CC_BENIGN_Crop_0.png 0 ```
     - Atente-se que o valor de TEST_DIR + a parte do caminho da imagem representam o caminho absoluto para cada imagem do conjunto de teste.
     - Verifique se o caminho está correto, tente abrir a imagem via terminal utilizando o *eog*.
     ```bash
     $ eog /home/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/manual_cropped_dataset/good/Calc-Training_P_00937_RIGHT_CC_BENIGN_Crop_0.png
     ```
     - Caso o caminho seja inválido, ajuste o valor de TEST_DIR.
       - Não é necessário colocar a / no final do caminho indicado em TEST_DIR. 

3. Salve o arquivo. É uma boa prática salvar o arquivo sempre que fizer alguma alteração. =D

4. Considerando que o ambiente virtual já está ativado, faça as adaptações necessárias no comando abaixo:
   ```bash
   $ python test_prob_cancer_tissue.py metrics/manual_cropped_dataset/test_set_03/test_2019_10_15.csv metrics/manual_cropped_dataset/test_set_03/confusion_matrix.txt metrics/manual_cropped_dataset/test_set_03/probabilities_test_2019_10_15.csv
   ```
   - Considerando que você está na pasta src/squeezeNet:
     - O primeiro argumento refere-se ao caminho e nome do arquivo que será criado para armazenar o caminho absoluto das imagens do conjunto de teste com suas respectivas classificações. 
     - O segundo argumento refere-se ao caminho e nome do arquivo que será criado para armazenar a matriz de confusão do teste. 
     - O terceiro argumento refere-se ao caminho e nome do arquivo que será criado para armazenar todas as probabilidades geradas durante a análise do conjunto de teste. 
   - Se você tiver acesso a este projeto, favor utilizar os nomes de arquivos sugeridos acima.
     - O _03_ é refente ao número da pasta do treinamento que está sendo testado.
     - Por enquanto é necessário criar manualmente esta pasta antes de executar este script. Vamos automatizar isso.
     ```bash
     $ mkdir metrics/manual_cropped_dataset/test_set_03
     ```
