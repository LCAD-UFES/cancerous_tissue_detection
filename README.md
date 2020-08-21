# Cancerous Tissue Detection

### Versão em Português
- Repositório do projeto "Análise de Imagens Médicas Usando Aprendizado Profundo" que pode ser acessado em http://www.informatica.ufes.br/pt-br/pos-graduacao/PPGI/detalhes-do-projeto?id=9678.

Para mais informações envie um e-mail para sabrina.panceri@lcad.inf.ufes.br ou sabrinapanceri@gmail.com

---

## Requisitos
- Ubuntu 16.04 LTS / 64bits 
- CUDA 10.1
- Python 3.5.2 
- Pip3
  - Página oficial do Pip - https://pypi.org/project/pip/
- VirtualEnv
  - Aconselhamos que utilize o VirtualEnv e crie um ambiente virtual específico para rodar esse projeto.
    - Utilize o python3 para criar o ambiente virtual. 
    ```bash
    $ virtualenv -p /usr/bin/python3.5 ~/breast_cancer 
    ```
    - Aqui tem um tutorial bem legal sobre o [VirtualEnv](https://gist.github.com/Geoyi/d9fab4f609e9f75941946be45000632b)
    - Página oficial do VirtualEnv - https://pypi.org/project/virtualenv/
- Git

### Instalação do Pytorch

-Acesse o repositório que contém o Pytorch

```
 git clone --recursive https://github.com/pytorch/pytorch
 cd pytorch
```  

```
git submodule sync
git submodule update --init --recursive
python setup.py install
```

### Instalação do Torchvision

-TorchVision requer PyTorch 1.4 ou um mais novo.


-Instalação pelo Pip:
  
```
pip install torchvision
```
-Pelo respositório:
  
```
python setup.py install
```

### Tutorial para a instalação do Cuda 10

-Installing CUDA 10(https://github.com/LCAD-UFES/carmen_lcad/wiki/Installing-CUDA-10)

---

## Passo a passo para a execução do treinamento 

1. Clone esse repositório para seu computador
   - Escolha o diretório onde deseja colocar o projeto e acesso o mesmo via terminal
   ```bash
   git clone https://github.com/LCAD-UFES/cancerous_tissue_detection.git
   ```

2. Faça o download dos arquivos abaixo:
   - Base CBIS-DDSM convertida em PNG (Arquivo [CALC_CC_flipped_dataset.tar.gz](https://drive.google.com/open?id=1Q3WGOcVmnrY21_Pf7RckzSZSfr3nqsPh))
   - Base gerada manualmente (Arquivo [manual_cropped_dataset.tar.gz](https://drive.google.com/open?id=1X6eZ8hrxsR7oPwYK5iiHx_21aPIRQv77))
   - Descompactar os arquivos para *cancerous_tissue_detection/dataset/*
     - Você pode usar a interface para fazer a descompactação dos arquivos. 
     - Ou, pelo terminal:
       Acesse via terminal a pasta aonde você salvou os arquivos e execute o comando abaixo.
       *Comando em uma linha*
       ```bash
       tar -zxvf CALC_CC_flipped_dataset.tar.gz && tar -zxvf manual_cropped_dataset.tar.gz
       ```
       *Ou*
       ```bash
       tar -zxvf CALC_CC_flipped_dataset.tar.gz 
       tar -zxvf manual_cropped_dataset.tar.gz
       ```

3. Via terminal, ative o ambiente virtual criado e acesse na pasta do projeto.
   ```bash
   source ~/breast_cancer/bin/activate
   cd cancerous_tissue_detection
   ```
   - Utilize o pip para instalar o requisitos necessários para rodar o projeto
   ```bash
   pip install --no-cache -r requirements.txt
   ```
4. Ainda na pasta do projeto, acesse o diretório onde se encontra o script.
   ```bash
    cd src/squeezenet
   ```
5. Agora, execute o script da rede com os parâmetros desejados. 
   ```bash
    python squeezenet.py --root /home/your_user --batch 128 --accum 1 --epochs 100 --saves 1 --lr 0.0004 --endLR 7 --decayRate 2 --stepSize 5 --shuffle 1 --initModel /home/your_user/cancerous_tissue_detection/src/squeezeNet/runs_manual_cropped_dataset
   ``` 
- **Parâmetros, passados por linha de comando no exemplo anterior, da rede**
  - --root caminho até o local onde está o dirtório do projeto.
    ```
    --root /home/your_user
    ```
  - --batch representa a quantidade de imagens que serão analisadas juntas. 
    ```
    --batch 128
    ``` 
  - --accum representa quantas vezes você deseja acumular os valores resultantes da análise do batch. Se --accum 2, significa que o valor resultante da análise será o equivalente a batch x 2.  
    ```
    --accum 1
    ``` 

  - --epochs representa a quantidade de épocas para análise do conjunto de treino. 
    ```
    --epochs 100
    ```

  - --saves representa a quantidade de check points que você deseja salvar a cada época. 
    ```
    --saves 1
    ```

  - --lr representa a taxa de aprendizado inicial da rede. Este valor é utilizado para atualizar o valor das sinapses e dos neurônios da rede.
    ```
    --lr 0.0004
    ```

  - --endLR representa em qual época será finalizada a diminuição do valor da taxa de aprendizado.
    ```
    --endLR 7
    ```

  - --decayRate representa o valor pelo qual a taxa de aprendizado será dividida a cada decaimento. 
    ```
    --decayRate 2
    ```

  - --stepSize representa a quantidade de épocas necessárias para fazer a atualização da taxa de aprendizado.
    ```
    --stepSize 2
    ```
  - --shuffle usar como 1 apenas no primeiro treino. Nos demais, usar 0.
    ```
    --shuffle 1
    ```
   - --initModel  o caminho absoluto da pasta onde você quer salvar a pasta do treino
    ```
    --initModel /home/your_user/cancerous_tissue_detection/src/squeezeNet/runs_manual_cropped_dataset
    ```
  
  - O ajuste destes hiperparâmetros são essenciais para melhorar o aprendizado da rede.


#### Pastas *runs_.../*
1. Todo treinamento cria uma pasta *squeezenet1_1/{numero}* dentro do caminho indicado no parâmetro *--initModel*.
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

