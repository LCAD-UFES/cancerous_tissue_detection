# _Mammo_ _PreProcessing_

Módulo que contém os códigos criados para realizar diversos tipos de pré-processamentos nas imagens, antes de elas serem analisadas pelas redes neurais.

*Acesso rápido*
- [Base de Dados - Binária](https://github.com/LCAD-UFES/breast_cancer_analyzer_LCAD/tree/master/src/mammo_preprocessing#Base-de-Dados---Binária)
- [Base de Dados - BIRADS](https://github.com/LCAD-UFES/breast_cancer_analyzer_LCAD/tree/master/src/mammo_preprocessing#Base-de-Dados---BIRADS)

--- 

## Base de Dados - Binária

Este dataset é gerado através do original fazendo um aumento de dados através do recorte de cada imagem, dada como entrada, em outras imagens menores de 224x224.   

### Criando o dataset

Para criar o dataset devemos rodar o script mergeSeg.py juntamente com a Base CBIS-DDSM convertida em PNG (Arquivo [CALC_CC_flipped_dataset.tar.gz](https://drive.google.com/open?id=1Q3WGOcVmnrY21_Pf7RckzSZSfr3nqsPh)) (Atenção: extraia o arquivo compactado no diretório '/seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/src/mammo_preprocessing/dataset'), também devemos incluir alguns parâmetros necessários para a sua execução, da seguinte forma:

```
cd /seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/src/mammo_preprocessing/
python mergeSeg.py /seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/dataset/Calc-CC_flipped_dataset/ /seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/aux_files/mamografias_completas.txt /seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/aux_files/mamografias_segmentadas.txt ../automatic_cropped_dataset 0
```  
Onde o primeiro parâmetro é o caminho até o dataset original com as imagens já rotacionadas para a mesma direção. O segundo e terceiro são, respectivamente, as localizações dos arquivos de textos que contém os caminhos das imagens dos exames e de suas imagens segmentadas. O quarto parâmetro serve para definir o caminho e o nome do diretório onde as imagens e os arquivos de textos referentes a elas serão armazenados na máquina. O quinto parâmetro é para definir a porcentagem de sobreposição de uma imagem recortada sobre outra. 

Após o término do processamento, vá até a pasta 'automatic_cropped_dataset' onde estão as imagens geradas. Lá você encontrará os arquivos de textos 'no_cancer.txt' e 'with_cancer.txt' que servem de entrada para o script da Squeezenet, ambos contendo os caminhos para as imagens do novo dataset e as labels.  


## Base de Dados - BIRADS 

Este dataset é gerado através do original_renamed fazendo um aumento de dados através do recorte de cada imagem, dada como entrada, em outras imagens menores de 224x224.   

### Criando o dataset

Para criar o dataset devemos rodar o script mergeSegBirads.py juntamente com a Base CBIS-DDSM convertida em PNG e renomeada (Arquivo [BIRADS_CALC_CC_flipped_dataset.tar.gz](https://drive.google.com/open?id=13CVQu9kr74-OLVkO_gP7xKHGn-fFUVPc)) (Atenção: extraia o arquivo compactado no diretório '/seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/src/mammo_preprocessing/dataset') , também devemos incluir alguns parâmetros necessários para a sua execução, da seguinte forma:

```
cd /seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/src/mammo_preprocessing/
python mergeSegBirads.py /seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/dataset/Birads_Calc-CC_flipped_dataset/ /seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/dataset/birads_dataset/aux_files/mamografias_completas.txt /seu_caminho_até_este_diretório/breast_cancer_analyzer_LCAD/dataset/birads_tissue_dataset/aux_files/mamografias_segmentadas.txt ../automatic_cropped_birads_dataset 0
```  
Onde o primeiro parâmetro é o caminho até o dataset original com as imagens já rotacionadas para a mesma direção. O segundo e terceiro são, respectivamente, as localizações dos arquivos de textos que contém os caminhos das imagens dos exames e de suas imagens segmentadas. O quarto parâmetro serve para definir o caminho e o nome do diretório onde as imagens e os arquivos de textos referentes a elas serão armazenados na máquina. O quinto parâmetro é para definir a porcentagem de sobreposição de uma imagem recortada sobre outra. 

Após o término do processamento, vá até a pasta automatic_cropped_birads_dataset' onde estão as imagens geradas. Lá você encontrará os arquivos de textos 'no_cancer.txt' e 'with_cancer.txt' que servem de entrada para o script da Squeezenet, ambos contendo os caminhos para as imagens do novo dataset e as labels.  



