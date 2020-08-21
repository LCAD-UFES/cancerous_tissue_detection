# Mammo Marker

Para ter acesso ao Mammo Marker é necessário compilar o arquivo "mammo_marker.cpp". 
 Para isso, siga os passos abaixo:
 1. Abra o terminal na pasta do projeto
 2. Entre na pasta
 	cd src/mammo_marker
 3. Compile o arquivo. Será gerado um arquivo binário chamado "mammo_marker"
 	make
 4. Crie uma pasta com nome labels 
 	mkdir labels
 
 5. Faça o download do arquivo original_dataset pelo link abaixo
 	https://drive.google.com/drive/folders/1zE6C8WPYQYwQvixJ_bhkzBLocZPyeyCB?usp=sharing
 	Descompacte o arquivo e salve a pasta "original_dataset" dentro de "breast_cancer_analyzer_LCAD/dataset/"

 6. Crie um arquivo com o caminho absoluto das imagens. Para o caso da base CBIS-DDSM, são utilizadas as mamografias completas e as mamografias segmentadas. É importante que ambos os arquivos estejam na mesma ordem, ou seja, os arquivos devem apresentar as imagens dos pacientes na mesma ordem.

 É necessário baixar a pasta 
 As pastas com as imagens estão disponíveis em 
 
 É necessário baixar os arquivos original_dataset.tar.gz e descompactar a pasta dentro da pasta "dataset"
 
 5. Para abrir o arquivo binário, digite
 	./mammo_marker mamografias_completas.txt mamografias_segmentadas.txt

