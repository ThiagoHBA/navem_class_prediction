# Navem Class Prediction
[![NPM](https://img.shields.io/npm/l/react)](https://github.com/ThiagoHBA/navem_class_prediction/blob/master/LICENSE) 

![](navem_example.gif)

## O que é Navem?
**Navem** nasceu devido à grande dificuldade que os deficientes visuais enfrentam todos os dias. Seu objetivo é facilitar a locomoção desses indivíduos nas áreas urbanas e garantir que tais pessoas possam ter sua vida mais fácil.

### O Sistema
Para cumprir o seu objetivo principal, a Navem utiliza algoritmos de aprendizagem profunda para detectar obstáculos nas imagens. Com arquiteturas como RestNet, Vgg16, Dronet,
ela usa um conjunto de dados carregado com imagens categorizadas com dados de sensores giroscópio e acelerômetro. O repositório para captura de dados e geração de conjunto de dados pode ser encontrado aqui no seguinte link: https://github.com/ThiagoHBA/navem

O exemplo completo do uso do Navem em um vídeo **No Stream** pode ser visto no seguinte link: https://youtu.be/REyySHAWXmw

Este repositório busca lidar e fazer previsões com os pesos gerados por essas redes neurais e usa uma captura de imagem em forma de **Stream** (Em tempo real) com bibliotecas como **Opencv** e **PiCamera**, essas são utilizadas para capturar as imagens, ajustar seu tamanho e modificar sua escala de cores para adequar a arquitetura CNN utilizada.

Após essa classificação das imagens, são utilizados sinais que serão enviados aos sensores vibracall para indicar ao caminhante em que direção ele deve se mover e com que intensidade deve fazer esse movimento.

## Tecnologias Abordadas Neste Repositório

### Python
* TensorFlow/Keras para compilar os pesos.
* OpenCV para capturar as imagens e redimensionar para a arquitetura CNN específica.
* PiCamera para obter as imagens em um dispositivo NAVEM.

## Iniciando
* Instalar o python seguindo sua documentação oficial: https://www.python.org/downloads/
* Clone este repositório usando `git clone`
* Execute o arquivo `predict_cnn_model.py`
