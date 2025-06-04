# Reconhecimento de Câncer de Mama (BreastCancerRecognition)

Este projeto, desenvolvido por [IceZin](https://github.com/IceZin), foca no reconhecimento de câncer de mama utilizando técnicas de aprendizado de máquina para segmentação de imagens.

## Técnicas e Métodos Utilizados no Treinamento

O treinamento do modelo de Inteligência Artificial neste projeto emprega as seguintes técnicas e metodologias, baseadas principalmente no framework TensorFlow e Keras:

### 1. Arquitetura do Modelo: U-Net com Backbone MobileNetV2

* **Modelo Base (Encoder):** Utiliza-se a arquitetura **MobileNetV2** pré-treinada como extrator de características (encoder). As camadas específicas (`block_1_expand_relu`, `block_3_expand_relu`, `block_6_expand_relu`, `block_13_expand_relu`, `block_16_project`) são aproveitadas para capturar características em diferentes níveis de profundidade. O `down_stack` é criado a partir dessas camadas e seus pesos são congelados (`trainable = False`) para aproveitar o aprendizado de transferência.
* **Decoder (Upsampling):** A parte de decodificação (decoder) é construída usando uma pilha de camadas de `Conv2DTranspose` (definidas na função `upsample`). Essas camadas realizam o upsampling para reconstruir a máscara de segmentação.
* **Conexões Skip (Skip Connections):** A arquitetura U-Net é caracterizada por conexões diretas (skip connections) entre as camadas do encoder e do decoder. Isso é implementado concatenando a saída das camadas do `down_stack` (invertidas) com as saídas correspondentes do `up_stack`.
* **Camada de Normalização:** Uma camada customizada `InstanceNormalization` é utilizada dentro dos blocos de upsampling, alternativamente à `BatchNormalization`.
* **Camada de Saída:** A camada final é uma `Conv2DTranspose` com ativação `softmax` para produzir a máscara de segmentação com `OUTPUT_CHANNELS` (definido como 3).

### 2. Preparação e Processamento de Dados

* **Fonte dos Dados:** As imagens e suas respectivas máscaras são carregadas a partir do diretório `processed_dataset`. Espera-se que cada subpasta contenha um `image.jpg` e sua máscara correspondente (ex: `mask_*.jpg`).
* **Leitura e Decodificação:** As imagens (JPEG) são lidas usando `tf.io.read_file` e decodificadas com `tf.image.decode_jpeg`.
* **Normalização:** Os pixels das imagens e máscaras são normalizados para o intervalo \[0, 1\] dividindo seus valores por 255.0.
* **Pipeline de Dados com `tf.data`:**
    * `tf.data.Dataset.from_tensor_slices`: Cria um dataset a partir dos caminhos das imagens e máscaras.
    * `.map(read_train_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)`: Aplica as funções de leitura e normalização de forma paralela e otimizada.
    * `.shuffle(BUFFER_SIZE)`: Embaralha os dados para evitar que o modelo aprenda a ordem.
    * `.batch(BATCH_SIZE)`: Agrupa os dados em lotes.
    * `.repeat()`: Permite que o dataset seja iterado múltiplas vezes (épocas).
    * `.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)`: Otimiza o carregamento dos dados, permitindo que o pré-processamento ocorra em paralelo com o treinamento.

### 3. Configuração do Treinamento

* **Otimizador:** `Adam` é utilizado como algoritmo de otimização.
* **Função de Perda (Loss Function):** `categorical_crossentropy` é empregada, adequada para problemas de segmentação multi-classe.
* **Métricas:** A `accuracy` (acurácia) é monitorada durante o treinamento.
* **Batch Size:** Definido por `BATCH_SIZE` (32 no script).
* **Épocas (Epochs):** O modelo é treinado por um número de `EPOCHS` (40 no script).
* **Passos por Época (Steps per Epoch):** Calculado como `TRAIN_LENGTH // BATCH_SIZE`.

### 4. Callbacks e Visualização

* **`DisplayCallback`:** Uma callback customizada é utilizada para:
    * Limpar a saída do console (`clear_output`).
    * Chamar a função `show_predictions` ao final de cada época para visualizar a imagem de entrada, a máscara verdadeira e a máscara predita pelo modelo em um exemplo. As predições são salvas em `./predictions/`.
* **`create_mask`:** Função auxiliar para converter a saída bruta da predição do modelo (logits) em uma máscara de segmentação visualizável, aplicando `tf.argmax` no último eixo.

### 5. Ambiente de Execução

* **Suporte a GPU:** O script verifica a disponibilidade de GPUs e tenta configurar o TensorFlow para utilizá-las, o que acelera significativamente o processo de treinamento.

Este conjunto de técnicas visa criar um modelo robusto para a segmentação de imagens médicas, especificamente para a identificação de regiões relevantes em imagens de câncer de mama.

## Estrutura do Projeto

O repositório está organizado da seguinte forma:

* **dataset/**: Contém o conjunto de dados original utilizado para treinar e testar os modelos.
* **predictions/**: Armazena as previsões geradas pelos modelos durante e após o treinamento.
* **processed_dataset/**: Contém versões pré-processadas e aumentadas do conjunto de dados, prontas para serem consumidas pelo script de treinamento.
* **src/**: Contém o código-fonte do projeto, incluindo:
    * `train.py`: Script principal para o treinamento do modelo de segmentação.
    * (Outros scripts relevantes para pré-processamento, avaliação, etc.)
* **requirements.txt**: Lista as dependências Python necessárias para executar o projeto.

## Primeiros Passos

Para executar este projeto e treinar o modelo:

### Pré-requisitos

* Python (versão compatível com as dependências, ex: 3.7+)
* Pip para instalar as dependências Python
* (Opcional, mas recomendado) Uma GPU NVIDIA com drivers CUDA e cuDNN compatíveis com a versão do TensorFlow listada em `requirements.txt` para treinamento acelerado.

### Instalação

1.  Clone o repositório:
    ```bash
    git clone [https://github.com/IceZin/BreastCancerRecognition.git](https://github.com/IceZin/BreastCancerRecognition.git)
    cd BreastCancerRecognition
    ```
2.  Crie e ative um ambiente virtual (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

### Preparação dos Dados

* Certifique-se de que o diretório `processed_dataset/` contém os dados de imagem e máscara no formato esperado pelo script `src/train.py`. Se houver um script de pré-processamento, execute-o primeiro.

### Execução do Treinamento

1.  Navegue até o diretório `src`:
    ```bash
    cd src
    ```
2.  Execute o script de treinamento:
    ```bash
    python train.py
    ```
    * O progresso do treinamento e as predições de exemplo (se `DisplayCallback` estiver ativo) serão exibidos no console.
    * As imagens de predição de exemplo serão salvas no diretório `predictions/` (relativo à raiz do projeto, então pode ser necessário ajustar o caminho no script ou criar o diretório na raiz).

## Contribuição

Como não há informações explícitas sobre contribuições, recomenda-se entrar em contato com o autor do repositório ([IceZin](https://github.com/IceZin)) ou abrir uma *issue* para discutir possíveis contribuições.

---
