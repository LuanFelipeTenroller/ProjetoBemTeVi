# 📊 Exemplos de Output Esperado

## 1. Treinamento com `--train all`

```
======================================================================
🎭 SISTEMA DE RECONHECIMENTO DE EMOÇÕES FACIAIS
======================================================================

🔹 Carregando dados de treino...
📊 Distribuição dos dados de TREINO:
  angry: 350 amostras
  disgust: 400 amostras
  fear: 380 amostras
  happy: 420 amostras
  neutral: 390 amostras
  sad: 410 amostras
  surprise: 395 amostras
✅ Dados carregados: (4135, 34), (4135,)

🔹 Aplicando SMOTE para balanceamento de classes...
✅ Dados balanceados: (4480, 34)
📚 Treino: (3584, 34), Validação: (896, 34)

🚀 Treinando MLP...
============================================================
📊 MÉTRICAS DO MODELO: MLP
============================================================
✅ Acurácia:  87.50%
✅ Precisão: 87.80%
✅ Recall:   86.95%
✅ F1-Score: 87.35%
============================================================

📋 Relatório de classificação detalhado:
              precision    recall  f1-score   support
       angry       0.89     0.87     0.88        128
     disgust       0.85     0.84     0.85        140
        fear       0.86     0.88     0.87        125
       happy       0.92     0.91     0.92        142
     neutral       0.82     0.83     0.82        135
         sad       0.86     0.86     0.86        140
    surprise       0.89     0.88     0.89        126
    accuracy                         0.87        896
   macro avg       0.87     0.87     0.87        896
weighted avg       0.87     0.87     0.87        896

🎯 Matriz de confusão (normalizada):
         angry  disgust  fear  happy  neutral  sad  surprise
angry     0.87    0.04  0.01   0.01    0.04  0.02     0.01
disgust   0.02    0.84  0.02   0.03    0.05  0.02     0.02
fear      0.01    0.02  0.88   0.01    0.02  0.03     0.03
happy     0.00    0.01  0.00   0.91    0.02  0.02     0.04
neutral   0.03    0.05  0.02   0.01    0.83  0.04     0.02
sad       0.02    0.02  0.04   0.02    0.04  0.86     0.00
surprise  0.01    0.02  0.03   0.03    0.03  0.00     0.88

🚀 Treinando RandomForest...
============================================================
📊 MÉTRICAS DO MODELO: RandomForest
============================================================
✅ Acurácia:  89.40%
✅ Precisão: 89.70%
✅ Recall:   89.15%
✅ F1-Score: 89.35%
============================================================

🚀 Treinando GradientBoosting...
============================================================
📊 MÉTRICAS DO MODELO: GradientBoosting
============================================================
✅ Acurácia:  90.95%
✅ Precisão: 91.20%
✅ Recall:   90.80%
✅ F1-Score: 91.00%
============================================================

============================================================
🏆 MELHOR MODELO: GradientBoosting (F1-Score: 91.00%)
============================================================

📊 Aplicando validação cruzada (5-fold) no GradientBoosting...
✅ CV Scores: [0.912 0.908 0.915 0.909 0.911]
✅ CV Mean F1-Score: 91.10% (+/- 0.26%)

💾 Melhor modelo (GradientBoosting) salvo!
💾 Comparação de modelos salva em model_comparison.csv
```

---

## 2. Teste com `--test`

```
📦 Carregando modelo...
✅ Modelo carregado: fer_gradientboosting_model.pkl

🔹 Testando imagens de data/test...
  Processando angry... (65 imagens)
  Processando disgust... (70 imagens)
  Processando fear... (68 imagens)
  Processando happy... (72 imagens)
  Processando neutral... (71 imagens)
  Processando sad... (69 imagens)
  Processando surprise... (66 imagens)

✅ Predições salvas em test_results_improved.csv

============================================================
📊 RESULTADOS DO TESTE
============================================================
✅ Total de imagens: 481
✅ Acurácia geral: 88.77%
✅ Confiança média: 79.34%
============================================================

🎯 Matriz de confusão (normalizada):
         angry  disgust  fear  happy  neutral  sad  surprise
angry     0.89    0.03  0.01   0.00    0.04  0.02     0.01
disgust   0.02    0.87  0.03   0.01    0.04  0.02     0.01
fear      0.01    0.03  0.87   0.01    0.02  0.03     0.03
happy     0.00    0.01  0.00   0.93    0.02  0.01     0.03
neutral   0.04    0.04  0.02   0.02    0.85  0.02     0.01
sad       0.02    0.02  0.04   0.01    0.03  0.88     0.00
surprise  0.02    0.02  0.04   0.04    0.02  0.00     0.86

📊 Acurácia por emoção:
  angry: 89.23%
  disgust: 87.14%
  fear: 85.29%
  happy: 93.06%
  neutral: 84.51%
  sad: 88.41%
  surprise: 86.36%

📊 Confiança média por emoção:
  angry: 82.45%
  disgust: 78.92%
  fear: 76.34%
  happy: 85.67%
  neutral: 75.23%
  sad: 77.89%
  surprise: 80.12%

✅ Primeiras 10 ACERTOS:
              image     true_emotion  confidence
         img_001.jpg            happy        0.94
         img_002.jpg            angry        0.91
         img_003.jpg         surprise        0.89
         img_004.jpg          disgust        0.87
         img_005.jpg             sad        0.85
         ...

❌ Primeiras 10 ERROS:
              image true_emotion predicted_emotion  confidence
         img_042.jpg           angry            fear        0.42
         img_051.jpg          fear           angry        0.48
         img_073.jpg         disgust          happy        0.51
         img_089.jpg          neutral           sad        0.45
         img_105.jpg         surprise          fear        0.41
         ...
```

---

## 3. Webcam em Tempo Real com `--realtime`

```
📦 Carregando modelo...
✅ Modelo carregado: fer_gradientboosting_model.pkl

[Janela de Vídeo Abre]

Exibição na tela:
┌─────────────────────────────────────┐
│                                     │
│  HAPPY (87%)                        │
│  Raw conf: 92% | Buffer: 15         │
│  FPS: 28.4                          │
│                                     │
│  [Rosto com expressão feliz]         │
│                                     │
└─────────────────────────────────────┘

Cores observadas:
- 🟢 Verde: Quando confiança > 60%
- 🟠 Laranja: Quando confiança entre 45-60%
- ⚪ Cinza: Quando confiança < 45% (mostra "INDEFINIDO")

[Pressione ESC para sair]

✅ Execução concluída! Total de frames processados: 842
```

---

## 4. Comparação de Modelos em `model_comparison.csv`

```
         model  accuracy  precision    recall  f1_score
           MLP      0.875      0.878      0.8695    0.8735
 RandomForest      0.894      0.897      0.8915    0.8935
GradientBoosting  0.9095      0.912      0.908      0.910
```

---

## 5. Resultados Detalhados em `test_results_improved.csv`

```
image,true_emotion,predicted_emotion,confidence,correct
img_001.jpg,happy,happy,0.942,True
img_002.jpg,angry,angry,0.876,True
img_003.jpg,fear,sad,0.423,False
img_004.jpg,disgust,disgust,0.891,True
img_005.jpg,neutral,neutral,0.654,True
img_006.jpg,surprise,happy,0.501,False
...
```

---

## 6. Output do CLI Interativo (`cli_interface.py`)

```
======================================================================
🎭 SISTEMA DE RECONHECIMENTO DE EMOÇÕES FACIAIS
======================================================================

📋 MENU PRINCIPAL:
  1. Treinar modelos
  2. Executar tempo real (webcam)
  3. Testar com imagens
  4. Ver resultados anteriores
  5. Comparar modelos
  6. Limpar arquivos
  0. Sair
----------------------------------------------------------------------
Escolha uma opção (0-6): 1

📚 TREINAR MODELO:
  1. MLPClassifier (Rede Neural)
  2. RandomForestClassifier (Ensemble)
  3. GradientBoostingClassifier (Ensemble Sequencial)
  4. Treinar TODOS e comparar
  0. Voltar
----------------------------------------------------------------------
Escolha uma opção (0-4): 4

🚀 Treinando TODOS os modelos (isso pode levar alguns minutos)...
[Processos de treinamento...]

======================================================================
🎭 SISTEMA DE RECONHECIMENTO DE EMOÇÕES FACIAIS
======================================================================

📋 MENU PRINCIPAL:
  ...
Escolha uma opção (0-6): 5

📊 COMPARAÇÃO DE MODELOS:
        model  accuracy  precision    recall  f1_score
          MLP      0.875      0.878      0.8695    0.8735
RandomForest      0.894      0.897      0.8915    0.8935
GradientBoosting  0.9095      0.912      0.908      0.910

🏆 Melhor modelo por métrica:
  Acurácia: GradientBoosting
  Precisão: GradientBoosting
  Recall:   GradientBoosting
  F1-Score: GradientBoosting
```

---

## 7. Estrutura de Features

Cada face gera um vetor com **34 features**:

```
[0-7]    Distâncias principais (8)
  - Distância inter-ocular
  - Distância nariz-queixo
  - Distância boca-olho esquerdo
  - ... (mais 5)

[8-11]   Ângulos (4)
  - Ângulo entre olhos-nariz
  - Ângulo entre sobrancelhas-nariz
  - ... (mais 2)

[12-14]  Abertura dos olhos (3)
  - Razão altura/largura olho esq
  - Razão altura/largura olho dir
  - Razão média

[15]     Abertura da boca (1)
  - Razão altura/largura boca

[16-17]  Simetria da boca (2)
  - Assimetria dos cantos
  - Diferença altura lados

[18-19]  Elevação da sobrancelha (2)
  - Distância sobrancelha esq-olho
  - Distância sobrancelha dir-olho

[20-21]  Assimetrias faciais (2)
  - Simetria vertical
  - Simetria horizontal

[22]     Curvatura dos lábios (1)

[23-25]  Proporções relativas (3)
  - Boca/face
  - ... (mais 2)

[26-29]  Posição relativa dos olhos (4)
  - Posição X olho esq
  - Posição Y olho esq
  - ... (mais 2)
```

---

## 8. Métricas Explicadas

### Acurácia
- **Fórmula**: (TP + TN) / (TP + TN + FP + FN)
- **Significado**: Percentual total de acertos
- **Quando usar**: Visão geral rápida

### Precisão (por emoção)
- **Fórmula**: TP / (TP + FP)
- **Significado**: "Quando modelo diz Happy, quantas vezes está certo?"
- **Quando usar**: Reduzir falsos positivos

### Recall (por emoção)
- **Fórmula**: TP / (TP + FN)
- **Significado**: "Consegue encontrar todos os Happy?"
- **Quando usar**: Reduzir falsos negativos

### F1-Score
- **Fórmula**: 2 × (Precisão × Recall) / (Precisão + Recall)
- **Significado**: Balanço entre Precisão e Recall
- **Quando usar**: Métrica unificada (prefer quando há desbalanceamento)

---

Desenvolvido com ❤️ para máxima estabilidade e precisão

