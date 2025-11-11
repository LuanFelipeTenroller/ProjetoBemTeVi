# 🚀 Guia Rápido de Uso

## Instalação de Dependências

```bash
pip install -r requirements.txt
```

## Interface Interativa (Recomendado)

```bash
python cli_interface.py
```

Menu interativo com as seguintes opções:
- 🎓 Treinar modelos
- 🎬 Executar webcam
- 🧪 Testar com imagens
- 📊 Ver resultados
- 🏆 Comparar modelos
- 🗑️  Limpar arquivos

---

## Uso via Linha de Comando

### 1. **TREINAR MODELOS**

```bash
# Treinar todos os modelos (recomendado)
python main.py --train all

# Treinar apenas um modelo
python main.py --train mlp    # MLPClassifier
python main.py --train rf     # RandomForest
python main.py --train gb     # GradientBoosting
```

**Saída esperada:**
```
🔹 Carregando dados de treino...
📊 Distribuição dos dados de TREINO:
  angry: 350 amostras
  disgust: 400 amostras
  ...
🚀 Treinando MLP...
✅ Acurácia: 89.50%
✅ Precisão: 89.20%
...
💾 Melhor modelo salvo!
```

**Arquivos gerados:**
- `fer_mlp_model.pkl` (ou RF/GB)
- `label_encoder.pkl`
- `model_comparison.csv` (se treinou todos)

---

### 2. **WEBCAM EM TEMPO REAL**

```bash
# Usar melhor modelo automaticamente
python main.py --realtime

# Usar modelo específico
python main.py --realtime --model fer_gradientboosting_model.pkl
python main.py --realtime --model fer_randomforest_model.pkl
```

**Controles:**
- ESC: Sair
- Mostra: Emoção detectada, Confiança, FPS

**Cores na tela:**
- 🟢 **Verde**: Confiança > 60%
- 🟠 **Laranja**: Confiança entre 45-60%
- 🔴 **Cinza**: Confiança < 45% ("INDEFINIDO")

---

### 3. **TESTAR COM IMAGENS**

```bash
# Testar com melhor modelo
python main.py --test

# Testar com modelo específico
python main.py --test --model fer_gradientboosting_model.pkl
```

**Saída:**
```
==========================================================
📊 RESULTADOS DO TESTE
==========================================================
✅ Total de imagens: 500
✅ Acurácia geral: 87.20%
✅ Confiança média: 78.45%
==========================================================

🎯 Matriz de confusão (normalizada):
         angry  disgust  fear  happy  neutral  sad  surprise
angry     0.92    0.02  0.01   0.00    0.04  0.01     0.00
disgust   0.01    0.95  0.01   0.00    0.02  0.01     0.00
...
```

**Arquivos gerados:**
- `test_results_improved.csv` (resultados detalhados)

---

## 📊 Visualizando Resultados

### Ver Comparação de Modelos

```bash
# Windows
type model_comparison.csv

# Linux/Mac
cat model_comparison.csv
```

Exemplo:
```
        model  accuracy  precision    recall  f1_score
          MLP      0.87       0.88      0.86      0.87
RandomForest      0.89       0.90      0.88      0.89
GradientBoosting  0.91       0.92      0.90      0.91
```

### Ver Resultados de Teste Detalhados

```bash
# Windows
more test_results_improved.csv

# Linux/Mac
less test_results_improved.csv
```

---

## 🎓 Exemplos de Fluxo Completo

### Exemplo 1: Setup Inicial Completo

```bash
# 1. Treinar todos os modelos
python main.py --train all

# 2. Testar com imagens
python main.py --test

# 3. Usar em tempo real
python main.py --realtime
```

### Exemplo 2: Comparar e Escolher Melhor Modelo

```bash
# 1. Treinar todos
python main.py --train all

# 2. Ver comparação
type model_comparison.csv

# 3. Usar melhor modelo (ex: GradientBoosting)
python main.py --realtime --model fer_gradientboosting_model.pkl
```

### Exemplo 3: Treinar Específico e Testar

```bash
# 1. Treinar apenas RandomForest
python main.py --train rf

# 2. Testar com este modelo
python main.py --test --model fer_randomforest_model.pkl
```

---

## ⚙️ Configurações Avançadas

### Ajustar Threshold de Confiança

Edite em `main.py` na função `realtime()`:

```python
# De: confidence_threshold=0.45
# Para: confidence_threshold=0.55  (mais restritivo)
predictor = SmoothedPredictor(
    buffer_size=20,
    confidence_threshold=0.55,  # ← Aumentar aqui
    use_weighted_avg=True
)
```

### Aumentar Suavização Temporal

```python
# De: buffer_size=20
# Para: buffer_size=30
predictor = SmoothedPredictor(
    buffer_size=30,  # ← Aumentar aqui (mais suave)
    confidence_threshold=0.45,
    use_weighted_avg=True
)
```

### Aumentar Augmentação de Dados

Edite em `main.py` na função `load_data()`:

```python
# De: for _ in range(3):
# Para: for _ in range(5):
if augment:
    for _ in range(5):  # ← Aumentar aqui
        try:
            img_aug = augment_image(img)
```

---

## 🐛 Troubleshooting

### Erro: "Nenhum modelo encontrado"

```bash
# Solução: Treinar primeiro
python main.py --train all
```

### Webcam não funciona

```python
# Edite em main.py:
cap = cv2.VideoCapture(0)  # Tente 1, 2, etc se 0 não funcionar
```

### Acurácia muito baixa

```bash
# 1. Verifique estrutura de dados
# Deve ter: data/train/{emotion}/*.jpg
# e: data/test/{emotion}/*.jpg

# 2. Retreine com mais dados
python main.py --train all
```

### Previsões oscilam muito

```python
# Aumentar buffer_size em realtime():
predictor = SmoothedPredictor(buffer_size=40)  # Mais suave
```

---

## 📈 Interpretar Métricas

| Métrica | O Que Significa | Ideal |
|---------|-----------------|-------|
| **Acurácia** | Percentual de acertos totais | >85% |
| **Precisão** | Quando acerta, qual % é correto | >85% |
| **Recall** | Consegue encontrar a emoção | >85% |
| **F1-Score** | Balanço entre Precisão e Recall | >85% |

### Exemplo:
```
✅ Acurácia:  87.20%  → 87 de 100 imagens corretas
✅ Precisão:  88.15%  → Quando diz que acertou, acertou 88% das vezes
✅ Recall:    86.45%  → Encontrou 86% das emoções corretas
✅ F1-Score:  87.25%  → Bom balanço geral
```

---

## 🎯 Dicas de Desempenho

### Para máxima precisão:
1. Use **GradientBoosting**: `--model fer_gradientboosting_model.pkl`
2. Aumentar `buffer_size=30` para mais suavização
3. Reduzir `confidence_threshold=0.50` para ser mais seletivo

### Para máxima velocidade:
1. Use **MLP**: `--model fer_mlp_model.pkl`
2. Reduzir `buffer_size=10` para resposta mais rápida
3. Aumentar `confidence_threshold=0.40` para aceitar mais

### Para uso móvel/edge:
1. Use **MLP** (menor footprint)
2. Considerar quantização do modelo
3. Aumentar buffer para reduzir cargas de processamento

---

## 📝 Estrutura de Arquivos Esperada

```
ProjetoBemTeVi/
├── main.py                          # Script principal
├── cli_interface.py                 # Interface interativa
├── requirements.txt                 # Dependências
├── MELHORIAS.md                     # Documentação de melhorias
├── GUIA_RAPIDO.md                   # Este arquivo
│
├── data/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   │
│   └── test/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
│
├── fer_mlp_model.pkl                 # Modelos treinados (gerados)
├── fer_randomforest_model.pkl
├── fer_gradientboosting_model.pkl
├── label_encoder.pkl
│
├── test_results_improved.csv         # Resultados (gerados)
└── model_comparison.csv
```

---

## 🎬 Exemplo Prático Passo-a-Passo

```bash
# 1. Instalar dependências (primeira vez)
pip install -r requirements.txt

# 2. Treinar modelos
python main.py --train all
# Isso levará alguns minutos...

# 3. Ver comparação
type model_comparison.csv

# 4. Testar com imagens
python main.py --test

# 5. Usar em webcam
python main.py --realtime

# 6. Feche com ESC quando terminar
```

---

## 📞 Suporte

Para mais detalhes, consulte:
- `MELHORIAS.md` - Documentação técnica
- Comentários no `main.py`
- Exemplos de uso nos argumentos: `python main.py --help`

---

**Desenvolvido com ❤️ para máxima estabilidade e precisão**

