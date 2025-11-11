# 🎭 Sistema de Reconhecimento de Emoções Faciais

Reconhecimento de emoções em tempo real usando **MediaPipe FaceMesh**, **extração de features geométricas avançadas** e **modelos de machine learning** (MLP, RandomForest, GradientBoosting).

## ✨ Características Principais

### 🚀 Performance Otimizada
- ✅ **Suavização temporal inteligente** com média ponderada por confiança
- ✅ **Threshold de confiança** para filtrar previsões indefinidas
- ✅ **34+ features geométricas** extraídas de landmarks faciais
- ✅ **Balanceamento automático** com SMOTE
- ✅ **Validação cruzada 5-fold** para garantir generalizabilidade

### 🤖 Modelos Disponíveis
- **MLPClassifier**: Rede neural com 4 camadas (512→256→128→64)
- **RandomForestClassifier**: 200 árvores com regularização
- **GradientBoostingClassifier**: Ensemble sequencial com early stopping
- **Comparação automática** com seleção do melhor modelo

### 📊 Métricas Robustas
- Acurácia, Precisão, Recall, F1-Score
- Matriz de confusão normalizada
- Acurácia por emoção
- Distribuição de confiança

### 🎬 Modos de Uso
- 📹 **Tempo Real**: Detecção contínua via webcam
- 🧪 **Teste**: Avaliação em lote com imagens
- 🎓 **Treinamento**: Comparação de múltiplos modelos

## 🔧 Instalação

```bash
# Clonar ou descompactar o projeto
cd ProjetoBemTeVi

# Instalar dependências
pip install -r requirements.txt
```

## 🚀 Uso Rápido

### Interface Interativa (Recomendado)
```bash
python cli_interface.py
```

Menu com as seguintes opções:
- 🎓 Treinar modelos
- 🎬 Executar webcam
- 🧪 Testar com imagens
- 📊 Ver resultados
- 🏆 Comparar modelos

### Linha de Comando

**Treinar modelos:**
```bash
python main.py --train all      # Treina todos os modelos
python main.py --train mlp      # Apenas MLP
python main.py --train rf       # Apenas RandomForest
python main.py --train gb       # Apenas GradientBoosting
```

**Executar webcam:**
```bash
python main.py --realtime                                      # Auto
python main.py --realtime --model fer_gradientboosting_model.pkl
```

**Testar com imagens:**
```bash
python main.py --test                                         # Auto
python main.py --test --model fer_randomforest_model.pkl
```

## 📊 Estrutura de Dados

```
data/
├── train/
│   ├── angry/        ← 300-400 imagens
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── ... (mesma estrutura)
    └── surprise/
```

## 📈 Exemplos de Output

### Treinamento
```
🚀 Treinando GradientBoosting...
============================================================
📊 MÉTRICAS DO MODELO: GradientBoosting
============================================================
✅ Acurácia:  90.95%
✅ Precisão: 91.20%
✅ Recall:   90.80%
✅ F1-Score: 91.00%
```

### Teste
```
============================================================
📊 RESULTADOS DO TESTE
============================================================
✅ Total de imagens: 481
✅ Acurácia geral: 88.77%
✅ Confiança média: 79.34%

🎯 Matriz de confusão (normalizada):
         angry  disgust  fear  happy  neutral  sad  surprise
angry     0.89    0.03  0.01   0.00    0.04  0.02     0.01
...
```

### Webcam
```
Exibição em tempo real:
- Emoção detectada (em cor)
- Confiança suavizada (%)
- Confiança bruta (%)
- FPS

Cores:
- 🟢 Verde: Confiança > 60%
- 🟠 Laranja: Confiança 45-60%
- ⚪ Cinza: Confiança < 45% ("INDEFINIDO")
```

## 🎯 Melhorias Implementadas

### 1. Suavização Temporal Inteligente ✅
- Classe `SmoothedPredictor` com:
  - Média móvel ponderada por confiança
  - Janela deslizante configurável (default: 20 frames)
  - Threshold de confiança (default: 0.45)
  - Histórico de predições válidas

### 2. Features Expandidas ✅
- Total: **34+ features geométricas**
- Novas features:
  - Abertura dos olhos (altura/largura)
  - Simetria da boca (cantos e altura dos lados)
  - Posição relativa dos olhos em relação ao nariz
  - Normalização robusta e invariante

### 3. Múltiplos Modelos ✅
- MLP, RandomForest, GradientBoosting
- Comparação automática com seleção do melhor
- Salva em `model_comparison.csv`

### 4. Balanceamento de Classes ✅
- SMOTE (Synthetic Minority Over-sampling Technique)
- Class weights balanceados
- Melhora recall em classes minoritárias

### 5. Métricas Robustas ✅
- Validação cruzada 5-fold estratificada
- Acurácia, Precisão, Recall, F1-Score
- Matriz de confusão normalizada
- Acurácia por emoção

### 6. Filtragem Inteligente ✅
- Threshold de confiança (< 0.45 = "INDEFINIDO")
- Código de cores na exibição
- Confiança ponderada em tempo real

## 📚 Documentação

- **MELHORIAS.md** - Documentação técnica detalhada
- **GUIA_RAPIDO.md** - Guia passo-a-passo de uso
- **EXEMPLOS_OUTPUT.md** - Exemplos de saída esperada

## 🔧 Configurações Avançadas

### Ajustar Threshold de Confiança
```python
# Em realtime(), linha ~470:
predictor = SmoothedPredictor(
    buffer_size=20,
    confidence_threshold=0.55,  # Aumentar para ser mais restritivo
    use_weighted_avg=True
)
```

### Aumentar Suavização Temporal
```python
# Em realtime(), linha ~470:
predictor = SmoothedPredictor(
    buffer_size=30,  # Aumentar para mais suavidade
    confidence_threshold=0.45,
    use_weighted_avg=True
)
```

### Aumentar Augmentação de Dados
```python
# Em load_data(), linha ~245:
if augment:
    for _ in range(5):  # De 3 para 5
        try:
            img_aug = augment_image(img)
```

## 📊 Comparação Antes vs Depois

| Aspecto | Antes | Depois |
|---------|-------|--------|
| Oscilação de previsões | Alta | Baixa (suavização) |
| Features | 27 | 34+ |
| Modelos | 1 (MLP) | 3 (MLP, RF, GB) |
| Balanceamento | Não | SMOTE |
| Métricas | Acurácia | Acc + Prec + Recall + F1 |
| Validação | Sem CV | 5-fold CV |
| Threshold | Não | 0.45 configurável |

## 🐛 Troubleshooting

**"Nenhum modelo encontrado"**
```bash
python main.py --train all
```

**Previsões oscilam muito**
```python
# Em realtime():
predictor = SmoothedPredictor(buffer_size=40)
```

**Webcam não funciona**
```python
# Em realtime(), tente:
cap = cv2.VideoCapture(1)  # Ao invés de 0
```

**Acurácia baixa**
```bash
# Retreine com mais dados e SMOTE
python main.py --train all
```

## 📁 Arquivos Gerados

- `fer_mlp_model.pkl` - Modelo MLP
- `fer_randomforest_model.pkl` - Modelo RandomForest
- `fer_gradientboosting_model.pkl` - Modelo GradientBoosting
- `label_encoder.pkl` - Encoder para labels
- `model_comparison.csv` - Comparação de modelos
- `test_results_improved.csv` - Resultados de teste detalhados

## 🎬 Exemplo Prático Completo

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Treinar todos os modelos
python main.py --train all

# 3. Ver comparação
type model_comparison.csv

# 4. Testar com imagens
python main.py --test

# 5. Usar melhor modelo em webcam
python main.py --realtime --model fer_gradientboosting_model.pkl

# 6. Pressione ESC para sair
```

## 💡 Dicas de Desempenho

### Máxima Precisão:
- Use GradientBoosting: `--model fer_gradientboosting_model.pkl`
- Aumentar `buffer_size=30`
- Reduzir `confidence_threshold=0.50`

### Máxima Velocidade:
- Use MLP: `--model fer_mlp_model.pkl`
- Reduzir `buffer_size=10`
- Aumentar `confidence_threshold=0.40`

### Uso Móvel/Edge:
- Use MLP (menor footprint)
- Aumentar buffer para reduzir processamento
- Considerar quantização do modelo

## 📦 Dependências

```
scikit-learn>=1.7.2
pandas>=2.0.0
opencv-python>=4.8.0
mediapipe>=0.10.0
albumentations>=1.4.0
imbalanced-learn>=0.12.0
numpy>=1.24.0
scipy>=1.10.0
```

## 🔐 Licença

Projeto acadêmico - Livre para uso educacional

## 📞 Suporte

Para mais detalhes:
- Consulte `MELHORIAS.md` para documentação técnica
- Veja `GUIA_RAPIDO.md` para exemplos práticos
- Confira `EXEMPLOS_OUTPUT.md` para outputs esperados
- Leia os comentários em `main.py`

---

**Desenvolvido com ❤️ para máxima estabilidade e precisão**

🎯 **Versão**: 2.0 (Melhorada)  
📅 **Última atualização**: 2025  
✨ **Status**: Production Ready
