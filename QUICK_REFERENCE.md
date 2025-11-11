# ⚡ Quick Reference - Guia Rápido

## 🚀 5 Minutos para Começar

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Treinar
python main.py --train all

# 3. Testar
python main.py --test

# 4. Webcam
python main.py --realtime

# 5. Pressione ESC para sair
```

---

## 📋 Comandos Essenciais

### Treino
```bash
python main.py --train all          # Todos os modelos
python main.py --train mlp          # Apenas MLP
python main.py --train rf           # Apenas RandomForest
python main.py --train gb           # Apenas GradientBoosting
```

### Teste
```bash
python main.py --test               # Com melhor modelo
python main.py --test --model fer_randomforest_model.pkl
```

### Webcam
```bash
python main.py --realtime           # Com melhor modelo
python main.py --realtime --model fer_gradientboosting_model.pkl
```

### Interface Interativa
```bash
python cli_interface.py              # Menu com tudo
```

---

## 🎯 Arquivos Importantes

### Input
- `data/train/{emotion}/*.jpg` - Imagens de treino
- `data/test/{emotion}/*.jpg` - Imagens de teste

### Output
- `fer_*_model.pkl` - Modelos treinados
- `label_encoder.pkl` - Encoder de labels
- `model_comparison.csv` - Comparação de modelos
- `test_results_improved.csv` - Resultados de teste

### Documentação
- `MELHORIAS.md` - Técnico detalhado
- `GUIA_RAPIDO.md` - Uso completo
- `EXEMPLOS_OUTPUT.md` - Exemplos de output
- `README_NOVO.md` - Overview

---

## 🎨 Cores na Webcam

| Cor | Confiança | Significado |
|-----|-----------|-------------|
| 🟢 Verde | > 60% | Muito confiante |
| 🟠 Laranja | 45-60% | Moderadamente confiante |
| ⚪ Cinza | < 45% | Indefinido |

---

## 📊 Métricas Explicadas (1 minuto)

```
Acurácia     = Percentual total de acertos
Precisão     = Quantos "sim" corretos quando diz "sim"
Recall       = Quantos "sim" encontra de todos os "sim"
F1-Score     = Balanço entre Precisão e Recall
```

### Exemplo
```
Acurácia:  87%  → 87 de 100 imagens certo
Precisão:  89%  → Quando diz "happy", 89% é realmente
Recall:    85%  → De todos os "happy", encontra 85%
F1-Score:  87%  → Métrica unificada
```

---

## 🔧 Ajustes Rápidos

### Mais Suave (menos oscilações)
```python
# Em realtime(), procure SmoothedPredictor:
predictor = SmoothedPredictor(
    buffer_size=40,  # Aumentar
    confidence_threshold=0.45,
    use_weighted_avg=True
)
```

### Mais Responsivo (menos latência)
```python
predictor = SmoothedPredictor(
    buffer_size=10,  # Diminuir
    confidence_threshold=0.45,
    use_weighted_avg=True
)
```

### Mais Rigoroso (menos "INDEFINIDO")
```python
predictor = SmoothedPredictor(
    buffer_size=20,
    confidence_threshold=0.55,  # Aumentar
    use_weighted_avg=True
)
```

---

## ❌ Problemas Comuns

| Problema | Solução |
|----------|---------|
| "Modelo não encontrado" | `python main.py --train all` |
| Oscila muito | Aumentar `buffer_size=40` |
| Webcam preta | Tente `cv2.VideoCapture(1)` |
| Acurácia baixa | Retreine: `python main.py --train all` |
| CSV vazio | Verifique `data/test/{emotion}/*.jpg` |

---

## 📁 Estrutura Esperada

```
ProjetoBemTeVi/
├── main.py
├── cli_interface.py
├── requirements.txt
├── data/
│   ├── train/{angry,disgust,fear,happy,neutral,sad,surprise}/
│   └── test/{angry,disgust,fear,happy,neutral,sad,surprise}/
└── (arquivos gerados após rodar)
```

---

## 🎓 3 Workflows Comuns

### Workflow 1: Setup + Treino + Teste
```bash
pip install -r requirements.txt
python main.py --train all
python main.py --test
```

### Workflow 2: Desenvolvimento
```bash
python cli_interface.py  # Menu interativo
# Escolha: Treinar → Testar → Webcam
```

### Workflow 3: Produção
```bash
python main.py --realtime --model fer_gradientboosting_model.pkl
# Use GradientBoosting (mais preciso)
```

---

## 📊 Ler CSV Resultados

### Windows
```cmd
type test_results_improved.csv
```

### Linux/Mac
```bash
cat test_results_improved.csv
```

### Python
```python
import pandas as pd
df = pd.read_csv('test_results_improved.csv')
print(df.head(10))
print(f"Acurácia: {(df['true_emotion'] == df['predicted_emotion']).mean():.2%}")
```

---

## 🏆 Qual Modelo Usar?

### Para Máxima Precisão
- Modelo: **GradientBoosting**
- Comando: `--model fer_gradientboosting_model.pkl`
- Precisão: ~91%

### Para Máxima Velocidade
- Modelo: **MLP**
- Comando: `--model fer_mlp_model.pkl`
- Velocidade: ~30 FPS

### Para Equilíbrio
- Modelo: **RandomForest**
- Comando: `--model fer_randomforest_model.pkl`
- Balanço: ~89% precisão, ~25 FPS

---

## 🎬 Exemplo Completo (5 min)

```bash
# Terminal 1: Setup
cd ProjetoBemTeVi
pip install -r requirements.txt

# Terminal 1: Treinar (3-5 min)
python main.py --train all

# Resultado esperado:
# "GradientBoosting F1-Score: 91.00%"

# Terminal 1: Testar (1 min)
python main.py --test

# Resultado esperado:
# "Acurácia geral: 88.77%"

# Terminal 1: Webcam (até 5 min)
python main.py --realtime --model fer_gradientboosting_model.pkl

# Press ESC to exit

# Ver resultados
type test_results_improved.csv
type model_comparison.csv
```

---

## 💡 Pro Tips

1. **Treina uma vez, usa muito**
   - Modelos são reutilizáveis
   - Uma vez treinados, não precisa retreinar

2. **GradientBoosting é melhor**
   - Geralmente melhor F1-score
   - Vale a pena usar em produção

3. **Monitore confiança**
   - Verde (>60%): Use com confiança
   - Laranja (45-60%): Análise manual
   - Cinza (<45%): Ignore

4. **Aumentar dados = Melhor modelo**
   - Mais imagens de treino = melhor performance
   - Use augmentação: `--train`

5. **Webcam estável**
   - Boa iluminação = melhores resultados
   - Evite sombras no rosto

---

## 📈 Benchmark Esperado

Com dados típicos (500-600 imagens por emoção):

```
Acurácia:  85-92%
Precisão:  86-93%
Recall:    84-91%
F1-Score:  85-92%
FPS:       25-30
```

---

## 🔐 Suporte Rápido

**Documentação:**
- Técnico? Leia `MELHORIAS.md`
- Usuário? Leia `GUIA_RAPIDO.md`
- Exemplos? Veja `EXEMPLOS_OUTPUT.md`

**Código:**
- Interface: `cli_interface.py`
- Lógica: `main.py`
- Classe: `SmoothedPredictor`

---

## ⏱️ Tempos Esperados

| Operação | Tempo |
|----------|-------|
| pip install | 2-5 min |
| train all | 5-10 min |
| test | 2-3 min |
| realtime (por frame) | 30-40 ms |
| FPS webcam | 25-30 FPS |

---

## 🎯 Checklist Antes de Usar

- [x] Python 3.8+
- [x] `pip install -r requirements.txt`
- [x] `data/train/` com imagens
- [x] `data/test/` com imagens
- [x] Webcam funcionando (se usar realtime)

---

**Tudo pronto? Comece com:**
```bash
python cli_interface.py
```

🎉 Divirta-se!

