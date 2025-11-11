#!/usr/bin/env python3
"""
Script auxiliar para o sistema de Reconhecimento de Emoções Faciais
Fornece uma interface interativa para treinamento, teste e uso em tempo real.
"""

import os
import sys
import subprocess

def print_header():
    print("\n" + "="*70)
    print("🎭 SISTEMA DE RECONHECIMENTO DE EMOÇÕES FACIAIS")
    print("="*70)

def print_menu():
    print("\n📋 MENU PRINCIPAL:")
    print("  1. Treinar modelos")
    print("  2. Executar tempo real (webcam)")
    print("  3. Testar com imagens")
    print("  4. Ver resultados anteriores")
    print("  5. Comparar modelos")
    print("  6. Limpar arquivos")
    print("  0. Sair")
    print("-" * 70)

def train_menu():
    print("\n📚 TREINAR MODELO:")
    print("  1. MLPClassifier (Rede Neural)")
    print("  2. RandomForestClassifier (Ensemble)")
    print("  3. GradientBoostingClassifier (Ensemble Sequencial)")
    print("  4. Treinar TODOS e comparar")
    print("  0. Voltar")
    print("-" * 70)
    
    choice = input("Escolha uma opção (0-4): ").strip()
    
    if choice == "1":
        print("🚀 Treinando MLPClassifier...")
        subprocess.run(["python", "main.py", "--train", "mlp"])
    elif choice == "2":
        print("🚀 Treinando RandomForestClassifier...")
        subprocess.run(["python", "main.py", "--train", "rf"])
    elif choice == "3":
        print("🚀 Treinando GradientBoostingClassifier...")
        subprocess.run(["python", "main.py", "--train", "gb"])
    elif choice == "4":
        print("🚀 Treinando TODOS os modelos (isso pode levar alguns minutos)...")
        subprocess.run(["python", "main.py", "--train", "all"])
    elif choice == "0":
        return
    else:
        print("❌ Opção inválida!")

def realtime_menu():
    print("\n🎬 TEMPO REAL:")
    
    # Lista modelos disponíveis
    models = []
    for name in ["fer_gradientboosting_model.pkl", "fer_randomforest_model.pkl", "fer_mlp_model.pkl"]:
        if os.path.exists(name):
            models.append(name)
    
    if not models:
        print("❌ Nenhum modelo encontrado! Treine um primeiro.")
        return
    
    print("\n📦 Modelos disponíveis:")
    for i, model in enumerate(models, 1):
        size = os.path.getsize(model) / (1024 * 1024)  # Convert to MB
        print(f"  {i}. {model} ({size:.2f} MB)")
    print(f"  {len(models)+1}. Usar padrão")
    print("  0. Cancelar")
    print("-" * 70)
    
    choice = input("Escolha um modelo (0-{}): ".format(len(models)+1)).strip()
    
    if choice == "0":
        return
    elif choice == str(len(models)+1):
        print("🎬 Iniciando tempo real...")
        subprocess.run(["python", "main.py", "--realtime"])
    else:
        try:
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                print(f"🎬 Iniciando tempo real com {models[model_idx]}...")
                subprocess.run(["python", "main.py", "--realtime", "--model", models[model_idx]])
            else:
                print("❌ Opção inválida!")
        except ValueError:
            print("❌ Opção inválida!")

def test_menu():
    print("\n🧪 TESTAR COM IMAGENS:")
    
    # Lista modelos disponíveis
    models = []
    for name in ["fer_gradientboosting_model.pkl", "fer_randomforest_model.pkl", "fer_mlp_model.pkl"]:
        if os.path.exists(name):
            models.append(name)
    
    if not models:
        print("❌ Nenhum modelo encontrado! Treine um primeiro.")
        return
    
    print("\n📦 Modelos disponíveis:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print(f"  {len(models)+1}. Usar padrão")
    print("  0. Cancelar")
    print("-" * 70)
    
    choice = input("Escolha um modelo (0-{}): ".format(len(models)+1)).strip()
    
    if choice == "0":
        return
    elif choice == str(len(models)+1):
        print("🧪 Testando com imagens...")
        subprocess.run(["python", "main.py", "--test"])
    else:
        try:
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                print(f"🧪 Testando com {models[model_idx]}...")
                subprocess.run(["python", "main.py", "--test", "--model", models[model_idx]])
            else:
                print("❌ Opção inválida!")
        except ValueError:
            print("❌ Opção inválida!")

def view_results():
    print("\n📊 RESULTADOS ANTERIORES:")
    
    files = [
        ("test_results_improved.csv", "Resultados de Teste"),
        ("model_comparison.csv", "Comparação de Modelos"),
        ("MELHORIAS.md", "Documentação de Melhorias")
    ]
    
    for i, (filename, desc) in enumerate(files, 1):
        exists = "✅" if os.path.exists(filename) else "❌"
        print(f"  {i}. {exists} {desc} ({filename})")
    
    print("  0. Voltar")
    print("-" * 70)
    
    choice = input("Escolha um arquivo para visualizar (0-3): ").strip()
    
    if choice == "0":
        return
    elif choice in ["1", "2", "3"]:
        idx = int(choice) - 1
        filename = files[idx][0]
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                # Mostra primeiras e últimas linhas se muito grande
                lines = content.split('\n')
                if len(lines) > 50:
                    print("\n" + "\n".join(lines[:25]))
                    print(f"\n... ({len(lines) - 50} linhas omitidas) ...\n")
                    print("\n".join(lines[-25:]))
                else:
                    print("\n" + content)
        else:
            print(f"❌ Arquivo não encontrado: {filename}")
    else:
        print("❌ Opção inválida!")

def compare_models():
    print("\n📊 COMPARAÇÃO DE MODELOS:")
    
    if not os.path.exists("model_comparison.csv"):
        print("❌ Arquivo model_comparison.csv não encontrado!")
        print("   Execute 'python main.py --train all' para gerar")
        return
    
    try:
        import pandas as pd
        df = pd.read_csv("model_comparison.csv")
        print("\n" + df.to_string(index=False))
        print("\n🏆 Melhor modelo por métrica:")
        print(f"  Acurácia: {df.loc[df['accuracy'].idxmax(), 'model']}")
        print(f"  Precisão: {df.loc[df['precision'].idxmax(), 'model']}")
        print(f"  Recall:   {df.loc[df['recall'].idxmax(), 'model']}")
        print(f"  F1-Score: {df.loc[df['f1_score'].idxmax(), 'model']}")
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")

def cleanup():
    print("\n🗑️  LIMPAR ARQUIVOS:")
    print("  1. Remover modelos treinados")
    print("  2. Remover resultados de teste")
    print("  3. Remover tudo")
    print("  0. Cancelar")
    print("-" * 70)
    
    choice = input("Escolha uma opção (0-3): ").strip()
    
    if choice == "0":
        return
    elif choice == "1":
        for model in ["fer_mlp_model.pkl", "fer_randomforest_model.pkl", 
                     "fer_gradientboosting_model.pkl", "label_encoder.pkl"]:
            if os.path.exists(model):
                os.remove(model)
                print(f"🗑️  Removido: {model}")
    elif choice == "2":
        for file in ["test_results_improved.csv", "model_comparison.csv"]:
            if os.path.exists(file):
                os.remove(file)
                print(f"🗑️  Removido: {file}")
    elif choice == "3":
        for file in ["fer_mlp_model.pkl", "fer_randomforest_model.pkl",
                    "fer_gradientboosting_model.pkl", "label_encoder.pkl",
                    "test_results_improved.csv", "model_comparison.csv"]:
            if os.path.exists(file):
                os.remove(file)
                print(f"🗑️  Removido: {file}")
    else:
        print("❌ Opção inválida!")

def main():
    while True:
        print_header()
        print_menu()
        
        choice = input("Escolha uma opção (0-6): ").strip()
        
        if choice == "0":
            print("\n👋 Até logo!")
            sys.exit(0)
        elif choice == "1":
            train_menu()
        elif choice == "2":
            realtime_menu()
        elif choice == "3":
            test_menu()
        elif choice == "4":
            view_results()
        elif choice == "5":
            compare_models()
        elif choice == "6":
            cleanup()
        else:
            print("❌ Opção inválida!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Programa interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)
