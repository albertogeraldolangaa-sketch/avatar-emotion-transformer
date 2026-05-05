"""
config.py - Configurações centralizadas do sistema
"""

import os
import logging
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================

# Diretórios
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Criar diretórios se não existirem
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# CONFIGURAÇÕES DO MODELO LLM
# ==============================================================================

# Caminho do modelo (ajuste conforme seu arquivo)
BASE_DIR = Path(__file__).resolve().parent
LLM_MODEL_PATH = BASE_DIR / "models" / "Llama-3-8B-Instruct-v0.1.Q4_K_M.gguf"

# Fallback se não encontrar
if not LLM_MODEL_PATH.exists():
    # Procura qualquer .gguf na pasta models
    gguf_files = list(MODELS_DIR.glob("*.gguf"))
    if gguf_files:
        LLM_MODEL_PATH = gguf_files[0]
        print(f"📦 Usando modelo encontrado: {LLM_MODEL_PATH.name}")
    else:
        print("⚠️ Nenhum modelo .gguf encontrado em models/")

# ==============================================================================
# CONFIGURAÇÕES DO TTS
# ==============================================================================

# API URL do servidor TTS (se estiver rodando separadamente)
TTS_API_URL = os.environ.get("TTS_API_URL", "")

# Para TTS local via Edge TTS (fallback)
USE_EDGE_TTS_FALLBACK = True

# ==============================================================================
# CONFIGURAÇÕES DE CONVERSA
# ==============================================================================

# Configurações padrão do modelo
DEFAULT_MODEL_CONFIG = {
    "n_ctx": 4096,          # Contexto máximo
    "n_batch": 512,         # Tamanho do batch
    "temperature": 0.7,     # Criatividade
    "top_p": 0.9,           # Nucleus sampling
    "top_k": 40,            # Top-k sampling
    "repeat_penalty": 1.1,  # Penalidade de repetição
}

# Prompt do sistema
SYSTEM_PROMPT = """Você é um assistente útil, amigável e prestativo chamado LocalAI.
Você foi projetado para responder perguntas de forma clara, precisa e concisa.
Você pode ajudar com qualquer assunto, desde ciência e tecnologia até conversas casuais.
Responda em português (pt-BR) sempre que possível.
"""

# ==============================================================================
# CONFIGURAÇÕES DE LOGGING
# ==============================================================================

def setup_logging(level=logging.INFO):
    """Configura o sistema de logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "app.log", encoding='utf-8')
        ]
    )
    
    # Reduzir logs de bibliotecas muito verbosas
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)