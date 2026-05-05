"""
conversation_manager.py - Gerencia histórico e contexto de conversas
"""

import json
import time
import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Representa uma mensagem na conversa"""
    role: str  # 'user' ou 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=data.get('timestamp', time.time())
        )


@dataclass
class Conversation:
    """Representa uma conversa completa"""
    id: str
    name: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def add_message(self, role: str, content: str):
        """Adiciona uma mensagem à conversa"""
        self.messages.append(Message(role=role, content=content))
        self.updated_at = time.time()
    
    def get_messages_for_api(self, system_prompt: str, max_tokens: int = None) -> List[Dict[str, str]]:
        """Prepara mensagens para a API do modelo"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Limita histórico se necessário (estimativa grosseira)
        tokens_estimate = len(system_prompt) // 4
        recent_messages = []
        
        for msg in reversed(self.messages):
            msg_tokens = len(msg.content) // 4 + 10
            if max_tokens and tokens_estimate + msg_tokens > max_tokens - 500:  # Reserva para resposta
                break
            tokens_estimate += msg_tokens
            recent_messages.insert(0, msg)
        
        for msg in recent_messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        return messages
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'messages': [m.to_dict() for m in self.messages],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'message_count': len(self.messages)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        conv = cls(
            id=data['id'],
            name=data['name'],
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time())
        )
        for msg_data in data.get('messages', []):
            conv.messages.append(Message.from_dict(msg_data))
        return conv


class ConversationManager:
    """Gerencia múltiplas conversas com persistência"""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversations: Dict[str, Conversation] = {}
        self.current_id: Optional[str] = None
        
        self._load_all()
        
        # Se não há conversas, cria uma padrão
        if len(self.conversations) == 0:
            self.create_conversation("Conversa Principal")
    
    def _get_file_path(self, conv_id: str) -> Path:
        """Retorna o caminho do arquivo de uma conversa"""
        return self.storage_dir / f"{conv_id}.json"
    
    def _save_conversation(self, conv: Conversation):
        """Salva uma conversa no disco"""
        try:
            file_path = self._get_file_path(conv.id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erro ao salvar conversa {conv.id}: {e}")
    
    def _load_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Carrega uma conversa do disco"""
        try:
            file_path = self._get_file_path(conv_id)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return Conversation.from_dict(data)
        except Exception as e:
            logger.error(f"Erro ao carregar conversa {conv_id}: {e}")
        return None
    
    def _load_all(self):
        """Carrega todas as conversas do diretório"""
        for file_path in self.storage_dir.glob("*.json"):
            conv_id = file_path.stem
            conv = self._load_conversation(conv_id)
            if conv:
                self.conversations[conv_id] = conv
        
        logger.info(f"📚 Carregadas {len(self.conversations)} conversas")
    
    def create_conversation(self, name: str = None) -> Conversation:
        """Cria uma nova conversa"""
        conv_id = str(uuid.uuid4())[:8]
        
        if not name:
            name = f"Conversa {len(self.conversations) + 1}"
        
        conv = Conversation(id=conv_id, name=name)
        self.conversations[conv_id] = conv
        self.current_id = conv_id
        
        self._save_conversation(conv)
        logger.info(f"📝 Nova conversa criada: {name}")
        
        return conv
    
    def get_current(self) -> Optional[Conversation]:
        """Retorna a conversa atual"""
        if self.current_id and self.current_id in self.conversations:
            return self.conversations[self.current_id]
        
        # Se não tem conversa atual mas tem conversas, pega a primeira
        if self.conversations:
            self.current_id = list(self.conversations.keys())[0]
            return self.conversations[self.current_id]
        
        # Se não tem nenhuma conversa, cria uma
        return self.create_conversation()
    
    def set_current(self, conv_id: str) -> bool:
        """Define a conversa atual"""
        if conv_id in self.conversations:
            self.current_id = conv_id
            return True
        return False
    
    def switch_or_create(self, conv_id: str = None) -> Conversation:
        """Muda para uma conversa ou cria nova"""
        if conv_id and conv_id in self.conversations:
            self.current_id = conv_id
            return self.conversations[conv_id]
        
        # Se não especificado ou não encontrado, cria nova
        return self.create_conversation()
    
    def add_message(self, role: str, content: str) -> Optional[Conversation]:
        """Adiciona mensagem à conversa atual"""
        conv = self.get_current()
        if conv:
            conv.add_message(role, content)
            self._save_conversation(conv)
            return conv
        return None
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """Lista todas as conversas"""
        convs = []
        for conv in self.conversations.values():
            convs.append({
                'id': conv.id,
                'name': conv.name,
                'message_count': len(conv.messages),
                'updated_at': conv.updated_at,
                'is_current': conv.id == self.current_id
            })
        return sorted(convs, key=lambda x: x['updated_at'], reverse=True)
    
    def delete_conversation(self, conv_id: str) -> bool:
        """Deleta uma conversa"""
        if conv_id in self.conversations:
            # Se é a atual, muda para outra
            if self.current_id == conv_id:
                self.current_id = None
            
            del self.conversations[conv_id]
            
            # Remove arquivo
            try:
                self._get_file_path(conv_id).unlink(missing_ok=True)
            except:
                pass
            
            logger.info(f"🗑️ Conversa {conv_id} deletada")
            return True
        return False
    
    def clear_history(self) -> bool:
        """Limpa o histórico da conversa atual"""
        conv = self.get_current()
        if conv:
            conv.messages = []
            conv.updated_at = time.time()
            self._save_conversation(conv)
            logger.info("🧹 Histórico da conversa limpo")
            return True
        return False
    
    def export_conversation(self, conv_id: str = None) -> str:
        """Exporta uma conversa para texto"""
        if conv_id is None:
            conv = self.get_current()
        else:
            conv = self.conversations.get(conv_id)
        
        if not conv:
            return "Nenhuma conversa encontrada"
        
        lines = [f"Conversa: {conv.name}", "=" * 50, ""]
        
        for msg in conv.messages:
            role = "Você" if msg.role == "user" else "Assistente"
            lines.append(f"[{role}]")
            lines.append(msg.content)
            lines.append("")
        
        return "\n".join(lines)