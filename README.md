 

```markdown
# Avatar Emotion Transformer
 
Implementação de *runtime* e núcleo de transformadores para avatares interativos, integrando processamento de linguagem natural (LLM), sincronização labial, micro-expressões faciais e dinâmica espacial em tempo real.

---

## 🛠️ Arquitetura e Módulos

O projeto é composto por módulos altamente desacoplados e baseados em transformadores:

* **`avatar_runtime_merged.py`**: Motor de execução principal que orquestra a renderização, o estado de fala e o pipeline de áudio.
* **`speech_lip_sync_extension.py`**: Camada adicional para sincronização labial (Lip-Sync) em português e processamento de fala.
* **`occupancy_visibility_transformer.py`**: Controlador de visibilidade e densidade espacial baseado em PyTorch.
* **`animation_fluidity_transformers.py`**: Transformações avançadas de animação secundária e fluidez de movimentos.
* **`advanced_animation_transformers.py`**: Otimizações adicionais e processamento de suavização dinâmica.
* **`facial_micro_expression_transformers.py`**: Geração dinâmica de micro-expressões faciais e sincronização de visemas.
* **`axial_rotation_transformers.py`**: Suporte para dinâmica de rotação e alinhamento do olhar/cabeça.
* **`body_motion_transformers.py`**: Lógica de suporte e suavização para cinemática e intenção de movimento.
* **`spatial_motion_intelligence.py`**: Módulo de intenção espacial, cinemática e inteligência de movimento.
* **`phoneme_viseme_sync.py`**: Mapeamento e alinhamento fonema-visema para fala natural.

---

## 📦 Instalação

### Pré-requisitos
Certifique-se de que tem o Python 3.10 ou superior instalado.

1. Clone o repositório:
   ```bash
   git clone [https://github.com/albertogeraldolangaa-sketch/avatar-emotion-transformer.git](https://github.com/albertogeraldolangaa-sketch/avatar-emotion-transformer.git)
   cd avatar-emotion-transformer
   ```

2. Instale as dependências principais:
   ```bash
   pip install torch numpy sounddevice pillow
   ```

---

##  Como Utilizar

Para inicializar o agente e o seu *runtime* principal:

```python
import tkinter as tk
from avatar_runtime_merged import AvatarApp

def main():
    root = tk.Tk()
    app = AvatarApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
```

---

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir *Issues* ou enviar *Pull Requests* para melhorias nos modelos de animação ou pipeline de áudio.

---

## 📝 Licença

Este projeto está licenciado sob a licença **MIT** - consulte o arquivo `LICENSE` para obter mais detalhes.
```
 
