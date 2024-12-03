"""
Arquivo com definições de execução quando o backend estiver em produção
"""
# Opções de desenvolvimento: use True se estiver programando, ou False se estiver apenas testando a ferramenta
DEBUG = True  # se o debugger deve ser executado
USE_RELOADER = False  # se modificações no código-fonte devem ser atualizadas na aplicação Web
USE_CPU = False  # se o modelo deve usar a CPU para fazer classificações, ou então a GPU
PORT = 5000  # Porta a ser usada pelo servidor
