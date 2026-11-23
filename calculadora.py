"""
Calculadora com agente em LangGraph + Claude.

Fluxo do grafo:

    START -> [agente] --(tem tool_calls?)--> [ferramentas] -> volta pro [agente]
                  \
                   --(não tem / limite atingido)--> END

Rodar (Windows, dentro do venv):
    python -m pip install -U langgraph langchain-anthropic python-dotenv
    copie .env.example para .env e preencha as chaves
    python calculadora.py

Com LANGSMITH_TRACING=true no .env, cada pergunta vira um trace em
https://smith.langchain.com, no projeto definido em LANGSMITH_PROJECT.
"""

# O .env precisa ser carregado ANTES de criar o LLM e o grafo,
# porque o tracing do LangSmith lê as variáveis de ambiente na inicialização.
from dotenv import load_dotenv

load_dotenv()

import os
from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MAX_PASSOS = 10  # trava de segurança contra loop infinito agente <-> ferramentas


# ---------------------------------------------------------------------------
# 1. ESTADO
# ---------------------------------------------------------------------------
class CalcState(TypedDict):
    # Com reducer: add_messages ACUMULA (o node devolve só as mensagens novas).
    messages: Annotated[list[BaseMessage], add_messages]
    # Sem reducer: SOBRESCREVE (último valor escrito vence).
    passos: int


# ---------------------------------------------------------------------------
# 2. TOOLS (o LLM vê o nome, a docstring e os tipos dos parâmetros)
# ---------------------------------------------------------------------------
@tool
def somar(a: float, b: float) -> float:
    """Soma dois números: a + b."""
    return a + b


@tool
def subtrair(a: float, b: float) -> float:
    """Subtrai dois números: a - b."""
    return a - b


@tool
def multiplicar(a: float, b: float) -> float:
    """Multiplica dois números: a * b."""
    return a * b


@tool
def dividir(a: float, b: float) -> float:
    """Divide dois números: a / b. Falha se b for zero."""
    if b == 0:
        raise ValueError("divisão por zero")
    return a / b


@tool
def potencia(base: float, expoente: float) -> float:
    """Eleva a base ao expoente: base ** expoente."""
    return base**expoente


TOOLS = [somar, subtrair, multiplicar, dividir, potencia]
TOOLS_POR_NOME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# 3. LLM
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Você é uma calculadora. NUNCA faça contas de cabeça: use sempre as tools, "
    "uma operação por vez, respeitando a precedência matemática. "
    "Quando tiver o resultado final, responda em português de forma curta."
)

llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
# bind_tools: anexa o "catálogo" de tools em toda chamada ao modelo.
llm_com_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# 4. NODES (recebem o estado inteiro, devolvem só o que mudou)
# ---------------------------------------------------------------------------
def agente(state: CalcState) -> dict:
    """Chama o Claude. Ele responde com texto OU com pedidos de tool (tool_calls)."""
    resposta = llm_com_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [resposta]}


def ferramentas(state: CalcState) -> dict:
    """Executa as tools que o Claude pediu na última mensagem."""
    ultima: AIMessage = state["messages"][-1]
    resultados = []

    for chamada in ultima.tool_calls:
        fn = TOOLS_POR_NOME[chamada["name"]]
        try:
            saida = fn.invoke(chamada["args"])
        except Exception as erro:  # o erro volta pro LLM, que decide o que dizer
            saida = f"Erro: {erro}"

        resultados.append(
            ToolMessage(
                content=str(saida),
                name=chamada["name"],
                tool_call_id=chamada["id"],  # liga a resposta ao pedido certo
            )
        )

    return {
        "messages": resultados,
        "passos": state.get("passos", 0) + len(resultados),
    }


# ---------------------------------------------------------------------------
# 5. EDGE CONDICIONAL (decide o próximo pulo)
# ---------------------------------------------------------------------------
def deve_continuar(state: CalcState) -> Literal["ferramentas", "__end__"]:
    if state.get("passos", 0) >= MAX_PASSOS:
        return END

    ultima = state["messages"][-1]
    if isinstance(ultima, AIMessage) and ultima.tool_calls:
        return "ferramentas"

    return END


# ---------------------------------------------------------------------------
# 6. MONTAGEM DO GRAFO
# ---------------------------------------------------------------------------
def montar_grafo():
    builder = StateGraph(CalcState)

    # nodes
    builder.add_node("agente", agente)
    builder.add_node("ferramentas", ferramentas)

    # edges
    builder.add_edge(START, "agente")
    builder.add_conditional_edges(
        "agente",
        deve_continuar,
        {"ferramentas": "ferramentas", END: END},  # retorno da função -> node
    )
    builder.add_edge("ferramentas", "agente")  # o ciclo: resultado volta pro LLM

    # checkpointer: guarda o estado por thread_id entre uma pergunta e outra
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 7. LANGSMITH (só configuração; o tracing é automático)
# ---------------------------------------------------------------------------
def status_langsmith() -> bool:
    ligado = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    if ligado and not os.getenv("LANGSMITH_API_KEY"):
        print("⚠ LANGSMITH_TRACING=true, mas falta LANGSMITH_API_KEY no .env")
        return False
    if ligado:
        projeto = os.getenv("LANGSMITH_PROJECT", "default")
        print(f"✔ LangSmith ligado: veja os traces no projeto '{projeto}' em https://smith.langchain.com")
    else:
        print("ℹ LangSmith desligado (defina LANGSMITH_TRACING=true no .env para ligar)")
    return ligado


# ---------------------------------------------------------------------------
# 8. EXECUÇÃO (mostra cada node rodando)
# ---------------------------------------------------------------------------
def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Falta ANTHROPIC_API_KEY no .env")
        return

    status_langsmith()
    grafo = montar_grafo()

    print("\n=== Grafo (Mermaid, cole em https://mermaid.live) ===")
    print(grafo.get_graph().draw_mermaid())

    thread_id = "sessao-1"
    print("Digite uma conta (ou 'sair'):")

    try:
        while True:
            pergunta = input("\nVocê: ").strip()
            if pergunta.lower() in {"sair", "exit", "q"}:
                break
            if not pergunta:
                continue

            entrada = {"messages": [HumanMessage(pergunta)], "passos": 0}

            config = {
                # usado pelo checkpointer (memória da conversa)
                "configurable": {"thread_id": thread_id},
                # usados pelo LangSmith (organizam e filtram os traces no painel)
                "run_name": "calculadora",
                "tags": ["estudo", "calculadora"],
                "metadata": {"thread_id": thread_id, "pergunta": pergunta},
            }

            # stream_mode="updates": a cada passo, devolve {nome_do_node: o_que_ele_retornou}
            for evento in grafo.stream(entrada, config, stream_mode="updates"):
                for node, update in evento.items():
                    print(f"\n--> node [{node}]")
                    for msg in update.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for c in msg.tool_calls:
                                print(f"    pede tool: {c['name']}({c['args']})")
                        elif isinstance(msg, ToolMessage):
                            print(f"    {msg.name} = {msg.content}")
                        else:
                            print(f"    resposta: {msg.content}")
                    if "passos" in update:
                        print(f"    passos: {update['passos']}")
    finally:
        # Os traces vão para o LangSmith em segundo plano.
        # Isso garante que os últimos sejam enviados antes do programa fechar.
        wait_for_all_tracers()


if __name__ == "__main__":
    main()