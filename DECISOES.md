# Decisões do projeto

Este registro separa escolhas aprovadas, regras do enunciado e questões ainda abertas. Os diagnósticos são reproduzidos no notebook.

## Escolhas aprovadas

| Data | Decisão | Justificativa e alcance |
| --- | --- | --- |
| 13/10/2026 | Manter o repositório privado nesta fase | A mudança para público será feita em uma etapa futura, ainda sem data definida. |
| 13/10/2026 | Comparar manter os registros com retirar grupos conflitantes | A comparação é um experimento futuro, aplicado somente ao treino de cada rodada. Não foi aprovada uma exclusão definitiva de todos os conflitos. |
| 14/10/2026 | Retirar `num_outbound_cmds` das entradas dos futuros modelos | Era zero em todo o treino original e continua constante na base oficial nova. A coluna permanece na base original e no diagnóstico. |
| 01/11/2026 | Integrar `feat/eda-data-quality` à `main` | Reaproveitar os cinco commits da equipe que adaptam a primeira entrega à base oficial, preservando o histórico e a autoria. |
| 01/11/2026 | Manter os CSVs em `data/raw/`, fora do versionamento | A pasta e os nomes dos três arquivos oficiais são ignorados pelo Git, inclusive se forem copiados para a raiz por engano. |
| 01/11/2026 | Completar o diagnóstico antes de escolher tratamentos | A etapa 1 investiga dados, relações com a classe e possíveis vazamentos. Não aplica imputação, balanceamento, remoção de registros ou treinamento de modelos. |

## Regras do enunciado atualizado

- Usar os arquivos oficiais disponibilizados pela professora.
- Alvo: `class`, com os valores `normal` e `anomaly`.
- Selecionar modelos e ajustar o pré-processamento usando somente o treino, por validação interna.
- Preservar o teste para gerar a previsão após definir o pipeline final. O gabarito não é divulgado às equipes.
- A submissão final tem `id` e `class_prevista`, preserva os IDs e sua ordem e deve ser reproduzível pelo notebook.
- A métrica oficial é F1-macro, calculada pela professora sobre o teste.

Essas regras são requisitos da atividade, não escolhas entre os cenários antigos de classes.

## O que mudou com a nova base

O treino oficial tem 69.833 registros, 49 atributos e o alvo binário. A descrição geral menciona 41 atributos, mas o CSV inclui oito adicionais e não contém `difficulty`. A leitura deve respeitar o cabeçalho fornecido.

O impacto da retirada dos conflitos aumentou: os grupos conflitantes envolvem 46,482% dos registros `anomaly`. Por isso, preservar a comparação aprovada não significa adotar automaticamente essa limpeza. O efeito deve ser avaliado por classe.

As quatro relações algébricas e a codificação de protocolo são diagnosticadas como redundâncias. Nenhuma coluna derivada foi excluída por isso nesta etapa. Sua eventual utilidade depende do modelo e da comparação posterior.

## Questões ainda abertas

- Confirmar a origem e a disponibilidade temporal de `score_ids_legado`, `tipo_alerta_soc` e `tempo_resposta_analista_min`. Associação com a classe é evidência descritiva, não prova de vazamento.
- Definir quais colunas adicionais podem entrar nos futuros modelos, com justificativa.
- Escolher o tratamento das ausências, valores incompatíveis com o domínio, `su_attempted = 2` e extremos.
- Comparar estratégias de redundância e conflito, verificando a perda de exemplos da classe minoritária.
- Definir os grupos/folds de validação, evitando separar cópias entre treino e validação.
- Escolher e comparar as técnicas de codificação, transformação numérica, balanceamento e redução de dimensionalidade ou seleção de instâncias.
- Escolher os classificadores nas etapas previstas, mantendo as decisões justificadas em conjunto.

## Critério de interpretação

Valores negativos de duração/bytes e frações fora de [0, 1] são apresentados separadamente de limites indicativos dos contadores. Sem documentação que estabeleça um máximo obrigatório, exceder um limite de referência não comprova erro. Nenhuma correção é aplicada automaticamente por essas verificações.
