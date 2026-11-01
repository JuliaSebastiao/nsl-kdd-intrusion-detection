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
| 01/11/2026 | Deixar `tempo_resposta_analista_min` fora dos experimentos por enquanto | Decisão provisória, assumindo que é o tempo observado de resposta do analista à conexão avaliada, posterior ao momento da detecção. Manter na base original e no diagnóstico. Reavaliar se a origem confirmar que é uma estimativa ou estatística histórica disponível antes da classificação. |
| 01/11/2026 | Deixar `tipo_alerta_soc` fora dos experimentos por enquanto | Decisão provisória, pois não foi esclarecido se o alerta é anterior à classificação ou resultado de investigação posterior. A associação com a classe não comprova vazamento. Manter a coluna original e reavaliar se sua disponibilidade anterior for confirmada. |
| 01/11/2026 | Deixar `score_ids_legado` fora da referência inicial por enquanto | Decisão provisória até confirmar como o escore foi produzido, com quais dados e se está disponível antes da classificação. Um detector anterior independente pode fornecer uma entrada legítima; a associação observada não comprova vazamento. Manter a coluna original para diagnóstico e reavaliação. |
| 01/11/2026 | Comparar três representações das colunas derivadas | A: manter as cinco derivadas (45 atributos candidatos); B: retirar `src_bytes_kb`, `dst_bytes_kb`, `same_srv_pct` e `protocolo_cod`, mantendo `total_bytes` (41); C: retirar as cinco derivadas (40). Contagens após as quatro exclusões aprovadas e antes da codificação. A comparação será interna ao treino, sem escolher antecipadamente uma versão vencedora ou alterar a base original. |
| 01/11/2026 | Preparar uma cópia independente com as funções dentro do notebook | Atender ao fechamento da Etapa 1 com diagnóstico breve revisado e cópia executada reproduzível, sem exigir o módulo local de quem recebe. Gerar a cópia a partir do notebook e de `preprocessamento.py`, preservados como fontes de manutenção. Manter as cópias executadas em `temp/`, fora do Git, e não incluir os CSVs. Essa cópia é um apoio; a entrega é o relatório em PDF, conforme a confirmação posterior da equipe. |
| 01/11/2026 | Entregar um relatório em PDF com o link do notebook no GitHub | Formato confirmado pela usuária para esta primeira etapa. Versionar o relatório em `output/pdf/`, com identificação do grupo Trabalho prático 3, achados, limites das conclusões, orientações de reprodução e o link direto para o notebook revisado na branch `feat/preprocessamento-etapa2`. |
| 01/11/2026 | Tornar o repositório público | Mudança expressamente autorizada pela usuária para permitir o acesso da professora ao link do relatório. Substitui a preferência inicial por privacidade. Os CSVs e os arquivos temporários continuam ignorados. |

## Regras do enunciado atualizado

- Usar os arquivos oficiais disponibilizados pela professora.
- Alvo: `class`, com os valores `normal` e `anomaly`.
- Selecionar modelos e ajustar o pré-processamento usando somente o treino, por validação interna.
- Preservar o teste para gerar a previsão após definir o pipeline final. O gabarito não é divulgado às equipes.
- A submissão final tem `id` e `class_prevista`, preserva os IDs e sua ordem e deve ser reproduzível pelo notebook.
- A métrica oficial é F1-macro, calculada pela professora sobre o teste.
- Na Etapa 1 (01/11), identificar e diagnosticar os atributos categóricos. Na Etapa 2 (07/11), implementar e justificar pelo menos uma técnica de conversão simbólico-numérica, discutindo o efeito da cardinalidade de `service`.

Essas regras são requisitos da atividade, não escolhas entre os cenários antigos de classes.

## O que mudou com a nova base

O treino oficial tem 69.833 registros, 49 atributos e o alvo binário. A descrição geral menciona 41 atributos, mas o CSV inclui oito adicionais e não contém `difficulty`. A leitura deve respeitar o cabeçalho fornecido.

O impacto da retirada dos conflitos aumentou: os grupos conflitantes envolvem 46,482% dos registros `anomaly`. Por isso, preservar a comparação aprovada não significa adotar automaticamente essa limpeza. O efeito deve ser avaliado por classe.

As quatro relações algébricas e a codificação de protocolo são diagnosticadas como redundâncias. Nenhuma coluna derivada foi excluída da base ou do diagnóstico. Foi aprovada a comparação futura das versões A, B e C; sua eventual utilidade depende do modelo e da validação interna.

## Entradas nominais: situação e proposta em discussão

O diagnóstico preserva `protocol_type`, `service` e `flag` como categorias sem ordem. `protocolo_cod` também é nominal, apesar do armazenamento numérico: a associação com a classe já é calculada por V de Cramér, sem interpretar seus códigos como grandezas. `tipo_alerta_soc` continua no diagnóstico categórico, mas está provisoriamente excluída dos experimentos. A identificação por dtype descreve o armazenamento, não substitui essa interpretação semântica.

Ainda não existe um codificador de entradas ajustado ou aplicado. A detecção de variantes usa `strip` e `casefold` em séries temporárias, sem padronizar a base original.

Proposta pendente de aprovação: usar one-hot encoding como primeira técnica para as entradas nominais mantidas em cada versão. Antes de implementar, definir a padronização das categorias, o tratamento das ausências e a política para categorias novas. O vocabulário deve ser aprendido somente na parcela de treino de cada rodada e reutilizado na validação, mantendo o mesmo esquema de colunas. Aplicar ao teste apenas após definir e ajustar o pipeline final com o treino oficial. Medir o aumento de dimensionalidade, especialmente de `service`. Se `protocolo_cod` permanecer na versão A, tratá-lo como nominal também na codificação.

Referências da proposta: [codificação de categorias](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) e [prevenção de vazamento](https://scikit-learn.org/stable/common_pitfalls.html).

## Questões ainda abertas

- Confirmar a origem e a disponibilidade temporal de `score_ids_legado`, `tipo_alerta_soc` e `tempo_resposta_analista_min`. As três foram excluídas provisoriamente da referência inicial, com possibilidade de reavaliação se sua disponibilidade e origem forem esclarecidas. Associação com a classe é evidência descritiva, não prova de vazamento.
- Executar a comparação aprovada das três representações das colunas derivadas e justificar a representação final.
- Escolher o tratamento das ausências, valores incompatíveis com o domínio, `su_attempted = 2` e extremos.
- Comparar as estratégias de conflito, verificando a perda de exemplos da classe minoritária.
- Definir os grupos/folds de validação, evitando separar cópias entre treino e validação.
- Escolher e comparar as técnicas de codificação, transformação numérica, balanceamento e redução de dimensionalidade ou seleção de instâncias.
- Escolher os classificadores nas etapas previstas, mantendo as decisões justificadas em conjunto.

## Critério de interpretação

Valores negativos de duração/bytes e frações fora de [0, 1] são apresentados separadamente de limites indicativos dos contadores. Sem documentação que estabeleça um máximo obrigatório, exceder um limite de referência não comprova erro. Nenhuma correção é aplicada automaticamente por essas verificações.
