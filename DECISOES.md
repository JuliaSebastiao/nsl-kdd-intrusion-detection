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
| 01/11/2026 | Adaptar o relatório ao padrão ABNT do guia da PUC Minas | Formatação solicitada pela usuária, que confirmou instituição PUC Minas, curso Ciência da Computação e cidade Belo Horizonte. Aplicar A4, margens superior/esquerda de 3 cm e inferior/direita de 2 cm, Arial 12, entrelinhas 1,5 no corpo, recuo de parágrafo de 1,25 cm, elementos de identificação, sumário, paginação, fontes de tabelas e figura, citações e referências, conforme o guia institucional de 2025. Preservar os achados e o link do notebook. |
| 01/11/2026 | Disponibilizar uma cópia que executa diretamente no Colab | Atender ao pedido de rodar a análise sem clonagem ou configuração manual. Gerar `01_analise_exploratoria_colab.ipynb` com todas as funções incluídas, preparação das bibliotecas e seleção do treino apenas quando ele estiver ausente. Reutilizar bibliotecas do ambiente e exibir suas versões, instalando somente as ausentes, para evitar substituir pacotes já carregados. Manter o ambiente local fixado como referência. A base não será publicada; o envio do CSV continua necessário quando não houver cópia na sessão. As nove células de análise após o carregamento são preservadas. |

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

As quatro relações algébricas e a codificação de protocolo são diagnosticadas como redundâncias. O diagnóstico e a base original preservam as colunas derivadas. As versões A, B e C foram comparadas na Etapa 2.

## Entradas nominais

O diagnóstico preserva `protocol_type`, `service` e `flag` como categorias sem ordem. `protocolo_cod` também é nominal, apesar do armazenamento numérico: a associação com a classe já é calculada por V de Cramér, sem interpretar seus códigos como grandezas. `tipo_alerta_soc` continua no diagnóstico categórico, mas está provisoriamente excluída dos experimentos. A identificação por dtype descreve o armazenamento, não substitui essa interpretação semântica.

Na Etapa 1, a detecção de variantes usa `strip` e `casefold` em séries temporárias. Na Etapa 2, a mesma padronização é aplicada nas cópias usadas pelos experimentos. O vocabulário de cada codificador é aprendido na parte de treino de cada divisão. Na validação, o one-hot completo representa categorias novas com zeros nas colunas daquele atributo. O codificador inteiro usa um código reservado para elas. Na versão A, `protocolo_cod` é tratado como nominal.

Referências de implementação — [codificação de categorias](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) e [prevenção de vazamento](https://scikit-learn.org/stable/common_pitfalls.html).

## Comparação de pré-processamento da Etapa 2

Os experimentos usaram somente o `treino.csv` oficial, com F1-macro como critério principal e recall de `anomaly` como complemento. As opções seguem as Partes 1, 2, 3, 4, 6 e 7 do material da disciplina. A referência foi a versão C, com mediana, categoria explícita para ausência, one-hot completo, Z-score, todas as linhas, sem balanceamento e sem redução. O diagnóstico de domínio serviu para marcar como ausentes durações e bytes negativos e taxas fora de [0, 1] nas cópias de trabalho.

Usamos cinco divisões estratificadas por grupo com semente 42. Os grupos correspondem a entradas iguais na versão C depois da padronização de categorias e das regras de domínio, ignorando o rótulo. As quatro primeiras divisões serviram para a comparação de alternativas. A quinta ficou fora dos ajustes e das medições dessas rodadas e foi usada para conferir as combinações predefinidas. Tinha 13.883 registros, dos quais 788 eram anomalias. A mesma divisão já havia entrado nas médias do piloto anterior. A validação manteve a composição original de cada divisão. Imputação, codificação, escala, seleção e amostragem foram ajustadas apenas nas respectivas parcelas de treino.

### Efeito de variar uma técnica

| Decisão | F1-macro médio nas quatro divisões | Interpretação |
| --- | --- | --- |
| Representação | A 0,8544 · B 0,8544 · C 0,8543 | Usar C, com 40 atributos antes da codificação. As derivadas não trouxeram ganho relevante na árvore de referência. |
| Linhas de treino | Todas 0,8543 · sem duplicatas de mesmo rótulo 0,8556 · sem conflitos 0,8514 · sem ambos 0,8519 | Remover apenas duplicatas de mesmo rótulo nas duas combinações recomendadas. Esse tratamento reduziu cerca de 16% das linhas de ajuste, enquanto excluir conflitos reduziu o recall de anomalias de 70,64% para 69,13% na comparação isolada. |
| Imputação numérica | Mediana 0,8543 · média 0,8553 · mediana com indicador 0,8541 · KNN 0,8570 | O KNN com cinco vizinhos e 2.000 referências amostradas do treino ficou à frente na árvore em quatro de quatro divisões. A mediana fica na referência e na configuração indicada para a rede. |
| Ausência nominal | Categoria explícita 0,8543 · moda 0,8568 | A moda ganhou pouco na árvore. A categoria explícita preserva a informação de que o valor estava ausente e fica na configuração da rede. |
| Codificação | One-hot 0,8543 · one-hot agrupando serviços raros 0,8543 · inteiros 0,8579 | Códigos inteiros ficaram à frente na árvore, mas a combinação com KNN e moda perdeu desempenho na rede. O agrupamento reduziu colunas, com perda de desempenho na divisão reservada da árvore. O one-hot completo continua a referência sem ordem artificial. |
| Escala na rede | Z-score 0,8386 · Min-Max 0,8353 · robusta 0,8000 · sem escala 0,7547 | Usar Z-score. Ele ganhou nas quatro divisões do teste de escala. A rede foi pequena, com 16 neurônios e limite de 16 iterações, atingido nesses testes. |
| Balanceamento na árvore | Nenhum 0,8543 · oversampling 0,8499 · undersampling 0,8261 | Manter a árvore sem amostragem. Os dois métodos usaram a proporção de 1 anomalia para 2 registros normais. O oversampling elevou o recall de 70,64% para 78,81%, com menor precisão de anomalias. |
| Redução na árvore | Nenhuma 0,8543 · filtro univariado de 60 entradas 0,8359 · PCA de 60 componentes 0,8533 | O PCA é a melhor redução entre as duas testadas. Para maximizar F1-macro, manter as entradas das configurações recomendadas. Na árvore com deduplicação, o PCA ficou ligeiramente à frente na divisão reservada, mas abaixo na média das quatro divisões iniciais. |

### Combinações recomendadas para reavaliar na Etapa 3

As duas combinações abaixo usam a versão C, marcam os valores incompatíveis com o domínio como ausentes, removem duplicatas de mesmo rótulo somente durante o ajuste e preservam os grupos conflitantes. Ambas foram testadas com Z-score e sem uma etapa adicional de redução.

| Modelo | Configuração com melhor F1-macro entre as combinações avaliadas para o modelo | Média das quatro divisões | Divisão reservada |
| --- | --- | --- | --- |
| Árvore de decisão, profundidade máxima 8 | KNN numérico, moda para ausências nominais, códigos inteiros e sem balanceamento | F1-macro 0,8592 · recall 72,10% | F1-macro 0,8624 · recall 72,84% |
| Rede neural com 16 neurônios | Mediana, ausência nominal explícita, one-hot completo e oversampling aleatório | F1-macro 0,8430 · recall 76,43% | F1-macro 0,8502 · recall 79,44% |

A árvore com KNN, moda e códigos inteiros superou a referência em todas as quatro comparações e na divisão reservada. A mesma combinação teve F1-macro de 0,8116 na rede nas quatro divisões. O oversampling elevou o recall da rede em relação à configuração deduplicada sem balanceamento, mas a vantagem em F1-macro variou entre divisões. A escolha final deve ser reavaliada com os classificadores da Etapa 3. O teste oficial fica reservado para as previsões finais.

O [notebook independente da Etapa 2](02_preprocessamento_colab.ipynb) reproduz as tabelas com o `treino.csv` oficial e grava os resultados por divisão durante a execução. Os resultados anteriores e o roteiro de comparação também estão em `temp/etapa2-comparacao/`, na cópia local. O ambiente utilizado tinha Python 3.12.14, pandas 3.0.1, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.7.2 e imbalanced-learn 0.14.1.

## Questões ainda abertas

- Confirmar a origem e a disponibilidade temporal de `score_ids_legado`, `tipo_alerta_soc` e `tempo_resposta_analista_min`. As três foram excluídas provisoriamente da referência inicial, com possibilidade de reavaliação se sua disponibilidade e origem forem esclarecidas. Associação com a classe é evidência descritiva, não prova de vazamento.
- Verificar `su_attempted = 2` e os valores extremos. Os experimentos preservaram essas observações.
- Reavaliar as combinações recomendadas ao comparar os classificadores da Etapa 3 e definir o pipeline final.

## Critério de interpretação

Valores negativos de duração/bytes e frações fora de [0, 1] são apresentados separadamente de limites indicativos dos contadores. Sem documentação que estabeleça um máximo obrigatório, exceder um limite de referência não comprova erro. O diagnóstico da Etapa 1 preserva a base original.
