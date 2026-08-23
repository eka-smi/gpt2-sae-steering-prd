# GPT-2 SAE Steering with Projection-Residual Denoising

Эксперимент по улучшению векторного steering GPT-2 Small с помощью обучаемого residual-денойзера и метода **Projection-Residual Denoising (PRD)**.

Обычный steering изменяет residual stream модели:

$$
h_{\mathrm{steer}} = h + \alpha v.
$$

При увеличении силы $\alpha$ целевой концепт усиливается, но perplexity быстро растёт. В работе обучен небольшой денойзер и предложен PRD: из поправки денойзера удаляется компонента вдоль steering-вектора $v$. Благодаря этому денойзер не должен отменять полезную часть вмешательства.

## Основной результат

Финальное сравнение выполнено на полностью отложенных концептах `Paris` и `ocean`. Гиперпараметр $\beta=1.0$ заранее выбран только на development-концептах `opera`, `banana`, `blanket` и `football`.

| Метод | Среднее сохранение целевого эффекта | Минимальное сохранение | Среднее изменение специфичности | Среднее PPL / ordinary | PPL-победы |
|---|---:|---:|---:|---:|---:|
| Full denoising | 0.9360 | 0.7243 | -0.2001 | 0.9258 | 10/10 |
| PRD | 0.9996 | 0.8864 | -0.0049 | 1.0694 | 6/10 |

**Вывод:** полный денойзинг лучше снижает perplexity, но заметнее ослабляет steering. PRD почти полностью сохраняет целевой эффект и специфичность, однако при сильном вмешательстве может ухудшать perplexity. Универсальной победы нет: методы реализуют разные стороны компромисса между силой концепта и fluency.

Подробная методика, формулы, таблицы и ограничения приведены в [REPORT.md](REPORT.md).

## Экспериментальная установка

- модель: GPT-2 Small через TransformerLens;
- точка вмешательства: `blocks.6.hook_resid_post`;
- размер residual stream: 768;
- SAE: OpenAI GPT-2 Small `resid_post_mlp_v5_32k`, слой 6;
- количество SAE-признаков: 32 768;
- обучающие активации: 27 000;
- валидационные активации: 3 000;
- финальный perplexity-бенчмарк: WikiText-2 validation, 5120 предсказываемых токенов;
- baseline perplexity: 54.0742.

## Проверяемые steering-векторы

| Роль | Концепт | SAE feature index |
|---|---|---:|
| Development | opera | 28999 |
| Development | banana | 31031 |
| Development | blanket | 31117 |
| Development | football | 11220 |
| Final holdout | Paris | 2252 |
| Final holdout | ocean | 24664 |

Все шесть признаков были исключены из случайного SAE-шума во время обучения денойзера. Аудит 20 000 обучающих искажений обнаружил `0` попаданий защищённых признаков.

## Структура репозитория

```text
.
├── notebooks/
│   └── gpt2_sae_prd_experiment.ipynb
├── protocols/
│   ├── ordinary_steering_protocol.json
│   ├── prd_development_protocol.json
│   ├── prd_final_protocol.json
│   └── prd_final_dist_protocol.json
├── result/
│   ├── prd_beta_summary.csv
│   ├── prd_development_combined.csv
│   ├── prd_final_concept_comparison.csv
│   ├── prd_final_perplexity.csv
│   ├── prd_final_joint_comparison.csv
│   ├── prd_final_method_summary.csv
│   └── prd_final_dist_summary.csv
├── src/
├── REPORT.md
├── README.md
└── LICENSE
```

## Как воспроизвести эксперимент

1. Откройте `notebooks/gpt2_sae_prd_experiment.ipynb` в Kaggle.
2. Включите Internet и ускоритель GPU T4.
3. Запускайте ячейки сверху вниз: установка зависимостей, загрузка GPT-2 и SAE, сбор активаций, обучение денойзера, development-оценка и финальная оценка.
4. Не используйте финальные концепты `Paris` и `ocean` для выбора архитектуры, checkpoint или $\beta$.
5. Сверьте итоговые показатели с CSV-файлами в `result/` и протоколами в `protocols/`.

Notebook содержит полный код подготовки данных, обучения, геометрического аудита PRD и вычисления concept score, perplexity и `dist-1/dist-2/dist-3`.

## Checkpoint

Лучший checkpoint денойзера опубликован в открытом репозитории Hugging Face:

**[eka-smi/gpt2-sae-prd-denoiser](https://huggingface.co/eka-smi/gpt2-sae-prd-denoiser)**

Checkpoint содержит веса небольшого денойзера. Веса GPT-2 и SAE в репозиторий не дублируются и загружаются из исходных публичных источников.

## Ограничения

- финальная проверка выполнена только на двух концептах;
- использовано 16 генераций на условие без доверительных интервалов;
- `dist-n` оценивает разнообразие, но не гарантирует связность текста;
- на больших силах PRD сохраняет концепт, но может ухудшать perplexity;
- результаты относятся к GPT-2 Small и выбранному слою 6.

Более подробное обсуждение ограничений и дальнейших улучшений находится в [REPORT.md](REPORT.md).
