"""Generate the final RL2 research report from checked experiment artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "stream_rl"
LOGS = CODE_ROOT / "logs"
OUTPUT = ROOT / "output" / "pdf" / "rl2_streaming_memory_cognitive_maps_report.pdf"


def read_monitor(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    returns = [
        float(row["info.returned_episode_returns"])
        for row in rows
        if row.get("info.returned_episode_returns", "") != ""
    ]
    success = [
        float(row["info.success"])
        for row in rows
        if row.get("info.success", "") != ""
    ]
    tail = min(500, len(returns))
    result = {
        "return": sum(returns[-tail:]) / tail,
        "success": sum(success[-min(500, len(success)):]) / min(500, len(success))
        if success
        else math.nan,
    }
    if rows and rows[0].get("info.reward_phase", "") != "":
        phases = {}
        for row in rows:
            phase = int(float(row["info.reward_phase"]))
            phases.setdefault(phase, []).append(
                float(row["info.returned_episode_returns"])
            )
        gains = []
        for phase in sorted(phases)[2:-1]:
            values = phases[phase]
            if len(values) >= 8:
                n = min(4, len(values) // 2)
                gains.append(sum(values[-n:]) / n - sum(values[:n]) / n)
        result["adaptation"] = sum(gains) / len(gains) if gains else math.nan
    return result


def monitor_values(run_dir: Path, field: str):
    return [read_monitor(path)[field] for path in sorted(run_dir.glob("monitor_seed_*.csv"))]


def mean_sd(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def fmt_mean_sd(values, digits=3):
    mean, sd = mean_sd(values)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def load_metrics(name):
    return json.loads(
        (LOGS / "representation_analysis_final" / name / "representation_metrics.json").read_text()
    )


def rows_for(metrics, run_fragment, label, representation, condition="trained"):
    return [
        row
        for row in metrics["rows"]
        if run_fragment in row["run"]
        and row["label"] == label
        and row["representation"] == representation
        and row["condition"] == condition
    ]


def row_mean(rows, key):
    values = [float(row[key]) for row in rows if math.isfinite(float(row.get(key, math.nan)))]
    return statistics.mean(values) if values else math.nan


def p(text, style):
    return Paragraph(text, style)


def report_table(data, widths, styles, font_size=8.2):
    converted = []
    for row_index, row in enumerate(data):
        style = styles["table_head"] if row_index == 0 else styles["table"]
        converted.append([p(str(value), style) for value in row])
    table = LongTable(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20364f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c4cf")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
            ]
        )
    )
    return table


def image_pair(left: Path, right: Path, left_caption: str, right_caption: str, styles):
    width = 7.4 * cm
    cells = []
    for path, caption in ((left, left_caption), (right, right_caption)):
        if path.exists():
            cells.append([Image(str(path), width=width, height=6.2 * cm), p(caption, styles["caption"])])
        else:
            cells.append([p("Визуализация недоступна", styles["caption"]), p(caption, styles["caption"])])
    table = Table(
        [[cells[0][0], cells[1][0]], [cells[0][1], cells[1][1]]],
        colWidths=[8.0 * cm, 8.0 * cm],
        hAlign="CENTER",
    )
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return KeepTogether([table])


def register_fonts():
    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    italic = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
    pdfmetrics.registerFont(TTFont("Arial", regular))
    pdfmetrics.registerFont(TTFont("Arial-Bold", bold))
    pdfmetrics.registerFont(TTFont("Arial-Italic", italic))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")


def make_styles():
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleRU", parent=sample["Title"], fontName="Arial-Bold", fontSize=22,
            leading=27, textColor=colors.HexColor("#17324d"), alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleRU", parent=sample["Normal"], fontName="Arial", fontSize=11,
            leading=15, textColor=colors.HexColor("#49627a"), alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1RU", parent=sample["Heading1"], fontName="Arial-Bold", fontSize=16,
            leading=20, textColor=colors.HexColor("#17324d"), spaceBefore=12, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2RU", parent=sample["Heading2"], fontName="Arial-Bold", fontSize=12.5,
            leading=16, textColor=colors.HexColor("#275a78"), spaceBefore=10, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyRU", parent=sample["BodyText"], fontName="Arial", fontSize=9.5,
            leading=13.2, alignment=TA_LEFT, textColor=colors.HexColor("#1f2933"),
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "BulletRU", parent=sample["BodyText"], fontName="Arial", fontSize=9.2,
            leading=12.8, leftIndent=13, firstLineIndent=-7, bulletIndent=4, spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "CalloutRU", parent=sample["BodyText"], fontName="Arial-Bold", fontSize=10,
            leading=14, leftIndent=10, rightIndent=10, borderColor=colors.HexColor("#7fa4bf"),
            borderWidth=0.8, borderPadding=8, backColor=colors.HexColor("#eef5f9"), spaceAfter=9,
        ),
        "table": ParagraphStyle(
            "TableRU", parent=sample["BodyText"], fontName="Arial", fontSize=7.7,
            leading=9.5, textColor=colors.HexColor("#1f2933"),
        ),
        "table_head": ParagraphStyle(
            "TableHeadRU", parent=sample["BodyText"], fontName="Arial-Bold", fontSize=7.7,
            leading=9.5, textColor=colors.white,
        ),
        "caption": ParagraphStyle(
            "CaptionRU", parent=sample["BodyText"], fontName="Arial-Italic", fontSize=7.8,
            leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#526575"),
        ),
        "small": ParagraphStyle(
            "SmallRU", parent=sample["BodyText"], fontName="Arial", fontSize=7.7,
            leading=10.2, textColor=colors.HexColor("#526575"),
        ),
    }


def build_report():
    register_fonts()
    styles = make_styles()
    passive = load_metrics("passive_tmaze")
    active = load_metrics("active_tmaze")
    popgym = load_metrics("popgym")
    popgym_rtu = load_metrics("popgym_rtu_stable")
    rtu_passive = load_metrics("stream_rtu_passive")

    easy_root = LOGS / "representation_study" / "easy"
    pilot_root = LOGS / "rtu_hparam_pilots" / "easy"
    ct_root = LOGS / "architecture_continuing_complex" / "complex"

    behavior = {
        "passive_ppo": monitor_values(easy_root / "passive_tmaze__ppo_lstm2_mlp64", "success"),
        "passive_stream": monitor_values(easy_root / "passive_tmaze__stream_lstm2_directent_alr01_clr03_ent001_k3", "success"),
        "passive_ppo_rtu": monitor_values(easy_root / "passive_tmaze__ppo_rtu1_h32_mlp64_bptt", "success"),
        "passive_stream_rtu": monitor_values(pilot_root / "passive_tmaze__stream_rtu1_h64_linear_rtrl_paper", "success"),
        "active_ppo": monitor_values(easy_root / "active_tmaze__ppo_lstm2_mlp64", "success"),
        "active_stream": monitor_values(easy_root / "active_tmaze__stream_lstm2_directent_alr01_clr03_ent001_k3", "success"),
        "pop_ppo": monitor_values(easy_root / "popgym__ppo_gru2_w128_mlp128", "return"),
        "pop_stream": monitor_values(easy_root / "popgym__stream_gru2_w128_directent_alr01_clr03_ent001_k3", "return"),
        "pop_rtu": monitor_values(pilot_root / "popgym__stream_rtu1_h64_linear_rtrl_paper", "return"),
        "pop_rtu_stable": monitor_values(
            LOGS / "rtu_stability_pilots/easy/popgym__stream_rtu1_h64_linear_rtrl_paper_rmax99",
            "return",
        ),
        "pop_medium_ppo": monitor_values(
            LOGS / "rtu_medium_pilots/complex/popgym__ppo_gru2_w128_mlp128",
            "return",
        ),
        "pop_medium_rtu": monitor_values(
            LOGS / "rtu_medium_pilots/complex/popgym__stream_rtu1_h64_linear_rtrl_paper_rmax99",
            "return",
        ),
        "ct_ppo": monitor_values(ct_root / "ctgraph__ppo_gru1_linear", "return"),
        "ct_stream": monitor_values(ct_root / "ctgraph__stream_gru2_directent_alr01_clr03_ent001_k3", "return"),
        "ct_ppo_adapt": monitor_values(ct_root / "ctgraph__ppo_gru1_linear", "adaptation"),
        "ct_stream_adapt": monitor_values(ct_root / "ctgraph__stream_gru2_directent_alr01_clr03_ent001_k3", "adaptation"),
    }

    ppo_pop_target = rows_for(popgym, "popgym__ppo", "hidden_target", "layer_1_h")
    stream_pop_target = rows_for(popgym, "popgym__stream", "hidden_target", "layer_1_h")
    ppo_pop_memory = rows_for(popgym, "popgym__ppo", "memory_state", "layer_1_h")
    stream_pop_memory = rows_for(popgym, "popgym__stream", "memory_state", "layer_1_h")
    ppo_pop_deck = rows_for(popgym, "popgym__ppo", "deck_count_0", "layer_1_h")
    stream_pop_deck = rows_for(popgym, "popgym__stream", "deck_count_0", "layer_1_h")
    ppo_pop_history = rows_for(popgym, "popgym__ppo", "hidden_target", "history_4")
    stream_pop_history = rows_for(popgym, "popgym__stream", "hidden_target", "history_4")
    rtu_pop_target = rows_for(popgym_rtu, "popgym__stream_rtu1", "hidden_target", "layer_0_h")
    rtu_pop_memory = rows_for(popgym_rtu, "popgym__stream_rtu1", "memory_state", "layer_0_h")

    story = []
    story += [Spacer(1, 1.0 * cm), p("RL2: от памяти к рабочей памяти и когнитивным картам", styles["title"])]
    story += [p("Streaming AC, e-prop, RTU/RTRL и PPO на POMDP и lifelong-задачах", styles["subtitle"])]
    story += [p("Итоговый экспериментальный отчёт · 3 независимых сида для основных сравнений · PPO num_envs = 1", styles["subtitle"])]
    story += [Spacer(1, 0.4 * cm)]
    story += [p(
        "Главный вывод: скрытые состояния действительно несут богатую информацию о латентном состоянии, "
        "но декодируемость, k-means homogeneity и даже сходная геометрия между алгоритмами не доказывают "
        "наличие когнитивной карты. Наиболее успешные представления являются селективной рабочей памятью "
        "для задачи, а не полным переносимым графом среды.", styles["callout"]
    )]
    story += [p("Краткие выводы", styles["h1"])]
    bullets = [
        "<b>Passive T-maze:</b> PPO-LSTM и Streaming AC-LSTM решают задачу, а полное состояние декодируется с accuracy 1.0. Такой же результат даёт необученная рекуррентная сеть, поэтому одной декодируемости недостаточно.",
        "<b>Active T-maze:</b> обе LSTM-модели кодируют позицию и время (probe ≈ 1.0), но не скрытую цель (≈ 0.49) и остаются на случайном success rate. Пространственная структура без belief-state не обеспечивает управление.",
        "<b>POPGym RepeatPreviousEasy:</b> PPO-GRU почти идеально декодирует требуемую карту (0.999) и окно памяти (0.966); Streaming AC-GRU заметно хуже (0.644 и 0.767) и не обучается по возврату.",
        "<b>RTU/RTRL:</b> однослойный Якобиан проверен против полного BPTT. Исходная конфигурация имеет NaN-коллапс на одном сиде; ограничение r_max=.99 даёт 0.798 ± 0.047 на трёх сидах POPGym. На Passive T-maze Streaming RTU остаётся на случайном уровне.",
        "<b>Когнитивная карта:</b> перенос позиции между контекстами цели остаётся около chance, корреляция графовых и латентных расстояний мала или нестабильна, а успешная PPO на POPGym почти не кодирует состояние колоды. Гипотеза о полной модели мира не подтверждена.",
        "<b>Lifelong CT-graph depth 4:</b> PPO и Streaming AC статистически неразличимы по финальному возврату; явного streaming-преимущества нет. Полный 5M запуск не прошёл заранее установленный критерий масштабирования.",
    ]
    for item in bullets:
        story.append(p("• " + item, styles["bullet"]))

    story += [PageBreak(), p("1. Исследовательская постановка", styles["h1"])]
    story += [p(
        "Рабочая гипотеза была разделена на четыре уровня: (1) хранение истории; (2) декодируемое belief-state; "
        "(3) факторизованная рабочая память, переносимая между контекстами; (4) когнитивная карта, чья геометрия "
        "и динамика отражают граф переходов и поддерживают обобщение. Такой порядок предотвращает подмену "
        "сильного утверждения простой линейной декодируемостью.", styles["body"]
    )]
    story += [p(
        "Обзор Whittington et al. трактует когнитивные карты как представления структуры среды, связанные с "
        "гибким поведением. CSCG-модель George et al. подчёркивает разделение алиасированных наблюдений по "
        "контекстам, перенос схем и планирование. Поэтому в экспериментальный протокол добавлены проверки "
        "алиасинга, межконтекстного переноса, следующего состояния и графовых расстояний, а не только k-means.",
        styles["body"],
    )]
    story += [p("Критерии и контроли", styles["h2"])]
    for item in [
        "Замороженный backbone: linear probe обучается после RL и не меняет представление.",
        "Episode-disjoint folds: траектории train/test не пересекаются.",
        "Untrained control: отделяет эффект обучения от случайных рекуррентных признаков.",
        "History-4 control: показывает, какая информация доступна из буквальной короткой истории.",
        "PCA ≤ 32 перед k-means; t-SNE используется только для визуализации, не как численная метрика.",
        "Три сида для основных сравнений; пилотные гиперпараметры явно отделены от подтверждающих прогонов.",
    ]:
        story.append(p("• " + item, styles["bullet"]))

    story += [p("2. Поведенческие результаты", styles["h1"])]
    behavior_table = [
        ["Среда", "Модель", "Метрика, mean ± sd", "Сиды", "Вывод"],
        ["Passive T-maze L=5", "PPO, LSTM×2", fmt_mean_sd(behavior["passive_ppo"]), "3", "Надёжно решает"],
        ["Passive T-maze L=5", "Streaming AC, LSTM×2", fmt_mean_sd(behavior["passive_stream"]), "3", "Надёжно решает"],
        ["Passive T-maze L=5", "PPO, RTU-BPTT", fmt_mean_sd(behavior["passive_ppo_rtu"]), "3", "1/3 сидов решает"],
        ["Passive T-maze L=5", "AC(λ), RTU-RTRL", fmt_mean_sd(behavior["passive_stream_rtu"]), "3", "Случайный уровень"],
        ["Active T-maze L=5", "PPO, LSTM×2", fmt_mean_sd(behavior["active_ppo"]), "3", "Около chance"],
        ["Active T-maze L=5", "Streaming AC, LSTM×2", fmt_mean_sd(behavior["active_stream"]), "3", "Около chance"],
        ["POPGym Easy", "PPO, GRU×2", fmt_mean_sd(behavior["pop_ppo"]), "3", "Решает"],
        ["POPGym Easy", "Streaming AC, GRU×2", fmt_mean_sd(behavior["pop_stream"]), "3", "Не обучается"],
        ["POPGym Easy", "AC(λ), RTU-RTRL", fmt_mean_sd(behavior["pop_rtu"]), "3", "2 успеха, 1 NaN-коллапс"],
        ["POPGym Easy", "RTU-RTRL, r_max=.99", fmt_mean_sd(behavior["pop_rtu_stable"]), "3", "Стабильно обучается"],
        ["POPGym Medium", "PPO-GRU", f"{behavior['pop_medium_ppo'][0]:.3f}", "1 pilot", "Около chance на 500k"],
        ["POPGym Medium", "RTU-RTRL, r_max=.99", f"{behavior['pop_medium_rtu'][0]:.3f}", "1 pilot", "Около chance на 500k"],
        ["CT-graph depth 4", "PPO, GRU", fmt_mean_sd(behavior["ct_ppo"]), "3", "Chance-подобная политика"],
        ["CT-graph depth 4", "Streaming AC, GRU×2", fmt_mean_sd(behavior["ct_stream"]), "3", "Преимущества нет"],
    ]
    story.append(report_table(behavior_table, [3.0*cm, 3.5*cm, 3.0*cm, 1.0*cm, 5.2*cm], styles))
    story += [Spacer(1, 0.2 * cm), p(
        f"На CT-graph средний within-phase adaptation gain равен {statistics.mean(behavior['ct_ppo_adapt']):.003f} "
        f"для PPO и {statistics.mean(behavior['ct_stream_adapt']):.003f} для Streaming AC. Разница мала на фоне "
        "межсидовой вариативности и не сопровождается ростом success rate.", styles["body"]
    )]
    story += [p(
        "Исправленные e-prop пилоты также не прошли критерий масштабирования: Active T-maze остаётся около "
        "chance (success 0.46–0.51 для symmetric/random feedback), а POPGym RepeatPreviousEasy даёт возврат "
        "около −0.48. Значения получены на диагностических seed-0 прогонах и не используются как "
        "publication-level сравнение.", styles["body"]
    )]

    story += [PageBreak(), p("3. Скрытые состояния: что именно кодируется", styles["h1"])]
    representation_table = [
        ["Среда / модель", "Target / state probe", "k-means homogeneity", "History-4", "Структурный тест"],
        ["Passive, PPO-LSTM", "task state 1.000", "1.000", "task state 0.824", "cross-context 0.167 (chance)"],
        ["Passive, Stream-LSTM", "task state 1.000", "1.000", "task state 0.824", "cross-context 0.192"],
        ["Passive, Stream-RTU", "task state 1.000", "1.000", "—", "success ≈ chance"],
        ["Active, PPO-LSTM", "goal 0.490; position 1.000", "goal 0.000", "—", "success ≈ chance"],
        ["Active, Stream-LSTM", "goal 0.490; position 1.000", "goal 0.000", "—", "success ≈ chance"],
        ["POPGym, PPO-GRU", f"target {row_mean(ppo_pop_target, 'linear_probe_accuracy'):.3f}; memory {row_mean(ppo_pop_memory, 'linear_probe_accuracy'):.3f}", f"target {row_mean(ppo_pop_target, 'homogeneity'):.3f}", f"target {row_mean(ppo_pop_history, 'linear_probe_accuracy'):.3f}", "graph ρ ≈ 0.014"],
        ["POPGym, Stream-GRU", f"target {row_mean(stream_pop_target, 'linear_probe_accuracy'):.3f}; memory {row_mean(stream_pop_memory, 'linear_probe_accuracy'):.3f}", f"target {row_mean(stream_pop_target, 'homogeneity'):.3f}", f"target {row_mean(stream_pop_history, 'linear_probe_accuracy'):.3f}", "graph ρ ≈ 0.152"],
        ["POPGym, Stream-RTU", f"target {row_mean(rtu_pop_target, 'linear_probe_accuracy'):.3f}; memory {row_mean(rtu_pop_memory, 'linear_probe_accuracy'):.3f}", f"target {row_mean(rtu_pop_target, 'homogeneity'):.3f}", "target 1.000", "graph ρ ≈ 0.010"],
    ]
    story.append(report_table(representation_table, [3.3*cm, 4.1*cm, 3.1*cm, 2.7*cm, 3.5*cm], styles))
    story += [p("Три наблюдения меняют интерпретацию:", styles["h2"])]
    for item in [
        "На Passive T-maze необученные LSTM и RTU также дают probe и homogeneity 1.0. Короткая детерминированная траектория легко разделяется случайной рекуррентной динамикой.",
        "На PPO-RTU все три сида имеют probe/homogeneity 1.0, хотя поведенчески решает только один. Следовательно, k-means не предсказывает успех внутри данного контроля.",
        "Стабилизированный Stream-RTU на POPGym имеет target probe 1.0 и deterministic success ≈ 0.95, но target homogeneity ≈ 0.002. Распределённый код успешно используется без компактных k-means-кластеров.",
        "На POPGym буквальная history-4 декодирует требуемую карту с accuracy 1.0. PPO-латент не демонстрирует информацию, недоступную из достаточной истории; он создаёт удобное нелинейное сжатие этой истории.",
    ]:
        story.append(p("• " + item, styles["bullet"]))

    passive_tsne = LOGS / "representation_analysis_final/passive_tmaze/passive_tmaze__ppo_lstm2_mlp64/seed_0/trained/tsne.png"
    active_tsne = LOGS / "representation_analysis_final/active_tmaze/active_tmaze__ppo_lstm2_mlp64/seed_0/trained/tsne.png"
    story.append(image_pair(passive_tsne, active_tsne, "Passive T-maze: успешная PPO-LSTM", "Active T-maze: неуспешная PPO-LSTM", styles))

    pop_ppo_tsne = LOGS / "representation_analysis_final/popgym/popgym__ppo_gru2_w128_mlp128/seed_0/trained/tsne.png"
    pop_stream_tsne = LOGS / "representation_analysis_final/popgym/popgym__stream_gru2_w128_directent_alr01_clr03_ent001_k3/seed_0/trained/tsne.png"
    story.append(image_pair(pop_ppo_tsne, pop_stream_tsne, "POPGym: PPO-GRU, seed 0", "POPGym: Streaming AC-GRU, seed 0", styles))
    rtu_pop_tsne = LOGS / "representation_analysis_final/popgym_rtu_stable/popgym__stream_rtu1_h64_linear_rtrl_paper_rmax99/seed_0/trained/tsne.png"
    story.append(image_pair(pop_stream_tsne, rtu_pop_tsne, "Неуспешный Stream-GRU", "Успешный стабилизированный Stream-RTU", styles))
    story += [p(
        "t-SNE служит качественной иллюстрацией локальной организации. Численные выводы основаны на исходных "
        "латентах, episode-disjoint probes и PCA+k-means, поскольку расстояния t-SNE не сохраняют глобальную геометрию.",
        styles["small"],
    )]

    story += [p("4. PPO против Streaming AC", styles["h1"])]
    story += [p(
        "Парные архитектуры были одинаковыми внутри каждой среды. На Passive T-maze геометрия центроидов "
        "умеренно сходна (CKA ≈ 0.76, RDM Spearman ≈ 0.67), а обе модели успешны. На Active T-maze сходство "
        "ещё выше (CKA ≈ 0.83, RDM ≈ 0.86), хотя обе модели не решают память цели. На POPGym CKA ≈ 0.71 и "
        "RDM ≈ 0.58, но поведение резко различается. Сходная геометрия не гарантирует одинаковую пригодность "
        "представления для policy readout.", styles["body"]
    )]
    story += [p(
        "Для POPGym Spearman-корреляция между target-homogeneity и success rate по шести точкам PPO/Streaming "
        "равна примерно 0.71, а между target linear-probe и success — 1.0. Оценка конфундирована семейством "
        "алгоритма и малым n. Для компоненты состояния колоды знак обратный: успешная PPO кодирует её слабее, "
        "чем неуспешный StreamAC. Политика сохраняет полезный target, а не полный Markov-state.", styles["body"]
    )]
    story += [p("5. RTU/RTRL: корректность, потенциал и ограничения", styles["h1"])]
    story += [p(
        "Автоматический тест сравнивает компактные RTU-чувствительности для ν, θ, B_real и B_imag с полным "
        "BPTT-Якобианом пятишаговой последовательности. Все элементы совпадают при rtol 2e-5. Проверка "
        "подтверждает точность однослойного рекуррентного ядра.", styles["body"]
    )]
    story += [p(
        "Стек из двух RTU в текущем BlockChain хранит чувствительности каждого слоя только к собственным "
        "параметрам. Перекрёстная производная прошлых выходов первого слоя через память второго отсутствует. "
        "Такой стек является layer-local RTRL, а не полным многослойным RTRL; результаты нельзя описывать как "
        "точный градиент всей иерархии.", styles["callout"]
    )]
    rtu_dyn = rtu_passive.get("rtu_dynamics", [])
    q90 = [item["layers"][0]["time_constant_q90"] for item in rtu_dyn]
    med = [item["layers"][0]["time_constant_median"] for item in rtu_dyn]
    story += [p(
        f"На Passive T-maze медианная временная константа RTU составляет {statistics.mean(med):.2f}, а q90 — "
        f"{statistics.mean(q90):.2f} шага. Память по масштабу достаточна для эпизода, и task-state декодируется "
        "идеально, но policy остаётся случайной. Ограничение находится в credit assignment/readout, а не в "
        "простом отсутствии длительных мод.", styles["body"]
    )]
    story += [p(
        f"На POPGym исходный RTU-RTRL даёт по трём сидам {fmt_mean_sd(behavior['pop_rtu'])}; два сида обучаются, "
        f"а один получает NaN entropy и возврат −1. После ограничения r_max=0.99 результат составляет "
        f"{fmt_mean_sd(behavior['pop_rtu_stable'])}. Однако на RepeatPreviousMedium он остаётся на "
        f"случайном уровне после 500k шагов ({behavior['pop_medium_rtu'][0]:.3f}); PPO-GRU также не "
        f"обучается ({behavior['pop_medium_ppo'][0]:.3f}). Предварительный критерий 5M-масштабирования не выполнен.",
        styles["body"]
    )]
    rtu_pop_dyn = popgym_rtu.get("rtu_dynamics", [])
    rtu_pop_q90 = [item["layers"][0]["time_constant_q90"] for item in rtu_pop_dyn]
    story += [p(
        f"Стабилизированный RTU использует распределённое представление: target linear probe = "
        f"{row_mean(rtu_pop_target, 'linear_probe_accuracy'):.3f}, но homogeneity = "
        f"{row_mean(rtu_pop_target, 'homogeneity'):.3f}. Средний q90 временных констант равен "
        f"{statistics.mean(rtu_pop_q90):.2f} шага. Низкая кластеризуемость не препятствует рабочей памяти.",
        styles["body"],
    )]

    story += [p("6. Lifelong CT-graph", styles["h1"])]
    story += [p(
        "Среда переведена в continuing-task режим: скрытое состояние и eligibility traces сохраняются через "
        "границы маршрутов, reward target меняется только на следующем корне после достижения порога, а "
        "PPO использует num_envs=1 и num_minibatches=1. Интервал переключения 128 равен длине PPO rollout, "
        "поэтому старый on-policy буфер действительно быстро устаревает.", styles["body"]
    )]
    story += [p(
        f"На depth 4 финальный возврат равен {fmt_mean_sd(behavior['ct_ppo'])} для PPO и "
        f"{fmt_mean_sd(behavior['ct_stream'])} для Streaming AC. Success rate близок к 1/16, то есть к выбору "
        "случайного листа. Easy depth 2 показывал небольшой положительный adaptation gain StreamAC, но эффект "
        "не перенёсся на сложное дерево. Гипотеза о преимуществе streaming-подхода для lifelong-задачи пока "
        "не подтверждена.", styles["body"]
    )]

    story += [p("7. Архитектуры и гиперпараметры", styles["h1"])]
    arch_table = [
        ["Модель", "Backbone / head", "Обновление", "Ключевые параметры"],
        ["PPO-LSTM", "embed64 → LSTM64 → LN → LSTM64 → LN → FC64(tanh) → out", "BPTT, Adam", "steps128; epochs4; lr3e-4; γ=.99; GAE=.95; entropy=.01"],
        ["Streaming AC-LSTM", "та же архитектура", "AC(λ), ObGD; direct entropy", "actor lr .01; critic .03; κ 3/2; λ=.95; entropy=.001"],
        ["PPO-GRU POPGym", "one-hot embed128 → GRU128×2 + LN → FC128 → out", "BPTT, Adam", "steps128; epochs4; lr3e-4"],
        ["Streaming AC-GRU POPGym", "та же архитектура", "AC(λ), one-step recurrence", "actor lr .01; critic .03; κ 3/2; λ=.95; entropy=.001"],
        ["AC(λ)-RTU-RTRL", "embed64 → RTU64(real+imag=128) → LN → out", "однослойный RTRL + traces", "actor/critic lr1; κ .2/.5; λ=.9; entropy=.01"],
        ["PPO-RTU", "embed64 → RTU32(output64) → LN → FC64 → out", "BPTT, Adam", "PPO-параметры как выше"],
        ["Stream e-prop", "one-hot/tanh embed → eprop GRU/LSTM → LN → MLP", "локальные eligibility traces", "λ=.9–.95; symmetric/random feedback"],
        ["CT PPO", "embed64 → GRU64 → out", "PPO", "num_envs1; steps128; minibatches1"],
        ["CT Stream", "embed64 → GRU64×2 + LN → FC64 → out", "AC(λ), direct entropy", "actor .01; critic .03; κ 3/2; switch128"],
    ]
    story.append(report_table(arch_table, [3.1*cm, 5.1*cm, 3.1*cm, 5.5*cm], styles))
    story += [p("Все основные архитектурные сравнения используют отдельные actor и critic backbones одинаковой формы.", styles["small"])]

    story += [p("8. Исправления реализации", styles["h1"])]
    for item in [
        "e-prop GRU/LSTM: исправлены цепные производные, типы trace, reset после done, categorical encoder, feedback seed и LayerNorm feedback.",
        "StreamAC: энтропийный градиент вынесен из TD-trace в прямое ограниченное обновление; старый вариант мог менять знак энтропийного давления через будущие TD errors.",
        "CT-graph: исправлена граница reward phase и добавлен continuing wrapper без сброса памяти агента.",
        "RTU: сериализация carry и извлечение real/imag латентов; однослойная RTRL-проверка против BPTT.",
        "Диагностика POPGym: исправлена интерпретация get_state(), обновление наблюдения и one-hot history baseline.",
    ]:
        story.append(p("• " + item, styles["bullet"]))
    story += [p("Набор из 16 целевых тестов проходит полностью.", styles["callout"])]

    story += [p("9. Ограничения и решение о длинных прогонах", styles["h1"])]
    for item in [
        "Три сида достаточны для обнаружения крупных эффектов и нестабильности, но недостаточны для узких доверительных интервалов.",
        "Active T-maze не решён ни PPO, ни Streaming AC; выводы по его латентам относятся к неуспешным политикам.",
        "Linear probe проверяет доступность информации, а не причинное использование. Интервенция в латент и reward-remapping transfer остаются более сильными будущими тестами.",
        "Эмпирическая графовая метрика зависит от посещённых состояний и не проверяет планирование по невидимым рёбрам.",
        "RTU стабилизирован на POPGym Easy, но 500k-контроль на Medium не прошёл критерий масштабирования; оба алгоритма оценены только на одном диагностическом сиде.",
    ]:
        story.append(p("• " + item, styles["bullet"]))
    story += [p(
        "Рекомендация: прежде чем увеличивать бюджет, исправить раннее схлопывание энтропии RTU на Medium, "
        "добавить gradient/sensitivity norm logging и finite-check, затем повторить 500k-gate на трёх сидах. Для когнитивной карты нужен "
        "reward-remapping/shortcut тест с замороженным backbone, а не ещё одна кластеризация.", styles["callout"]
    )]

    story += [PageBreak(), p("10. Литература", styles["h1"])]
    references = [
        ("Whittington et al., How to build a cognitive map, Nature Neuroscience (2022)", "https://www.nature.com/articles/s41593-022-01153-y"),
        ("George et al., Clone-structured graph representations..., Nature Communications (2021)", "https://www.nature.com/articles/s41467-021-22559-5"),
        ("Elelimy et al., Real-Time Recurrent Learning using Trace Units in Reinforcement Learning", "https://arxiv.org/abs/2409.01449"),
        ("Elsayed et al., Streaming Deep Reinforcement Learning Finally Works", "https://arxiv.org/abs/2410.14606"),
        ("Morad et al., POPGym", "https://arxiv.org/abs/2303.01859"),
        ("Schulman et al., Proximal Policy Optimization Algorithms", "https://arxiv.org/abs/1707.06347"),
        ("Bellec et al., e-prop", "https://www.nature.com/articles/s41467-020-17236-y"),
    ]
    for index, (label, url) in enumerate(references, start=1):
        story.append(p(f'{index}. <a href="{url}" color="#245b78">{label}</a>', styles["body"]))

    story += [p("11. Воспроизводимость", styles["h1"])]
    story += [p(
        "Код, конфигурации, checkpoints, сырые monitor CSV, frozen-latent datasets и итоговые JSON находятся "
        "в отдельной папке RL2/Streaming_RL. Каждый run сохраняет agent.yaml, env.yaml и runner.yaml, поэтому "
        "архитектура и режим обучения восстанавливаются без ручного переноса параметров.", styles["body"]
    )]
    repro_table = [
        ["Артефакт", "Путь относительно RL2/Streaming_RL"],
        ["Основной код", "stream_rl/stream_rl/src"],
        ["Эксперименты", "stream_rl/stream_rl/experiments"],
        ["Финальные representation metrics", "stream_rl/logs/representation_analysis_final"],
        ["RTU stability runs", "stream_rl/logs/rtu_stability_pilots"],
        ["CT-graph depth-4", "stream_rl/logs/architecture_continuing_complex"],
        ["PDF", "output/pdf/rl2_streaming_memory_cognitive_maps_report.pdf"],
    ]
    story.append(report_table(repro_table, [5.0*cm, 11.8*cm], styles))
    story += [Spacer(1, 0.2*cm), p(
        "Проверка: 16 целевых unit/integration tests; RTU-RTRL sensitivity дополнительно сопоставлена с BPTT. "
        "Графики t-SNE и probe-метрики строятся скриптом compare_representations.py только после заморозки "
        "обученного backbone.", styles["body"]
    )]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=1.55*cm, leftMargin=1.55*cm,
        topMargin=1.55*cm, bottomMargin=1.45*cm, title="RL2: память, рабочая память и когнитивные карты",
        author="RL2 research report",
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(colors.HexColor("#657786"))
        canvas.drawString(1.55*cm, 0.75*cm, "RL2 · Streaming RL / PPO / RTU-RTRL")
        canvas.drawRightString(A4[0]-1.55*cm, 0.75*cm, f"стр. {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
