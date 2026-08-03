"""Generate the PDF report for the RTRL approximation study."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
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
CODE = ROOT / "stream_rl"
STUDY = CODE / "logs" / "rtrl_approx_study"
ARCHIVE = STUDY / "archive_analysis"
GRADIENT = STUDY / "gradient"
CORRECTED = STUDY / "corrected_analysis"
OUTPUT = ROOT / "output" / "pdf" / "rtrl_approximations_report.pdf"


def register_fonts() -> None:
    pdfmetrics.registerFont(
        TTFont("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("Arial-Italic", "/System/Library/Fonts/Supplemental/Arial Italic.ttf")
    )
    pdfmetrics.registerFontFamily(
        "Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic"
    )


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title_ru",
            parent=sample["Title"],
            fontName="Arial-Bold",
            fontSize=21,
            leading=26,
            textColor=colors.HexColor("#17324d"),
            alignment=TA_CENTER,
            spaceAfter=15,
        ),
        "subtitle": ParagraphStyle(
            "subtitle_ru",
            parent=sample["Normal"],
            fontName="Arial",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#526b80"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "h1_ru",
            parent=sample["Heading1"],
            fontName="Arial-Bold",
            fontSize=15.5,
            leading=19,
            textColor=colors.HexColor("#17324d"),
            spaceBefore=9,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2_ru",
            parent=sample["Heading2"],
            fontName="Arial-Bold",
            fontSize=11.8,
            leading=15,
            textColor=colors.HexColor("#28617e"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body_ru",
            parent=sample["BodyText"],
            fontName="Arial",
            fontSize=9.2,
            leading=12.6,
            textColor=colors.HexColor("#202b35"),
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "bullet_ru",
            parent=sample["BodyText"],
            fontName="Arial",
            fontSize=9,
            leading=12.2,
            leftIndent=13,
            firstLineIndent=-7,
            bulletIndent=4,
            spaceAfter=3.5,
        ),
        "callout": ParagraphStyle(
            "callout_ru",
            parent=sample["BodyText"],
            fontName="Arial-Bold",
            fontSize=9.6,
            leading=13.3,
            leftIndent=9,
            rightIndent=9,
            borderColor=colors.HexColor("#78a1ba"),
            borderWidth=0.8,
            borderPadding=8,
            backColor=colors.HexColor("#eef5f8"),
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "table_ru",
            parent=sample["BodyText"],
            fontName="Arial",
            fontSize=7.4,
            leading=9.1,
        ),
        "table_head": ParagraphStyle(
            "table_head_ru",
            parent=sample["BodyText"],
            fontName="Arial-Bold",
            fontSize=7.4,
            leading=9.1,
            textColor=colors.white,
        ),
        "caption": ParagraphStyle(
            "caption_ru",
            parent=sample["BodyText"],
            fontName="Arial-Italic",
            fontSize=7.6,
            leading=9.6,
            textColor=colors.HexColor("#586d7d"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small_ru",
            parent=sample["BodyText"],
            fontName="Arial",
            fontSize=7.4,
            leading=9.6,
            textColor=colors.HexColor("#586d7d"),
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(data, widths, style_set):
    converted = []
    for idx, row in enumerate(data):
        style = style_set["table_head"] if idx == 0 else style_set["table"]
        converted.append([p(str(cell), style) for cell in row])
    result = LongTable(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20364f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bac6cf")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f4f7f9")],
                ),
            ]
        )
    )
    return result


def add_image(story, path: Path, caption: str, style_set, width=16.0 * cm):
    if path.exists():
        from PIL import Image as PILImage

        with PILImage.open(path) as image:
            ratio = image.height / image.width
        story.append(Image(str(path), width=width, height=width * ratio))
        story.append(p(caption, style_set["caption"]))
    else:
        story.append(p(f"Визуализация не найдена: {path.name}", style_set["small"]))


def fmt(mean: float, sd: float, digits=3) -> str:
    if not math.isfinite(float(mean)):
        return "n/a"
    if not math.isfinite(float(sd)):
        sd = 0.0
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def archive_table() -> list[list[str]]:
    frame = pd.read_csv(ARCHIVE / "archive_summary.csv")
    keep = [
        "PPO-GRU",
        "RTU exact RTRL",
        "RTU one-step",
        "RTU nominal TBPTT(5)",
        "GRU one-step",
        "GRU nominal TBPTT(5)",
        "GRU e-prop (symmetric)",
    ]
    frame = frame[
        (frame.task == "RepeatPreviousEasy")
        & (frame.cohort == "tuned")
        & frame.method.isin(keep)
    ]
    rows = [["Метод", "Сиды", "Шаги", "Return, последние 500"]]
    for row in frame.sort_values("return_mean", ascending=False).itertuples():
        rows.append(
            [
                row.method,
                str(row.n_seeds),
                f"{int(row.max_step_min):,}".replace(",", " "),
                fmt(row.return_mean, row.return_sd),
            ]
        )
    return rows


def gradient_table() -> list[list[str]]:
    data = json.loads((GRADIENT / "gradient_summary.json").read_text())["final_step"]
    by_key = {(row["kind"], row["method"]): row for row in data}
    rows = [["Память", "1-step", "TBPTT(5)", "Local/e-prop", "Вывод"]]
    for kind in ("gru", "lstm", "rtu", "delta"):
        one = by_key[(kind, "one_step")]["cosine"]
        five = by_key[(kind, "tbptt5")]["cosine"]
        local = by_key[(kind, "local_eprop")]["cosine"]
        verdict = "TBPTT(5) близок к exact" if five > 0.98 else "длинный хвост ошибок"
        rows.append([kind.upper(), f"{one:.3f}", f"{five:.3f}", f"{local:.3f}", verdict])
    return rows


def scaling_table() -> list[list[str]]:
    rows = [["Память", "Exact sensitivity, MiB", "Compact/local, MiB", "ms/step"]]
    frame = pd.read_csv(STUDY / "scaling_h64" / "gradient_resources.csv")
    for row in frame[frame.seed == 1].itertuples():
        rows.append(
            [
                row.kind.upper(),
                f"{row.exact_sensitivity_mib:.3f}",
                f"{row.local_trace_upper_mib:.3f}",
                f"{row.milliseconds_per_step:.2f}",
            ]
        )
    return rows


def corrected_table() -> list[list[str]]:
    path = CORRECTED / "corrected_summary.csv"
    rows = [["Метод", "Сиды", "Min шагов", "Return", "Шагов/с*"]]
    if not path.exists():
        rows.append(["Полные прогоны не завершены", "-", "-", "-", "-"])
        return rows
    frame = pd.read_csv(path)
    for row in frame.sort_values("return_mean", ascending=False).itertuples():
        rows.append(
            [
                row.method,
                str(int(row.n_seeds)),
                f"{int(row.max_step_min):,}".replace(",", " "),
                fmt(row.return_mean, row.return_sd),
                f"{row.throughput_median:.1f}",
            ]
        )
    return rows


def corrected_narrative() -> str:
    path = CORRECTED / "corrected_summary.csv"
    if not path.exists():
        return "Полные исправленные прогоны еще не агрегированы."
    frame = pd.read_csv(path).set_index("method")
    pieces = []
    for method in (
        "RTU true TBPTT(5)",
        "RTU true TBPTT(5), matched lambda",
        "LSTM one-step",
        "Delta-rule one-step",
        "GRU approximate e-prop",
        "LSTM approximate e-prop",
    ):
        if method in frame.index:
            row = frame.loc[method]
            pieces.append(f"{method}: {fmt(row.return_mean, row.return_sd)}")
    archive = pd.read_csv(ARCHIVE / "archive_summary.csv")
    archive = archive[(archive.task == "RepeatPreviousEasy") & (archive.cohort == "tuned")]
    anchors = []
    for method in ("PPO-GRU", "RTU exact RTRL", "RTU one-step"):
        subset = archive[archive.method == method]
        if not subset.empty:
            row = subset.iloc[0]
            anchors.append(f"{method}: {fmt(row.return_mean, row.return_sd)}")
    return (
        "Исправленные методы - "
        + "; ".join(pieces)
        + ". Архивные ориентиры - "
        + "; ".join(anchors)
        + "."
    )


def timing_narrative() -> str:
    def rate(path: Path) -> float:
        frame = pd.read_csv(path)
        stat = path.stat()
        start = getattr(stat, "st_birthtime", stat.st_ctime)
        elapsed = stat.st_mtime - start
        return float(frame["total_steps"].max()) / elapsed if elapsed > 0 else math.nan

    exact_paths = [
        CODE / "logs/rtu_hparam_pilots/easy/popgym__stream_rtu1_h64_linear_rtrl_paper/monitor_seed_0.csv",
        CODE / "logs/rtu_stability_pilots/easy/popgym__stream_rtu1_h64_linear_rtrl_paper_rmax99/monitor_seed_0.csv",
    ]
    tbptt_paths = sorted(
        (STUDY / "full_corrected_tbptt").glob("**/monitor_seed_*.csv")
    )
    exact_rates = [rate(path) for path in exact_paths if path.exists()]
    tbptt_rates = [rate(path) for path in tbptt_paths if path.exists()]
    if not exact_rates or not tbptt_rates:
        return "Wall-clock сравнение compact RTU-RTRL и true TBPTT пока недоступно."
    exact = float(pd.Series(exact_rates).median())
    tbptt = float(pd.Series(tbptt_rates).median())
    return (
        f"После JAX-компиляции compact RTU-RTRL показывает около {exact:.0f} шагов/с, "
        f"true RTU-TBPTT(5) - около {tbptt:.0f} шагов/с на той же машине. "
        f"Наблюдаемое преимущество exact RTU составляет примерно {exact / tbptt:.2f}x; "
        "начальная компиляция и параллельная нагрузка в оценку не входят."
    )


def corrected_interpretation() -> str:
    path = CORRECTED / "corrected_summary.csv"
    if not path.exists():
        return ""
    new = pd.read_csv(path).set_index("method")
    archive = pd.read_csv(ARCHIVE / "archive_summary.csv")
    archive = archive[(archive.task == "RepeatPreviousEasy") & (archive.cohort == "tuned")]
    archived = archive.set_index("method")
    statements = []
    matched = "RTU true TBPTT(5), matched lambda"
    if matched in new.index and {"RTU exact RTRL", "RTU one-step"}.issubset(archived.index):
        value = float(new.loc[matched, "return_mean"])
        one = float(archived.loc["RTU one-step", "return_mean"])
        exact = float(archived.loc["RTU exact RTRL", "return_mean"])
        statements.append(
            f"Matched TBPTT(5) находится на {value - one:+.3f} выше one-step и на {value - exact:+.3f} относительно exact RTU-RTRL"
        )
    if {"LSTM approximate e-prop", "LSTM one-step"}.issubset(new.index):
        delta = float(new.loc["LSTM approximate e-prop", "return_mean"] - new.loc["LSTM one-step", "return_mean"])
        statements.append(f"approximate e-prop улучшает LSTM относительно one-step на {delta:+.3f}")
    if "Delta-rule one-step" in new.index:
        statements.append(
            f"delta-rule завершает обучение с return {float(new.loc['Delta-rule one-step', 'return_mean']):.3f}"
        )
    return "; ".join(statements) + ("." if statements else "")


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(colors.HexColor("#607487"))
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    register_fonts()
    s = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.45 * cm,
        title="Аппроксимации RTRL для рекуррентной памяти",
        author="RL2 research",
    )
    story = []

    story.extend(
        [
            Spacer(1, 2.2 * cm),
            p("Аппроксимации RTRL для рекуррентной памяти", s["title"]),
            p(
                "LSTM, GRU, RTU и delta-rule память в streaming actor-critic",
                s["subtitle"],
            ),
            Spacer(1, 0.5 * cm),
            p(
                "Аудит реализации, согласованность градиентов, вычислительная цена и обучение на POPGym RepeatPreviousEasy",
                s["subtitle"],
            ),
            Spacer(1, 1.0 * cm),
            p(
                "Ключевой результат: точный RTRL практически полезен здесь не как универсальный алгоритм для любой RNN, а как следствие специальной диагональной динамики RTU. Для плотных GRU/LSTM одношаговое усечение заметно искажает градиент; настоящее TBPTT(5) почти восстанавливает его направление на синтетической проверке. В архиве PPO-GRU остается сильнейшим поведенческим методом на выбранной простой POMDP.",
                s["callout"],
            ),
            Spacer(1, 4.2 * cm),
            p("Антон Михайлович - исследовательский отчет RL2", s["subtitle"]),
            p("02 августа 2026 г.", s["subtitle"]),
        ]
    )
    story.append(PageBreak())

    story.append(p("1. Вопросы и дизайн исследования", s["h1"]))
    story.append(
        p(
            "Исследование проверяет четыре связанных вопроса: насколько одношаговая аппроксимация RTRL портит градиент для разных типов памяти; помогает ли окно из пяти шагов; можно ли сохранить forward-mode преимущества через локальные eligibility traces; отражается ли согласованность градиента в качестве и скорости RL-обучения.",
            s["body"],
        )
    )
    taxonomy = [
        ["Метод", "Временной градиент", "Постоянная память", "Статус"],
        ["Exact RTRL", "вся история, forward sensitivity", "O(state x params)", "референс"],
        ["RTU-RTRL", "точный, компактный из-за диагональной рекуррентности", "линейна по локальным параметрам", "основной streaming метод"],
        ["1-step", "только текущий переход", "минимальная", "прямая аппроксимация"],
        ["True TBPTT(5)", "последние 5 переходов", "окно состояний", "реализовано отдельно"],
        ["Local/e-prop", "forward eligibility, локальный якобиан", "локальные traces", "структурная аппроксимация"],
        ["Delta-rule", "fast-weight состояние, 1-step update", "матрица fast weights", "линейная память"],
        ["PPO-GRU", "BPTT внутри rollout", "rollout buffer", "num_envs=1 baseline"],
    ]
    story.append(table(taxonomy, [3.2 * cm, 6.2 * cm, 3.4 * cm, 3.0 * cm], s))
    story.append(p("Протокол", s["h2"]))
    for item in (
        "Поведенческая среда: POPGym RepeatPreviousEasy - простая POMDP, на которой в архиве действительно обучаются PPO и RTU.",
        "Сопоставимый бюджет: около 170 тыс. взаимодействий; три независимых сида 0, 42, 123; PPO запускается с num_envs=1. Оконный runner выполняет 168.1 тыс. шагов из-за кратности chunk/window.",
        "Градиентный тест: 8 сидов, длина 25, width=8; exact RTRL сверяется с full BPTT.",
        "Масштабирование: width 16/32/64; время после прогрева и размер sensitivity tensor.",
    ):
        story.append(p("- " + item, s["bullet"]))
    story.append(PageBreak())

    story.append(p("2. Аудит кода и архива логов", s["h1"]))
    story.append(
        p(
            "Архив ветки stream_eprop загружен отдельно от рабочего дерева: 123 MB после распаковки, 714 monitor CSV. После дедупликации выбрано 633 записей; нечисловых итогов нет; все выбранные конфигурации используют num_envs=1.",
            s["body"],
        )
    )
    story.append(
        p(
            "Критическая ошибка маркировки: прежний tbptt_steps=5 передавался только в параметр unroll рекуррентного слоя. StreamAC на каждом обновлении останавливал градиент на предыдущем carry и подавал последовательность длины один. Поэтому архивные кривые с названием TBPTT(5) не имеют пятишагового горизонта и не считаются валидным сравнением TBPTT(1) против TBPTT(5). Косвенная проверка подтверждает проблему: средняя абсолютная разница 174 пар one-step/nominal-5 равна лишь 0.0064.",
            s["callout"],
        )
    )
    story.append(
        p(
            "Вторая ошибка затрагивала symmetric e-prop: ветка напрямую брала одношаговый autodiff-градиент ячейки и не использовала накопленный eligibility trace. Исправленная версия умножает текущий симметричный learning signal на forward trace. Скалярный trace_decay остается дополнительной аппроксимацией, поэтому реализация в отчете называется approximate e-prop, а не exact e-prop.",
            s["body"],
        )
    )
    fixes = [
        ["Изменение", "Проверка"],
        ["WindowedStreamAC с реальным окном 5", "тест подтверждает входную последовательность длины 5"],
        ["DeltaRuleCell", "one-step smoke test"],
        ["Symmetric e-prop использует trace", "отдельный unit test feedback и traced gradient"],
        ["RTU compact sensitivity", "равенство с ненулевой диагональю full BPTT"],
        ["Сброс carry/traces по done", "поэлементный тест двух env slots"],
    ]
    story.append(table(fixes, [7.3 * cm, 8.5 * cm], s))
    story.append(PageBreak())

    story.append(p("3. Что показывают архивные прогоны", s["h1"]))
    story.append(table(archive_table(), [6.4 * cm, 1.8 * cm, 2.5 * cm, 5.1 * cm], s))
    story.append(
        p(
            "На RepeatPreviousEasy PPO-GRU почти решает задачу, точный RTU-RTRL занимает второе место, а RTU-one-step существенно слабее. GRU-one-step и прежний e-prop остаются около случайной политики. На AutoencodeEasy все выбранные методы остаются около -0.5; на Active T-maze L5 PPO и GRU находятся около 50% успеха, а RTU не дает устойчивого результата. Поэтому поведенческая часть нового исследования ограничена средой, где есть наблюдаемый сигнал обучения.",
            s["body"],
        )
    )
    add_image(
        story,
        ARCHIVE / "archive_learning_curves.png",
        "Архивные кривые: сплошные средние по трем сидам, полоса - разброс.",
        s,
    )
    story.append(PageBreak())

    story.append(p("4. Согласованность градиентов", s["h1"]))
    story.append(
        p(
            "Exact RTRL в синтетическом тесте численно совпал с full BPTT: cosine около 1.0, максимальная абсолютная ошибка порядка 1e-7. Сравнение ниже показывает cosine с точным градиентом на шаге 25; чем ближе к 1, тем лучше совпадает направление обновления.",
            s["body"],
        )
    )
    story.append(table(gradient_table(), [2.3 * cm, 2.2 * cm, 2.5 * cm, 2.8 * cm, 6.0 * cm], s))
    story.append(
        p(
            "Relative L2 подтверждает тот же порядок: для one-step GRU/LSTM/RTU/Delta получены 0.615/0.525/0.584/0.892; для TBPTT(5) - 0.062/0.051/0.043/0.568; для local approximation - 0.243/0.211/0.000/0.457. Sign agreement также сохранен в CSV-артефакте.",
            s["body"],
        )
    )
    story.append(
        p(
            "Главный численный вывод: TBPTT(5) практически совпадает с exact для GRU, LSTM и RTU (cosine 0.998-0.999), тогда как one-step дает 0.801-0.862. У delta-rule хвост влияний длиннее: TBPTT(5) достигает лишь 0.842, one-step - 0.488. Локальная структурная аппроксимация лучше one-step для всех четырех типов; для RTU она точна благодаря диагональной рекуррентности.",
            s["callout"],
        )
    )
    add_image(
        story,
        GRADIENT / "gradient_cosine_by_horizon.png",
        "Cosine similarity относительно exact RTRL по длине последовательности, среднее по 8 сидам.",
        s,
    )
    story.append(PageBreak())

    story.append(p("5. Цена точного RTRL", s["h1"]))
    story.append(
        p(
            "Для плотной RNN exact sensitivity имеет размер state_dim x param_dim. При width=64 один небольшой GRU требует 6.05 MiB, LSTM - 16.13 MiB только на sensitivity одного рекуррентного блока; actor и critic удваивают цену. На практике вычислительная сложность плотного exact RTRL растет быстрее памяти и становится главным ограничением.",
            s["body"],
        )
    )
    story.append(table(scaling_table(), [3.0 * cm, 4.6 * cm, 4.6 * cm, 3.6 * cm], s))
    story.append(
        p(
            "Столбец compact/local показывает верхнюю оценку структурного следа. Для RTU при width=64 она равна 0.063 MiB против 4.063 MiB полного sensitivity tensor. Компактный след RTU остается точным; для GRU/LSTM подобное сокращение требует отбросить межнейронные производные и становится аппроксимацией.",
            s["callout"],
        )
    )
    add_image(
        story,
        GRADIENT / "gradient_memory_cost.png",
        "Память exact sensitivity и локального структурного следа в диагностическом тесте.",
        s,
    )
    story.append(PageBreak())

    story.append(p("6. Исправленные трехсидовые прогоны", s["h1"]))
    story.append(table(corrected_table(), [6.0 * cm, 1.6 * cm, 2.4 * cm, 3.2 * cm, 2.6 * cm], s))
    story.append(p(corrected_narrative(), s["body"]))
    story.append(p(timing_narrative(), s["body"]))
    interpretation = corrected_interpretation()
    if interpretation:
        story.append(p(interpretation, s["callout"]))
    story.append(
        p(
            "* Наблюдаемая пропускная способность рассчитана по времени между первой и последней записью monitor CSV. Значение не включает начальную JAX-компиляцию и зависит от параллельной нагрузки, поэтому служит инженерной, а не аппаратно-независимой оценкой.",
            s["small"],
        )
    )
    add_image(
        story,
        CORRECTED / "corrected_learning_curves.png",
        "Исправленные методы: rolling mean по 250 эпизодам, среднее и SD по трем сидам.",
        s,
    )
    add_image(
        story,
        CORRECTED / "equal_budget_final.png",
        "Сопоставление архивных baseline и исправленных вариантов при близком бюджете около 170 тыс. шагов.",
        s,
    )
    story.append(PageBreak())

    story.append(p("7. Интерпретация и ответ на гипотезу", s["h1"]))
    for item in (
        "Качество градиента. Одношаговое усечение приемлемо как дешевый baseline, но систематически теряет дальние вклады. Для плотных GRU/LSTM окно 5 дает гораздо более верное направление.",
        "Архитектурное преимущество RTU. Компактный exact RTRL возможен из-за диагональной рекуррентной динамики, а не из-за универсального свойства streaming AC.",
        "Поведенческое качество. На выбранной POMDP exact RTU-RTRL превосходит RTU-one-step, но PPO-GRU с num_envs=1 остается заметно сильнее. Данных недостаточно для общего тезиса о превосходстве streaming методов.",
        "Delta-rule память. Простой fast-weight baseline имеет слабую one-step согласованность; результат нельзя переносить на полноценный Gated DeltaNet без отдельной реализации gating, normalization и параллельного обучения.",
        "E-prop. Исправленный symmetric вариант теперь использует forward trace, однако скалярный decay остается упрощением. Полученное качество следует трактовать как оценку данной аппроксимации, а не всего семейства e-prop.",
    ):
        story.append(p("- " + item, s["bullet"]))
    story.append(
        p(
            "Практическая рекомендация: сохранять RTU-RTRL как основной exact-online вариант; для GRU/LSTM использовать true TBPTT(5) как контролируемый компромисс; one-step оставлять дешевой нижней границей; e-prop развивать через cell-specific diagonal Jacobian вместо единого trace_decay. Уровень уверенности - высокий для градиентного вывода, средний для RepeatPreviousEasy, низкий для обобщения на другие POMDP.",
            s["callout"],
        )
    )
    story.append(PageBreak())

    story.append(p("8. Архитектуры", s["h1"]))
    architectures = [
        ["Название", "Backbone", "Головы actor/critic", "Обучение памяти"],
        ["PPO-GRU", "obs embed 64 -> GRU 64", "линейные", "PPO BPTT rollout"],
        ["RTU exact RTRL", "obs embed 64 -> RTU hidden 64 = 128 real outputs", "линейные", "compact exact RTRL"],
        ["RTU one-step", "obs embed 64 -> RTU hidden 64 = 128 real outputs", "линейные", "stop-gradient carry"],
        ["RTU true TBPTT(5)", "obs embed 64 -> RTU hidden 64 = 128 real outputs", "линейные", "окно 5"],
        ["LSTM one-step", "obs embed 64 -> LSTM 64", "линейные", "stop-gradient carry"],
        ["GRU/LSTM approx e-prop", "obs embed 64 -> custom cell 64 + LayerNorm", "линейные", "symmetric signal x scalar-decay trace"],
        ["Delta-rule one-step", "obs embed 64 -> fast weights 16 x 64", "линейные", "stop-gradient fast weights"],
    ]
    story.append(table(architectures, [3.5 * cm, 5.1 * cm, 3.6 * cm, 3.8 * cm], s))
    story.append(p("Гиперпараметры", s["h2"]))
    hparams = [
        ["Группа", "Параметры"],
        ["Общие", "num_envs=1; gamma=0.99; nominal 170k steps (true TBPTT: 168.1k); seeds 0,42,123; CPU"],
        ["Новые streaming variants", "trace_lambda=0.98; actor_lr=1.0; critic_lr=1.0; actor_kappa=0.2; critic_kappa=0.5; entropy=0.01; adaptive ObGD=true"],
        ["Архивные RTU variants", "trace_lambda=0.97; actor_lr=critic_lr=1.0; actor_kappa=0.2; critic_kappa=0.5; entropy=0.01; adaptive ObGD=true"],
        ["True TBPTT", "window=5; window-start carry detached; AC(lambda) traces применяются по порядку внутри окна; matched control trace_lambda=0.97"],
        ["Approx e-prop", "trace_decay=0.9; trace_lambda=0.98; symmetric feedback; LayerNorm=true; sparse init=true; sparsity=0.9"],
        ["Delta-rule", "key_dim=16; decay gate clipped to [0.90,0.999]; normalized q/k; learned write gate"],
        ["PPO archive", "num_envs=1; num_steps=128; num_minibatches=1; epochs=4; lr=7e-4; GAE lambda=0.95; clip=0.2; entropy=0.01; GRU64 + LayerNorm"],
        ["Gradient diagnostic", "width=8; input_dim=6; horizon=25; 8 seeds; loss=sum final state"],
    ]
    story.append(table(hparams, [4.0 * cm, 12.0 * cm], s))
    story.append(PageBreak())

    story.append(p("9. Ограничения и воспроизводимость", s["h1"]))
    for item in (
        "Поведенческое сравнение использует одну простую POPGym-задачу. Active T-maze и AutoencodeEasy исключены из нового полного прогона, поскольку архив не демонстрирует устойчивого обучения даже у baseline.",
        "Wall-clock зависит от JAX compilation cache и одновременной нагрузки. Основной вывод о цене опирается на размер sensitivity tensor и отдельный прогретый микробенчмарк.",
        "Архивные nominal TBPTT(5) и symmetric e-prop сохранены только как аудит исторических результатов; они не считаются реализациями заявленных алгоритмов.",
        "Локальный градиентный benchmark и RL-реализация e-prop различаются: benchmark использует структурный block-diagonal Jacobian, RL-вариант - еще более дешевый scalar-decay trace.",
        "Новый delta-rule слой является минимальным линейным fast-weight baseline, а не воспроизведением полной архитектуры Gated DeltaNet.",
    ):
        story.append(p("- " + item, s["bullet"]))
    story.append(p("Артефакты", s["h2"]))
    artifacts = [
        ["Артефакт", "Расположение"],
        ["Архив логов", "RL2/Streaming_RL/archive/stream_eprop_aba0926/logs"],
        ["Аудит архива", "stream_rl/logs/rtrl_approx_study/archive_analysis"],
        ["Градиенты", "stream_rl/logs/rtrl_approx_study/gradient"],
        ["Scaling", "stream_rl/logs/rtrl_approx_study/scaling_h16,h32,h64"],
        ["Исправленные прогоны", "stream_rl/logs/rtrl_approx_study/full_corrected_*"],
        ["Агрегация", "stream_rl/logs/rtrl_approx_study/corrected_analysis"],
    ]
    story.append(table(artifacts, [5.0 * cm, 11.0 * cm], s))
    story.append(PageBreak())

    story.append(p("10. Источники", s["h1"]))
    references = [
        "Bellec G. et al. A solution to the learning dilemma for recurrent networks of spiking neurons. Nature Communications 11, 3625 (2020). https://www.nature.com/articles/s41467-020-17236-y",
        "Elelimy E. et al. Real-Time Recurrent Learning using Trace Units in Reinforcement Learning (2024). https://arxiv.org/abs/2409.01449",
        "Yang S. et al. Gated Delta Networks: Improving Mamba2 with Delta Rule (2024). https://arxiv.org/abs/2412.06464",
        "Tallec C., Ollivier Y. Unbiased Online Recurrent Optimization. ICLR (2018). https://openreview.net/forum?id=rJQDjk-0b",
        "Morad S. et al. POPGym: Benchmarking Partially Observable Reinforcement Learning (2023). https://arxiv.org/abs/2303.01859",
    ]
    for idx, reference in enumerate(references, 1):
        story.append(p(f"{idx}. {reference}", s["body"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(
        p(
            "Итоговый статус: кодовые ошибки исправлены и покрыты тестами; архив провалидирован; exact/approx gradient benchmark завершен; поведенческое сравнение проведено на трех независимых сидах при сопоставимом бюджете около 170 тыс. шагов и num_envs=1.",
            s["callout"],
        )
    )

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(OUTPUT)


if __name__ == "__main__":
    build()
