#!/usr/bin/env python3
"""Export the v1.1 delivery results as a dependency-free Excel workbook."""
from __future__ import annotations

import csv
import math
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures_v1_1_optimal" / "SPR_SC_v1_1_complete_results.xlsx"
DISPLAY = {
    "enron": "Enron", "school": "P.School", "school2": "H.School",
    "directors": "Directors", "crime": "Crime", "fb_auto": "FB-AUTO",
    "jf17k": "JF17K", "m_fb15k": "M-FB15K", "wikipeople": "WikiPeople",
}
CURVE_DIRS = (
    "results_v1_1_optimal_core_awgn_0_20_step2",
    "results_v1_1_optimal_core_rayleigh_0_20_step2",
    "results_v1_1_optimal_fb_auto_awgn_0_20_step2",
    "results_v1_1_optimal_fb_auto_rayleigh_0_20_step2",
    "results_v1_1_optimal_jf17k_fast_awgn_0_20_step2",
    "results_v1_1_optimal_jf17k_fast_rayleigh_0_20_step2",
    "results_v1_1_optimal_m_fb15k_fast_awgn_0_20",
    "results_v1_1_optimal_m_fb15k_fast_rayleigh_0_20",
    "results_v1_1_optimal_wikipeople_fast_awgn_0_20",
    "results_v1_1_optimal_wikipeople_fast_rayleigh_0_20",
)
METHOD_ORDER = {"SHyRe-count": 0, "SHyRe-fast-count": 0,
                "Max Clique": 1, "ECC": 2, "DEMON": 3}


def load_snr_rows():
    rows = []
    for directory_name in CURVE_DIRS:
        directory = ROOT / directory_name
        source = directory / "snr_baselines.csv"
        if not source.is_file():
            continue
        fast = "_fast_" in directory_name
        with source.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("returncode") != "0" or row["dataset"] not in DISPLAY:
                    continue
                method = row["method"]
                if method == "SHyRe":
                    method = "SHyRe-fast-count" if fast else "SHyRe-count"
                rows.append({
                    "dataset_key": row["dataset"], "dataset": DISPLAY[row["dataset"]],
                    "channel": row["channel"].upper(), "snr_db": int(row["snr_db"]),
                    "seed": int(row.get("seed", 123)), "method": method,
                    "f1": float(row["f1"]), "protocol": "SHyRe-fast" if fast else "SHyRe",
                    "shown_in_fig12": "是" if row["dataset"] in {"enron", "fb_auto"} else "否",
                    "source": directory_name,
                })
    # Keep only complete 11-SNR × 3-seed SHyRe datasets per channel.
    complete = set()
    cells = defaultdict(set)
    for row in rows:
        if row["method"].startswith("SHyRe"):
            cells[(row["dataset_key"], row["channel"])].add((row["snr_db"], row["seed"]))
    for key, values in cells.items():
        if len(values) == 33:
            complete.add(key)
    rows = [row for row in rows if (row["dataset_key"], row["channel"]) in complete]
    rows.sort(key=lambda row: (list(DISPLAY).index(row["dataset_key"]), row["channel"],
                               row["snr_db"], row["seed"], METHOD_ORDER.get(row["method"], 99)))
    return rows


def aggregate_snr(raw):
    groups = defaultdict(list)
    metadata = {}
    for row in raw:
        key = (row["dataset_key"], row["dataset"], row["channel"], row["snr_db"], row["method"])
        groups[key].append(row["f1"])
        metadata[key] = (row["protocol"], row["shown_in_fig12"])
    result = []
    for key, values in groups.items():
        protocol, shown = metadata[key]
        result.append({
            "dataset_key": key[0], "dataset": key[1], "channel": key[2],
            "snr_db": key[3], "method": key[4], "f1_mean": statistics.mean(values),
            "f1_std": statistics.pstdev(values), "n_seeds": len(values),
            "protocol": protocol, "shown_in_fig12": shown,
        })
    result.sort(key=lambda row: (list(DISPLAY).index(row["dataset_key"]), row["channel"],
                                  row["snr_db"], METHOD_ORDER.get(row["method"], 99)))
    return result


def load_ablation():
    path = ROOT / "figures_v1_1_optimal" / "ablation_all_datasets_awgn20.csv"
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def column_name(index):
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def xml_value(value, row, column):
    ref = f"{column_name(column)}{row}"
    if value is None or value == "":
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            value = ""
        else:
            style = 2 if isinstance(value, float) else 0
            return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    text = escape(str(value))
    preserve = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f'<c r="{ref}" t="inlineStr"><is><t{preserve}>{text}</t></is></c>'


def sheet_xml(rows, freeze=True, autofilter=True):
    width_count = max((len(row) for row in rows), default=1)
    widths = []
    for column in range(width_count):
        width = max((len(str(row[column])) if column < len(row) and row[column] is not None else 0
                     for row in rows), default=8)
        widths.append(min(max(width + 2, 10), 48))
    columns = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{width}" customWidth="1"/>'
        for i, width in enumerate(widths))
    body = []
    for row_number, values in enumerate(rows, 1):
        cells = []
        for column, value in enumerate(values):
            cell = xml_value(value, row_number, column)
            if row_number == 1:
                cell = cell.replace(f'r="{column_name(column)}{row_number}"',
                                    f'r="{column_name(column)}{row_number}" s="1"', 1)
            cells.append(cell)
        body.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    pane = '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>' if freeze else ""
    selection = '<selection pane="bottomLeft" activeCell="A2" sqref="A2"/>' if freeze else '<selection activeCell="A1" sqref="A1"/>'
    filter_xml = (f'<autoFilter ref="A1:{column_name(width_count - 1)}{len(rows)}"/>'
                  if autofilter and len(rows) > 1 else "")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetViews><sheetView workbookViewId="0">{pane}{selection}</sheetView></sheetViews>
<sheetFormatPr defaultRowHeight="15"/><cols>{columns}</cols><sheetData>{''.join(body)}</sheetData>
{filter_xml}<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>
</worksheet>'''


def make_workbook(sheets):
    content_sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, len(sheets) + 1))
    content_types = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
{content_sheets}</Types>'''
    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, (name, _) in enumerate(sheets, 1))
    workbook = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{workbook_sheets}</sheets></workbook>'''
    relationships = "".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, len(sheets) + 1))
    workbook_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{relationships}
<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000"/></numFmts>
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    package_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles)
        for index, (_, rows) in enumerate(sheets, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml(rows))


def main():
    raw = load_snr_rows()
    aggregate = aggregate_snr(raw)
    ablation = load_ablation()
    snrs = list(range(0, 21, 2))

    info = [
        ["项目", "SPR-SC v1.1 实验结果交付"],
        ["生成时间", datetime.now().astimezone().isoformat(timespec="seconds")],
        ["主曲线协议", "AWGN/Rayleigh；0–20 dB，间隔2 dB；3个随机种子"],
        ["图1/图2展示", "仅 Enron 与 FB-AUTO"],
        ["Excel覆盖", "九个已完整数据集；逐种子、均值、标准差、所有基线"],
        ["未纳入", "DBLP仍在运行；Hosts-Virus和Foursquare按实验安排排除"],
        ["fast数据集", "JF17K、M-FB15K、WikiPeople使用SHyRe-fast候选路径"],
        ["评价指标", "Exact-match reconstruction F1"],
        ["说明", "模型优化与超参数寻优仍在进行，当前结果不是最终定型参数"],
    ]
    raw_sheet = [["数据集", "内部名称", "信道", "SNR(dB)", "随机种子", "方法", "F1", "协议", "图1/2展示", "来源目录"]]
    raw_sheet += [[r["dataset"], r["dataset_key"], r["channel"], r["snr_db"], r["seed"],
                   r["method"], r["f1"], r["protocol"], r["shown_in_fig12"], r["source"]] for r in raw]
    mean_sheet = [["数据集", "信道", "SNR(dB)", "方法", "F1均值", "F1标准差", "种子数", "协议", "图1/2展示"]]
    mean_sheet += [[r["dataset"], r["channel"], r["snr_db"], r["method"], r["f1_mean"],
                    r["f1_std"], r["n_seeds"], r["protocol"], r["shown_in_fig12"]] for r in aggregate]

    def pivot(channel):
        rows = [["数据集", "方法"] + [f"{snr}dB均值" for snr in snrs]]
        lookup = {(r["dataset"], r["method"], r["snr_db"]): r["f1_mean"]
                  for r in aggregate if r["channel"] == channel}
        pairs = sorted({(r["dataset"], r["method"]) for r in aggregate if r["channel"] == channel},
                       key=lambda pair: (list(DISPLAY.values()).index(pair[0]), METHOD_ORDER.get(pair[1], 99)))
        rows += [[dataset, method] + [lookup.get((dataset, method, snr)) for snr in snrs]
                 for dataset, method in pairs]
        return rows

    ablation_headers = list(ablation[0]) if ablation else []
    ablation_sheet = [ablation_headers]
    for row in ablation:
        values = []
        for header in ablation_headers:
            value = row[header]
            if header in {"f1_mean", "f1_std", "delta_vs_base", "candidate_recall_mean", "runtime_mean_seconds"} and value:
                value = float(value)
            elif header == "seeds" and value:
                value = int(value)
            values.append(value)
        ablation_sheet.append(values)

    best_sheet = [["数据集", "Base F1", "最佳模块", "最佳 F1", "相对Base增益", "最佳F1标准差"]]
    by_dataset = defaultdict(list)
    for row in ablation:
        if row.get("f1_mean"):
            by_dataset[row["dataset"]].append(row)
    for dataset in DISPLAY:
        if dataset not in by_dataset:
            continue
        rows = by_dataset[dataset]
        best = max(rows, key=lambda row: float(row["f1_mean"]))
        base = next(row for row in rows if row["module"] == "base")
        best_sheet.append([DISPLAY[dataset], float(base["f1_mean"]), best["label"],
                           float(best["f1_mean"]), float(best["delta_vs_base"]), float(best["f1_std"])])

    sheets = [
        ("说明", info), ("SNR全部逐种子", raw_sheet), ("SNR均值标准差", mean_sheet),
        ("AWGN透视", pivot("AWGN")), ("Rayleigh透视", pivot("RAYLEIGH")),
        ("消融完整结果", ablation_sheet), ("消融最优配置", best_sheet),
    ]
    make_workbook(sheets)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes, {len(raw)} raw SNR rows)")


if __name__ == "__main__":
    main()
