import { ChangeEvent, FormEvent, ReactElement, useEffect, useMemo, useState } from "react";

type StepStatus = "pending" | "running" | "done" | "failed" | "skipped";
type JobStatus = "queued" | "running" | "completed" | "failed";
type Locale = "zh" | "en";

type StepState = {
  status: StepStatus;
  message: string;
  progress: number;
};

type Job = {
  job_id: string;
  file_name: string;
  file_type: "image" | "video";
  status: JobStatus;
  error: string;
  summary: Record<string, unknown>;
  steps: Record<string, StepState>;
  active_step: string;
  progress_percent: number;
  created_at: number;
  updated_at: number;
  options: {
    call_api: boolean;
    run_segmentation: boolean;
    report_language?: Locale;
    conf: number;
    iou: number;
    imgsz: number;
    device: string;
    tracker_backend: string;
  };
};

type Artifact = {
  name: string;
  relative_path: string;
  kind: "image" | "video" | "report" | "csv" | "json" | "file";
  size_bytes: number;
  modified_at?: number;
  url: string;
};

type AreaRow = {
  itemId: string;
  classCode: string;
  className: string;
  confidence: number;
  methodId: "M1" | "M3" | "M4";
  methodName: string;
  area: number;
  status: string;
  widthPx?: number;
  heightPx?: number;
  bboxAreaPx2?: number;
  widthM?: number;
  heightM?: number;
  scaleAssumption?: string;
  cameraAssumption?: string;
  depthMedian?: number;
};

type AreaGroup = {
  itemId: string;
  classCode: string;
  className: string;
  confidence: number;
  M1?: AreaRow;
  M3?: AreaRow;
  M4?: AreaRow;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

const COPY = {
  zh: {
    eyebrow: "RDD2022 Road Inspection Workbench",
    title: "道路病害智能分析工作台",
    lead: "单文件上传，串行执行检测、去重、面积估计和 Qwen 报告生成。每一步都输出可视化证据。",
    waiting: "等待上传",
    language: "语言",
    reset: "新任务",
    uploadTitle: "上传图片或视频",
    uploadHint: "选择单个 .jpg / .png / .mp4 / .mov",
    segmentation: "附加语义分割探索可视化",
    callApi: "调用 SiliconFlow Qwen 生成真实报告",
    apiMissing: "后端未检测到 SILICONFLOW_API_KEY；会生成 request preview，不会真实调用 Qwen。",
    apiReady: "后端已检测到 SILICONFLOW_API_KEY，会尝试真实生成 Qwen 报告。",
    apiSource: "API 来源",
    progress: "任务进度",
    currentStep: "当前步骤",
    conf: "conf 阈值",
    iou: "IoU 阈值",
    submit: "开始分析",
    submitting: "提交中...",
    chooseFile: "请先上传一张图片或一个视频。",
    uploadFailed: "上传失败",
    jobFailed: "任务创建失败",
    rulesTitle: "上传规则",
    rulesText: "一次只处理一个文件。图片进入检测、面积和报告；视频额外进入跨帧去重和代表帧面积计算。",
    visualTitle: "主可视化",
    visualWaiting: "等待 pipeline 输出",
    visualEmpty: "上传后，这里会展示检测框、面积 board、去重时间线或最终报告。",
    stepsTitle: "步骤状态",
    summaryTitle: "统计摘要",
    detections: "检测数",
    events: "去重事件",
    areaRows: "面积条目",
    artifacts: "产物数",
    noStats: "暂无统计",
    areaTitle: "面积计算",
    areaExplain: "Rule-based bbox geometry 使用检测框宽高和固定像素尺度计算：D00=框高×0.8，D10=框宽×1.2，D20/D40=框面积/3；Depth Anything V2-assisted 和 Metric3D-assisted 方法在同一 bbox 区域内读取深度图并进行几何辅助估计。当前都是估计面积，不是真实标定面积。",
    areaEmpty: "面积 CSV 生成后会显示在这里。",
    areaBbox: "检测框",
    areaScale: "尺度假设",
    areaCamera: "相机假设",
    ruleBased: "Rule-based bbox geometry",
    depthAnything: "Depth Anything V2-assisted",
    metric3d: "Metric3D-assisted",
    localReportTitle: "结构化摘要",
    qwenReportTitle: "Qwen 报告",
    savePdf: "打印 / 保存 PDF",
    reportRunning: "Qwen 报告正在等待结构化证据或正在生成中。请看上方进度条和步骤状态。",
    noReportWithApi: "真实 Qwen 报告未生成。请检查“报告生成”步骤的错误信息。若提示缺少 API key，请停止后端，执行 export SILICONFLOW_API_KEY=\"$(cat apikey.txt)\" 后重新启动后端，再提交新任务。",
    noReportWithoutApi: "未勾选真实 API 调用。本次只生成 qwen_request_preview.json，不生成 Qwen 报告。",
    artifactsTitle: "全部中间产物",
    noArtifacts: "暂无产物。",
    artifactHint: "CSV/JSON 是论文和前端二次可视化的可信数据来源，报告只作为自然语言解释层。",
    selectArtifact: "点击产物会在主面板预览；下载按钮才会打开或保存文件。",
    download: "下载",
    reportInputHint: "已读取 report_input.json；摘要和报告均来自同一份结构化证据。",
    localEmpty: "暂无报告。",
    localOverview: "结构化摘要",
    file: "文件",
    status: "状态",
    classStats: "类别统计",
    noClassStats: "暂无类别统计",
    areaMethodExplain: "面积估计解释",
    priorityAreas: "重点面积结果",
    noAreaResults: "暂无面积结果",
    confidenceShort: "置信度",
    stepLabels: {
      upload: "上传",
      segmentation: "语义分割探索",
      detection: "病害检测",
      dedup: "视频去重",
      area: "面积计算",
      report: "报告生成"
    },
    statuses: {
      pending: "等待",
      queued: "排队",
      running: "运行中",
      done: "完成",
      completed: "完成",
      failed: "失败",
      skipped: "跳过"
    }
  },
  en: {
    eyebrow: "RDD2022 Road Inspection Workbench",
    title: "Road Damage Intelligence Workbench",
    lead: "Upload one file and run detection, deduplication, area estimation, and Qwen report generation with visual evidence at every step.",
    waiting: "Waiting for upload",
    language: "Language",
    reset: "New job",
    uploadTitle: "Upload image or video",
    uploadHint: "Choose one .jpg / .png / .mp4 / .mov",
    segmentation: "Attach segmentation exploration visuals",
    callApi: "Call SiliconFlow Qwen for a real report",
    apiMissing: "The backend does not see SILICONFLOW_API_KEY; it will generate a request preview only.",
    apiReady: "The backend sees SILICONFLOW_API_KEY and will try to generate a real Qwen report.",
    apiSource: "API source",
    progress: "Job progress",
    currentStep: "Current step",
    conf: "conf threshold",
    iou: "IoU threshold",
    submit: "Start analysis",
    submitting: "Submitting...",
    chooseFile: "Upload one image or video first.",
    uploadFailed: "Upload failed",
    jobFailed: "Job creation failed",
    rulesTitle: "Upload rules",
    rulesText: "Only one file is processed per job. Images run detection, area, and report. Videos also run cross-frame deduplication and representative-frame area estimation.",
    visualTitle: "Main visualization",
    visualWaiting: "Waiting for pipeline output",
    visualEmpty: "After upload, this panel shows bbox detections, area boards, dedup timelines, or the final report.",
    stepsTitle: "Step status",
    summaryTitle: "Summary",
    detections: "Detections",
    events: "Unique events",
    areaRows: "Area rows",
    artifacts: "Artifacts",
    noStats: "No statistics yet",
    areaTitle: "Area estimation",
    areaExplain: "Rule-based bbox geometry uses the detection-box width, height and a fixed pixel scale: D00=box height×0.8, D10=box width×1.2, and D20/D40=one third of the box area. Depth Anything V2-assisted and Metric3D-assisted estimates read depth maps inside the same bbox region. All values are estimated areas, not calibrated physical ground truth.",
    areaEmpty: "Area rows will appear after the CSV is generated.",
    areaBbox: "Bounding box",
    areaScale: "Scale assumption",
    areaCamera: "Camera assumption",
    ruleBased: "Rule-based bbox geometry",
    depthAnything: "Depth Anything V2-assisted",
    metric3d: "Metric3D-assisted",
    localReportTitle: "Structured summary",
    qwenReportTitle: "Qwen report",
    savePdf: "Print / Save PDF",
    reportRunning: "The Qwen report is waiting for structured evidence or is being generated. Check the progress bar and step status above.",
    noReportWithApi: "A real Qwen report was not generated. Check the Report Generation step message. If the message says the API key is missing, stop the backend, run export SILICONFLOW_API_KEY=\"$(cat apikey.txt)\", restart the backend, and submit a new job.",
    noReportWithoutApi: "Real API calling is disabled. This job only generated qwen_request_preview.json, not a Qwen report.",
    artifactsTitle: "All intermediate artifacts",
    noArtifacts: "No artifacts yet.",
    artifactHint: "CSV/JSON files are the trusted data sources for the paper and frontend visualizations. Reports are the natural-language explanation layer.",
    selectArtifact: "Click an artifact to preview it in the main panel; use Download only when you want to open or save the file.",
    download: "Download",
    reportInputHint: "report_input.json has been loaded; summaries and reports are based on the same structured evidence.",
    localEmpty: "No report yet.",
    localOverview: "Structured summary",
    file: "File",
    status: "Status",
    classStats: "Class statistics",
    noClassStats: "No class statistics yet",
    areaMethodExplain: "Area-estimation notes",
    priorityAreas: "Priority area results",
    noAreaResults: "No area results yet",
    confidenceShort: "conf",
    stepLabels: {
      upload: "Upload",
      segmentation: "Segmentation exploration",
      detection: "Damage detection",
      dedup: "Video deduplication",
      area: "Area estimation",
      report: "Report generation"
    },
    statuses: {
      pending: "Pending",
      queued: "Queued",
      running: "Running",
      done: "Done",
      completed: "Completed",
      failed: "Failed",
      skipped: "Skipped"
    }
  }
} as const;

function statusText(status: StepStatus | JobStatus, locale: Locale): string {
  return COPY[locale].statuses[status] ?? status;
}

function artifactUrl(url: string, version?: string | number): string {
  const base = `${API_BASE}${url}`;
  if (version === undefined || version === "") return base;
  const separator = base.includes("?") ? "&" : "?";
  return `${base}${separator}v=${encodeURIComponent(String(version))}`;
}

function artifactCacheKey(item?: Artifact): string {
  if (!item) return "";
  const modified = item.modified_at ? Math.floor(item.modified_at * 1000) : 0;
  return `${item.size_bytes}-${modified}`;
}

function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let current = "";
  let row: string[] = [];
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(current);
      current = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(current);
      if (row.some((cell) => cell.trim() !== "")) rows.push(row);
      row = [];
      current = "";
    } else {
      current += char;
    }
  }
  if (current || row.length) {
    row.push(current);
    rows.push(row);
  }
  const [headers = [], ...body] = rows;
  return body.map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""])));
}

function normalizeAreaRows(rows: Record<string, string>[]): AreaRow[] {
  if (!rows.length) return [];
  const parseBbox = (value: string) => {
    try {
      const parsed = JSON.parse(value || "[]");
      if (Array.isArray(parsed) && parsed.length >= 4) return parsed.map(Number);
    } catch {
      return [];
    }
    return [];
  };
  const bboxMetrics = (row: Record<string, string>) => {
    const bbox = parseBbox(row.bbox_xyxy || "");
    const widthPx = Number(row.width_px || (bbox.length >= 4 ? Math.max(bbox[2] - bbox[0], 0) : 0));
    const heightPx = Number(row.height_px || (bbox.length >= 4 ? Math.max(bbox[3] - bbox[1], 0) : 0));
    return {
      widthPx,
      heightPx,
      bboxAreaPx2: Number(row.bbox_area_px2 || widthPx * heightPx || 0),
      widthM: row.width_m ? Number(row.width_m) : undefined,
      heightM: row.height_m ? Number(row.height_m) : undefined,
      scaleAssumption: row.scale_assumption,
      cameraAssumption: row.camera_assumption
    };
  };
  if ("method_id" in rows[0]) {
    return rows.map((row) => ({
      itemId: row.detection_id || row.event_id || "",
      classCode: row.class_code || "",
      className: row.class_name || "",
      confidence: Number(row.confidence || row.max_confidence || 0),
      methodId: row.method_id as "M1" | "M3" | "M4",
      methodName: row.method_name || methodLabel(row.method_id as "M1" | "M3" | "M4", "en"),
      area: Number(row.estimated_area_m2 || 0),
      status: row.status || "",
      depthMedian: row.depth_median_m ? Number(row.depth_median_m) : undefined,
      ...bboxMetrics(row)
    }));
  }
  return rows.flatMap((row) => {
    const base = {
      itemId: row.detection_id || row.event_id || "",
      classCode: row.class_code || "",
      className: row.class_name || "",
      confidence: Number(row.confidence || row.max_confidence || 0),
      ...bboxMetrics(row)
    };
    return [
      { ...base, methodId: "M1" as const, methodName: methodLabel("M1", "en"), area: Number(row.M1_area_m2 || 0), status: "success" },
      { ...base, methodId: "M3" as const, methodName: methodLabel("M3", "en"), area: Number(row.M3_area_m2 || 0), status: row.M3_status || "success" },
      { ...base, methodId: "M4" as const, methodName: methodLabel("M4", "en"), area: Number(row.M4_area_m2 || 0), status: row.M4_status || "success" }
    ];
  });
}

function groupAreaRows(rows: AreaRow[]): AreaGroup[] {
  const groups = new Map<string, AreaGroup>();
  for (const row of rows) {
    const group = groups.get(row.itemId) ?? {
      itemId: row.itemId,
      classCode: row.classCode,
      className: row.className,
      confidence: row.confidence
    };
    group[row.methodId] = row;
    groups.set(row.itemId, group);
  }
  return [...groups.values()].sort((a, b) => b.confidence - a.confidence);
}

function formatArea(value?: number): string {
  if (value === undefined || Number.isNaN(value)) return "-";
  return `${value.toFixed(value >= 10 ? 2 : 3)} m²`;
}

function methodLabel(methodId: "M1" | "M3" | "M4", locale: Locale): string {
  const labels = {
    M1: locale === "zh" ? "Rule-based bbox geometry" : "Rule-based bbox geometry",
    M3: locale === "zh" ? "Depth Anything V2-assisted" : "Depth Anything V2-assisted",
    M4: locale === "zh" ? "Metric3D-assisted" : "Metric3D-assisted"
  };
  return labels[methodId];
}

function markdownTable(lines: string[], keyPrefix: string) {
  const splitRow = (line: string) =>
    line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());
  const headers = splitRow(lines[0]);
  const rows = lines.slice(2).map(splitRow);
  return (
    <table className="markdownTable" key={keyPrefix}>
      <thead>
        <tr>{headers.map((header, index) => <th key={`${keyPrefix}-h-${index}`}>{renderInline(header)}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr key={`${keyPrefix}-r-${rowIndex}`}>
            {headers.map((_, cellIndex) => <td key={`${keyPrefix}-c-${rowIndex}-${cellIndex}`}>{renderInline(row[cellIndex] ?? "")}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function renderInline(text: string): ReactElement | string | (ReactElement | string)[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  if (parts.length === 1) return text.replaceAll("`", "");
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`strong-${index}`}>{part.slice(2, -2).replaceAll("`", "")}</strong>;
    }
    return part.replaceAll("`", "");
  });
}

function MarkdownView({ text }: { text: string }) {
  const lines = text.split(/\r?\n/);
  const elements: ReactElement[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = Math.min(heading[1].length, 4);
      const Tag = (["h2", "h3", "h4", "h5"] as const)[level - 1];
      elements.push(<Tag key={`h-${index}`}>{renderInline(heading[2])}</Tag>);
      index += 1;
      continue;
    }
    if (line.includes("|") && lines[index + 1]?.match(/^\s*\|?\s*:?-{3,}/)) {
      const tableLines = [line, lines[index + 1]];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        tableLines.push(lines[index]);
        index += 1;
      }
      elements.push(markdownTable(tableLines, `table-${index}`));
      continue;
    }
    if (line.match(/^\s*[-*]\s+/)) {
      const items: string[] = [];
      while (index < lines.length && lines[index].match(/^\s*[-*]\s+/)) {
        items.push(lines[index].replace(/^\s*[-*]\s+/, ""));
        index += 1;
      }
      elements.push(
        <ul key={`ul-${index}`}>
          {items.map((item, itemIndex) => <li key={`ul-${index}-${itemIndex}`}>{renderInline(item)}</li>)}
        </ul>
      );
      continue;
    }
    if (line.match(/^\s*\d+\.\s+/)) {
      const items: string[] = [];
      while (index < lines.length && lines[index].match(/^\s*\d+\.\s+/)) {
        items.push(lines[index].replace(/^\s*\d+\.\s+/, ""));
        index += 1;
      }
      elements.push(
        <ol key={`ol-${index}`}>
          {items.map((item, itemIndex) => <li key={`ol-${index}-${itemIndex}`}>{renderInline(item)}</li>)}
        </ol>
      );
      continue;
    }
    const paragraph: string[] = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].match(/^(#{1,4})\s+/) &&
      !lines[index].match(/^\s*[-*]\s+/) &&
      !lines[index].match(/^\s*\d+\.\s+/) &&
      !(lines[index].includes("|") && lines[index + 1]?.match(/^\s*\|?\s*:?-{3,}/))
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    elements.push(<p key={`p-${index}`}>{renderInline(paragraph.join(" "))}</p>);
  }
  return <div className="markdownReport">{elements}</div>;
}

function escapeHtml(text: string): string {
  return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function markdownToHtml(markdown: string): string {
  const inlineHtml = (value: string) => escapeHtml(value).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "$1");
  const lines = markdown.split(/\r?\n/);
  const html: string[] = [];
  for (const line of lines) {
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = Math.min(heading[1].length + 1, 5);
      html.push(`<h${level}>${inlineHtml(heading[2])}</h${level}>`);
    } else if (line.trim().startsWith("|")) {
      html.push(`<pre>${escapeHtml(line)}</pre>`);
    } else if (line.match(/^\s*[-*]\s+/)) {
      html.push(`<p>• ${inlineHtml(line.replace(/^\s*[-*]\s+/, ""))}</p>`);
    } else if (line.trim()) {
      html.push(`<p>${inlineHtml(line)}</p>`);
    }
  }
  return html.join("\n");
}

function printMarkdownReport(markdown: string, title: string) {
  const popup = window.open("", "_blank", "width=980,height=1200");
  if (!popup) return;
  popup.document.write(`<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(title)}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; color: #111; margin: 42px; line-height: 1.6; }
    h1, h2, h3, h4, h5 { margin: 1.2em 0 .45em; line-height: 1.25; }
    p { margin: .55em 0; }
    pre { white-space: pre-wrap; background: #f4f4f4; padding: 8px 10px; border-radius: 6px; }
  </style>
</head>
<body>
${markdownToHtml(markdown)}
<script>window.onload = () => setTimeout(() => window.print(), 200);</script>
</body>
</html>`);
  popup.document.close();
  popup.focus();
}

function pickHeroArtifact(artifacts: Artifact[]): Artifact | undefined {
  const preference = [
    "_annotated.mp4",
    "predicted_review_grid.jpg",
    "area_board.jpg",
    "event_timeline.png",
    "density_timeline.png",
    "report.md"
  ];
  const candidates = artifacts.filter((item) => !isRawUploadArtifact(item));
  const pool = candidates.length ? candidates : artifacts;
  for (const key of preference) {
    const found = pool.find((item) => item.name.includes(key) || item.relative_path.includes(key));
    if (found) return found;
  }
  return pool.find((item) => item.kind === "video" || item.kind === "image" || item.kind === "report");
}

function isRawUploadArtifact(item: Artifact): boolean {
  return (
    item.relative_path.startsWith("upload/") ||
    item.relative_path.startsWith("input_images/") ||
    item.relative_path.includes("/representative_raw_frames/")
  );
}

function artifactDisplayRank(item: Artifact): number {
  const path = item.relative_path;
  if (path.includes("_annotated.mp4")) return 0;
  if (path.includes("predicted_review_grid")) return 1;
  if (path.includes("area_board")) return 2;
  if (path.includes("representative_frames/")) return 3;
  if (path.includes("density_timeline") || path.includes("event_timeline")) return 4;
  if (item.kind === "report") return 5;
  if (item.kind === "video") return 6;
  if (item.kind === "image") return 7;
  return 9;
}

export function App() {
  const [locale, setLocale] = useState<Locale>(() => (localStorage.getItem("rdp-locale") === "en" ? "en" : "zh"));
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [runSegmentation, setRunSegmentation] = useState(false);
  const [callApi, setCallApi] = useState(false);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.5);
  const [job, setJob] = useState<Job | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [reportText, setReportText] = useState("");
  const [selectedArtifactPath, setSelectedArtifactPath] = useState("");
  const [selectedArtifactText, setSelectedArtifactText] = useState("");
  const [areaRows, setAreaRows] = useState<AreaRow[]>([]);
  const [reportInput, setReportInput] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [apiReady, setApiReady] = useState<boolean | null>(null);
  const [apiSource, setApiSource] = useState("");
  const [apiHint, setApiHint] = useState("");

  const text = COPY[locale];
  const selectedArtifact = useMemo(
    () => artifacts.find((item) => item.relative_path === selectedArtifactPath),
    [artifacts, selectedArtifactPath]
  );
  const heroArtifact = selectedArtifact ?? pickHeroArtifact(artifacts);
  const visualArtifacts = artifacts
    .filter((item) => (item.kind === "image" || item.kind === "video" || item.kind === "report") && !isRawUploadArtifact(item))
    .slice()
    .sort((a, b) => artifactDisplayRank(a) - artifactDisplayRank(b) || a.relative_path.localeCompare(b.relative_path));
  const tableArtifacts = artifacts.filter((item) => item.kind === "csv" || item.kind === "json");
  const areaGroups = useMemo(() => groupAreaRows(areaRows), [areaRows]);
  const reportLooksLikePlaceholder =
    !reportText ||
    reportText.includes("API call was not executed") ||
    reportText.includes("SILICONFLOW_API_KEY is not set") ||
    reportText.includes("Report generation failed");

  const localReport = useMemo(() => {
    if (!job) return text.localEmpty;
    const counts = (job.summary?.counts_by_class ?? job.summary?.unique_counts_by_class ?? {}) as Record<string, number>;
    const countLines = Object.entries(counts)
      .map(([key, value]) => `- ${key}: ${value}`)
      .join("\n");
    const topAreas = areaGroups
      .slice(0, 5)
      .map((item) => {
        const confLabel = locale === "zh" ? "置信度" : "conf";
        return `- ${item.itemId} ${item.classCode} ${confLabel}=${item.confidence.toFixed(2)} | Rule-based ${formatArea(item.M1?.area)} | Depth Anything V2 ${formatArea(item.M3?.area)} | Metric3D ${formatArea(item.M4?.area)}`;
      })
      .join("\n");
    if (locale === "en") {
      return [
        `## ${text.localOverview}`,
        `- ${text.file}: ${job.file_name}`,
        `- ${text.status}: ${statusText(job.status, locale)}`,
        `- ${text.detections}: ${String(job.summary?.total_detections ?? 0)}`,
        job.summary?.unique_events !== undefined ? `- ${text.events}: ${String(job.summary.unique_events)}` : "",
        "",
        `### ${text.classStats}`,
        countLines || `- ${text.noClassStats}`,
        "",
        `### ${text.areaMethodExplain}`,
        "- Rule-based bbox geometry: detection-box geometry with fixed pixel scale. D00 uses box height×0.8, D10 uses box width×1.2, and D20/D40 use one third of the box rectangle area.",
        "- Depth Anything V2-assisted: depth map over the bbox rectangle, then scaled by the same class-specific effective-area ratio.",
        "- Metric3D-assisted: depth map over the bbox rectangle, then scaled by the same class-specific effective-area ratio.",
        "- All values are estimated areas because no camera intrinsics, calibration object, or physical area GT is available.",
        "",
        `### ${text.priorityAreas}`,
        topAreas || `- ${text.noAreaResults}`
      ]
        .filter(Boolean)
        .join("\n");
    }
    return [
      `## ${text.localOverview}`,
      `- ${text.file}: ${job.file_name}`,
      `- ${text.status}: ${statusText(job.status, locale)}`,
      `- ${text.detections}: ${String(job.summary?.total_detections ?? 0)}`,
      job.summary?.unique_events !== undefined ? `- ${text.events}: ${String(job.summary.unique_events)}` : "",
      "",
      `### ${text.classStats}`,
      countLines || `- ${text.noClassStats}`,
      "",
      `### ${text.areaMethodExplain}`,
      "- Rule-based bbox geometry: 使用检测框几何尺寸和固定像素尺度计算。D00 按框高×0.8，D10 按框宽×1.2，D20/D40 按框矩形面积的 1/3。",
      "- Depth Anything V2-assisted: 在 bbox 矩形区域内读取 Depth Anything V2 深度图，再乘以同类别的有效面积比例。",
      "- Metric3D-assisted: 在 bbox 矩形区域内读取 Metric3D 深度图，再乘以同类别的有效面积比例。",
      "- 三种方法都属于 estimated area，因为当前没有相机内参、标定尺或真实面积 GT。",
      "",
      `### ${text.priorityAreas}`,
      topAreas || `- ${text.noAreaResults}`
    ]
      .filter(Boolean)
      .join("\n");
  }, [areaGroups, job, locale, text]);

  useEffect(() => {
    localStorage.setItem("rdp-locale", locale);
  }, [locale]);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((res) => res.json())
      .then((data) => {
        setApiReady(Boolean(data.siliconflow_api_ready));
        setApiSource(String(data.siliconflow_api_source ?? ""));
        setApiHint(String(data.siliconflow_api_hint ?? ""));
      })
      .catch(() => setApiReady(null));
  }, []);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      const next = await fetch(`${API_BASE}/api/jobs/${job.job_id}`).then((res) => res.json());
      setJob(next);
      const artifactResponse = await fetch(`${API_BASE}/api/jobs/${job.job_id}/artifacts`).then((res) => res.json());
      setArtifacts(artifactResponse.artifacts ?? []);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.job_id, job?.status]);

  useEffect(() => {
    if (!job || !["completed", "failed"].includes(job.status)) return;
    fetch(`${API_BASE}/api/jobs/${job.job_id}/artifacts`)
      .then((res) => res.json())
      .then((data) => setArtifacts(data.artifacts ?? []))
      .catch(() => undefined);
  }, [job?.job_id, job?.status]);

  useEffect(() => {
    const report = artifacts.find((item) => item.name === "report.md");
    if (!report) {
      setReportText("");
      return;
    }
    fetch(artifactUrl(report.url, artifactCacheKey(report)))
      .then((res) => res.text())
      .then(setReportText)
      .catch(() => setReportText(locale === "zh" ? "报告文件读取失败。" : "Failed to read report file."));
  }, [artifacts, locale]);

  useEffect(() => {
    if (!artifacts.length) {
      setSelectedArtifactPath("");
      return;
    }
    const current = artifacts.find((item) => item.relative_path === selectedArtifactPath);
    const next = pickHeroArtifact(artifacts);
    if (current && (!isRawUploadArtifact(current) || !next || isRawUploadArtifact(next))) return;
    setSelectedArtifactPath(next?.relative_path ?? "");
  }, [artifacts, selectedArtifactPath]);

  useEffect(() => {
    const item = selectedArtifact;
    if (!item || !["csv", "json", "file"].includes(item.kind)) {
      setSelectedArtifactText("");
      return;
    }
    fetch(artifactUrl(item.url, artifactCacheKey(item)))
      .then((res) => res.text())
      .then((textValue) => setSelectedArtifactText(textValue.slice(0, 20000)))
      .catch(() => setSelectedArtifactText(""));
  }, [selectedArtifact]);

  useEffect(() => {
    const area = artifacts.find((item) => item.name === "area_estimates.csv" || item.name === "event_area_estimates.csv");
    if (!area) {
      setAreaRows([]);
      return;
    }
    fetch(artifactUrl(area.url, artifactCacheKey(area)))
      .then((res) => res.text())
      .then((csvText) => setAreaRows(normalizeAreaRows(parseCsv(csvText))))
      .catch(() => setAreaRows([]));
  }, [artifacts]);

  useEffect(() => {
    const input = artifacts.find((item) => item.name === "report_input.json");
    if (!input) {
      setReportInput(null);
      return;
    }
    fetch(artifactUrl(input.url, artifactCacheKey(input)))
      .then((res) => res.json())
      .then(setReportInput)
      .catch(() => setReportInput(null));
  }, [artifacts]);

  function clearJob() {
    setFile(null);
    setJob(null);
    setArtifacts([]);
    setReportText("");
    setSelectedArtifactPath("");
    setSelectedArtifactText("");
    setAreaRows([]);
    setReportInput(null);
    setError("");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl("");
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setJob(null);
    setArtifacts([]);
    setReportText("");
    setSelectedArtifactPath("");
    setSelectedArtifactText("");
    setAreaRows([]);
    setReportInput(null);
    setError("");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(selected ? URL.createObjectURL(selected) : "");
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError(text.chooseFile);
      return;
    }
    setSubmitting(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("run_segmentation", String(runSegmentation));
    formData.append("call_api", String(callApi));
    formData.append("report_language", locale);
    formData.append("conf", String(conf));
    formData.append("iou", String(iou));
    formData.append("imgsz", "832");
    formData.append("device", "auto");
    formData.append("tracker_backend", "bytetrack");
    try {
      const response = await fetch(`${API_BASE}/api/jobs`, { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? text.uploadFailed);
      setJob(payload);
      const artifactResponse = await fetch(`${API_BASE}/api/jobs/${payload.job_id}/artifacts`).then((res) => res.json());
      setArtifacts(artifactResponse.artifacts ?? []);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : text.jobFailed);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="appShell">
      <section className="masthead">
        <div>
          <p className="eyebrow">{text.eyebrow}</p>
          <h1>{text.title}</h1>
          <p className="lead">{text.lead}</p>
        </div>
        <div className="mastheadActions">
          <div className="languageToggle" aria-label={text.language}>
            <button className={locale === "zh" ? "active" : ""} type="button" onClick={() => setLocale("zh")}>
              中文
            </button>
            <button className={locale === "en" ? "active" : ""} type="button" onClick={() => setLocale("en")}>
              EN
            </button>
          </div>
          <button className="ghostButton" type="button" onClick={clearJob}>
            {text.reset}
          </button>
          <div className="statusPill">{job ? `Job ${job.job_id} · ${statusText(job.status, locale)}` : text.waiting}</div>
          {job && (
            <div className="progressCard" aria-label={text.progress}>
              <div className="progressMeta">
                <span>{text.progress}</span>
                <strong>{job.progress_percent}%</strong>
              </div>
              <div className="progressTrack">
                <div style={{ width: `${job.progress_percent}%` }} />
              </div>
              <small>
                {text.currentStep}: {text.stepLabels[job.active_step as keyof typeof text.stepLabels] ?? job.active_step}
              </small>
            </div>
          )}
        </div>
      </section>

      <div className="workspaceGrid">
        <aside className="controlPanel">
          <form onSubmit={onSubmit}>
            <label className="uploadBox">
              <span>{text.uploadTitle}</span>
              <strong>{file ? file.name : text.uploadHint}</strong>
              <input type="file" accept=".jpg,.jpeg,.png,.mp4,.mov" onChange={onFileChange} />
            </label>

            {previewUrl && file && (
              <div className="localPreview">
                {file.type.startsWith("video") ? <video src={previewUrl} controls /> : <img src={previewUrl} alt="uploaded preview" />}
              </div>
            )}

            <label className="switchRow">
              <input type="checkbox" checked={runSegmentation} onChange={(event) => setRunSegmentation(event.target.checked)} />
              <span>{text.segmentation}</span>
            </label>
            <label className="switchRow">
              <input type="checkbox" checked={callApi} onChange={(event) => setCallApi(event.target.checked)} />
              <span>{text.callApi}</span>
            </label>
            {callApi && apiReady === false && <p className="warningText">{text.apiMissing}</p>}
            {callApi && apiReady === true && <p className="okText">{text.apiReady}</p>}
            {callApi && apiSource && (
              <p className={apiReady ? "okText" : "warningText"}>
                {text.apiSource}: {apiSource}
                {apiHint ? ` · ${apiHint}` : ""}
              </p>
            )}

            <div className="sliderBlock">
              <span>
                {text.conf} {conf.toFixed(2)}
              </span>
              <input type="range" min="0.1" max="0.8" step="0.05" value={conf} onChange={(event) => setConf(Number(event.target.value))} />
            </div>
            <div className="sliderBlock">
              <span>
                {text.iou} {iou.toFixed(2)}
              </span>
              <input type="range" min="0.3" max="0.8" step="0.05" value={iou} onChange={(event) => setIou(Number(event.target.value))} />
            </div>

            <button className="runButton" disabled={submitting || !file}>
              {submitting ? text.submitting : text.submit}
            </button>
            {error && <p className="errorText">{error}</p>}
          </form>

          <div className="rulesCard">
            <h2>{text.rulesTitle}</h2>
            <p>{text.rulesText}</p>
          </div>
        </aside>

        <section className="visualStage">
          <div className="stageHeader">
            <h2>{text.visualTitle}</h2>
            <span>{heroArtifact ? heroArtifact.relative_path : text.visualWaiting}</span>
          </div>
          <div className="heroCanvas">
            {!heroArtifact && <p>{text.visualEmpty}</p>}
            {heroArtifact?.kind === "image" && <img src={artifactUrl(heroArtifact.url, artifactCacheKey(heroArtifact))} alt={heroArtifact.name} />}
            {heroArtifact?.kind === "video" && <video src={artifactUrl(heroArtifact.url, artifactCacheKey(heroArtifact))} controls preload="metadata" />}
            {heroArtifact?.kind === "report" && <MarkdownView text={reportText} />}
            {heroArtifact && ["csv", "json", "file"].includes(heroArtifact.kind) && (
              <pre>{selectedArtifactText || `${heroArtifact.name}\n${heroArtifact.relative_path}`}</pre>
            )}
          </div>

          <div className="artifactStrip">
            {visualArtifacts.slice(0, 12).map((item) => (
              <button
                className={item.relative_path === heroArtifact?.relative_path ? "active" : ""}
                key={item.relative_path}
                type="button"
                onClick={() => setSelectedArtifactPath(item.relative_path)}
              >
                {item.kind === "image" && <img src={artifactUrl(item.url, artifactCacheKey(item))} alt={item.name} />}
                {item.kind === "video" && <video src={artifactUrl(item.url, artifactCacheKey(item))} muted preload="metadata" />}
                {item.kind === "report" && <span className="reportThumb">REPORT</span>}
                <span>{item.name}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="inspectorPanel">
          <section className="card stepCard">
            <h2>{text.stepsTitle}</h2>
            <div className="stepList">
              {Object.entries(text.stepLabels).map(([key, label]) => {
                const step = job?.steps?.[key] ?? { status: "pending" as StepStatus, message: "", progress: 0 };
                return (
                  <div className={`stepItem ${step.status}`} key={key}>
                    <span>{label}</span>
                    <strong>{statusText(step.status, locale)}</strong>
                    <div className="stepProgress" aria-label={`${label} ${step.progress ?? 0}%`}>
                      <div style={{ width: `${step.progress ?? 0}%` }} />
                    </div>
                    {step.message && <small>{step.message}</small>}
                  </div>
                );
              })}
            </div>
          </section>

          <section className="card metricCard">
            <h2>{text.summaryTitle}</h2>
            {job ? (
              <div className="metricGrid">
                <div>
                  <span>{text.detections}</span>
                  <strong>{String(job.summary?.total_detections ?? 0)}</strong>
                </div>
                <div>
                  <span>{text.events}</span>
                  <strong>{String(job.summary?.unique_events ?? "-")}</strong>
                </div>
                <div>
                  <span>{text.areaRows}</span>
                  <strong>{areaGroups.length}</strong>
                </div>
                <div>
                  <span>{text.artifacts}</span>
                  <strong>{artifacts.length}</strong>
                </div>
              </div>
            ) : (
              <p>{text.noStats}</p>
            )}
            {Boolean(job?.summary?.api_warning) && <p className="warningText">{String(job?.summary?.api_warning)}</p>}
          </section>

          <section className="card areaCard">
            <h2>{text.areaTitle}</h2>
            <p>{text.areaExplain}</p>
            <div className="areaList">
              {areaGroups.slice(0, 8).map((item) => (
                <div className="areaItem" key={item.itemId}>
                  <div>
                    <strong>{item.itemId}</strong>
                    <span>
                      {item.classCode} · {text.confidenceShort} {item.confidence.toFixed(2)}
                    </span>
                  </div>
                  <dl>
                    <div>
                      <dt>{methodLabel("M1", locale)}</dt>
                      <dd>{formatArea(item.M1?.area)}</dd>
                    </div>
                    <div>
                      <dt>{methodLabel("M3", locale)}</dt>
                      <dd>{formatArea(item.M3?.area)}</dd>
                    </div>
                    <div>
                      <dt>{methodLabel("M4", locale)}</dt>
                      <dd>{formatArea(item.M4?.area)}</dd>
                    </div>
                  </dl>
                  <p className="bboxDetail">
                    {text.areaBbox}: {Math.round(item.M1?.widthPx ?? item.M3?.widthPx ?? item.M4?.widthPx ?? 0)}×
                    {Math.round(item.M1?.heightPx ?? item.M3?.heightPx ?? item.M4?.heightPx ?? 0)} px
                    {item.M1?.widthM !== undefined && item.M1?.heightM !== undefined
                      ? ` (${item.M1.widthM.toFixed(3)}×${item.M1.heightM.toFixed(3)} m)`
                      : ""}
                  </p>
                </div>
              ))}
              {areaGroups.length === 0 && <p>{text.areaEmpty}</p>}
            </div>
          </section>

          <section className="card reportCard">
            <h2>{text.localReportTitle}</h2>
            <MarkdownView text={localReport} />
          </section>

          <section className="card reportCard">
            <div className="reportHeader">
            <h2>{text.qwenReportTitle}</h2>
            {!reportLooksLikePlaceholder && (
              <button type="button" onClick={() => printMarkdownReport(reportText, text.qwenReportTitle)}>
                {text.savePdf}
              </button>
            )}
          </div>
          {reportLooksLikePlaceholder ? (
              <MarkdownView
                text={
                  job?.steps?.report?.status === "running"
                    ? text.reportRunning
                    : job?.summary?.api_warning
                      ? String(job.summary.api_warning)
                      : job?.options.call_api
                        ? text.noReportWithApi
                        : text.noReportWithoutApi
                }
              />
            ) : (
              <MarkdownView text={reportText} />
            )}
          </section>
        </section>
      </div>

      <section className="artifactTable">
        <h2>{text.artifactsTitle}</h2>
        <div className="tableGrid">
          {artifacts.map((item) => (
            <div className={`artifactRow ${item.relative_path === heroArtifact?.relative_path ? "active" : ""}`} key={item.relative_path}>
              <button type="button" onClick={() => setSelectedArtifactPath(item.relative_path)}>
                <span>{item.kind}</span>
                <strong>{item.relative_path}</strong>
                <em>{Math.ceil(item.size_bytes / 1024)} KB</em>
              </button>
              <a href={artifactUrl(item.url, artifactCacheKey(item))} download>
                {text.download}
              </a>
            </div>
          ))}
          {artifacts.length === 0 && <p>{text.noArtifacts}</p>}
        </div>
        {tableArtifacts.length > 0 && <p className="hint">{text.artifactHint}</p>}
        {reportInput && <p className="hint">{text.reportInputHint}</p>}
      </section>
    </main>
  );
}
