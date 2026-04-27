import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type StepStatus = "pending" | "running" | "done" | "failed" | "skipped";
type JobStatus = "queued" | "running" | "completed" | "failed";
type Locale = "zh" | "en";

type StepState = {
  status: StepStatus;
  message: string;
};

type Job = {
  job_id: string;
  file_name: string;
  file_type: "image" | "video";
  status: JobStatus;
  error: string;
  summary: Record<string, unknown>;
  steps: Record<string, StepState>;
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
  url: string;
};

type AreaRow = {
  itemId: string;
  classCode: string;
  className: string;
  confidence: number;
  methodId: "M1" | "M3" | "M4";
  area: number;
  status: string;
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
    areaExplain: "M1 由检测框宽高和固定像素尺度计算：D00=框高×0.8，D10=框宽×1.2，D20/D40=框面积/3；M3/M4 在此基础上引入 Depth Anything V2 或 Metric3D 深度图。当前都是估计面积，不是真实标定面积。",
    areaEmpty: "面积 CSV 生成后会显示在这里。",
    localReportTitle: "结构化摘要",
    qwenReportTitle: "Qwen 报告",
    noReportWithApi: "真实 Qwen 报告未生成。请确认启动后端前已经 export SILICONFLOW_API_KEY，然后重新提交任务。",
    noReportWithoutApi: "未勾选真实 API 调用。本次只生成 qwen_request_preview.json，不生成 Qwen 报告。",
    artifactsTitle: "全部中间产物",
    noArtifacts: "暂无产物。",
    artifactHint: "CSV/JSON 是论文和前端二次可视化的可信数据来源，报告只作为自然语言解释层。",
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
    areaExplain: "M1 is computed from detection-box geometry and fixed pixel scale: D00=box height×0.8, D10=box width×1.2, D20/D40=box area/3. M3/M4 add Depth Anything V2 or Metric3D depth maps. All values are estimated areas, not calibrated physical ground truth.",
    areaEmpty: "Area rows will appear after the CSV is generated.",
    localReportTitle: "Structured summary",
    qwenReportTitle: "Qwen report",
    noReportWithApi: "A real Qwen report was not generated. Export SILICONFLOW_API_KEY before starting the backend, then submit the job again.",
    noReportWithoutApi: "Real API calling is disabled. This job only generated qwen_request_preview.json, not a Qwen report.",
    artifactsTitle: "All intermediate artifacts",
    noArtifacts: "No artifacts yet.",
    artifactHint: "CSV/JSON files are the trusted data sources for the paper and frontend visualizations. Reports are the natural-language explanation layer.",
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

function artifactUrl(url: string): string {
  return `${API_BASE}${url}`;
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
  if ("method_id" in rows[0]) {
    return rows.map((row) => ({
      itemId: row.detection_id || row.event_id || "",
      classCode: row.class_code || "",
      className: row.class_name || "",
      confidence: Number(row.confidence || row.max_confidence || 0),
      methodId: row.method_id as "M1" | "M3" | "M4",
      area: Number(row.estimated_area_m2 || 0),
      status: row.status || "",
      depthMedian: row.depth_median_m ? Number(row.depth_median_m) : undefined
    }));
  }
  return rows.flatMap((row) => {
    const base = {
      itemId: row.detection_id || row.event_id || "",
      classCode: row.class_code || "",
      className: row.class_name || "",
      confidence: Number(row.confidence || row.max_confidence || 0)
    };
    return [
      { ...base, methodId: "M1" as const, area: Number(row.M1_area_m2 || 0), status: "success" },
      { ...base, methodId: "M3" as const, area: Number(row.M3_area_m2 || 0), status: row.M3_status || "success" },
      { ...base, methodId: "M4" as const, area: Number(row.M4_area_m2 || 0), status: row.M4_status || "success" }
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

function pickHeroArtifact(artifacts: Artifact[]): Artifact | undefined {
  const preference = [
    "predicted_review_grid.jpg",
    "area_board.jpg",
    "event_timeline.png",
    "density_timeline.png",
    "annotated.mp4",
    "report.md"
  ];
  for (const key of preference) {
    const found = artifacts.find((item) => item.name.includes(key) || item.relative_path.includes(key));
    if (found) return found;
  }
  return artifacts.find((item) => item.kind === "image" || item.kind === "video" || item.kind === "report");
}

function isRawUploadArtifact(item: Artifact): boolean {
  return (
    item.relative_path.startsWith("upload/") ||
    item.relative_path.startsWith("input_images/") ||
    item.relative_path.includes("/representative_raw_frames/")
  );
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
  const [areaRows, setAreaRows] = useState<AreaRow[]>([]);
  const [reportInput, setReportInput] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [apiReady, setApiReady] = useState<boolean | null>(null);

  const text = COPY[locale];
  const heroArtifact = useMemo(() => pickHeroArtifact(artifacts), [artifacts]);
  const imageArtifacts = artifacts.filter((item) => item.kind === "image");
  const visualImageArtifacts = imageArtifacts.filter((item) => !isRawUploadArtifact(item));
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
        return `- ${item.itemId} ${item.classCode} ${confLabel}=${item.confidence.toFixed(2)} | M1 ${formatArea(item.M1?.area)} | M3 ${formatArea(item.M3?.area)} | M4 ${formatArea(item.M4?.area)}`;
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
        "- M1: detection-box geometry with fixed pixel scale. D00 uses box height×0.8, D10 uses box width×1.2, and D20/D40 use one third of the box rectangle area.",
        "- M3: Depth Anything V2 depth map over the bbox rectangle, then scaled by the same class-specific effective-area ratio.",
        "- M4: Metric3D depth map over the bbox rectangle, then scaled by the same class-specific effective-area ratio.",
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
      "- M1: 使用检测框几何尺寸和固定像素尺度计算。D00 按框高×0.8，D10 按框宽×1.2，D20/D40 按框矩形面积的 1/3。",
      "- M3: 在 bbox 矩形区域内读取 Depth Anything V2 深度图，再乘以同类别的有效面积比例。",
      "- M4: 在 bbox 矩形区域内读取 Metric3D 深度图，再乘以同类别的有效面积比例。",
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
      .then((data) => setApiReady(Boolean(data.siliconflow_api_ready)))
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
    fetch(artifactUrl(report.url))
      .then((res) => res.text())
      .then(setReportText)
      .catch(() => setReportText(locale === "zh" ? "报告文件读取失败。" : "Failed to read report file."));
  }, [artifacts, locale]);

  useEffect(() => {
    const area = artifacts.find((item) => item.name === "area_estimates.csv" || item.name === "event_area_estimates.csv");
    if (!area) {
      setAreaRows([]);
      return;
    }
    fetch(artifactUrl(area.url))
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
    fetch(artifactUrl(input.url))
      .then((res) => res.json())
      .then(setReportInput)
      .catch(() => setReportInput(null));
  }, [artifacts]);

  function clearJob() {
    setFile(null);
    setJob(null);
    setArtifacts([]);
    setReportText("");
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
            {heroArtifact?.kind === "image" && <img src={artifactUrl(heroArtifact.url)} alt={heroArtifact.name} />}
            {heroArtifact?.kind === "video" && <video src={artifactUrl(heroArtifact.url)} controls />}
            {heroArtifact?.kind === "report" && <pre>{reportText}</pre>}
          </div>

          <div className="artifactStrip">
            {visualImageArtifacts.slice(0, 10).map((item) => (
              <a key={item.relative_path} href={artifactUrl(item.url)} target="_blank" rel="noreferrer">
                <img src={artifactUrl(item.url)} alt={item.name} />
                <span>{item.name}</span>
              </a>
            ))}
          </div>
        </section>

        <section className="inspectorPanel">
          <section className="card stepCard">
            <h2>{text.stepsTitle}</h2>
            <div className="stepList">
              {Object.entries(text.stepLabels).map(([key, label]) => {
                const step = job?.steps?.[key] ?? { status: "pending" as StepStatus, message: "" };
                return (
                  <div className={`stepItem ${step.status}`} key={key}>
                    <span>{label}</span>
                    <strong>{statusText(step.status, locale)}</strong>
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
                      <dt>M1</dt>
                      <dd>{formatArea(item.M1?.area)}</dd>
                    </div>
                    <div>
                      <dt>M3</dt>
                      <dd>{formatArea(item.M3?.area)}</dd>
                    </div>
                    <div>
                      <dt>M4</dt>
                      <dd>{formatArea(item.M4?.area)}</dd>
                    </div>
                  </dl>
                </div>
              ))}
              {areaGroups.length === 0 && <p>{text.areaEmpty}</p>}
            </div>
          </section>

          <section className="card reportCard">
            <h2>{text.localReportTitle}</h2>
            <pre>{localReport}</pre>
          </section>

          <section className="card reportCard">
            <h2>{text.qwenReportTitle}</h2>
            {reportLooksLikePlaceholder ? (
              <pre>{job?.options.call_api ? text.noReportWithApi : text.noReportWithoutApi}</pre>
            ) : (
              <pre>{reportText}</pre>
            )}
          </section>
        </section>
      </div>

      <section className="artifactTable">
        <h2>{text.artifactsTitle}</h2>
        <div className="tableGrid">
          {artifacts.map((item) => (
            <a key={item.relative_path} href={artifactUrl(item.url)} target="_blank" rel="noreferrer">
              <span>{item.kind}</span>
              <strong>{item.relative_path}</strong>
              <em>{Math.ceil(item.size_bytes / 1024)} KB</em>
            </a>
          ))}
          {artifacts.length === 0 && <p>{text.noArtifacts}</p>}
        </div>
        {tableArtifacts.length > 0 && <p className="hint">{text.artifactHint}</p>}
        {reportInput && <p className="hint">{text.reportInputHint}</p>}
      </section>
    </main>
  );
}
