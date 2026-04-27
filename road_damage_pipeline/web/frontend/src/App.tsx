import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type StepStatus = "pending" | "running" | "done" | "failed" | "skipped";
type JobStatus = "queued" | "running" | "completed" | "failed";

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
const STEP_LABELS: Record<string, string> = {
  upload: "上传",
  segmentation: "语义分割探索",
  detection: "病害检测",
  dedup: "视频去重",
  area: "面积计算",
  report: "报告生成"
};

function statusText(status: StepStatus | JobStatus): string {
  return {
    pending: "等待",
    queued: "排队",
    running: "运行中",
    done: "完成",
    completed: "完成",
    failed: "失败",
    skipped: "跳过"
  }[status] ?? status;
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
    if (!job) return "暂无报告。";
    const counts = (job.summary?.counts_by_class ?? job.summary?.unique_counts_by_class ?? {}) as Record<string, number>;
    const countLines = Object.entries(counts)
      .map(([key, value]) => `- ${key}: ${value}`)
      .join("\n");
    const topAreas = areaGroups
      .slice(0, 5)
      .map(
        (item) =>
          `- ${item.itemId} ${item.classCode} conf=${item.confidence.toFixed(2)} | M1 ${formatArea(item.M1?.area)} | M3 ${formatArea(item.M3?.area)} | M4 ${formatArea(item.M4?.area)}`
      )
      .join("\n");
    return [
      "## 结构化摘要",
      `- 文件: ${job.file_name}`,
      `- 状态: ${statusText(job.status)}`,
      `- 检测数量: ${String(job.summary?.total_detections ?? 0)}`,
      job.summary?.unique_events !== undefined ? `- 去重后事件数: ${String(job.summary.unique_events)}` : "",
      "",
      "### 类别统计",
      countLines || "- 暂无类别统计",
      "",
      "### 面积估计解释",
      "- M1: bbox 经验规则，基于检测框几何尺寸进行固定比例估计。",
      "- M3: Depth Anything V2 深度图 + bbox 区域 + bbox 经验规则修正。",
      "- M4: Metric3D 深度图 + bbox 区域 + bbox 经验规则修正。",
      "- 三种方法都属于 estimated area，因为当前没有相机内参、标定尺或真实面积 GT。",
      "",
      "### 重点面积结果",
      topAreas || "- 暂无面积结果",
    ]
      .filter(Boolean)
      .join("\n");
  }, [areaGroups, job, reportLooksLikePlaceholder, reportText]);

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
      .catch(() => setReportText("报告文件读取失败。"));
  }, [artifacts]);

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

  useEffect(() => {
    const area = artifacts.find((item) => item.name === "area_estimates.csv" || item.name === "event_area_estimates.csv");
    if (!area) {
      setAreaRows([]);
      return;
    }
    fetch(artifactUrl(area.url))
      .then((res) => res.text())
      .then((text) => setAreaRows(normalizeAreaRows(parseCsv(text))))
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

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("请先上传一张图片或一个视频。");
      return;
    }
    setSubmitting(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("run_segmentation", String(runSegmentation));
    formData.append("call_api", String(callApi));
    formData.append("conf", String(conf));
    formData.append("iou", String(iou));
    formData.append("imgsz", "832");
    formData.append("device", "auto");
    formData.append("tracker_backend", "bytetrack");
    try {
      const response = await fetch(`${API_BASE}/api/jobs`, { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "上传失败");
      setJob(payload);
      const artifactResponse = await fetch(`${API_BASE}/api/jobs/${payload.job_id}/artifacts`).then((res) => res.json());
      setArtifacts(artifactResponse.artifacts ?? []);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "任务创建失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="appShell">
      <section className="masthead">
        <div>
          <p className="eyebrow">RDD2022 Road Inspection Workbench</p>
          <h1>道路病害智能分析工作台</h1>
          <p className="lead">单文件上传，串行执行检测、去重、面积估计和 Qwen 报告生成。所有中间产物都会保存并可视化。</p>
        </div>
        <div className="statusPill">{job ? `Job ${job.job_id} · ${statusText(job.status)}` : "等待上传"}</div>
      </section>

      <div className="workspaceGrid">
        <aside className="controlPanel">
          <form onSubmit={onSubmit}>
            <label className="uploadBox">
              <span>上传图片或视频</span>
              <strong>{file ? file.name : "选择单个 .jpg / .png / .mp4 / .mov"}</strong>
              <input type="file" accept=".jpg,.jpeg,.png,.mp4,.mov" onChange={onFileChange} />
            </label>

            {previewUrl && file && (
              <div className="localPreview">
                {file.type.startsWith("video") ? <video src={previewUrl} controls /> : <img src={previewUrl} alt="uploaded preview" />}
              </div>
            )}

            <label className="switchRow">
              <input type="checkbox" checked={runSegmentation} onChange={(event) => setRunSegmentation(event.target.checked)} />
              <span>附加语义分割探索可视化</span>
            </label>
            <label className="switchRow">
              <input type="checkbox" checked={callApi} onChange={(event) => setCallApi(event.target.checked)} />
              <span>调用 SiliconFlow Qwen 生成真实报告</span>
            </label>
            {callApi && apiReady === false && (
              <p className="warningText">后端当前没有检测到 SILICONFLOW_API_KEY；任务会生成 request preview，但不会真实调用 Qwen。</p>
            )}
            {callApi && apiReady === true && <p className="okText">后端已检测到 SILICONFLOW_API_KEY，会尝试真实生成 Qwen 报告。</p>}

            <div className="sliderBlock">
              <span>conf 阈值 {conf.toFixed(2)}</span>
              <input type="range" min="0.1" max="0.8" step="0.05" value={conf} onChange={(event) => setConf(Number(event.target.value))} />
            </div>
            <div className="sliderBlock">
              <span>IoU 阈值 {iou.toFixed(2)}</span>
              <input type="range" min="0.3" max="0.8" step="0.05" value={iou} onChange={(event) => setIou(Number(event.target.value))} />
            </div>

            <button className="runButton" disabled={submitting || !file}>
              {submitting ? "提交中..." : "开始分析"}
            </button>
            {error && <p className="errorText">{error}</p>}
          </form>

          <div className="rulesCard">
            <h2>上传规则</h2>
            <p>一次只处理一个文件。图片进入检测、面积和报告；视频额外进入跨帧去重和代表帧面积计算。</p>
          </div>
        </aside>

        <section className="visualStage">
          <div className="stageHeader">
            <h2>主可视化</h2>
            <span>{heroArtifact ? heroArtifact.relative_path : "等待 pipeline 输出"}</span>
          </div>
          <div className="heroCanvas">
            {!heroArtifact && <p>上传文件后，这里会展示检测框、面积 board、去重时间线或最终报告。</p>}
            {heroArtifact?.kind === "image" && <img src={artifactUrl(heroArtifact.url)} alt={heroArtifact.name} />}
            {heroArtifact?.kind === "video" && <video src={artifactUrl(heroArtifact.url)} controls />}
            {heroArtifact?.kind === "report" && <pre>{reportText}</pre>}
          </div>

          <div className="artifactStrip">
            {visualImageArtifacts.slice(0, 10).map((item) => (
              <a key={item.relative_path} href={artifactUrl(item.url)} target="_blank">
                <img src={artifactUrl(item.url)} alt={item.name} />
                <span>{item.name}</span>
              </a>
            ))}
          </div>
        </section>

        <aside className="inspectorPanel">
          <section className="card">
            <h2>步骤状态</h2>
            <div className="stepList">
              {Object.entries(STEP_LABELS).map(([key, label]) => {
                const step = job?.steps?.[key] ?? { status: "pending", message: "" };
                return (
                  <div className={`stepItem ${step.status}`} key={key}>
                    <span>{label}</span>
                    <strong>{statusText(step.status)}</strong>
                    {step.message && <small>{step.message}</small>}
                  </div>
                );
              })}
            </div>
          </section>

          <section className="card">
            <h2>统计摘要</h2>
            {job ? (
              <div className="metricGrid">
                <div>
                  <span>检测数</span>
                  <strong>{String(job.summary?.total_detections ?? 0)}</strong>
                </div>
                <div>
                  <span>去重事件</span>
                  <strong>{String(job.summary?.unique_events ?? "-")}</strong>
                </div>
                <div>
                  <span>面积条目</span>
                  <strong>{areaGroups.length}</strong>
                </div>
                <div>
                  <span>产物数</span>
                  <strong>{artifacts.length}</strong>
                </div>
              </div>
            ) : (
              <p>暂无统计</p>
            )}
            {Boolean(job?.summary?.api_warning) && <p className="warningText">{String(job?.summary?.api_warning)}</p>}
          </section>

          <section className="card areaCard">
            <h2>面积计算</h2>
            <p>M1 是 bbox 经验规则；M3 是 Depth Anything V2；M4 是 Metric3D。当前都是估计面积，不是真实标定面积。</p>
            <div className="areaList">
              {areaGroups.slice(0, 8).map((item) => (
                <div className="areaItem" key={item.itemId}>
                  <div>
                    <strong>{item.itemId}</strong>
                    <span>
                      {item.classCode} · conf {item.confidence.toFixed(2)}
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
              {areaGroups.length === 0 && <p>面积 CSV 生成后会显示在这里。</p>}
            </div>
          </section>

          <section className="card reportCard">
            <h2>结构化摘要</h2>
            <pre>{localReport}</pre>
          </section>

          <section className="card reportCard">
            <h2>Qwen 报告</h2>
            {reportLooksLikePlaceholder ? (
              <pre>
                {job?.options.call_api
                  ? "真实 Qwen 报告未生成。请确认启动后端前已经 export SILICONFLOW_API_KEY，然后重新提交任务。"
                  : "未勾选真实 API 调用。本次只生成 qwen_request_preview.json，不生成 Qwen 报告。"}
              </pre>
            ) : (
              <pre>{reportText}</pre>
            )}
          </section>
        </aside>
      </div>

      <section className="artifactTable">
        <h2>全部中间产物</h2>
        <div className="tableGrid">
          {artifacts.map((item) => (
            <a key={item.relative_path} href={artifactUrl(item.url)} target="_blank">
              <span>{item.kind}</span>
              <strong>{item.relative_path}</strong>
              <em>{Math.ceil(item.size_bytes / 1024)} KB</em>
            </a>
          ))}
          {artifacts.length === 0 && <p>暂无产物。</p>}
        </div>
        {tableArtifacts.length > 0 && <p className="hint">CSV/JSON 是论文和前端二次可视化的可信数据来源，报告只作为自然语言解释层。</p>}
        {reportInput && <p className="hint">已读取 report_input.json；右侧摘要和本地报告均来自同一份结构化证据。</p>}
      </section>
    </main>
  );
}
