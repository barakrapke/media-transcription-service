"use strict";

/*
 * Media Transcription UI. Uses only the public API:
 *   POST   /v1/transcriptions                 upload (XHR, for upload progress)
 *   GET    /v1/transcriptions/{id}/events     live progress and segments (SSE)
 *   GET    /v1/transcriptions/{id}[?format=]  job state / exports (also the SSE fallback)
 *   DELETE /v1/transcriptions/{id}            cancel
 */

const API = "/v1/transcriptions";
const POLL_INTERVAL_MS = 2000;
const STATUS_LABELS = {
  queued: "Queued",
  processing: "Transcribing",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};
const TERMINAL = new Set(["completed", "failed", "cancelled"]);

const $ = (id) => document.getElementById(id);
const languageNames = new Intl.DisplayNames(["en"], { type: "language" });

const state = {
  file: null,
  mediaUrl: null,
  media: null,
  jobId: null,
  status: null,
  events: null,
  pollTimer: null,
  segments: new Map(), // id -> { data, element }
  attempt: null,
  activeSegment: null, // currently highlighted { data, element }
  activeWord: null, // currently highlighted word <span>
};

// ---- Helpers -------------------------------------------------------------------------------------

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function languageName(code) {
  try {
    return languageNames.of(code) || code;
  } catch {
    return code;
  }
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    return body?.error?.message || `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

function show(el, visible) {
  el.hidden = !visible;
}

// ---- Service status ----------------------------------------------------------------------------

function setServiceStatus(status, text) {
  const pill = $("service-status");
  pill.dataset.state = status;
  pill.title = text;
  pill.querySelector(".label").textContent = text;
}

async function checkService() {
  try {
    const response = await fetch("/readyz", { cache: "no-store" });
    const body = await response.json();
    if (response.ok) {
      setServiceStatus("ready", `Ready · ${body.workers} worker${body.workers === 1 ? "" : "s"}`);
    } else if (body.redis) {
      setServiceStatus("warming", "Worker starting (loading model)…");
    } else {
      setServiceStatus("down", "Service unavailable");
    }
  } catch {
    setServiceStatus("down", "Service unreachable");
  }
}

// ---- Upload --------------------------------------------------------------------------------------

function selectFile(file) {
  if (!file) return;
  state.file = file;
  $("dropzone").classList.add("has-file");
  $("dropzone-title").textContent = file.name;
  $("dropzone-hint").textContent = `${formatBytes(file.size)} · click to choose a different file`;
  $("submit-btn").disabled = false;
  show($("upload-error"), false);
}

function showUploadError(message) {
  const el = $("upload-error");
  el.textContent = message;
  show(el, true);
}

function upload(event) {
  event.preventDefault();
  if (!state.file) return;

  const form = new FormData();
  form.append("file", state.file);
  const language = $("language").value;
  if (language) form.append("language", language);
  form.append("word_timestamps", $("word-timestamps").checked ? "true" : "false");

  const xhr = new XMLHttpRequest();
  xhr.open("POST", API);
  xhr.responseType = "json";
  $("submit-btn").disabled = true;
  show($("upload-error"), false);
  show($("upload-progress"), true);

  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    $("upload-bar").style.width = `${pct}%`;
    $("upload-label").textContent = pct < 100 ? `Uploading… ${pct}%` : "Validating…";
  };
  xhr.onload = () => {
    show($("upload-progress"), false);
    $("upload-bar").style.width = "0";
    $("submit-btn").disabled = false;
    if (xhr.status === 202) {
      openJob(xhr.response, { withMedia: true });
    } else {
      showUploadError(xhr.response?.error?.message || `Upload failed (${xhr.status}).`);
    }
  };
  xhr.onerror = () => {
    show($("upload-progress"), false);
    $("submit-btn").disabled = false;
    showUploadError("Upload failed: the server could not be reached.");
  };
  xhr.send(form);
}

function setupDropzone() {
  const zone = $("dropzone");
  const input = $("file-input");
  input.addEventListener("change", () => selectFile(input.files[0]));
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    }),
  );
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
    }),
  );
  zone.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));
}

// ---- Job view ------------------------------------------------------------------------------------

function openJob(job, { withMedia }) {
  resetJobView();
  state.jobId = job.id;
  history.replaceState(null, "", `#job=${job.id}`);
  show($("upload-view"), false);
  show($("job-view"), true);
  $("job-title").textContent = job.filename;
  $("job-title").title = job.filename;

  if (withMedia && state.file) attachPlayer(state.file);
  else show($("player-missing"), true);
  applyJob(job);
  if (!TERMINAL.has(job.status)) subscribe();
}

function attachPlayer(file) {
  state.mediaUrl = URL.createObjectURL(file);
  const media = document.createElement(file.type.startsWith("video/") ? "video" : "audio");
  media.controls = true;
  media.preload = "metadata";
  media.src = state.mediaUrl;
  media.addEventListener("timeupdate", () => highlight(media.currentTime));
  $("player-wrap").replaceChildren(media);
  show($("player-wrap"), true);
  show($("player-missing"), false);
  state.media = media;
  document.querySelectorAll(".segment-time").forEach((button) => {
    button.disabled = false;
    button.title = "Play from here";
  });
}

function reattachMedia(file) {
  if (!file) return;
  // Playback is local: the file is opened in the browser, nothing is uploaded again.
  if (state.mediaUrl) URL.revokeObjectURL(state.mediaUrl);
  attachPlayer(file);
}

function resetJobView() {
  closeStream();
  if (state.mediaUrl) URL.revokeObjectURL(state.mediaUrl);
  Object.assign(state, { mediaUrl: null, media: null, jobId: null, status: null, attempt: null });
  clearTranscript();
  $("player-wrap").replaceChildren();
  show($("player-wrap"), false);
  show($("player-missing"), false);
  $("reattach-input").value = "";
  show($("result-bar"), false);
  show($("job-error"), false);
  show($("transcript-empty"), false);
  show($("job-progress"), true);
}

function applyJob(job) {
  setStatus(job.status, job.attempts);
  setProgress(job.progress);
  if (job.status === "completed" && job.result) finish(job);
  if (job.status === "failed" && job.error) fail(job.error);
}

function clearTranscript() {
  state.segments.clear();
  state.activeSegment = null;
  state.activeWord = null;
  $("transcript").replaceChildren();
}

function setStatus(status, attempt) {
  if (attempt != null) {
    if (state.attempt != null && attempt !== state.attempt) {
      // The job is being re-run after its worker was lost; segments are produced again.
      clearTranscript();
    }
    state.attempt = attempt;
  }
  state.status = status;
  const badge = $("job-badge");
  badge.dataset.status = status;
  badge.textContent = STATUS_LABELS[status] || status;
  show($("cancel-btn"), !TERMINAL.has(status));
  if (status === "cancelled") {
    show($("job-progress"), false);
    closeStream();
  }
  updateProgressLabel();
}

function setProgress(progress) {
  state.progress = progress;
  $("job-bar").style.width = `${Math.round(progress * 100)}%`;
  updateProgressLabel();
}

function updateProgressLabel() {
  const label = $("job-progress-label");
  const bar = $("job-bar").parentElement;
  const pct = Math.round((state.progress || 0) * 100);
  const indeterminate = state.status === "queued" || (state.status === "processing" && pct === 0);
  bar.classList.toggle("indeterminate", indeterminate);
  if (state.status === "queued") label.textContent = "Waiting for a worker…";
  else if (state.status === "processing") {
    label.textContent = pct === 0 ? "Preparing audio…" : `${pct}% transcribed`;
  } else if (state.status === "completed") label.textContent = "Done";
  else label.textContent = "";
}

function addSegment(segment) {
  if (state.segments.has(segment.id)) return;
  const item = document.createElement("li");
  item.className = "segment";

  const time = document.createElement("button");
  time.type = "button";
  time.className = "segment-time";
  time.textContent = formatTime(segment.start);
  time.title = state.media ? "Play from here" : "Choose the file above to enable playback";
  time.disabled = !state.media;
  time.addEventListener("click", () => seek(segment.start));

  const text = document.createElement("p");
  text.className = "segment-text";
  text.dir = "auto";
  if (segment.words?.length) {
    segment.words.forEach((w) => {
      const leading = w.word.match(/^\s*/)[0];
      if (leading) text.append(leading);
      const span = document.createElement("span");
      span.className = "word";
      span.textContent = w.word.slice(leading.length);
      span.dataset.start = w.start;
      span.dataset.end = w.end;
      text.append(span);
    });
  } else {
    text.textContent = segment.text;
  }

  item.append(time, text);
  $("transcript").append(item);
  state.segments.set(segment.id, { data: segment, element: item });
}

function seek(seconds) {
  if (!state.media) return;
  state.media.currentTime = seconds;
  state.media.play().catch(() => {});
}

function highlight(currentTime) {
  // Called ~4×/s during playback: find the active segment/word, but touch the DOM only when they
  // change, so the cost doesn't grow with the length of the transcript.
  let segment = null;
  for (const entry of state.segments.values()) {
    if (currentTime >= entry.data.start && currentTime < entry.data.end) {
      segment = entry;
      break;
    }
  }
  if (segment !== state.activeSegment) {
    state.activeSegment?.element.classList.remove("active");
    segment?.element.classList.add("active");
    state.activeSegment = segment;
  }

  let word = null;
  if (segment?.data.words) {
    for (const span of segment.element.querySelectorAll(".word")) {
      if (currentTime >= +span.dataset.start && currentTime < +span.dataset.end) {
        word = span;
        break;
      }
    }
  }
  if (word !== state.activeWord) {
    state.activeWord?.classList.remove("active");
    word?.classList.add("active");
    state.activeWord = word;
  }
}

function finish(job) {
  const result = job.result;
  closeStream();
  setStatus("completed", job.attempts);
  setProgress(1);
  show($("job-progress"), false);
  result.segments.forEach(addSegment);
  show($("transcript-empty"), result.segments.length === 0);

  const speed = result.processing_time > 0 ? result.duration / result.processing_time : 0;
  const meta = [
    ["Language", `${languageName(result.language)}${job.language ? "" : " (detected)"}`],
    ["Duration", formatTime(result.duration)],
    ["Processed in", `${result.processing_time.toFixed(1)} s${speed ? ` (${speed.toFixed(1)}× real time)` : ""}`],
    ["Model", result.model],
  ];
  $("meta").replaceChildren(
    ...meta.map(([term, value]) => {
      const row = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = value;
      row.append(dt, dd);
      return row;
    }),
  );
  for (const fmt of ["txt", "srt", "vtt"]) $(`dl-${fmt}`).href = `${API}/${job.id}?format=${fmt}`;
  $("copy-btn").onclick = () => copyText(result.text);
  $("footer-model").textContent = result.model;
  show($("result-bar"), true);
}

function fail(error) {
  closeStream();
  show($("job-progress"), false);
  const el = $("job-error");
  el.textContent = error.message;
  show(el, true);
}

async function copyText(text) {
  const button = $("copy-btn");
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied ✓";
  } catch {
    button.textContent = "Copy failed";
  }
  setTimeout(() => (button.textContent = "Copy text"), 1500);
}

async function cancelJob() {
  if (!state.jobId) return;
  const response = await fetch(`${API}/${state.jobId}`, { method: "DELETE" });
  if (response.status === 202) applyJob(await response.json());
  else if (!response.ok) fail({ message: await errorMessage(response) });
}

function newTranscription() {
  resetJobView();
  history.replaceState(null, "", location.pathname);
  show($("job-view"), false);
  show($("upload-view"), true);
}

// ---- Live updates (SSE, with polling fallback) ----------------------------------------------

function subscribe() {
  const jobId = state.jobId;
  const events = new EventSource(`${API}/${jobId}/events`);
  state.events = events;
  const data = (e) => JSON.parse(e.data);

  events.addEventListener("status", (e) => setStatus(data(e).status, data(e).attempt));
  events.addEventListener("progress", (e) => setProgress(data(e).progress));
  events.addEventListener("segment", (e) => addSegment(data(e)));
  events.addEventListener("completed", (e) => finish(data(e)));
  events.addEventListener("failed", (e) => fail(data(e).error));
  events.onerror = () => {
    // The browser reconnects automatically (the server replays on connect). If the stream
    // was refused outright, fall back to polling.
    if (events.readyState === EventSource.CLOSED && state.jobId === jobId) {
      closeStream();
      poll(jobId);
    }
  };
}

function closeStream() {
  state.events?.close();
  state.events = null;
  clearTimeout(state.pollTimer);
}

async function poll(jobId) {
  if (state.jobId !== jobId) return;
  try {
    const response = await fetch(`${API}/${jobId}`, { cache: "no-store" });
    if (!response.ok) return fail({ message: await errorMessage(response) });
    const job = await response.json();
    applyJob(job);
    if (TERMINAL.has(job.status)) return;
  } catch {
    /* transient network error: keep polling */
  }
  state.pollTimer = setTimeout(() => poll(jobId), POLL_INTERVAL_MS);
}

// ---- Boot ----------------------------------------------------------------------------------------

async function restoreFromUrl() {
  const match = location.hash.match(/job=([0-9a-f]{32})/);
  if (!match) return;
  const response = await fetch(`${API}/${match[1]}`, { cache: "no-store" });
  if (response.ok) {
    openJob(await response.json(), { withMedia: false });
  } else {
    history.replaceState(null, "", location.pathname);
    showUploadError(response.status === 404 ? "That transcription no longer exists (results expire after 24 hours)." : await errorMessage(response));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupDropzone();
  $("upload-form").addEventListener("submit", upload);
  $("cancel-btn").addEventListener("click", cancelJob);
  $("new-btn").addEventListener("click", newTranscription);
  $("reattach-input").addEventListener("change", (e) => reattachMedia(e.target.files[0]));
  checkService();
  setInterval(checkService, 10000);
  restoreFromUrl();
});
