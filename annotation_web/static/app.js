const state = {
  username: null,
  activeTask: null,
  selectedImageIndex: null,
  selectedRating: null,
};

const el = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    credentials: "same-origin",
    ...options,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function setMessage(text, kind = "") {
  const message = el("taskMessage");
  message.textContent = text || "";
  message.className = "message" + (kind ? ` message--${kind}` : "");
}

function showDashboard(show) {
  el("loginCard").hidden = show;
  el("dashboardCard").hidden = !show;
}

function showTask(show) {
  el("taskEmpty").hidden = show;
  el("taskCard").hidden = !show;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function shuffleWithTracking(images) {
  // Create array of {url, originalIndex} to track original positions
  const tracked = images.map((url, idx) => ({ url, originalIndex: idx + 1 }));
  // Fisher-Yates shuffle
  for (let i = tracked.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [tracked[i], tracked[j]] = [tracked[j], tracked[i]];
  }
  return tracked;
}

function renderWords(words) {
  if (!Array.isArray(words) || !words.length) return "";
  return words.map((word) => `<span>${escapeHtml(word)}</span>`).join("");
}

function renderTopicContext(task) {
  const representation = task.topic_representation || task.topic_name || `Topic ${task.topic}`;
  const docs = Array.isArray(task.representative_docs) ? task.representative_docs.slice(0, 3) : [];
  const keywords = renderWords(task.words || []);

  let html = `
    <section class="topic-panel">
      <div class="topic-panel__header">
        <div class="topic-panel__eyebrow">Topic Representation</div>
        <h3 class="topic-panel__title">${escapeHtml(representation)}</h3>
      </div>
  `;

  if (keywords) {
    html += `
      <div class="topic-panel__section">
        <div class="topic-panel__label">Helper Keywords</div>
        <div class="word-chips word-chips--compact">${keywords}</div>
      </div>
    `;
  }

  if (docs.length) {
    html += '<div class="topic-panel__section"><div class="topic-panel__label">Representative Excerpts</div><div class="topic-doc-grid">';
    html += docs.map((doc, index) => `
      <article class="topic-doc-card">
        <div class="topic-doc-card__index">${index + 1}</div>
        <p>${escapeHtml(doc)}</p>
      </article>
    `).join("");
    html += "</div></div>";
  }

  html += "</section>";
  return html;
}

function renderTask(task) {
  const displayTopic = task.topic_representation || task.topic_name || `Topic ${task.topic}`;
  const helperKeywords = Array.isArray(task.words) ? task.words.slice(0, 8) : [];
  const helperDocs = Array.isArray(task.representative_docs) ? task.representative_docs.slice(0, 2) : [];

  state.activeTask = task;
  state.selectedImageIndex = null;
  state.selectedRating = null;
  el("responseFields").innerHTML = "";
  el("commentInput").value = "";
  el("responseBox").hidden = false;
  el("nextBtn").hidden = true;
  el("submitBtn").disabled = false;
  setMessage("");

  showTask(true);

  el("taskTypeBadge").textContent = task.task_type === "image_intrusion" ? "Image intrusion" : "Topic matching";
  el("taskTitle").textContent = task.task_type === "image_intrusion"
    ? "Find the intruder"
    : displayTopic;
  el("taskVideo").textContent = task.video || "Unknown video";

  el("taskTopic").innerHTML = `
  <div class="topic-banner__eyebrow">Main topic</div>
  <div class="topic-banner__title"><bold>${escapeHtml(displayTopic)}</bold></div>
  <div class="topic-banner__meta">Topic: ${task.topic} • Point ${escapeHtml(task.point_id)} • Video ${escapeHtml(task.video || "Unknown video")}</div>
`;
  el("taskInstructions").textContent = task.instructions || "Review the annotation and submit a response.";

  if (task.task_type === "topic_matching") {
    const keywordsText = helperKeywords.length ? helperKeywords.map((word) => escapeHtml(word)).join(" · ") : "No keywords available";
    const docsText = helperDocs.length ? helperDocs.map((doc) => escapeHtml(doc)).join(" ") : "No representative excerpts available";
    el("taskUsage").innerHTML = `
      <div class="usage-summary__title">How this task uses the topic</div>
      <div class="usage-summary__grid">
        <div class="usage-summary__item"><span>Title</span><strong>${escapeHtml(displayTopic)}</strong></div>
        <div class="usage-summary__item"><span>Helper keywords</span><strong>${keywordsText}</strong></div>
        <div class="usage-summary__item"><span>Representative docs</span><strong>${docsText}</strong></div>
      </div>
    `;
  } else {
    el("taskUsage").innerHTML = "";
  }

  if (task.task_type === "topic_matching") {
    el("taskWords").innerHTML = renderTopicContext(task);
  } else {
    el("taskWords").innerHTML = renderWords(task.words || []);
  }

  const body = el("taskBody");
  body.innerHTML = "";

  if (task.task_type === "image_intrusion") {
    const grid = document.createElement("div");
    // shuffle the images so the intruder isn't always in the same position
    const originalImages = task.images || [];
    console.log("Original images order:", originalImages);
    const shuffledImages = shuffleWithTracking(originalImages);
    console.log("Shuffled images order:", shuffledImages.map(i => i.originalIndex));
    
    grid.className = "image-grid image-grid--intrusion";
    shuffledImages.forEach((item, index) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "image-option";
      card.dataset.shuffledIndex = index;
      card.dataset.originalIndex = item.originalIndex;
      card.innerHTML = `
        <img src="${item.url}" alt="candidate ${index + 1}" />
        <div class="image-option__label">${index + 1}</div>
        <div class="image-option__caption">Candidate ${index + 1}</div>
        <div class="image-option__origin" title="Original position (randomized)">• orig ${item.originalIndex}</div>
      `;
      card.addEventListener("click", () => {
        state.selectedImageIndex = index;
        body.querySelectorAll(".image-option").forEach((node) => node.classList.remove("selected"));
        card.classList.add("selected");
      });
      grid.appendChild(card);
    });
    body.appendChild(grid);
    el("responseFields").innerHTML = `<div class="pill pill--muted">Images are randomized • Select the intruder from the grid above</div>`;
  } else {
    const grid = document.createElement("div");
    grid.className = "image-grid image-grid--matching";
    const shuffledImages = shuffleWithTracking(task.images || []);
    
    shuffledImages.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "image-option";
      card.dataset.shuffledIndex = index;
      card.dataset.originalIndex = item.originalIndex;
      card.innerHTML = `
        <img src="${item.url}" alt="topic image ${index + 1}" />
        <div class="image-option__label">${index + 1}</div>
        <div class="image-option__caption">Topic image ${index + 1}</div>
        <div class="image-option__origin" title="Original position (randomized)">• orig ${item.originalIndex}</div>
      `;
      grid.appendChild(card);
    });
    body.appendChild(grid);

    const rating = document.createElement("div");
    rating.className = "response-box";
    rating.innerHTML = `
      <h3>Match score</h3>
      <div class="rating-group" id="ratingGroup"></div>
      <p class="hint">1 = poor match, 5 = strong match.</p>
    `;
    body.appendChild(rating);
    const ratingGroup = rating.querySelector("#ratingGroup");
    for (let score = 1; score <= 5; score += 1) {
      const label = document.createElement("label");
      label.innerHTML = `<input type="radio" name="matchScore" value="${score}" /><span>${score}</span>`;
      label.addEventListener("click", () => {
        state.selectedRating = score;
        ratingGroup.querySelectorAll("label").forEach((node) => node.classList.remove("selected"));
        label.classList.add("selected");
      });
      ratingGroup.appendChild(label);
    }
    el("responseFields").innerHTML = `<div class="pill pill--muted">Images are randomized • Rate the match using context below</div>`;
  }
}

async function refreshDashboard() {
  const dashboard = await api("/api/dashboard");
  el("userBadge").textContent = `Signed in as ${dashboard.username}`;
  el("logoutBtn").hidden = false;
  showDashboard(true);
  el("statOpen").textContent = dashboard.counts.open;
  el("statClaimed").textContent = dashboard.counts.claimed;
  el("statDone").textContent = dashboard.counts.completed;
  el("statActive").textContent = dashboard.counts.active;

  if (dashboard.active_task) {
    renderTask(dashboard.active_task);
    setMessage("Resumed your active task.");
  } else {
    showTask(false);
  }
}

async function refreshDashboardCountsOnly() {
  try {
    const dashboard = await api("/api/dashboard");
    el("statOpen").textContent = dashboard.counts.open;
    el("statClaimed").textContent = dashboard.counts.claimed;
    el("statDone").textContent = dashboard.counts.completed;
    el("statActive").textContent = dashboard.counts.active;
  } catch (error) {
    console.warn(error);
  }
}

async function login(username) {
  await api("/api/session", { method: "POST", body: JSON.stringify({ username }) });
  state.username = username;
  await refreshDashboard();
}

async function claimTask(taskType) {
  const result = await api("/api/tasks/claim", { method: "POST", body: JSON.stringify({ task_type: taskType }) });
  if (!result.task) {
    setMessage(result.message || "No task available.", "error");
    return;
  }
  renderTask(result.task);
  setMessage("Task claimed. Complete it before taking a new one.");
  await refreshDashboardCountsOnly();
}

async function submitTask() {
  if (!state.activeTask) return;
  const taskType = state.activeTask.task_type;
  const response = { comment: el("commentInput").value.trim() };
  if (taskType === "image_intrusion") {
    if (state.selectedImageIndex === null) {
      setMessage("Choose the intruder image before submitting.", "error");
      return;
    }
    response.selected_image_index = state.selectedImageIndex;
  } else {
    if (state.selectedRating === null) {
      setMessage("Choose a match score before submitting.", "error");
      return;
    }
    response.match_score = state.selectedRating;
  }
  const result = await api(`/api/tasks/${state.activeTask.task_db_id}/submit`, {
    method: "POST",
    body: JSON.stringify({ response }),
  });
  setMessage(`Saved response for ${result.point_id}.`, "success");
  el("submitBtn").disabled = true;
  el("nextBtn").hidden = false;
  el("nextBtn").textContent = `Claim next ${taskType.replace("_", " ")}`;
  el("nextBtn").dataset.taskType = taskType;
  state.activeTask = null;
  await refreshDashboardCountsOnly();
}

async function releaseTask() {
  if (!state.activeTask) return;
  await api(`/api/tasks/${state.activeTask.task_db_id}/release`, { method: "POST" });
  state.activeTask = null;
  setMessage("Task released.");
  await refreshDashboardCountsOnly();
  showTask(false);
}

function initBindings() {
  el("loginBtn").addEventListener("click", async () => {
    const username = el("usernameInput").value.trim();
    if (!username) {
      setMessage("Enter a username first.", "error");
      return;
    }
    try {
      await login(username);
    } catch (error) {
      setMessage(error.message, "error");
    }
  });
  el("usernameInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") el("loginBtn").click();
  });
  document.querySelectorAll("[data-claim]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await claimTask(button.dataset.claim);
      } catch (error) {
        setMessage(error.message, "error");
      }
    });
  });
  el("submitBtn").addEventListener("click", async () => {
    try {
      await submitTask();
    } catch (error) {
      setMessage(error.message, "error");
    }
  });
  el("releaseBtn").addEventListener("click", async () => {
    try {
      await releaseTask();
    } catch (error) {
      setMessage(error.message, "error");
    }
  });
  el("nextBtn").addEventListener("click", async () => {
    try {
      await claimTask(el("nextBtn").dataset.taskType || "any");
    } catch (error) {
      setMessage(error.message, "error");
    }
  });
  el("logoutBtn").addEventListener("click", async () => {
    await api("/api/logout", { method: "POST" });
    window.location.reload();
  });
}

async function boot() {
  initBindings();
  const session = await api("/api/session");
  if (session.username) {
    state.username = session.username;
    await refreshDashboard();
  } else {
    showDashboard(false);
    showTask(false);
    el("logoutBtn").hidden = true;
  }
}

boot().catch((error) => {
  console.error(error);
  setMessage(error.message, "error");
});