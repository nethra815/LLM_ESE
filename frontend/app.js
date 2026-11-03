const messages = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const list = document.getElementById("disease-list");
const filter = document.getElementById("filter");
const diseaseCount = document.getElementById("disease-count");
const noResults = document.getElementById("no-results");

function addMessage(text, who, meta) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.textContent = text;          // textContent: model output is never parsed as HTML
  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    div.appendChild(m);
  }
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

async function ask(question) {
  addMessage(question, "user");
  sendBtn.disabled = true;
  input.disabled = true;
  const thinking = addMessage("Thinking...", "bot");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    if (!res.ok) throw new Error("Server error " + res.status);
    const data = await res.json();
    thinking.remove();
    const meta = data.disease ? "Source: " + data.disease + " (match " + data.score.toFixed(2) + ")" : null;
    addMessage(data.answer, "bot", meta);
  } catch (err) {
    thinking.remove();
    addMessage("Something went wrong: " + err.message, "bot error");
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  ask(q);
});

async function loadDiseases() {
  try {
    const res = await fetch("/api/diseases");
    const data = await res.json();
    renderList(data.diseases);
  } catch (err) {
    list.innerHTML = "<li>Could not load diseases.</li>";
  }
}

function renderList(names) {
  list.innerHTML = "";
  diseaseCount.textContent = names.length;
  noResults.hidden = names.length > 0;
  names.forEach((name) => {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.addEventListener("click", () => ask("Tell me about " + name + "."));
    li.appendChild(button);
    list.appendChild(li);
  });
}

filter.addEventListener("input", () => {
  const q = filter.value.toLowerCase();
  const visible = [...list.children].filter((li) =>
    li.textContent.toLowerCase().includes(q)
  );
  [...list.children].forEach((li) => {
    li.hidden = !visible.includes(li);
  });
  diseaseCount.textContent = visible.length;
  noResults.hidden = visible.length > 0;
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => ask(button.dataset.question));
});

loadDiseases();
