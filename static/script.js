async function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  appendMessage(message, "user");
  input.value = "";
  input.focus();

  appendMessage("⏳ Thinking...", "bot", true);  // temp message

  try {
    const response = await fetch("/predict", {  // Changed from "/predictions" to "/predict"
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: message })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    replaceLastBotMessage(data.response || data.prediction || "I didn't understand that. Can you rephrase?");
  } catch (error) {
    console.error("Error:", error);
    replaceLastBotMessage("❌ Sorry, I'm having trouble responding. Please try again later.");
  }
}

function appendMessage(text, type, isTemp = false) {
  const chatBox = document.getElementById("chat-box");
  const msg = document.createElement("div");
  msg.className = `message ${type}`;
  msg.textContent = text;
  if (isTemp) msg.dataset.temp = "true";
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function replaceLastBotMessage(newText) {
  const chatBox = document.getElementById("chat-box");
  const messages = chatBox.querySelectorAll('.message.bot[data-temp="true"]');
  if (messages.length) {
    const last = messages[messages.length - 1];
    last.textContent = newText;
    delete last.dataset.temp;
  }
}