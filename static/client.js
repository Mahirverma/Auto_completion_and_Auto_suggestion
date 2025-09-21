// static/client.js (REPLACE file)
(() => {
  const editor = document.getElementById("editor");
  const ghost = document.getElementById("ghost");
  const suggestionsDiv = document.getElementById("suggestions");
  const status = document.getElementById("status");

  let ws;
  let latestQuick = [];
  let latestRanked = [];
  let latestSeq = 0;       // latest seq we have sent
  let appliedSeq = 0;      // latest seq we applied suggestions for

  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => { status.innerText = "Status: connected"; };
    ws.onclose = () => { status.innerText = "Status: disconnected (refresh to retry)"; };
    ws.onerror = (e) => { status.innerText = "Status: error"; console.error(e); };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        // ignore messages for older seqs
        const seq = msg.seq || 0;
        if (seq < latestSeq) {
          // old message, ignore
          return;
        }
        // only apply messages whose seq equals latestSeq (snapshot matching)
        // quick messages can be applied even if seq == latestSeq
        if (msg.type === "quick" && seq === latestSeq) {
          latestQuick = msg.suggestions || [];
          showSuggestions(latestQuick);
          showGhost(msg.inline || "");
          appliedSeq = seq;
        } else if (msg.type === "ranked" && seq === latestSeq) {
          latestRanked = msg.suggestions || [];
          showSuggestions(latestRanked);
          showGhost(msg.inline || "");
          appliedSeq = seq;
        }
      } catch (e) {
        console.error("Invalid message:", ev.data);
      }
    };
  }

  function sendUpdate() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const text = editor.value;
    const cursor = editor.selectionStart;
    // increment seq (monotonic)
    latestSeq += 1;
    const payload = { text, cursor, seq: latestSeq };
    try {
      ws.send(JSON.stringify(payload));
    } catch (e) {
      console.error("WS send error", e);
    }
  }

  // show ghost suggestion (only appended part)
  function showGhost(inlineFull) {
    if (!inlineFull) {
      ghost.innerText = "";
      return;
    }
    const text = editor.value;
    const cursor = editor.selectionStart;
    const left = text.slice(0, cursor);
    // compute append (only show the part after the prefix)
    let append = inlineFull;
    const lastTokenMatch = left.match(/(\S+)$/);
    if (lastTokenMatch) {
      const last = lastTokenMatch[1];
      if (inlineFull.toLowerCase().startsWith(last.toLowerCase())) {
        append = inlineFull.slice(last.length);
      }
    }
    // put ghost only the appended suggestion (so ghost won't duplicate typed text)
    ghost.innerText = left + (append || "");
    // But visually show ghost only the appended part by hiding the left portion with a zero-width trick:
    // We'll render the whole left+append, but the textarea sits on top so user sees their real text.
    // Keep it simple: ghost contains entire left+append, but overlayed behind textarea.
  }

  function showSuggestions(list) {
    suggestionsDiv.innerHTML = "";
    (list || []).forEach((s) => {
      const chip = document.createElement("div");
      chip.className = "chip";
      chip.innerText = s;
      chip.onclick = () => acceptSuggestion(s);
      suggestionsDiv.appendChild(chip);
    });
  }

  function acceptSuggestion(s) {
    // Accept suggestion relative to current caret position
    const text = editor.value;
    const cursor = editor.selectionStart;
    const left = text.slice(0, cursor);
    const right = text.slice(cursor);
    const m = left.match(/(\S+)$/);
    if (m) {
      const prefix = m[1];
      const newLeft = left.slice(0, left.length - prefix.length) + s + " ";
      editor.value = newLeft + right;
      const newCursor = newLeft.length;
      editor.setSelectionRange(newCursor, newCursor);
    } else {
      const newLeft = left + s + " ";
      editor.value = newLeft + right;
      const newCursor = newLeft.length;
      editor.setSelectionRange(newCursor, newCursor);
    }
    editor.focus();
    // After accepting, bump seq and send full update (so server can compute new suggestions for new text)
    latestSeq += 1;
    sendUpdate();
  }

  // capture Tab to accept the top suggestion
  editor.addEventListener("keydown", (ev) => {
  if (ev.key === "Tab") {
    ev.preventDefault();
    // Prefer ghost text (what user sees)
    const ghostText = ghost.innerText || "";
    const editorText = editor.value;
    if (ghostText.startsWith(editorText)) {
      const append = ghostText.slice(editorText.length);
      if (append) {
        editor.value = editorText + append + " ";
        const newCursor = editor.value.length;
        editor.setSelectionRange(newCursor, newCursor);
        sendUpdate();
        return;
      }
    }
    // fallback to suggestion list
    const top = (latestRanked && latestRanked.length) ? latestRanked[0] : (latestQuick && latestQuick.length ? latestQuick[0] : null);
    if (top) acceptSuggestion(top);
  }
});

  // debounce sending updates to server
  let timer = null;
  editor.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(sendUpdate, 80); // slightly increased to 80ms
  });

  // initial connect & first update after short delay
  connect();
  setTimeout(sendUpdate, 200);

})();