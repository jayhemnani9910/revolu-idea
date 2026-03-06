      (() => {
        const url = new URL(window.location.href);
        url.hash = "";
        url.search = "";
        url.pathname = url.pathname.replace(/index\.html$/, "");

        const sleep = (ms, signal) =>
          new Promise((resolve, reject) => {
            const timer = window.setTimeout(resolve, ms);
            if (!signal) return;
            signal.addEventListener(
              "abort",
              () => {
                window.clearTimeout(timer);
                reject(new DOMException("Aborted", "AbortError"));
              },
              { once: true },
            );
          });

        const loadScript = (src) =>
          new Promise((resolve, reject) => {
            const el = document.createElement("script");
            el.src = src;
            el.async = true;
            el.onload = () => resolve();
            el.onerror = () => reject(new Error(`Failed to load: ${src}`));
            document.head.appendChild(el);
          });

        const siteLink = document.getElementById("site-link");
        if (siteLink) {
          siteLink.href = url.toString();
          siteLink.textContent = url.toString();
        }

        const copyBtn = document.getElementById("copy-btn");
        if (copyBtn) {
          copyBtn.addEventListener("click", async () => {
            try {
              await navigator.clipboard.writeText(url.toString());
              copyBtn.textContent = "Copied";
              window.setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
            } catch {
              copyBtn.textContent = "Copy failed";
              window.setTimeout(() => (copyBtn.textContent = "Copy"), 1400);
            }
          });
        }

        const finePointer = window.matchMedia && window.matchMedia("(pointer: fine)").matches;
        if (finePointer) {
          for (const tile of document.querySelectorAll(".tile")) {
            tile.addEventListener("mousemove", (e) => {
              const rect = tile.getBoundingClientRect();
              const x = ((e.clientX - rect.left) / rect.width) * 100;
              const y = ((e.clientY - rect.top) / rect.height) * 100;
              tile.style.setProperty("--mx", `${x}%`);
              tile.style.setProperty("--my", `${y}%`);
            });

            tile.addEventListener("mouseleave", () => {
              tile.style.removeProperty("--mx");
              tile.style.removeProperty("--my");
            });
          }
        }

        // Mermaid (render diagrams if available; otherwise keep plain text).
        (async () => {
          const mermaidEl = document.querySelector(".mermaid");
          if (!mermaidEl) return;

          try {
            await loadScript("https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js");
            if (!window.mermaid) return;

            window.mermaid.initialize({
              startOnLoad: false,
              theme: "dark",
              securityLevel: "strict",
            });
            await window.mermaid.run({ querySelector: ".mermaid" });
          } catch {
            // No-op: fallback is the plain-text Mermaid code already on the page.
          }
        })();

        // Browser demo (simulated run).
        const demoRunBtn = document.getElementById("demo-run");
        const demoStopBtn = document.getElementById("demo-stop");
        const demoQuery = document.getElementById("demo-query");
        const demoSteps = document.getElementById("demo-steps");
        const demoLog = document.getElementById("demo-log");
        const demoReport = document.getElementById("demo-report");

        if (demoRunBtn && demoStopBtn && demoQuery && demoSteps && demoLog && demoReport) {
          demoStopBtn.disabled = true;

          const steps = [
            { id: "planner", label: "Plan research", ms: 500 },
            { id: "search", label: "Web search", ms: 850 },
            { id: "analyze", label: "Analyze & extract evidence", ms: 900 },
            { id: "graph", label: "Build knowledge graph", ms: 650 },
            { id: "report", label: "Generate report", ms: 700 },
            { id: "audit", label: "Audit/verify", ms: 750 },
            { id: "save", label: "Save artifacts to output/", ms: 450 },
          ];

          const renderSteps = (stateById) => {
            demoSteps.innerHTML = "";
            for (const s of steps) {
              const row = document.createElement("div");
              row.className = "stepRow";

              const label = document.createElement("div");
              label.className = "stepLabel";
              label.textContent = s.label;

              const status = document.createElement("div");
              const st = stateById[s.id] || "pending";
              status.className = `stepStatus ${st === "pending" ? "" : st}`;
              status.textContent = st;

              row.appendChild(label);
              row.appendChild(status);
              demoSteps.appendChild(row);
            }
          };

          let controller = null;
          const appendLog = (line) => {
            demoLog.textContent += `${line}\n`;
            if (demoLog.parentElement)
              demoLog.parentElement.scrollTop = demoLog.parentElement.scrollHeight;
          };

          const stop = () => {
            if (controller) controller.abort();
            controller = null;
          };

          demoStopBtn.addEventListener("click", () => {
            stop();
            appendLog("[demo] stopped");
            demoStopBtn.disabled = true;
          });

          demoRunBtn.addEventListener("click", async () => {
            stop();
            controller = new AbortController();
            const { signal } = controller;

            demoRunBtn.disabled = true;
            demoStopBtn.disabled = false;

            const query = (demoQuery.value || "").trim() || "Demo topic";
            const state = Object.fromEntries(steps.map((s) => [s.id, "pending"]));
            renderSteps(state);

            demoLog.textContent = "";
            appendLog("============================================================");
            appendLog("CAG DEEP RESEARCH SYSTEM (DEMO)");
            appendLog("============================================================");
            appendLog(`Query: ${query}`);
            appendLog("Note: simulated in-browser demo (no API calls).");
            appendLog("------------------------------------------------------------");

            demoReport.textContent = "Running demo…";

            try {
              for (const s of steps) {
                state[s.id] = "running";
                renderSteps(state);
                appendLog(`[${s.id}] started`);
                await sleep(s.ms, signal);
                state[s.id] = "done";
                renderSteps(state);
                appendLog(`[${s.id}] completed`);
              }

              let md = "";
              try {
                const res = await fetch("demo/demo_report.md", { cache: "no-store", signal });
                md = await res.text();
              } catch {
                md = "# Demo report\n\n(Unable to load demo report file. Try refreshing.)\n";
              }

              md = md.replace(/^# .*/m, `# ${query} (demo)`);
              demoReport.textContent = md;
              appendLog("------------------------------------------------------------");
              appendLog("RESEARCH COMPLETE (DEMO)");
              appendLog("Saved to: output/reports/<demo>.md");
            } catch (e) {
              if (e && e.name === "AbortError") {
                demoReport.textContent = "Demo stopped.";
              } else {
                demoReport.textContent = "Demo failed to run. Try refreshing.";
              }
              for (const k of Object.keys(state)) state[k] = "error";
              renderSteps(state);
            } finally {
              demoRunBtn.disabled = false;
              demoStopBtn.disabled = true;
            }
          });

          // Initial UI.
          renderSteps(Object.fromEntries(steps.map((s) => [s.id, "pending"])));
        }
      })();
    
