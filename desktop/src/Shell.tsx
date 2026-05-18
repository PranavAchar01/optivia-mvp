import { useEffect, useRef } from "react";
import { Terminal as XTerm } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import "@xterm/xterm/css/xterm.css";

const SESSION_ID = "main";

export function Shell() {
  const hostRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<XTerm | null>(null);
  const fitRef = useRef<FitAddon | null>(null);

  useEffect(() => {
    if (!hostRef.current || termRef.current) return;

    const term = new XTerm({
      fontFamily: '"Geist Mono", ui-monospace, "SF Mono", Menlo, Monaco, monospace',
      fontSize: 12,
      lineHeight: 1.35,
      cursorBlink: true,
      cursorStyle: "bar",
      cursorWidth: 1,
      allowProposedApi: true,
      convertEol: true,
      scrollback: 5000,
      theme: {
        background: "#000000",
        foreground: "#e5e5e5",
        cursor: "#ffffff",
        cursorAccent: "#000000",
        selectionBackground: "rgba(255,255,255,0.18)",
        black: "#000000",
        red: "#f87171",
        green: "#4ade80",
        yellow: "#fbbf24",
        blue: "#60a5fa",
        magenta: "#a78bfa",
        cyan: "#22d3ee",
        white: "#e5e5e5",
        brightBlack: "rgba(255,255,255,0.35)",
        brightRed: "#fca5a5",
        brightGreen: "#86efac",
        brightYellow: "#fde68a",
        brightBlue: "#93c5fd",
        brightMagenta: "#c4b5fd",
        brightCyan: "#67e8f9",
        brightWhite: "#ffffff",
      },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.loadAddon(new WebLinksAddon());
    term.open(hostRef.current);
    fit.fit();

    termRef.current = term;
    fitRef.current = fit;

    const cols = term.cols;
    const rows = term.rows;

    let unlistenOutput: UnlistenFn | null = null;
    let unlistenExit: UnlistenFn | null = null;
    let disposed = false;

    (async () => {
      try {
        await invoke("terminal_open", { id: SESSION_ID, rows, cols });
      } catch (e) {
        term.writeln(`\x1b[31moptivia: failed to open shell: ${e}\x1b[0m`);
        return;
      }
      if (disposed) return;

      unlistenOutput = await listen<string>(
        `terminal:output:${SESSION_ID}`,
        (e) => term.write(e.payload),
      );
      unlistenExit = await listen(`terminal:exit:${SESSION_ID}`, () => {
        term.writeln("\r\n\x1b[2m[shell exited]\x1b[0m");
      });

      term.onData((data) => {
        invoke("terminal_write", { id: SESSION_ID, data }).catch(() => {});
      });
      term.onResize(({ cols, rows }) => {
        invoke("terminal_resize", { id: SESSION_ID, rows, cols }).catch(() => {});
      });
    })();

    const onWinResize = () => {
      try {
        fit.fit();
      } catch {
        /* noop */
      }
    };
    window.addEventListener("resize", onWinResize);

    const ro = new ResizeObserver(() => onWinResize());
    ro.observe(hostRef.current);

    return () => {
      disposed = true;
      window.removeEventListener("resize", onWinResize);
      ro.disconnect();
      unlistenOutput?.();
      unlistenExit?.();
      invoke("terminal_close", { id: SESSION_ID }).catch(() => {});
      term.dispose();
      termRef.current = null;
      fitRef.current = null;
    };
  }, []);

  return (
    <div
      style={{
        background: "#000",
        border: "1px solid rgba(255,255,255,0.09)",
        borderRadius: 10,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      {/* Title bar */}
      <div
        style={{
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          padding: "0.5rem 1rem",
          display: "flex",
          alignItems: "center",
          gap: "0.4rem",
          flexShrink: 0,
        }}
      >
        {["#ff5f57", "#febc2e", "#28c840"].map((c) => (
          <div
            key={c}
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: c,
              opacity: 0.7,
            }}
          />
        ))}
        <span
          style={{
            marginLeft: "0.5rem",
            fontSize: "0.58rem",
            letterSpacing: "0.12em",
            color: "rgba(255,255,255,0.2)",
            textTransform: "uppercase",
          }}
        >
          shell · type `claude` to dispatch
        </span>
      </div>

      {/* xterm host */}
      <div
        ref={hostRef}
        style={{
          flex: "1 1 0",
          minHeight: 0,
          padding: "0.5rem 0.6rem",
          background: "#000",
        }}
      />
    </div>
  );
}
