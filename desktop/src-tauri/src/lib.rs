mod pty;

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize)]
pub struct DispatchPayload {
    prompt: String,
    model: Option<String>,
    n_agents: Option<u32>,
    task_type: Option<String>,
    trace_id: Option<String>,
    proxy_base: Option<String>,
}

#[derive(Serialize)]
pub struct DispatchResult {
    queue_path: String,
    timestamp: u64,
}

fn queue_dir() -> Result<PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME env var not set".to_string())?;
    let dir = PathBuf::from(home).join(".optivia").join("queue");
    fs::create_dir_all(&dir).map_err(|e| format!("create queue dir: {e}"))?;
    Ok(dir)
}

#[tauri::command]
fn dispatch_to_claude(payload: DispatchPayload) -> Result<DispatchResult, String> {
    let dir = queue_dir()?;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    let path = dir.join(format!("{ts}.json"));
    let json = serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?;
    let mut f = fs::File::create(&path).map_err(|e| format!("create queue file: {e}"))?;
    f.write_all(json.as_bytes()).map_err(|e| format!("write queue file: {e}"))?;

    // Also write/overwrite ~/.optivia/current.json so the embedded shell shim
    // picks it up the next time the user types `claude` in the terminal.
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let current = PathBuf::from(home).join(".optivia").join("current.json");
    if let Some(parent) = current.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&current, &json);

    Ok(DispatchResult {
        queue_path: path.to_string_lossy().to_string(),
        timestamp: ts,
    })
}

const SHIM_MARKER_BEGIN: &str = "# >>> optivia dispatch shim >>>";
const SHIM_MARKER_END: &str = "# <<< optivia dispatch shim <<<";
const SHIM_BODY: &str = r#"# >>> optivia dispatch shim >>>
# Auto-installed by Optivia. Wraps `claude` to consume queued prompts from ~/.optivia/queue.
optivia_claude_shim() {
  if [ "$#" -eq 0 ]; then
    local q="$HOME/.optivia/queue"
    if [ -d "$q" ]; then
      local pending
      pending="$(ls -t "$q"/[0-9]*.json 2>/dev/null | head -n 1)"
      if [ -n "$pending" ]; then
        local payload prompt model slash_commands args
        payload="$(/usr/bin/python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d.get("prompt",""))
print("---MODEL---")
print(d.get("model","") or "")
print("---SLASH---")
cmds=d.get("slash_commands") or []
print(" ".join(cmds) if cmds else "")
' "$pending")"
        prompt="$(echo "$payload" | awk '/^---MODEL---$/{exit} {print}')"
        model="$(echo "$payload" | awk '/^---MODEL---$/,/^---SLASH---$/{if(!/^---/){print}}')"
        slash_commands="$(echo "$payload" | awk '/^---SLASH---$/{found=1;next} found{print}')"
        mv "$pending" "${pending%.json}.consumed"
        if [ -n "$prompt" ]; then
          args=()
          [ -n "$model" ] && args+=(--model "$model")
          if [ -n "$slash_commands" ]; then
            full_prompt="$slash_commands
$prompt"
          else
            full_prompt="$prompt"
          fi
          command claude "${args[@]}" "$full_prompt"
          return $?
        fi
      fi
    fi
  fi
  command claude "$@"
}
alias claude=optivia_claude_shim
# <<< optivia dispatch shim <<<
"#;

fn shell_rc_paths() -> Vec<PathBuf> {
    let mut out = vec![];
    if let Ok(home) = std::env::var("HOME") {
        let h = PathBuf::from(home);
        out.push(h.join(".zshrc"));
        out.push(h.join(".bashrc"));
    }
    out
}

#[tauri::command]
fn install_shim() -> Result<Vec<String>, String> {
    let paths = shell_rc_paths();
    let mut updated = vec![];
    for path in paths {
        let existing = fs::read_to_string(&path).unwrap_or_default();
        if existing.contains(SHIM_MARKER_BEGIN) {
            updated.push(format!("{} (already installed)", path.display()));
            continue;
        }
        let new = format!("{}\n\n{}\n", existing.trim_end(), SHIM_BODY);
        fs::write(&path, new).map_err(|e| format!("write {}: {e}", path.display()))?;
        updated.push(format!("{} (installed)", path.display()));
    }
    Ok(updated)
}

#[tauri::command]
fn uninstall_shim() -> Result<Vec<String>, String> {
    let paths = shell_rc_paths();
    let mut updated = vec![];
    for path in paths {
        let existing = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if !existing.contains(SHIM_MARKER_BEGIN) {
            updated.push(format!("{} (not installed)", path.display()));
            continue;
        }
        let mut out = String::new();
        let mut skipping = false;
        for line in existing.lines() {
            if line.contains(SHIM_MARKER_BEGIN) {
                skipping = true;
                continue;
            }
            if line.contains(SHIM_MARKER_END) {
                skipping = false;
                continue;
            }
            if !skipping {
                out.push_str(line);
                out.push('\n');
            }
        }
        fs::write(&path, out.trim_end_matches('\n').to_string() + "\n")
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        updated.push(format!("{} (removed)", path.display()));
    }
    Ok(updated)
}

#[tauri::command]
fn shim_status() -> Result<bool, String> {
    let paths = shell_rc_paths();
    for path in paths {
        let existing = fs::read_to_string(&path).unwrap_or_default();
        if existing.contains(SHIM_MARKER_BEGIN) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_http::init())
        .invoke_handler(tauri::generate_handler![
            dispatch_to_claude,
            install_shim,
            uninstall_shim,
            shim_status,
            pty::terminal_open,
            pty::terminal_write,
            pty::terminal_resize,
            pty::terminal_close
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
