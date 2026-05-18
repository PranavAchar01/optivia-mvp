use once_cell::sync::Lazy;
use parking_lot::Mutex;
use portable_pty::{CommandBuilder, MasterPty, NativePtySystem, PtyPair, PtySize, PtySystem};
use serde::Serialize;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;
use std::thread;
use tauri::{AppHandle, Emitter};

struct Session {
    master: Box<dyn MasterPty + Send>,
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    _child: Box<dyn portable_pty::Child + Send + Sync>,
}

static SESSIONS: Lazy<Mutex<HashMap<String, Session>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Serialize, Clone)]
pub struct TerminalOutput {
    id: String,
    data: String,
}

const INIT_SCRIPT: &str = r#"# Optivia shell init — auto-injects optimised prompts into `claude`
__optivia_consume() {
  local current="$HOME/.optivia/current.json"
  [ -f "$current" ] || return 1
  if ! command -v /usr/bin/python3 >/dev/null 2>&1; then
    return 1
  fi
  local prompt base trace
  prompt="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prompt",""), end="")' "$current" 2>/dev/null)"
  base="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("proxy_base",""), end="")' "$current" 2>/dev/null)"
  trace="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("trace_id",""), end="")' "$current" 2>/dev/null)"
  [ -n "$prompt" ] || return 1
  mv "$current" "${current%.json}.$(date +%s).consumed.json"
  if [ -n "$base" ] && [ -n "$trace" ]; then
    ANTHROPIC_BASE_URL="${base}/proxy/req/${trace}" command claude "$prompt"
  else
    command claude "$prompt"
  fi
}

optivia_claude() {
  if [ "$#" -eq 0 ]; then
    __optivia_consume && return $?
  fi
  command claude "$@"
}
alias claude=optivia_claude

# Friendly banner
if [ -f "$HOME/.optivia/current.json" ]; then
  printf '\033[2m\033[38;5;245moptivia\033[0m  prompt queued — type \033[1mclaude\033[0m to dispatch\n'
else
  printf '\033[2m\033[38;5;245moptivia\033[0m  shell ready — submit a prompt to queue a dispatch\n'
fi
"#;

fn write_init_script() -> Result<std::path::PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let dir = std::path::PathBuf::from(&home).join(".optivia").join("shell");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("init.sh");
    std::fs::write(&path, INIT_SCRIPT).map_err(|e| e.to_string())?;
    Ok(path)
}

#[tauri::command]
pub fn terminal_open(
    app: AppHandle,
    id: String,
    rows: u16,
    cols: u16,
) -> Result<(), String> {
    if SESSIONS.lock().contains_key(&id) {
        return Ok(());
    }

    let init_path = write_init_script()?;

    let pty_system = NativePtySystem::default();
    let PtyPair { master, slave } = pty_system
        .openpty(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| e.to_string())?;

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string());
    let mut cmd = CommandBuilder::new(&shell);
    if shell.ends_with("zsh") {
        cmd.args([
            "-c",
            &format!(
                "source {init} 2>/dev/null; exec {shell} -i",
                init = init_path.display(),
                shell = shell
            ),
        ]);
    } else if shell.ends_with("bash") {
        cmd.args([
            "--rcfile",
            init_path.to_str().unwrap_or(""),
            "-i",
        ]);
    } else {
        cmd.args([
            "-c",
            &format!(
                ". {init}; exec {shell} -i",
                init = init_path.display(),
                shell = shell
            ),
        ]);
    }

    if let Ok(home) = std::env::var("HOME") {
        cmd.cwd(home);
    }
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");
    cmd.env("LANG", "en_US.UTF-8");

    // Prepend common tool directories so `claude`, `npm`, `brew` binaries are
    // found even though Tauri inherits a minimal PATH without loading login profile.
    let existing_path = std::env::var("PATH").unwrap_or_default();
    let extra_dirs = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/local/sbin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ];
    let mut path_parts: Vec<&str> = extra_dirs.iter().copied().collect();
    for segment in existing_path.split(':') {
        if !segment.is_empty() && !path_parts.contains(&segment) {
            path_parts.push(segment);
        }
    }
    cmd.env("PATH", path_parts.join(":"));

    let child = slave
        .spawn_command(cmd)
        .map_err(|e| e.to_string())?;
    drop(slave);

    let reader = master.try_clone_reader().map_err(|e| e.to_string())?;
    let writer = master.take_writer().map_err(|e| e.to_string())?;
    let writer = Arc::new(Mutex::new(writer));

    // Reader thread → emits "terminal:<id>:output" events with chunks
    let app_handle = app.clone();
    let session_id = id.clone();
    thread::spawn(move || {
        let mut reader = reader;
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let chunk = String::from_utf8_lossy(&buf[..n]).to_string();
                    let _ = app_handle.emit(
                        &format!("terminal:output:{}", session_id),
                        chunk,
                    );
                }
                Err(_) => break,
            }
        }
        let _ = app_handle.emit(&format!("terminal:exit:{}", session_id), ());
    });

    SESSIONS.lock().insert(
        id,
        Session {
            master,
            writer,
            _child: child,
        },
    );

    Ok(())
}

#[tauri::command]
pub fn terminal_write(id: String, data: String) -> Result<(), String> {
    let sessions = SESSIONS.lock();
    let s = sessions.get(&id).ok_or_else(|| "no such session".to_string())?;
    let mut w = s.writer.lock();
    w.write_all(data.as_bytes()).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn terminal_resize(id: String, rows: u16, cols: u16) -> Result<(), String> {
    let sessions = SESSIONS.lock();
    let s = sessions.get(&id).ok_or_else(|| "no such session".to_string())?;
    s.master
        .resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn terminal_close(id: String) -> Result<(), String> {
    SESSIONS.lock().remove(&id);
    Ok(())
}
