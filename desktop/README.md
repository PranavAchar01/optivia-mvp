# Optivia Desktop

A minimal black terminal for Optivia. Type a prompt, press Enter, see the LangGraph engine output: classification, scoring, clarification, synthesis, routing, sub-agent decomposition.

Built with Tauri 2 + React + Vite. Bundles to native `.app`/`.dmg` on macOS and `.AppImage`/`.deb` on Linux.

## Configure

On first run, click `SETTINGS` (top right) and paste the URL of your Optivia backend. The app stores it in `localStorage` and POSTs prompts to `{base}/optimize`.

The desktop app holds no API keys — your backend handles Anthropic auth.

## Develop

```bash
npm install
npm run tauri dev
```

Opens a native window pointed at the Vite dev server with hot reload. Rust changes recompile automatically.

## Build locally

```bash
npm run tauri build
```

macOS output: `src-tauri/target/release/bundle/dmg/Optivia_<version>_<arch>.dmg`
Linux output: `src-tauri/target/release/bundle/{appimage,deb}/`

Linux builds must run on a Linux host — the toolchain doesn't cross-compile cleanly from macOS.

## Release via GitHub Actions

The workflow at `.github/workflows/release-desktop.yml` builds macOS arm64, macOS x64, and Linux x64 in parallel and publishes a GitHub Release.

```bash
git tag desktop-v0.1.0
git push origin desktop-v0.1.0
```

The release will appear at `github.com/<owner>/<repo>/releases` with downloadable installers for each platform.

## Stack

- Tauri 2 (Rust shell, ~5 MB bundle, system webview)
- React 19 + Vite 7
- Geist Mono via `@fontsource/geist-mono`
- No CSS framework — direct styles match the Optivia website terminal 1:1
