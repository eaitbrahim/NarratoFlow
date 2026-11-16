# Installing NarratoFlow

Each release on GitHub has one download per system:

| System | File | Requirements |
| --- | --- | --- |
| macOS | `narratoflow-X.Y.Z-macos.zip` | Apple Silicon (M1 or later), macOS 14 or later |
| Linux | `narratoflow-X.Y.Z-linux.tar.gz` | x86_64 or arm64, Debian/Ubuntu (apt) or Fedora (dnf) |
| Windows | `narratoflow-X.Y.Z-windows.zip` | Windows 10 or 11, 64-bit (x86) |

The installer installs everything: NarratoFlow with every optional extra (web UI, Arabic tools,
voices, alignment, diffusion), Python 3.12 (through `uv`), ffmpeg with libass, the C++ build
tools needed by the alignment extra, and the 274 MB CAMeL dialect model. The first install
downloads several GB and can take 15 to 30 minutes. Running the installer again is safe: it skips
what is already there, and it is also how you upgrade to a newer release.

The downloads are not signed with an Apple Developer ID or a Windows code-signing certificate,
so macOS and Windows block them by default. The steps below show how to allow them once.

Licence note: the alignment extra (`ctc-forced-aligner` and its default model) is for
non-commercial use only. See the README's Alignment section.

## macOS

1. Download `narratoflow-X.Y.Z-macos.zip` and double-click it to unpack it, for example in
   `Downloads`.
2. Open **Terminal** (Applications > Utilities > Terminal).
3. Remove the download quarantine, which is what makes macOS say the file "cannot be opened
   because it is from an unidentified developer":

   ```sh
   xattr -dr com.apple.quarantine ~/Downloads/narratoflow-X.Y.Z-macos
   ```

4. Run the installer:

   ```sh
   sh ~/Downloads/narratoflow-X.Y.Z-macos/install.sh
   ```

   If Homebrew is missing, the installer installs it first. Homebrew asks for your Mac password
   and installs Apple's command line tools. The installer then adds
   `NARRATOFLOW_FFMPEG` to `~/.zprofile` and `~/.bash_profile`, because Homebrew's
   `ffmpeg-full` is not on `PATH`.
5. Open a new Terminal window and check: `narratoflow --version`.

If you double-clicked `install.sh` and saw "Apple could not verify ...", close the dialog and use
the Terminal steps above. As a fallback, you can also open System Settings > Privacy & Security
and click **Open Anyway** next to the message about the blocked file.

## Linux

1. Download `narratoflow-X.Y.Z-linux.tar.gz` and unpack it:

   ```sh
   tar -xzf narratoflow-X.Y.Z-linux.tar.gz
   ```

2. Run the installer. It uses `sudo` to install `ffmpeg`, a C++ compiler and `curl`:

   ```sh
   sh narratoflow-X.Y.Z-linux/install.sh
   ```

   On Fedora, enable RPM Fusion first so that `dnf` can install ffmpeg. Other distributions:
   install ffmpeg (with libass) and a C++ compiler yourself, then run the installer.
3. Open a new terminal and check: `narratoflow --version`.

Linux does not block unsigned downloads. If you see "Permission denied", you ran `./install.sh`
on a file without the execute bit; `sh install.sh` as shown above avoids that.

## Windows

1. Download `narratoflow-X.Y.Z-windows.zip`.
2. Unblock it **before** unpacking, so that Windows does not mark every unpacked file as coming
   from the internet: right-click the zip > **Properties** > tick **Unblock** at the bottom of the
   General tab > **OK**. (If there is no Unblock box, the file is already allowed.)
3. Right-click the zip > **Extract All...**.
4. Open **PowerShell** (Start menu, type "PowerShell"). Do not use "Run as administrator"; the
   installer asks for administrator rights itself when it needs them.
5. Go to the unpacked folder and run the installer. The `-ExecutionPolicy Bypass` part allows
   this one unsigned script to run without changing your system's policy:

   ```powershell
   cd $HOME\Downloads\narratoflow-X.Y.Z-windows\narratoflow-X.Y.Z-windows
   powershell -ExecutionPolicy Bypass -File .\install.ps1
   ```

   If you skipped step 2, run `Get-ChildItem -Recurse | Unblock-File` in that folder first.
   The installer uses `winget` to install the Visual Studio C++ Build Tools (approve the
   administrator prompt; this step alone can take 10 minutes) and ffmpeg (the gyan.dev build).
6. Open a new PowerShell window and check: `narratoflow --version`.

If Windows SmartScreen shows "Windows protected your PC" while a tool is being installed, click
**More info** > **Run anyway**.

## After installing

- Run `narratoflow --help` to see the commands; `narratoflow ui SERIES_JSON --channel CHANNEL.toml`
  opens the review station in your browser.
- API keys and settings go in the environment or in a `.env` file in the folder where you run
  `narratoflow` (see "Where settings live" in the README).
- Uninstall with `uv tool uninstall narratoflow`.
