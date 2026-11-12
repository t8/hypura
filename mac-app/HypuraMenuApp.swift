import Cocoa

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var process: Process?
    let port: Int = 6000
    let modelsDir: String = "/Volumes/RealtekData/models"
    let defaultContext: Int = 32768

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory) // Menu bar only (no Dock icon), like Ollama

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            if #available(macOS 11.0, *) {
                button.image = NSImage(systemSymbolName: "bolt.circle.fill", accessibilityDescription: "Hypura")
            } else {
                button.title = "⚡H"
            }
        }

        ensureCliInstalled()
        startServer()
        buildMenu()
    }

    func ensureCliInstalled() {
        let bundlePath = Bundle.main.bundlePath
        let binaryPath = "\(bundlePath)/Contents/MacOS/hypura-bin"
        guard FileManager.default.fileExists(atPath: binaryPath) else { return }

        // Symlink to ~/.local/bin/hypura
        let home = NSHomeDirectory()
        let localBin = "\(home)/.local/bin"
        let symlink = "\(localBin)/hypura"

        try? FileManager.default.createDirectory(atPath: localBin, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(atPath: symlink)
        try? FileManager.default.createSymbolicLink(atPath: symlink, withDestinationPath: binaryPath)
    }

    func startServer() {
        let bundlePath = Bundle.main.bundlePath
        var binaryPath = "\(bundlePath)/Contents/MacOS/hypura-bin"

        if !FileManager.default.fileExists(atPath: binaryPath) {
            // Fallback to relative project target
            binaryPath = "/Volumes/RealtekData/projects/hypura/target/release/hypura"
        }

        guard FileManager.default.fileExists(atPath: binaryPath) else {
            NSLog("Hypura binary not found at \(binaryPath)")
            return
        }

        let task = Process()
        task.executableURL = URL(fileURLWithPath: binaryPath)
        task.arguments = [
            "serve",
            "--models-dir", modelsDir,
            "--port", "\(port)",
            "--context", "\(defaultContext)",
            "--idle-timeout", "1800",
            "--kv-quant", "q8_0"
        ]

        try? task.run()
        self.process = task
    }

    func stopServer() {
        if let p = process, p.isRunning {
            p.terminate()
        }
        process = nil
    }

    func restartServer() {
        stopServer()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.startServer()
            self?.buildMenu()
        }
    }

    func buildMenu() {
        let menu = NSMenu()

        let titleItem = NSMenuItem(title: "Hypura Server", action: nil, keyEquivalent: "")
        titleItem.isEnabled = false
        menu.addItem(titleItem)

        let isRunning = process?.isRunning ?? false
        let statusText = isRunning ? "🟢 Running on port \(port)" : "🔴 Stopped"
        let statusItem = NSMenuItem(title: statusText, action: nil, keyEquivalent: "")
        statusItem.isEnabled = false
        menu.addItem(statusItem)

        menu.addItem(NSMenuItem.separator())

        let copyUrlItem = NSMenuItem(title: "Copy Local URL (http://localhost:\(port))", action: #selector(copyUrl), keyEquivalent: "c")
        copyUrlItem.target = self
        menu.addItem(copyUrlItem)

        let modelsDirItem = NSMenuItem(title: "Open Models Directory...", action: #selector(openModelsDir), keyEquivalent: "m")
        modelsDirItem.target = self
        menu.addItem(modelsDirItem)

        let installCliItem = NSMenuItem(title: "Install CLI (hypura in PATH)", action: #selector(installCliAction), keyEquivalent: "i")
        installCliItem.target = self
        menu.addItem(installCliItem)

        menu.addItem(NSMenuItem.separator())

        let restartItem = NSMenuItem(title: "Restart Server", action: #selector(restartAction), keyEquivalent: "r")
        restartItem.target = self
        menu.addItem(restartItem)

        let quitItem = NSMenuItem(title: "Quit Hypura", action: #selector(quitAction), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        self.statusItem.menu = menu
    }

    @objc func copyUrl() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("http://localhost:\(port)", forType: .string)
    }

    @objc func openModelsDir() {
        let url = URL(fileURLWithPath: modelsDir)
        NSWorkspace.shared.open(url)
    }

    @objc func installCliAction() {
        ensureCliInstalled()
        let alert = NSAlert()
        alert.messageText = "CLI Installed"
        alert.informativeText = "hypura is installed at ~/.local/bin/hypura. You can now run 'hypura list', 'hypura pull', and 'hypura serve' from any terminal."
        alert.runModal()
    }

    @objc func restartAction() {
        restartServer()
    }

    @objc func quitAction() {
        stopServer()
        NSApp.terminate(nil)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
