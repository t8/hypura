import Cocoa

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var process: Process?
    var settingsWindow: NSWindow?

    // Configuration keys for UserDefaults
    let keyModelsDir = "hypura_models_dir"
    let keyPort = "hypura_port"
    let keyContext = "hypura_context"
    let keyIdleTimeout = "hypura_idle_timeout"
    let keyKvQuant = "hypura_kv_quant"

    var modelsDir: String {
        get {
            if let saved = UserDefaults.standard.string(forKey: keyModelsDir), !saved.isEmpty {
                return saved
            }
            if FileManager.default.fileExists(atPath: "/Volumes/RealtekData/models") {
                return "/Volumes/RealtekData/models"
            }
            if FileManager.default.fileExists(atPath: "/Volumes/RealtekData/OllamaModels") {
                return "/Volumes/RealtekData/OllamaModels"
            }
            return "\(NSHomeDirectory())/.ollama/models"
        }
        set {
            UserDefaults.standard.set(newValue, forKey: keyModelsDir)
        }
    }

    var port: Int {
        get {
            let p = UserDefaults.standard.integer(forKey: keyPort)
            return p > 0 ? p : 6000
        }
        set {
            UserDefaults.standard.set(newValue, forKey: keyPort)
        }
    }

    var defaultContext: Int {
        get {
            let c = UserDefaults.standard.integer(forKey: keyContext)
            return c > 0 ? c : 32768
        }
        set {
            UserDefaults.standard.set(newValue, forKey: keyContext)
        }
    }

    var idleTimeout: Int {
        get {
            let t = UserDefaults.standard.integer(forKey: keyIdleTimeout)
            return t > 0 ? t : 1800
        }
        set {
            UserDefaults.standard.set(newValue, forKey: keyIdleTimeout)
        }
    }

    var kvQuant: String {
        get {
            let q = UserDefaults.standard.string(forKey: keyKvQuant) ?? ""
            return !q.isEmpty ? q : "q8_0"
        }
        set {
            UserDefaults.standard.set(newValue, forKey: keyKvQuant)
        }
    }

    // UI elements in settings window
    var modelsDirField: NSTextField?
    var portField: NSTextField?
    var contextPopup: NSPopUpButton?
    var idleTimeoutField: NSTextField?
    var kvQuantPopup: NSPopUpButton?
    var statusLabel: NSTextField?
    var toggleServerButton: NSButton?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory) // Menu bar agent app (no Dock icon)

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

        let home = NSHomeDirectory()
        let localBin = "\(home)/.local/bin"
        let symlink = "\(localBin)/hypura"

        try? FileManager.default.createDirectory(atPath: localBin, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(atPath: symlink)
        try? FileManager.default.createSymbolicLink(atPath: symlink, withDestinationPath: binaryPath)
    }

    func startServer() {
        if process?.isRunning == true {
            return
        }

        let bundlePath = Bundle.main.bundlePath
        var binaryPath = "\(bundlePath)/Contents/MacOS/hypura-bin"

        if !FileManager.default.fileExists(atPath: binaryPath) {
            binaryPath = "/Volumes/RealtekData/projects/hypura/target/release/hypura"
        }

        guard FileManager.default.fileExists(atPath: binaryPath) else {
            NSLog("Hypura binary not found at \(binaryPath)")
            return
        }

        let task = Process()
        task.executableURL = URL(fileURLWithPath: binaryPath)
        // Set working directory to /tmp or modelsDir to prevent any home/Music scanning
        task.currentDirectoryURL = URL(fileURLWithPath: "/tmp")
        task.arguments = [
            "serve",
            "--models-dir", modelsDir,
            "--port", "\(port)",
            "--context", "\(defaultContext)",
            "--idle-timeout", "\(idleTimeout)",
            "--kv-quant", kvQuant
        ]

        try? task.run()
        self.process = task
        updateStatusUI()
    }

    func stopServer() {
        if let p = process, p.isRunning {
            p.terminate()
        }
        process = nil
        updateStatusUI()
    }

    func restartServer() {
        stopServer()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.startServer()
            self?.buildMenu()
            self?.updateStatusUI()
        }
    }

    func updateStatusUI() {
        let isRunning = process?.isRunning ?? false
        let statusText = isRunning ? "🟢 Running on port \(port)" : "🔴 Server Stopped"
        statusLabel?.stringValue = statusText
        toggleServerButton?.title = isRunning ? "Stop Server" : "Start Server"
        buildMenu()
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

        let settingsItem = NSMenuItem(title: "Settings...", action: #selector(openSettings), keyEquivalent: ",")
        settingsItem.target = self
        menu.addItem(settingsItem)

        let copyUrlItem = NSMenuItem(title: "Copy Local URL (http://localhost:\(port))", action: #selector(copyUrl), keyEquivalent: "c")
        copyUrlItem.target = self
        menu.addItem(copyUrlItem)

        let modelsDirItem = NSMenuItem(title: "Open Models Directory...", action: #selector(openModelsDir), keyEquivalent: "m")
        modelsDirItem.target = self
        menu.addItem(modelsDirItem)

        menu.addItem(NSMenuItem.separator())

        let restartItem = NSMenuItem(title: "Restart Server", action: #selector(restartAction), keyEquivalent: "r")
        restartItem.target = self
        menu.addItem(restartItem)

        let quitItem = NSMenuItem(title: "Quit Hypura", action: #selector(quitAction), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        self.statusItem.menu = menu
    }

    @objc func openSettings() {
        if let window = settingsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 480, height: 320),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Hypura Settings"
        window.center()
        window.isReleasedWhenClosed = false

        let contentView = NSView(frame: NSRect(x: 0, y: 0, width: 480, height: 320))

        // 1. Status label
        let isRunning = process?.isRunning ?? false
        let status = NSTextField(labelWithString: isRunning ? "🟢 Running on port \(port)" : "🔴 Server Stopped")
        status.frame = NSRect(x: 20, y: 275, width: 280, height: 24)
        status.font = NSFont.boldSystemFont(ofSize: 14)
        contentView.addSubview(status)
        self.statusLabel = status

        // Toggle Server Button
        let toggleBtn = NSButton(title: isRunning ? "Stop Server" : "Start Server", target: self, action: #selector(toggleServerAction))
        toggleBtn.frame = NSRect(x: 320, y: 272, width: 140, height: 28)
        toggleBtn.bezelStyle = .rounded
        contentView.addSubview(toggleBtn)
        self.toggleServerButton = toggleBtn

        // Separator
        let sep = NSBox(frame: NSRect(x: 20, y: 260, width: 440, height: 1))
        sep.boxType = .separator
        contentView.addSubview(sep)

        // 2. Models Folder
        let modelsLabel = NSTextField(labelWithString: "Models Directory:")
        modelsLabel.frame = NSRect(x: 20, y: 225, width: 140, height: 20)
        modelsLabel.alignment = .right
        contentView.addSubview(modelsLabel)

        let modelsField = NSTextField(frame: NSRect(x: 170, y: 223, width: 200, height: 24))
        modelsField.stringValue = modelsDir
        contentView.addSubview(modelsField)
        self.modelsDirField = modelsField

        let browseBtn = NSButton(title: "Browse...", target: self, action: #selector(browseFolderAction))
        browseBtn.frame = NSRect(x: 375, y: 221, width: 85, height: 28)
        browseBtn.bezelStyle = .rounded
        contentView.addSubview(browseBtn)

        // 3. Port
        let portLabel = NSTextField(labelWithString: "Server Port:")
        portLabel.frame = NSRect(x: 20, y: 185, width: 140, height: 20)
        portLabel.alignment = .right
        contentView.addSubview(portLabel)

        let pField = NSTextField(frame: NSRect(x: 170, y: 183, width: 100, height: 24))
        pField.stringValue = "\(port)"
        contentView.addSubview(pField)
        self.portField = pField

        // 4. Context Size
        let ctxLabel = NSTextField(labelWithString: "Context Size:")
        ctxLabel.frame = NSRect(x: 20, y: 145, width: 140, height: 20)
        ctxLabel.alignment = .right
        contentView.addSubview(ctxLabel)

        let ctxPopup = NSPopUpButton(frame: NSRect(x: 170, y: 143, width: 180, height: 26), pullsDown: false)
        ctxPopup.addItems(withTitles: ["32768 (32k)", "16384 (16k)", "8192 (8k)", "4096 (4k)"])
        switch defaultContext {
        case 16384: ctxPopup.selectItem(at: 1)
        case 8192:  ctxPopup.selectItem(at: 2)
        case 4096:  ctxPopup.selectItem(at: 3)
        default:    ctxPopup.selectItem(at: 0)
        }
        contentView.addSubview(ctxPopup)
        self.contextPopup = ctxPopup

        // 5. Idle Timeout
        let timeoutLabel = NSTextField(labelWithString: "Idle Timeout (sec):")
        timeoutLabel.frame = NSRect(x: 20, y: 105, width: 140, height: 20)
        timeoutLabel.alignment = .right
        contentView.addSubview(timeoutLabel)

        let tField = NSTextField(frame: NSRect(x: 170, y: 103, width: 100, height: 24))
        tField.stringValue = "\(idleTimeout)"
        contentView.addSubview(tField)
        self.idleTimeoutField = tField

        // 6. KV Quantization
        let kvLabel = NSTextField(labelWithString: "KV Cache Quant:")
        kvLabel.frame = NSRect(x: 20, y: 65, width: 140, height: 20)
        kvLabel.alignment = .right
        contentView.addSubview(kvLabel)

        let kvPopup = NSPopUpButton(frame: NSRect(x: 170, y: 63, width: 220, height: 26), pullsDown: false)
        kvPopup.addItems(withTitles: ["q8_0 (Recommended)", "q4_0 (Low Memory)", "f16 (Full Precision)"])
        switch kvQuant {
        case "q4_0": kvPopup.selectItem(at: 1)
        case "f16":  kvPopup.selectItem(at: 2)
        default:     kvPopup.selectItem(at: 0)
        }
        contentView.addSubview(kvPopup)
        self.kvQuantPopup = kvPopup

        // 7. Save & Restart Button
        let saveBtn = NSButton(title: "Save & Restart Server", target: self, action: #selector(saveSettingsAction))
        saveBtn.frame = NSRect(x: 270, y: 15, width: 190, height: 32)
        saveBtn.bezelStyle = .rounded
        saveBtn.keyEquivalent = "\r"
        contentView.addSubview(saveBtn)

        window.contentView = contentView
        self.settingsWindow = window

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc func browseFolderAction() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Select Models Folder"

        if panel.runModal() == .OK, let url = panel.url {
            modelsDirField?.stringValue = url.path
        }
    }

    @objc func toggleServerAction() {
        if process?.isRunning == true {
            stopServer()
        } else {
            startServer()
        }
        updateStatusUI()
    }

    @objc func saveSettingsAction() {
        if let dir = modelsDirField?.stringValue, !dir.isEmpty {
            self.modelsDir = dir
        }
        if let pStr = portField?.stringValue, let p = Int(pStr), p > 0 {
            self.port = p
        }
        if let ctxIdx = contextPopup?.indexOfSelectedItem {
            switch ctxIdx {
            case 1: self.defaultContext = 16384
            case 2: self.defaultContext = 8192
            case 3: self.defaultContext = 4096
            default: self.defaultContext = 32768
            }
        }
        if let tStr = idleTimeoutField?.stringValue, let t = Int(tStr) {
            self.idleTimeout = t
        }
        if let kvIdx = kvQuantPopup?.indexOfSelectedItem {
            switch kvIdx {
            case 1: self.kvQuant = "q4_0"
            case 2: self.kvQuant = "f16"
            default: self.kvQuant = "q8_0"
            }
        }

        restartServer()
        settingsWindow?.close()
    }

    @objc func copyUrl() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("http://localhost:\(port)", forType: .string)
    }

    @objc func openModelsDir() {
        let url = URL(fileURLWithPath: modelsDir)
        NSWorkspace.shared.open(url)
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
