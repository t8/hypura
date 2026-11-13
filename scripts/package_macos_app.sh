#!/usr/bin/env bash
set -e

# ==============================================================================
# Hypura macOS Application Packager
# ==============================================================================
# Builds a native macOS menu bar application bundle (Hypura.app) like Ollama:
# - Self-contained App bundle in dist/Hypura.app
# - Menu bar daemon (LSUIElement = true)
# - Embedded release binary (Contents/MacOS/hypura-bin)
# - Automatic CLI installer (symlinks ~/.local/bin/hypura on launch)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_BUNDLE="$DIST_DIR/Hypura.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES_DIR="$CONTENTS/Resources"

echo "Building Hypura release binary with Cargo..."
(cd "$ROOT_DIR" && cargo build --release)

echo "Compiling native Swift menu bar launcher..."
mkdir -p "$ROOT_DIR/mac-app"
swiftc -O "$ROOT_DIR/mac-app/HypuraMenuApp.swift" -o "$ROOT_DIR/mac-app/Hypura"

echo "Assembling macOS App bundle structure at $APP_BUNDLE..."
rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Copy executables
cp "$ROOT_DIR/mac-app/Hypura" "$MACOS_DIR/Hypura"
cp "$ROOT_DIR/target/release/hypura" "$MACOS_DIR/hypura-bin"
chmod +x "$MACOS_DIR/Hypura" "$MACOS_DIR/hypura-bin"

# Write Info.plist
cat <<EOF > "$CONTENTS/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>Hypura</string>
    <key>CFBundleExecutable</key>
    <string>Hypura</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.hypura.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Hypura</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.2.7</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo ""
echo "=========================================================="
echo " Hypura.app packaged successfully!"
echo " Location: $APP_BUNDLE"
echo "=========================================================="
echo ""
echo "To install to /Applications:"
echo "  cp -R \"$APP_BUNDLE\" /Applications/"
echo ""
echo "To launch now:"
echo "  open \"$APP_BUNDLE\""
echo ""
