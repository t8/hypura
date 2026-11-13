#!/usr/bin/env bash
set -e

# ==============================================================================
# Hypura Xcode Archive (.xcarchive) Creator
# ==============================================================================
# Creates an official Xcode Archive bundle (dist/Hypura.xcarchive) that can be:
# 1. Opened directly in Xcode Organizer (open dist/Hypura.xcarchive)
# 2. Uploaded to App Store Connect with Xcode's "Distribute App" wizard
# 3. Notarized for Developer ID distribution with one click
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
ARCHIVE_PATH="$DIST_DIR/Hypura.xcarchive"
PRODUCTS_APP_DIR="$ARCHIVE_PATH/Products/Applications"
VERSION="$(grep '^version' "$ROOT_DIR/Cargo.toml" | head -n1 | cut -d '"' -f 2)"
BUILD_NUMBER="${BUILD_NUMBER:-1}"
TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "==> Building Hypura.app bundle..."
"$SCRIPT_DIR/package_macos_app.sh"

echo "==> Constructing Xcode Archive at $ARCHIVE_PATH..."
rm -rf "$ARCHIVE_PATH"
mkdir -p "$PRODUCTS_APP_DIR"

# Copy App into Products/Applications
cp -R "$DIST_DIR/Hypura.app" "$PRODUCTS_APP_DIR/Hypura.app"

# Sign embedded binaries with App Store or Developer ID entitlements if requested
ENTITLEMENTS="$ROOT_DIR/mac-app/Hypura-AppStore.entitlements"
if [ "$1" == "--developer-id" ]; then
    ENTITLEMENTS="$ROOT_DIR/mac-app/Hypura-DeveloperID.entitlements"
fi

if [ -n "$SIGNING_IDENTITY" ]; then
    echo "==> Signing binaries with identity: $SIGNING_IDENTITY..."
    codesign --deep --force --options runtime --entitlements "$ENTITLEMENTS" --sign "$SIGNING_IDENTITY" "$PRODUCTS_APP_DIR/Hypura.app/Contents/MacOS/hypura-bin"
    codesign --deep --force --options runtime --entitlements "$ENTITLEMENTS" --sign "$SIGNING_IDENTITY" "$PRODUCTS_APP_DIR/Hypura.app"
else
    echo "==> Ad-hoc signing with entitlements (Xcode Organizer will re-sign on distribution)..."
    codesign --deep --force --entitlements "$ENTITLEMENTS" -s - "$PRODUCTS_APP_DIR/Hypura.app/Contents/MacOS/hypura-bin"
    codesign --deep --force --entitlements "$ENTITLEMENTS" -s - "$PRODUCTS_APP_DIR/Hypura.app"
fi

# Create archive Info.plist
cat <<EOF > "$ARCHIVE_PATH/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>ApplicationProperties</key>
    <dict>
        <key>ApplicationPath</key>
        <string>Applications/Hypura.app</string>
        <key>CFBundleIdentifier</key>
        <string>com.hypura.app</string>
        <key>CFBundleShortVersionString</key>
        <string>${VERSION:-0.2.7}</string>
        <key>CFBundleVersion</key>
        <string>${BUILD_NUMBER}</string>
        <key>SigningIdentity</key>
        <string>${SIGNING_IDENTITY:--}</string>
    </dict>
    <key>ArchiveVersion</key>
    <integer>2</integer>
    <key>CreationDate</key>
    <date>${TIMESTAMP}</date>
    <key>Name</key>
    <string>Hypura</string>
    <key>SchemeName</key>
    <string>Hypura</string>
</dict>
</plist>
EOF

echo ""
echo "=========================================================="
echo " Xcode Archive created successfully!"
echo " Location: $ARCHIVE_PATH"
echo "=========================================================="
echo ""
echo "To open in Xcode Organizer for App Store Connect upload:"
echo "  open \"$ARCHIVE_PATH\""
echo ""
echo "Inside Xcode Organizer:"
echo "  1. Click 'Distribute App'"
echo "  2. Choose 'App Store Connect' (or 'Developer ID')"
echo "  3. Select your Team and click 'Upload'"

