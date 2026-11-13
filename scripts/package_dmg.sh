#!/usr/bin/env bash
set -e

# ==============================================================================
# Hypura macOS DMG Packager
# ==============================================================================
# Creates a drag-and-drop .dmg installer containing:
# - Hypura.app
# - Drag-to-Applications symlink (/Applications)
# - Code signing (Developer ID if configured, or ad-hoc)
# - Apple Notarization support (via xcrun notarytool)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_BUNDLE="$DIST_DIR/Hypura.app"
DMG_STAGING="$DIST_DIR/dmg_staging"
VERSION="$(grep '^version' "$ROOT_DIR/Cargo.toml" | head -n1 | cut -d '"' -f 2)"
DMG_NAME="Hypura-${VERSION:-0.2.7}-macOS-arm64.dmg"
DMG_PATH="$DIST_DIR/$DMG_NAME"

echo "==> Ensuring Hypura.app is built..."
"$SCRIPT_DIR/package_macos_app.sh"

echo "==> Preparing DMG staging area at $DMG_STAGING..."
rm -rf "$DMG_STAGING" "$DMG_PATH"
mkdir -p "$DMG_STAGING"

# Copy App bundle
cp -R "$APP_BUNDLE" "$DMG_STAGING/Hypura.app"

# Create drag-to-Applications symlink
ln -s /Applications "$DMG_STAGING/Applications"

# Code sign if Developer ID identity is present
if [ -n "$DEVELOPER_ID" ]; then
    echo "==> Signing with Developer ID: $DEVELOPER_ID..."
    codesign --deep --force --options runtime --sign "$DEVELOPER_ID" "$DMG_STAGING/Hypura.app"
else
    echo "==> Signing with local ad-hoc signature (set DEVELOPER_ID for official release)..."
    codesign --deep --force -s - "$DMG_STAGING/Hypura.app"
fi

echo "==> Building compressed DMG image: $DMG_NAME..."
hdiutil create \
    -volname "Hypura" \
    -srcfolder "$DMG_STAGING" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

rm -rf "$DMG_STAGING"

# Optional Apple Notarization
if [ -n "$APPLE_ID" ] && [ -n "$APPLE_TEAM_ID" ] && [ -n "$APPLE_APP_SPECIFIC_PASSWORD" ]; then
    echo "==> Submitting $DMG_NAME to Apple Notary Service..."
    xcrun notarytool submit "$DMG_PATH" \
        --apple-id "$APPLE_ID" \
        --team-id "$APPLE_TEAM_ID" \
        --password "$APPLE_APP_SPECIFIC_PASSWORD" \
        --wait

    echo "==> Stapling notarization ticket to $DMG_NAME..."
    xcrun stapler staple "$DMG_PATH"
fi

echo ""
echo "=========================================================="
echo " Hypura DMG created successfully!"
echo " Location: $DMG_PATH"
echo " Size: $(du -h "$DMG_PATH" | cut -f1)"
echo "=========================================================="
echo ""
echo "Users can open $DMG_NAME and drag Hypura to Applications."

