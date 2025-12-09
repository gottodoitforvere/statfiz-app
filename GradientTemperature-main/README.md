# GradientTemperature

## macOS signing and notarization
- GitHub Actions now signs and notarizes the macOS bundle when the following secrets are present: `APPLE_CERTIFICATE` (base64 Developer ID Application .p12), `APPLE_CERT_PASSWORD`, `APPLE_CODESIGN_IDENTITY`, `APPLE_TEAM_ID`, `APPLE_NOTARIZE_APPLE_ID`, `APPLE_NOTARIZE_PASSWORD`, and optionally `APPLE_BUNDLE_ID`.
- The entitlements file (`scripts/entitlements.plist`) disables library validation so the embedded Python frameworks are accepted by macOS.
- Local build example (on macOS with Xcode tools installed):
  `CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" NOTARIZE_APPLE_ID="apple-id@example.com" NOTARIZE_APPLE_PASSWORD="xxxx-xxxx-xxxx-xxxx" NOTARIZE_TEAM_ID="TEAMID" ./scripts/build-mac.sh`
