# MultiStream - Multi-Platform Live Streaming App

MultiStream is a native Android application that enables content creators to broadcast live video to multiple streaming platforms (TikTok, Instagram, Twitch, YouTube) simultaneously from a single mobile device.

## Features

- **Multi-Platform Streaming**: Stream to TikTok Live, Instagram Live, Twitch, and YouTube simultaneously
- **Automated Authentication**: Streamlined authentication with automatic stream key retrieval
- **Unified Comment Overlay**: Real-time comment aggregation from all platforms with color-coding
- **Freemium Model**: 3 free streams, then upgrade to unlimited

## Project Status

🚧 **In Development** - Phase 1: Foundation

### Current Phase: Foundation Setup ✅

- [x] Android project structure
- [x] Gradle configuration with dependencies
- [x] Domain models and repository interfaces
- [x] Basic UI framework (Jetpack Compose)
- [ ] Camera capture implementation
- [ ] Video/Audio encoding
- [ ] RTMP streaming

## Tech Stack

- **Language**: Kotlin
- **UI Framework**: Jetpack Compose
- **Minimum SDK**: Android 10 (API 29)
- **Target SDK**: Android 14 (API 34)

### Key Dependencies

- **CameraX**: Camera capture and preview
- **rtmp-rtsp-stream-client-java**: RTMP streaming
- **OkHttp**: Network operations and WebSocket support
- **AppAuth**: OAuth 2.0 for YouTube and Twitch
- **Retrofit**: REST API communication
- **Coroutines**: Asynchronous operations
- **DataStore**: Preferences storage
- **Security Crypto**: Secure credential storage

## Architecture

The app follows Clean Architecture principles with clear separation:

```
app/
├── domain/           # Business logic and models
│   ├── model/       # Data models (Platform, StreamConfig, Comment, etc.)
│   └── repository/  # Repository interfaces
├── data/            # Data layer (TODO)
│   ├── repository/  # Repository implementations
│   ├── api/         # API clients for each platform
│   └── local/       # Local storage (DataStore, Keystore)
├── ui/              # Presentation layer
│   ├── MainActivity.kt
│   ├── stream/      # Streaming screen
│   └── theme/       # UI theming
└── service/         # Background services (streaming service)
```

## Building the Project

### Prerequisites

- Android Studio Hedgehog (2023.1.1) or later
- JDK 17
- Android SDK with API 34
- Gradle 8.1+

### Setup

1. Clone the repository
2. Open the project in Android Studio
3. Sync Gradle dependencies
4. Build and run on an Android device (API 29+)

```bash
./gradlew build
```

### Running the App

```bash
./gradlew installDebug
```

## Development Roadmap

### Phase 1: Foundation (Current) ⏳
- Set up Android project structure ✅
- Implement basic camera capture and preview
- Implement single-platform RTMP streaming (YouTube)
- Implement YouTube OAuth and stream key retrieval

### Phase 2: Multi-Platform Streaming
- Implement Twitch OAuth and stream key retrieval
- Implement RTMP stream splitting to multiple endpoints
- Build platform toggle UI
- Test simultaneous YouTube + Twitch streaming

### Phase 3: Instagram & TikTok Integration
- Implement WebView authentication for Instagram
- Research and implement Instagram stream key capture
- Implement WebView authentication for TikTok
- Research and implement TikTok stream key capture

### Phase 4: Comment Aggregation
- Implement YouTube Live Chat API polling
- Implement Twitch IRC client
- Research and implement Instagram comment capture
- Research and implement TikTok comment capture

### Phase 5: Overlay & Polish
- Design and implement comment overlay UI
- Implement platform color coding
- Add auto-reconnection logic
- Implement stream count tracking (freemium)

### Phase 6: Testing & Beta
- Internal testing across all platforms
- Beta testing with creators
- Bug fixes and optimizations

## Required Permissions

- `CAMERA`: Video capture
- `RECORD_AUDIO`: Audio capture
- `INTERNET`: Stream transmission
- `ACCESS_NETWORK_STATE`: Monitor connection quality
- `FOREGROUND_SERVICE`: Keep streaming alive in background
- `WAKE_LOCK`: Prevent screen timeout during setup

## Platform Requirements

### For Users

| Platform | Follower Requirement | Other Requirements |
|----------|---------------------|-------------------|
| TikTok Live | 1,000+ followers | Age 16+ |
| Instagram Live | None | Account in good standing |
| Twitch | None | Account in good standing |
| YouTube Live | 50+ subscribers (mobile) | Channel in good standing |

## Security & Privacy

- Stream keys and auth tokens are stored securely using Android Keystore (encrypted)
- No credentials are transmitted to third-party servers
- WebView sessions are cleared on logout
- Sensitive data excluded from backups

## Known Limitations (MVP)

- Android only (iOS not supported)
- Portrait orientation only
- Fixed overlay position
- No stream recording
- No comment reply functionality
- 720p max resolution

## Contributing

This is currently a solo development project. Contributions may be accepted post-MVP.

## License

TBD

## Disclaimer

This app uses unofficial methods to access some platform APIs (Instagram, TikTok). Users should be aware that:
- This may violate platform Terms of Service
- Platform changes may break functionality
- Account actions are user responsibility

Use at your own risk.

## Contact

Developer: Kable
Project: MultiStream MVP
Budget: R1,000

---

**Status**: In active development 🚀
**Last Updated**: November 2024
