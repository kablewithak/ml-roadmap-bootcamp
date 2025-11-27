# Product Requirements Document (PRD)

## MultiStream — Multi-Platform Live Streaming App

**Version:** 1.0 (MVP)
**Last Updated:** November 2024
**Document Status:** Draft
**Author:** Kable

---

## 1. Executive Summary

MultiStream is a native Android application that enables content creators to broadcast live video to multiple streaming platforms simultaneously from a single mobile device. The app automates the retrieval of stream keys from TikTok Live, Instagram Live, Twitch, and YouTube, and provides a unified, transparent comment overlay that aggregates real-time comments from all connected platforms.

The product targets established content creators with approximately 20,000+ combined followers who meet each platform's live streaming eligibility requirements. The core value proposition is simplifying multi-platform streaming to a single-tap experience while solving the industry-wide pain point of fragmented audience engagement across platforms.

---

## 2. Problem Statement

### 2.1 Current Pain Points

Content creators who have built audiences across multiple platforms face significant challenges when going live:

**Fragmented Streaming Experience:** Creators must choose one platform per stream or use expensive, complex desktop solutions to multi-stream, leaving portions of their audience unreached.

**Technical Complexity:** Existing multi-streaming solutions (Restream, StreamYard, Prism Live) require manual stream key retrieval, are primarily desktop-focused, and demand technical knowledge most influencers don't possess.

**Comment Blindness:** When streaming to multiple platforms, creators cannot see comments from all platforms simultaneously, making audience engagement nearly impossible.

**Platform Lock-In:** Platforms intentionally make it difficult to stream elsewhere simultaneously, forcing creators to pick favourites rather than maximising reach.

### 2.2 Market Gap

While solutions exist for desktop multi-streaming, the mobile-first, automated, comment-aggregated experience does not exist in a user-friendly package. Instagram and TikTok's restrictive APIs have prevented competitors from solving this problem comprehensively.

---

## 3. Product Vision & Goals

### 3.1 Vision Statement

Empower content creators to go live everywhere their audience exists, from anywhere, with zero technical friction.

### 3.2 MVP Goals

1. Enable single-device, simultaneous live streaming to TikTok Live, Instagram Live, Twitch, and YouTube
2. Automate stream key acquisition for all four platforms without requiring manual user intervention
3. Provide a unified comment overlay displaying real-time comments from all platforms, colour-coded by source
4. Achieve a functional prototype within R1,000 budget using Claude Code

### 3.3 Success Metrics (Post-MVP)

| Metric | Target |
|--------|--------|
| Successful multi-platform stream completion rate | >85% |
| Average platforms per stream | 2.5+ |
| User retention after 3 free streams | >40% conversion to paid |
| Average stream duration | >15 minutes |
| App crash rate during streams | <2% |

---

## 4. Target Users

### 4.1 Primary Persona

**The Multi-Platform Influencer**

- **Follower count:** 20,000+ combined across platforms
- **Platform presence:** Active on at least 2 of the 4 supported platforms
- **Eligibility:** Meets streaming requirements (e.g., TikTok's 1,000 follower minimum)
- **Technical proficiency:** Low to moderate — expects apps to "just work"
- **Streaming frequency:** 2-8 times per month
- **Content type:** Lifestyle, entertainment, Q&A, product reviews, gaming, music
- **Device:** Android smartphone (primary content creation device)
- **Pain point:** "I want to go live for all my followers, not just the ones on one app"

### 4.2 User Requirements

Users must:
- Own an Android device running Android 10 or higher
- Have active accounts on at least one supported platform
- Meet each platform's live streaming eligibility criteria
- Have stable internet connection (minimum 5 Mbps upload recommended)

---

## 5. Platform Priority & Rationale

| Priority | Platform | Rationale |
|----------|----------|-----------|
| 1 | TikTok Live | Highest growth platform for creators; most restrictive API (solving this first proves core value) |
| 2 | Instagram Live | Second-most restrictive; massive creator user base; natural pairing with TikTok audience |
| 3 | Twitch | Established streaming infrastructure; well-documented API; gaming/creator crossover audience |
| 4 | YouTube | Most open API; largest potential reach; stable foundation |

---

## 6. Feature Specifications

### 6.1 Core Feature: Multi-Platform Simultaneous Streaming

#### 6.1.1 Description
The user initiates a single live stream from within the app. The app captures video from the device's camera, encodes it on-device, and simultaneously transmits the stream to all selected platforms via their respective RTMP endpoints.

#### 6.1.2 User Flow

1. User opens MultiStream app
2. User views connected platforms dashboard (shows which platforms are authenticated)
3. User selects which platforms to go live on (toggle on/off)
4. User taps "Go Live" button
5. App displays camera preview with comment overlay
6. Stream is broadcast to all selected platforms simultaneously
7. User taps "End Stream" to terminate all streams

#### 6.1.3 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F1.1 | App shall capture video from device's rear or front camera | Must Have |
| F1.2 | App shall capture audio from device microphone | Must Have |
| F1.3 | App shall encode video/audio stream on-device | Must Have |
| F1.4 | App shall transmit encoded stream to multiple RTMP endpoints simultaneously | Must Have |
| F1.5 | App shall allow user to toggle individual platforms on/off before going live | Must Have |
| F1.6 | App shall allow user to toggle individual platforms on/off during an active stream | Must Have |
| F1.7 | App shall auto-reconnect to a platform if connection drops during stream | Must Have |
| F1.8 | App shall display connection status for each platform during stream | Must Have |
| F1.9 | App shall support both front and rear camera switching during stream | Should Have |
| F1.10 | App shall maintain stream stability for minimum 60 minutes | Must Have |

#### 6.1.4 Technical Specifications

**Video Encoding:**
- Codec: H.264 (hardware-accelerated where available)
- Resolution: 720p default (1280x720), adaptive based on device capability
- Bitrate: 2500-4000 kbps (adaptive based on network conditions)
- Frame rate: 30 fps

**Audio Encoding:**
- Codec: AAC
- Bitrate: 128 kbps
- Sample rate: 44.1 kHz

**Network:**
- Protocol: RTMP/RTMPS
- Minimum upload speed: 5 Mbps recommended
- Adaptive bitrate adjustment based on network quality

---

### 6.2 Core Feature: Automated Stream Key Acquisition

#### 6.2.1 Description
The app automatically retrieves and manages stream keys for all supported platforms without requiring manual user intervention. This is achieved through a combination of official APIs (where available) and creative technical approaches for restricted platforms.

#### 6.2.2 Platform-Specific Approaches

**YouTube (Official API)**
- OAuth 2.0 authentication flow
- Use YouTube Live Streaming API to create broadcast and retrieve stream key
- Official, stable, well-documented approach
- Reference: YouTube Live Streaming API v3

**Twitch (Official API)**
- OAuth 2.0 authentication via Twitch Developer API
- Retrieve stream key from authenticated user's channel settings
- Official, stable approach
- Reference: Twitch API - Get Stream Key endpoint

**Instagram Live (Creative Approach)**
- Embedded WebView browser within app for user authentication
- User logs into Instagram through embedded browser
- App intercepts/captures stream key during the "Go Live" initiation flow
- Technical approach: Monitor network traffic within WebView for RTMP URL extraction
- Alternative: Leverage Instagram's internal API endpoints used by the web/mobile app (requires reverse engineering and session token management)
- Risk level: Medium — Instagram may update endpoints; does not typically result in account bans if mimicking normal user behaviour

**TikTok Live (Creative Approach)**
- Embedded WebView browser within app for user authentication
- User logs into TikTok through embedded browser
- Approach 1: Intercept stream key from TikTok Live Studio web flow
- Approach 2: Monitor WebSocket connections during live initiation for RTMP credentials
- Approach 3: Leverage TikTok's internal APIs used by their own apps (session-based authentication)
- Risk level: Medium-High — TikTok actively patches unofficial access methods; however, approaches that mimic normal user behaviour (embedded browser login) have lower ban risk
- Fallback: If automated retrieval fails, prompt user to retrieve key from TikTok Live Studio (desktop) with guided instructions

#### 6.2.3 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F2.1 | App shall provide OAuth authentication flow for YouTube | Must Have |
| F2.2 | App shall automatically retrieve YouTube stream key via official API | Must Have |
| F2.3 | App shall provide OAuth authentication flow for Twitch | Must Have |
| F2.4 | App shall automatically retrieve Twitch stream key via official API | Must Have |
| F2.5 | App shall provide embedded browser authentication for Instagram | Must Have |
| F2.6 | App shall automatically capture Instagram stream key during authentication flow | Must Have |
| F2.7 | App shall provide embedded browser authentication for TikTok | Must Have |
| F2.8 | App shall automatically capture TikTok stream key during authentication flow | Must Have |
| F2.9 | App shall securely store retrieved stream keys locally on device | Must Have |
| F2.10 | App shall handle stream key expiration and automatic refresh | Must Have |
| F2.11 | App shall provide fallback manual entry option if automated retrieval fails | Should Have |
| F2.12 | App shall notify user if platform authentication expires | Must Have |

#### 6.2.4 Security Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| S2.1 | Stream keys shall be stored in Android Keystore (encrypted) | Must Have |
| S2.2 | Authentication tokens shall be stored securely and never logged | Must Have |
| S2.3 | App shall not transmit credentials to any third-party servers | Must Have |
| S2.4 | WebView shall clear session data when user logs out | Must Have |

---

### 6.3 Core Feature: Unified Comment Overlay

#### 6.3.1 Description
A transparent overlay displayed on the streamer's screen (not visible to viewers) that shows a unified, real-time feed of comments from all connected platforms. Comments are colour-coded by platform for easy identification.

#### 6.3.2 Visual Design (MVP)

**Overlay Properties:**
- Position: Fixed (bottom third of screen, above camera controls)
- Transparency: 70% background opacity
- Width: Full screen width with padding
- Height: Approximately 25-30% of screen height
- Scrolling: Auto-scroll to newest comment; manual scroll to pause auto-scroll

**Comment Display:**
- Format: `[Username]: [Comment text]`
- Font: System default, readable at arm's length
- Text colour: White
- Background pill colour: Platform-specific (see below)

**Platform Colour Coding:**

| Platform | Colour | Hex Code |
|----------|--------|----------|
| TikTok | Cyan/Teal | #00F2EA |
| Instagram | Gradient Pink/Purple (simplified to Magenta) | #E1306C |
| Twitch | Purple | #9146FF |
| YouTube | Red | #FF0000 |

#### 6.3.3 Comment Aggregation Technical Approaches

**YouTube (Official API)**
- Use YouTube Live Streaming API - LiveChatMessages endpoint
- Poll every 2-5 seconds for new messages
- Official, stable approach

**Twitch (Official - IRC)**
- Connect to Twitch IRC (irc.chat.twitch.tv)
- Join channel's chat room
- Listen for PRIVMSG events
- Real-time, official approach

**Instagram (Creative Approach)**
- Maintain authenticated session via embedded WebView
- Approach 1: WebSocket listener for live comment stream (Instagram uses WebSockets for real-time updates)
- Approach 2: Poll Instagram's internal comment API endpoint at regular intervals
- Approach 3: Inject JavaScript into WebView to scrape comments from the live interface and pass to native app
- Risk level: Medium — requires maintaining active session; mimics normal user behaviour

**TikTok (Creative Approach)**
- Maintain authenticated session via embedded WebView
- Approach 1: WebSocket connection interception (TikTok uses WebSockets for live interactions)
- Approach 2: Inject JavaScript into WebView displaying TikTok Live to capture incoming comment events
- Approach 3: Reverse-engineer TikTok's proprietary real-time messaging protocol
- Alternative: Leverage third-party libraries that have already reverse-engineered TikTok Live (e.g., TikTok-Live-Connector)
- Risk level: Medium-High — TikTok frequently updates protocols; requires ongoing maintenance

#### 6.3.4 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F3.1 | App shall display transparent comment overlay visible only to streamer | Must Have |
| F3.2 | Overlay shall not be encoded into the outgoing video stream | Must Have |
| F3.3 | App shall aggregate comments from all connected platforms in real-time | Must Have |
| F3.4 | Each comment shall display the username and comment text | Must Have |
| F3.5 | Each comment shall be colour-coded by source platform | Must Have |
| F3.6 | Comments shall appear in chronological order (newest at bottom) | Must Have |
| F3.7 | Overlay shall auto-scroll to show newest comments | Must Have |
| F3.8 | User shall be able to manually scroll to view older comments | Should Have |
| F3.9 | Overlay position shall be fixed (not customisable in MVP) | Must Have |
| F3.10 | App shall handle high comment volume without performance degradation | Must Have |
| F3.11 | App shall gracefully handle comment retrieval failures per platform | Must Have |

#### 6.3.5 Performance Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| P3.1 | Comment latency (time from post to display) | <5 seconds |
| P3.2 | Maximum comments displayed simultaneously | 50 (older comments removed from memory) |
| P3.3 | Comment polling/refresh rate | Platform-dependent, 1-5 seconds |
| P3.4 | Overlay rendering impact on stream quality | <5% CPU overhead |

---

### 6.4 Feature: Stream Management & Controls

#### 6.4.1 Pre-Stream Controls
- Platform selection toggles (on/off for each platform)
- Camera selection (front/rear)
- Stream title input (applied to all platforms where supported)
- Connection status indicator per platform

#### 6.4.2 Mid-Stream Controls
- End individual platform streams (selective termination)
- End all streams
- Toggle comment overlay visibility
- Switch camera (front/rear)
- Mute/unmute microphone

#### 6.4.3 Auto-Reconnection Logic

When a platform disconnects during an active stream:
1. App detects disconnection within 5 seconds
2. App displays "Reconnecting..." status for that platform
3. App attempts reconnection every 10 seconds, up to 3 attempts
4. If reconnection succeeds, stream resumes automatically
5. If all attempts fail, platform is marked as "Disconnected" and user is notified
6. Other platform streams continue uninterrupted

#### 6.4.4 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F4.1 | User shall be able to select/deselect platforms before going live | Must Have |
| F4.2 | User shall be able to end individual platform streams mid-stream | Must Have |
| F4.3 | User shall be able to end all streams with single action | Must Have |
| F4.4 | App shall auto-reconnect dropped platforms (3 attempts, 10s intervals) | Must Have |
| F4.5 | App shall display real-time connection status per platform | Must Have |
| F4.6 | User shall be able to switch cameras during stream | Should Have |
| F4.7 | User shall be able to mute/unmute microphone during stream | Should Have |
| F4.8 | User shall be able to toggle comment overlay visibility | Should Have |

---

## 7. Business Model

### 7.1 Freemium Structure

**Free Tier:**
- 3 multi-platform live streams (lifetime limit)
- All platforms supported
- Full comment overlay functionality
- No time limit per stream

**Paid Tier (Pricing TBD):**
- Unlimited multi-platform live streams
- All features included
- Priority support

### 7.2 Implementation

| ID | Requirement | Priority |
|----|-------------|----------|
| B1.1 | App shall track number of completed streams per user | Must Have |
| B1.2 | App shall enforce 3-stream limit for free users | Must Have |
| B1.3 | App shall display remaining free streams to user | Must Have |
| B1.4 | App shall integrate payment gateway for subscription (post-MVP) | Deferred |

---

## 8. Technical Architecture

### 8.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ANDROID DEVICE                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Camera     │───▶│   Encoder    │───▶│  RTMP Splitter   │  │
│  │   Input      │    │  (H.264/AAC) │    │                  │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                    │            │
│         ┌──────────────────────────────────────────┼──────┐     │
│         │                    │                     │      │     │
│         ▼                    ▼                     ▼      ▼     │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐  ...   │
│  │ TikTok   │         │Instagram │         │  Twitch  │        │
│  │ RTMP     │         │  RTMP    │         │   RTMP   │        │
│  └──────────┘         └──────────┘         └──────────┘        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  COMMENT AGGREGATOR                      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │TikTok WS│  │Insta WS │  │Twitch   │  │YouTube  │    │   │
│  │  │Listener │  │Listener │  │IRC      │  │API Poll │    │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │   │
│  │       └───────────┴───────────┴───────────┘            │   │
│  │                        │                                │   │
│  │                        ▼                                │   │
│  │              ┌──────────────────┐                       │   │
│  │              │ Unified Comment  │                       │   │
│  │              │     Queue        │                       │   │
│  │              └────────┬─────────┘                       │   │
│  └───────────────────────│─────────────────────────────────┘   │
│                          ▼                                      │
│               ┌──────────────────┐                              │
│               │ Overlay Renderer │                              │
│               │  (Streamer View) │                              │
│               └──────────────────┘                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    AUTH MANAGER                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │TikTok   │  │Instagram│  │ Twitch  │  │ YouTube │    │   │
│  │  │WebView  │  │WebView  │  │ OAuth   │  │ OAuth   │    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  SECURE STORAGE                          │   │
│  │           (Android Keystore - Encrypted)                 │   │
│  │      Stream Keys | Auth Tokens | User Preferences        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Key Technical Components

| Component | Technology/Approach |
|-----------|---------------------|
| Video Capture | Android Camera2 API |
| Video Encoding | MediaCodec (hardware H.264) |
| RTMP Transmission | rtmp-rtsp-stream-client-java library or custom implementation |
| Multi-endpoint Streaming | Fork encoded output to multiple RTMP sockets |
| WebView Authentication | Android WebView with JavaScript interface |
| OAuth Flows | AppAuth for Android (YouTube, Twitch) |
| Secure Storage | Android Keystore |
| Comment WebSockets | OkHttp WebSocket client |
| IRC Client | Custom lightweight implementation or PircBotX |
| UI Framework | Jetpack Compose (modern) or XML layouts |

### 8.3 Third-Party Libraries & Dependencies

| Library | Purpose | License |
|---------|---------|---------|
| rtmp-rtsp-stream-client-java | RTMP streaming | Apache 2.0 |
| OkHttp | Network requests, WebSockets | Apache 2.0 |
| AppAuth | OAuth 2.0 flows | Apache 2.0 |
| Gson/Moshi | JSON parsing | Apache 2.0 |
| Coroutines | Async operations | Apache 2.0 |
| Jetpack Compose | UI (optional) | Apache 2.0 |

---

## 9. Risk Assessment & Mitigations

### 9.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TikTok patches stream key extraction method | High | High | Build modular auth system allowing rapid updates; implement fallback manual entry; monitor TikTok developer community for changes |
| Instagram API changes break comment retrieval | High | Medium | Abstract comment retrieval behind interface; build multiple fallback methods; monitor Instagram reverse-engineering community |
| Device overheating during encoding | Medium | Medium | Implement adaptive bitrate; add thermal monitoring; suggest external cooling to users |
| High comment volume causes lag | Medium | Low | Implement comment queue with max buffer; drop oldest comments; render asynchronously |
| Platform bans users for unofficial API use | Low | High | Mimic normal user behaviour; avoid automation patterns; use official APIs where possible; include disclaimer |

### 9.2 Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Platforms block multi-streaming entirely | Low | Critical | Monitor platform ToS changes; build goodwill with platforms; position as "audience growth tool" not "platform bypass" |
| Competitor launches similar solution | Medium | Medium | Leverage creator network for rapid adoption; focus on UX simplicity; build community |
| Low conversion from free to paid | Medium | High | Optimise free trial experience; gather feedback; adjust pricing |

### 9.3 Legal Considerations

- **Terms of Service:** Review ToS for each platform regarding third-party streaming tools and automated access
- **User Disclaimer:** Users must acknowledge that unofficial API methods may violate platform ToS
- **Data Privacy:** Ensure compliance with POPIA (South Africa) for user data handling
- **No Account Liability:** Clear terms that app is not responsible for platform account actions

---

## 10. Out of Scope (MVP)

The following features are explicitly excluded from the MVP:

| Feature | Reason for Exclusion |
|---------|----------------------|
| iOS version | Android-first strategy; resource constraints |
| Reply to comments from overlay | Complexity; not core value proposition |
| Stream alerts (donations, follows) | Nice-to-have; deferred to future version |
| Guest hosting / multi-person streams | Significant complexity |
| Cloud relay / server infrastructure | Cost constraints; phone-based processing first |
| Custom overlay positioning/sizing | MVP uses fixed design |
| Stream recording/archiving | Storage and complexity concerns |
| Analytics dashboard | Post-MVP feature |
| Desktop/web version | Mobile-first strategy |

---

## 11. Development Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Set up Android project structure
- [ ] Implement basic camera capture and preview
- [ ] Implement single-platform RTMP streaming (YouTube first — easiest API)
- [ ] Implement YouTube OAuth and stream key retrieval

### Phase 2: Multi-Platform Streaming (Week 3-4)
- [ ] Implement Twitch OAuth and stream key retrieval
- [ ] Implement RTMP stream splitting to multiple endpoints
- [ ] Build platform toggle UI (on/off per platform)
- [ ] Test simultaneous YouTube + Twitch streaming

### Phase 3: Instagram & TikTok Integration (Week 5-6)
- [ ] Implement embedded WebView authentication for Instagram
- [ ] Research and implement Instagram stream key capture
- [ ] Implement embedded WebView authentication for TikTok
- [ ] Research and implement TikTok stream key capture
- [ ] Test 4-platform simultaneous streaming

### Phase 4: Comment Aggregation (Week 7-8)
- [ ] Implement YouTube Live Chat API polling
- [ ] Implement Twitch IRC client
- [ ] Research and implement Instagram comment capture
- [ ] Research and implement TikTok comment capture
- [ ] Build unified comment queue

### Phase 5: Overlay & Polish (Week 9-10)
- [ ] Design and implement comment overlay UI
- [ ] Implement platform colour coding
- [ ] Add auto-reconnection logic
- [ ] Add mid-stream platform toggling
- [ ] Implement stream count tracking (freemium limit)

### Phase 6: Testing & Beta (Week 11-12)
- [ ] Internal testing across all platforms
- [ ] Recruit beta testers from creator network
- [ ] Gather feedback and fix critical bugs
- [ ] Prepare for limited release

---

## 12. Success Criteria (MVP)

The MVP will be considered successful if:

1. **Functional:** App can successfully stream to all 4 platforms simultaneously for at least 30 minutes without crash
2. **Authentication:** Stream keys can be retrieved automatically for at least 3 of 4 platforms (with manual fallback for the 4th)
3. **Comments:** Unified comment feed displays comments from at least 3 of 4 platforms with <5 second latency
4. **Stability:** <5% crash rate during beta testing
5. **Usability:** Beta testers can complete a multi-platform stream without technical support

---

## 13. Open Questions & Decisions Needed

| Question | Status | Decision Owner |
|----------|--------|----------------|
| What is the paid tier pricing? | Open | Kable |
| App name (placeholder: MultiStream) | Open | Kable |
| Which payment gateway to use (post-MVP)? | Deferred | Kable |
| Should we pursue official Instagram/TikTok partnerships? | Deferred | Kable |
| Minimum Android version to support? | Proposed: Android 10+ | Kable |

---

## 14. Appendices

### Appendix A: Platform Streaming Requirements

| Platform | Follower Requirement | Other Requirements |
|----------|----------------------|-------------------|
| TikTok Live | 1,000+ followers | Age 16+ (18+ for gifts) |
| Instagram Live | None (but reach matters) | Account in good standing |
| Twitch | None | Affiliate/Partner status for some features |
| YouTube Live | None (mobile: 50+ subscribers) | Channel in good standing |

### Appendix B: Useful Resources & References

**Official APIs:**
- YouTube Live Streaming API: https://developers.google.com/youtube/v3/live/docs
- Twitch API: https://dev.twitch.tv/docs/api/
- Twitch IRC: https://dev.twitch.tv/docs/irc/

**Reverse Engineering Resources (for research only):**
- TikTok-Live-Connector (Node.js): Community library for TikTok Live interaction
- Instagram Private API documentation: Various community-maintained repositories
- HTTP Toolkit: For inspecting mobile app network traffic

**Android Development:**
- Camera2 API: https://developer.android.com/reference/android/hardware/camera2/package-summary
- MediaCodec: https://developer.android.com/reference/android/media/MediaCodec
- rtmp-rtsp-stream-client-java: https://github.com/pedroSG94/rtmp-rtsp-stream-client-java

### Appendix C: Glossary

| Term | Definition |
|------|------------|
| RTMP | Real-Time Messaging Protocol — standard for streaming video |
| Stream Key | Secret credential that authorises streaming to a specific channel |
| OAuth | Open standard for access delegation (used by YouTube, Twitch) |
| WebView | Embedded browser component within a native app |
| WebSocket | Protocol for real-time bidirectional communication |
| IRC | Internet Relay Chat — protocol used by Twitch for chat |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | November 2024 | Kable | Initial draft |

---

*This document is a living specification and will be updated as development progresses and requirements evolve.*
