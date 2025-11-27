# MultiStream Implementation Guide

This document provides a detailed implementation guide for building the MultiStream app.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Foundation](#phase-1-foundation)
3. [Phase 2: Multi-Platform Streaming](#phase-2-multi-platform-streaming)
4. [Phase 3: Instagram & TikTok Integration](#phase-3-instagram--tiktok-integration)
5. [Phase 4: Comment Aggregation](#phase-4-comment-aggregation)
6. [Phase 5: Overlay & Polish](#phase-5-overlay--polish)
7. [Testing Strategy](#testing-strategy)

## Architecture Overview

The app follows Clean Architecture principles:

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                    │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ MainActivity│  │StreamActivity│ │ UI Components   │  │
│  └────────────┘  └────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     Domain Layer                         │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │  Models    │  │ Use Cases  │  │   Repositories   │  │
│  │            │  │            │  │   (Interfaces)   │  │
│  └────────────┘  └────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                      Data Layer                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │Repository  │  │   API      │  │   Storage        │  │
│  │Impl        │  │  Clients   │  │   (Keystore)     │  │
│  └────────────┘  └────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Phase 1: Foundation

### 1.1 Camera Capture Implementation

**File**: `app/src/main/java/com/kable/multistream/camera/CameraManager.kt`

```kotlin
class CameraManager(private val context: Context) {
    private lateinit var cameraProvider: ProcessCameraProvider
    private var camera: Camera? = null

    suspend fun startCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        useFrontCamera: Boolean = true
    ) {
        // Initialize CameraX
        // Bind preview use case
        // Configure video capture
    }

    fun switchCamera() {
        // Toggle between front and rear camera
    }
}
```

### 1.2 Video/Audio Encoding

**File**: `app/src/main/java/com/kable/multistream/encoder/VideoEncoder.kt`

The rtmp-rtsp-stream-client-java library handles encoding internally. We'll wrap it:

```kotlin
class StreamEncoder {
    private val rtmpCamera: RtmpCamera2 // from library

    fun prepareVideo(width: Int, height: Int, fps: Int, bitrate: Int) {
        rtmpCamera.prepareVideo(width, height, fps, bitrate, 0)
    }

    fun prepareAudio(bitrate: Int, sampleRate: Int, isStereo: Boolean) {
        rtmpCamera.prepareAudio(bitrate, sampleRate, isStereo, 0)
    }
}
```

### 1.3 Single Platform RTMP Streaming (YouTube)

**File**: `app/src/main/java/com/kable/multistream/data/repository/StreamRepositoryImpl.kt`

```kotlin
class StreamRepositoryImpl : StreamRepository {
    override suspend fun startStream(config: StreamConfig): Result<StreamSession> {
        // Get stream key for YouTube
        // Configure encoder with quality settings
        // Start RTMP connection
        // Return session
    }
}
```

### 1.4 YouTube OAuth Implementation

**File**: `app/src/main/java/com/kable/multistream/auth/oauth/YouTubeAuthProvider.kt`

```kotlin
class YouTubeAuthProvider(private val context: Context) {
    private val authService = AuthorizationService(context)

    suspend fun authenticate(): Result<String> {
        // Build OAuth request using AppAuth
        // Launch browser for user consent
        // Exchange code for token
        // Retrieve stream key from YouTube API
    }

    suspend fun getStreamKey(accessToken: String): String {
        // Call YouTube Live Streaming API
        // Create live broadcast
        // Return stream key
    }
}
```

**YouTube API Endpoints**:
- Create broadcast: `POST https://www.googleapis.com/youtube/v3/liveBroadcasts`
- Get stream key: `GET https://www.googleapis.com/youtube/v3/liveStreams`

## Phase 2: Multi-Platform Streaming

### 2.1 Twitch OAuth Implementation

**File**: `app/src/main/java/com/kable/multistream/auth/oauth/TwitchAuthProvider.kt`

Similar to YouTube, using AppAuth library:

```kotlin
class TwitchAuthProvider(private val context: Context) {
    suspend fun authenticate(): Result<String> {
        // OAuth flow for Twitch
    }

    suspend fun getStreamKey(accessToken: String): String {
        // GET https://api.twitch.tv/helix/streams/key
    }
}
```

### 2.2 RTMP Stream Splitter

**File**: `app/src/main/java/com/kable/multistream/streaming/MultiStreamManager.kt`

```kotlin
class MultiStreamManager {
    private val activeStreams = mutableMapOf<Platform, RtmpCamera2>()

    suspend fun startMultiStream(platforms: Set<Platform>) {
        platforms.forEach { platform ->
            val streamKey = getStreamKey(platform)
            val rtmpUrl = getRtmpUrl(platform)

            val rtmpCamera = RtmpCamera2(/* ... */)
            rtmpCamera.setAuthorization(streamKey)

            if (rtmpCamera.startStream(rtmpUrl)) {
                activeStreams[platform] = rtmpCamera
            }
        }
    }

    private fun getRtmpUrl(platform: Platform): String {
        return when (platform) {
            YOUTUBE -> "rtmp://a.rtmp.youtube.com/live2"
            TWITCH -> "rtmp://live.twitch.tv/app"
            INSTAGRAM -> /* Retrieved from auth flow */
            TIKTOK -> /* Retrieved from auth flow */
        }
    }
}
```

**Note**: We'll need to create separate RtmpCamera2 instances for each platform since the library doesn't natively support multi-streaming. An alternative approach is to fork the encoded stream at the network level.

## Phase 3: Instagram & TikTok Integration

### 3.1 Instagram WebView Authentication

**File**: `app/src/main/java/com/kable/multistream/auth/webview/InstagramAuthActivity.kt`

```kotlin
class InstagramAuthActivity : AppCompatActivity() {
    private lateinit var webView: WebView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        webView = WebView(this).apply {
            settings.javaScriptEnabled = true

            // Inject JavaScript to intercept network calls
            webViewClient = object : WebViewClient() {
                override fun shouldInterceptRequest(
                    view: WebView?,
                    request: WebResourceRequest?
                ): WebResourceResponse? {
                    val url = request?.url.toString()

                    // Look for RTMP URLs in requests
                    if (url.contains("rtmp") || url.contains("live")) {
                        extractStreamKey(url)
                    }

                    return super.shouldInterceptRequest(view, request)
                }
            }

            addJavascriptInterface(
                StreamKeyExtractor(),
                "Android"
            )
        }

        // Load Instagram live page
        webView.loadUrl("https://www.instagram.com/")
    }

    private fun extractStreamKey(url: String) {
        // Parse RTMP URL and stream key
        // Save to secure storage
        // Return to main app
    }
}
```

### 3.2 TikTok WebView Authentication

Similar approach to Instagram:

**File**: `app/src/main/java/com/kable/multistream/auth/webview/TikTokAuthActivity.kt`

```kotlin
class TikTokAuthActivity : AppCompatActivity() {
    // Similar WebView setup
    // Monitor network traffic for RTMP credentials
    // TikTok uses WebSocket for live setup
}
```

**Alternative approach**: Use HTTP proxy to intercept traffic:

```kotlin
class StreamKeyInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        val response = chain.proceed(request)

        // Parse response for stream keys
        val body = response.body?.string()
        if (body?.contains("rtmp") == true) {
            // Extract and store stream key
        }

        return response
    }
}
```

## Phase 4: Comment Aggregation

### 4.1 YouTube Comment Polling

**File**: `app/src/main/java/com/kable/multistream/comments/YouTubeCommentListener.kt`

```kotlin
class YouTubeCommentListener(
    private val accessToken: String,
    private val liveChatId: String
) {
    private val api = YouTubeApiClient()

    suspend fun startListening(onComment: (Comment) -> Unit) {
        var pageToken: String? = null

        while (isActive) {
            val response = api.getLiveChatMessages(liveChatId, pageToken)

            response.items.forEach { message ->
                val comment = Comment(
                    id = message.id,
                    platform = Platform.YOUTUBE,
                    username = message.authorDetails.displayName,
                    text = message.snippet.displayMessage,
                    timestamp = message.snippet.publishedAt
                )
                onComment(comment)
            }

            pageToken = response.nextPageToken
            delay(response.pollingIntervalMillis)
        }
    }
}
```

### 4.2 Twitch IRC Client

**File**: `app/src/main/java/com/kable/multistream/comments/TwitchIrcClient.kt`

```kotlin
class TwitchIrcClient(
    private val channel: String,
    private val oauthToken: String
) {
    private var socket: Socket? = null

    fun connect(onComment: (Comment) -> Unit) {
        socket = Socket("irc.chat.twitch.tv", 6667)
        val writer = PrintWriter(socket?.getOutputStream(), true)
        val reader = BufferedReader(InputStreamReader(socket?.getInputStream()))

        // Authenticate
        writer.println("PASS oauth:$oauthToken")
        writer.println("NICK yourusername")
        writer.println("JOIN #$channel")

        // Listen for messages
        var line: String?
        while (reader.readLine().also { line = it } != null) {
            if (line?.startsWith("PRIVMSG") == true) {
                val comment = parseIrcMessage(line!!)
                onComment(comment)
            }
        }
    }

    private fun parseIrcMessage(message: String): Comment {
        // Parse IRC PRIVMSG format
        // :username!username@username.tmi.twitch.tv PRIVMSG #channel :message text
    }
}
```

### 4.3 Instagram Comment Capture

**File**: `app/src/main/java/com/kable/multistream/comments/InstagramCommentListener.kt`

```kotlin
class InstagramCommentListener(private val sessionToken: String) {
    private val websocket: WebSocket

    fun connect(onComment: (Comment) -> Unit) {
        // Connect to Instagram's WebSocket endpoint
        // Listen for comment events

        val request = Request.Builder()
            .url("wss://edge-chat.instagram.com/chat")
            .addHeader("Cookie", "sessionid=$sessionToken")
            .build()

        websocket = OkHttpClient().newWebSocket(request, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) {
                // Parse incoming comment JSON
                val comment = parseInstagramComment(text)
                onComment(comment)
            }
        })
    }
}
```

**Note**: Instagram's WebSocket endpoint and protocol may require reverse engineering. Monitor network traffic during a live stream to identify the correct endpoint and message format.

### 4.4 Unified Comment Aggregator

**File**: `app/src/main/java/com/kable/multistream/data/repository/CommentRepositoryImpl.kt`

```kotlin
class CommentRepositoryImpl : CommentRepository {
    private val commentQueue = CommentQueue()
    private val listeners = mutableMapOf<Platform, CommentListener>()

    override suspend fun startListening(platforms: Set<Platform>) {
        platforms.forEach { platform ->
            val listener = when (platform) {
                YOUTUBE -> YouTubeCommentListener(/* ... */)
                TWITCH -> TwitchIrcClient(/* ... */)
                INSTAGRAM -> InstagramCommentListener(/* ... */)
                TIKTOK -> TikTokCommentListener(/* ... */)
            }

            listener.start { comment ->
                commentQueue.add(comment)
            }

            listeners[platform] = listener
        }
    }

    override fun getCommentStream(): Flow<List<Comment>> {
        return flow {
            while (true) {
                emit(commentQueue.getAll())
                delay(100) // Emit updates every 100ms
            }
        }
    }
}
```

## Phase 5: Overlay & Polish

### 5.1 Comment Overlay UI

**File**: `app/src/main/java/com/kable/multistream/ui/stream/CommentOverlay.kt`

```kotlin
@Composable
fun CommentOverlay(
    comments: List<Comment>,
    visible: Boolean,
    modifier: Modifier = Modifier
) {
    if (!visible) return

    Box(
        modifier = modifier
            .fillMaxWidth()
            .fillMaxHeight(0.3f)
            .background(Color.Black.copy(alpha = 0.7f))
            .padding(8.dp)
    ) {
        LazyColumn(
            reverseLayout = true, // Newest at bottom
            modifier = Modifier.fillMaxSize()
        ) {
            items(comments.takeLast(50)) { comment ->
                CommentItem(comment)
            }
        }
    }
}

@Composable
fun CommentItem(comment: Comment) {
    Surface(
        color = comment.platform.color.copy(alpha = 0.8f),
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .padding(vertical = 2.dp)
            .fillMaxWidth()
    ) {
        Text(
            text = comment.getDisplayText(),
            color = Color.White,
            fontSize = 14.sp,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
        )
    }
}
```

### 5.2 Stream Screen

**File**: `app/src/main/java/com/kable/multistream/ui/stream/StreamScreen.kt`

```kotlin
@Composable
fun StreamScreen(
    viewModel: StreamViewModel
) {
    val comments by viewModel.comments.collectAsState()
    val connectionStatuses by viewModel.connectionStatuses.collectAsState()
    val showOverlay by viewModel.showCommentOverlay.collectAsState()

    Box(modifier = Modifier.fillMaxSize()) {
        // Camera preview (from CameraX)
        AndroidView(
            factory = { context ->
                PreviewView(context).apply {
                    // Setup camera preview
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // Connection status indicators (top)
        Row(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(16.dp)
        ) {
            connectionStatuses.forEach { (platform, status) ->
                ConnectionStatusChip(platform, status)
            }
        }

        // Comment overlay (bottom)
        CommentOverlay(
            comments = comments,
            visible = showOverlay,
            modifier = Modifier.align(Alignment.BottomCenter)
        )

        // Controls (floating action buttons)
        StreamControls(
            onEndStream = { viewModel.endStream() },
            onSwitchCamera = { viewModel.switchCamera() },
            onToggleMute = { viewModel.toggleMute() },
            onToggleComments = { viewModel.toggleCommentOverlay() },
            modifier = Modifier.align(Alignment.BottomEnd)
        )
    }
}
```

### 5.3 Auto-Reconnection Logic

**File**: `app/src/main/java/com/kable/multistream/streaming/ConnectionMonitor.kt`

```kotlin
class ConnectionMonitor {
    private val reconnectionAttempts = mutableMapOf<Platform, Int>()
    private val maxAttempts = 3

    fun monitorConnection(
        platform: Platform,
        rtmpCamera: RtmpCamera2,
        onStatusChange: (ConnectionStatus) -> Unit
    ) {
        rtmpCamera.setReTries(0) // Disable library auto-retry, we'll handle it

        // Monitor connection status
        CoroutineScope(Dispatchers.IO).launch {
            while (isActive) {
                if (!rtmpCamera.isStreaming) {
                    handleDisconnection(platform, rtmpCamera, onStatusChange)
                }
                delay(5000) // Check every 5 seconds
            }
        }
    }

    private suspend fun handleDisconnection(
        platform: Platform,
        rtmpCamera: RtmpCamera2,
        onStatusChange: (ConnectionStatus) -> Unit
    ) {
        val attempts = reconnectionAttempts.getOrDefault(platform, 0)

        if (attempts < maxAttempts) {
            onStatusChange(ConnectionStatus.RECONNECTING)
            delay(10000) // Wait 10 seconds

            if (rtmpCamera.startStream(getRtmpUrl(platform))) {
                reconnectionAttempts[platform] = 0
                onStatusChange(ConnectionStatus.CONNECTED)
            } else {
                reconnectionAttempts[platform] = attempts + 1
            }
        } else {
            onStatusChange(ConnectionStatus.DISCONNECTED)
        }
    }
}
```

### 5.4 Freemium Stream Counter

**File**: `app/src/main/java/com/kable/multistream/data/repository/UserPreferencesRepositoryImpl.kt`

```kotlin
class UserPreferencesRepositoryImpl(
    private val dataStore: DataStore<Preferences>
) : UserPreferencesRepository {

    override suspend fun incrementStreamCount() {
        dataStore.edit { prefs ->
            val current = prefs[STREAM_COUNT_KEY] ?: 0
            prefs[STREAM_COUNT_KEY] = current + 1
        }
    }

    override suspend fun canStartNewStream(): Boolean {
        val prefs = dataStore.data.first()
        val streamCount = prefs[STREAM_COUNT_KEY] ?: 0
        val isPremium = prefs[IS_PREMIUM_KEY] ?: false

        return isPremium || streamCount < 3
    }

    companion object {
        private val STREAM_COUNT_KEY = intPreferencesKey("stream_count")
        private val IS_PREMIUM_KEY = booleanPreferencesKey("is_premium")
    }
}
```

## Testing Strategy

### Unit Tests

Test individual components:
- Repository implementations
- Comment parsers
- OAuth flows
- Stream key extraction

### Integration Tests

Test component interactions:
- Camera → Encoder → RTMP flow
- Multi-platform streaming
- Comment aggregation from multiple sources

### Manual Testing Checklist

- [ ] Camera preview displays correctly
- [ ] Can authenticate with YouTube
- [ ] Can authenticate with Twitch
- [ ] Stream starts on YouTube
- [ ] Stream starts on Twitch
- [ ] Simultaneous streaming to YouTube + Twitch works
- [ ] Instagram WebView authentication works
- [ ] TikTok WebView authentication works
- [ ] Comments appear from YouTube
- [ ] Comments appear from Twitch
- [ ] Comment overlay displays correctly
- [ ] Color coding matches platform
- [ ] Camera switching works mid-stream
- [ ] Microphone mute works
- [ ] Auto-reconnection triggers on disconnect
- [ ] Stream counter increments correctly
- [ ] Free limit enforced after 3 streams

## Platform-Specific Notes

### YouTube
- Requires API key and OAuth client ID
- Must create app in Google Cloud Console
- Enable YouTube Data API v3
- Add redirect URI: `com.kable.multistream://oauth`

### Twitch
- Register app at dev.twitch.tv
- Get Client ID and Client Secret
- Redirect URI: `com.kable.multistream://oauth`

### Instagram
- No official live streaming API
- Must reverse engineer web/app traffic
- High risk of changes breaking functionality
- Consider fallback to manual stream key entry

### TikTok
- Most restrictive platform
- Actively patches unofficial access
- Requires ongoing maintenance
- Manual fallback is essential

## Security Considerations

1. **Stream Keys**: Store in Android Keystore, never log
2. **OAuth Tokens**: Encrypted storage, short-lived access tokens
3. **WebView Sessions**: Clear cookies on logout
4. **Network Traffic**: Use HTTPS where possible, validate certificates
5. **Permissions**: Request only when needed, explain rationale

## Performance Optimization

1. **Encoding**: Use hardware encoding (MediaCodec)
2. **Comments**: Limit queue size, async rendering
3. **Network**: Monitor bandwidth, adaptive bitrate
4. **Memory**: Release camera resources when not streaming
5. **Battery**: Disable unnecessary services, use foreground service efficiently

---

**Next Steps**: Begin implementing Phase 1 components in order listed above.
