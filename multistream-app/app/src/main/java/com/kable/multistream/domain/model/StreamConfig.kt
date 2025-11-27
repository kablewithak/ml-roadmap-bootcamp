package com.kable.multistream.domain.model

/**
 * Stream quality settings
 */
data class StreamQuality(
    val width: Int,
    val height: Int,
    val videoBitrate: Int,
    val fps: Int
) {
    companion object {
        val QUALITY_720P = StreamQuality(
            width = 1280,
            height = 720,
            videoBitrate = 3000000, // 3 Mbps
            fps = 30
        )

        val QUALITY_480P = StreamQuality(
            width = 854,
            height = 480,
            videoBitrate = 1500000, // 1.5 Mbps
            fps = 30
        )

        val QUALITY_360P = StreamQuality(
            width = 640,
            height = 360,
            videoBitrate = 800000, // 800 Kbps
            fps = 30
        )
    }
}

/**
 * Audio encoding settings
 */
data class AudioConfig(
    val bitrate: Int = 128000, // 128 Kbps
    val sampleRate: Int = 44100, // 44.1 KHz
    val channelCount: Int = 1 // Mono
)

/**
 * Complete stream configuration
 */
data class StreamConfig(
    val streamQuality: StreamQuality = StreamQuality.QUALITY_720P,
    val audioConfig: AudioConfig = AudioConfig(),
    val isFrontCamera: Boolean = true,
    val isMuted: Boolean = false,
    val showCommentOverlay: Boolean = true,
    val enabledPlatforms: Set<Platform> = emptySet()
)

/**
 * Stream session information
 */
data class StreamSession(
    val sessionId: String,
    val startTime: Long,
    val platforms: Set<Platform>,
    val isActive: Boolean = true,
    val endTime: Long? = null
) {
    val duration: Long
        get() = (endTime ?: System.currentTimeMillis()) - startTime
}

/**
 * Stream statistics
 */
data class StreamStats(
    val totalBytesSent: Long = 0,
    val currentBitrate: Int = 0,
    val droppedFrames: Int = 0,
    val fps: Int = 0,
    val latency: Long = 0
)
