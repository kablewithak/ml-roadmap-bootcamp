package com.kable.multistream.domain.model

/**
 * User preferences and freemium tracking
 */
data class UserPreferences(
    val completedStreamCount: Int = 0,
    val isPremiumUser: Boolean = false,
    val defaultQuality: StreamQuality = StreamQuality.QUALITY_720P,
    val defaultFrontCamera: Boolean = true,
    val showCommentOverlay: Boolean = true
) {
    /**
     * Check if user can start a new stream (freemium limit)
     */
    fun canStartNewStream(): Boolean {
        return isPremiumUser || completedStreamCount < FREE_STREAM_LIMIT
    }

    /**
     * Get remaining free streams
     */
    fun getRemainingFreeStreams(): Int {
        return if (isPremiumUser) {
            Int.MAX_VALUE
        } else {
            (FREE_STREAM_LIMIT - completedStreamCount).coerceAtLeast(0)
        }
    }

    companion object {
        const val FREE_STREAM_LIMIT = 3
    }
}
