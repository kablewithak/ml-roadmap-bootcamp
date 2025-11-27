package com.kable.multistream.domain.model

import androidx.compose.ui.graphics.Color

/**
 * Represents a streaming platform
 */
enum class Platform(
    val displayName: String,
    val color: Color,
    val colorHex: String
) {
    YOUTUBE(
        displayName = "YouTube",
        color = Color(0xFFFF0000),
        colorHex = "#FF0000"
    ),
    TWITCH(
        displayName = "Twitch",
        color = Color(0xFF9146FF),
        colorHex = "#9146FF"
    ),
    INSTAGRAM(
        displayName = "Instagram",
        color = Color(0xFFE1306C),
        colorHex = "#E1306C"
    ),
    TIKTOK(
        displayName = "TikTok",
        color = Color(0xFF00F2EA),
        colorHex = "#00F2EA"
    );

    companion object {
        fun fromDisplayName(name: String): Platform? {
            return values().find { it.displayName.equals(name, ignoreCase = true) }
        }
    }
}

/**
 * Authentication status for a platform
 */
enum class AuthStatus {
    NOT_AUTHENTICATED,
    AUTHENTICATING,
    AUTHENTICATED,
    EXPIRED,
    ERROR
}

/**
 * Connection status for a streaming platform
 */
enum class ConnectionStatus {
    IDLE,
    CONNECTING,
    CONNECTED,
    STREAMING,
    RECONNECTING,
    DISCONNECTED,
    ERROR
}

/**
 * Platform configuration with authentication and connection details
 */
data class PlatformConfig(
    val platform: Platform,
    val authStatus: AuthStatus = AuthStatus.NOT_AUTHENTICATED,
    val connectionStatus: ConnectionStatus = ConnectionStatus.IDLE,
    val streamKey: String? = null,
    val rtmpUrl: String? = null,
    val authToken: String? = null,
    val isEnabled: Boolean = false,
    val lastError: String? = null
)
