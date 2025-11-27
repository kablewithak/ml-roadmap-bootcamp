package com.kable.multistream.domain.repository

import com.kable.multistream.domain.model.AuthStatus
import com.kable.multistream.domain.model.Platform
import com.kable.multistream.domain.model.PlatformConfig
import kotlinx.coroutines.flow.Flow

/**
 * Repository interface for platform authentication
 */
interface AuthRepository {

    /**
     * Get authentication status for all platforms
     */
    fun getPlatformConfigs(): Flow<List<PlatformConfig>>

    /**
     * Get authentication status for a specific platform
     */
    suspend fun getPlatformConfig(platform: Platform): PlatformConfig?

    /**
     * Start OAuth authentication flow for YouTube or Twitch
     */
    suspend fun authenticateOAuth(platform: Platform): Result<String>

    /**
     * Start WebView authentication flow for Instagram or TikTok
     */
    suspend fun authenticateWebView(platform: Platform): Result<String>

    /**
     * Store stream key for a platform
     */
    suspend fun storeStreamKey(platform: Platform, streamKey: String, rtmpUrl: String)

    /**
     * Store auth token for a platform
     */
    suspend fun storeAuthToken(platform: Platform, token: String)

    /**
     * Get stream key for a platform
     */
    suspend fun getStreamKey(platform: Platform): String?

    /**
     * Get RTMP URL for a platform
     */
    suspend fun getRtmpUrl(platform: Platform): String?

    /**
     * Update authentication status
     */
    suspend fun updateAuthStatus(platform: Platform, status: AuthStatus)

    /**
     * Refresh expired authentication
     */
    suspend fun refreshAuth(platform: Platform): Result<Unit>

    /**
     * Logout from a platform (clear credentials)
     */
    suspend fun logout(platform: Platform)

    /**
     * Check if platform is authenticated and ready to stream
     */
    suspend fun isAuthenticated(platform: Platform): Boolean
}
