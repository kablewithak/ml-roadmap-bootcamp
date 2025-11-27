package com.kable.multistream.domain.repository

import com.kable.multistream.domain.model.ConnectionStatus
import com.kable.multistream.domain.model.Platform
import com.kable.multistream.domain.model.StreamConfig
import com.kable.multistream.domain.model.StreamSession
import com.kable.multistream.domain.model.StreamStats
import kotlinx.coroutines.flow.Flow

/**
 * Repository interface for streaming operations
 */
interface StreamRepository {

    /**
     * Start streaming to specified platforms
     */
    suspend fun startStream(config: StreamConfig): Result<StreamSession>

    /**
     * Stop streaming to all platforms
     */
    suspend fun stopStream()

    /**
     * Stop streaming to a specific platform
     */
    suspend fun stopPlatformStream(platform: Platform)

    /**
     * Enable streaming to a platform mid-stream
     */
    suspend fun enablePlatform(platform: Platform): Result<Unit>

    /**
     * Get current stream session
     */
    fun getCurrentSession(): Flow<StreamSession?>

    /**
     * Get connection status for all platforms
     */
    fun getConnectionStatuses(): Flow<Map<Platform, ConnectionStatus>>

    /**
     * Get connection status for a specific platform
     */
    fun getConnectionStatus(platform: Platform): Flow<ConnectionStatus>

    /**
     * Get stream statistics
     */
    fun getStreamStats(): Flow<StreamStats>

    /**
     * Check if currently streaming
     */
    fun isStreaming(): Flow<Boolean>

    /**
     * Update stream configuration (quality, camera, etc.)
     */
    suspend fun updateStreamConfig(config: StreamConfig)
}
