package com.kable.multistream.domain.repository

import com.kable.multistream.domain.model.StreamQuality
import com.kable.multistream.domain.model.UserPreferences
import kotlinx.coroutines.flow.Flow

/**
 * Repository interface for user preferences and freemium tracking
 */
interface UserPreferencesRepository {

    /**
     * Get user preferences as a flow
     */
    fun getUserPreferences(): Flow<UserPreferences>

    /**
     * Increment completed stream count
     */
    suspend fun incrementStreamCount()

    /**
     * Set premium status
     */
    suspend fun setPremiumStatus(isPremium: Boolean)

    /**
     * Set default stream quality
     */
    suspend fun setDefaultQuality(quality: StreamQuality)

    /**
     * Set default camera preference
     */
    suspend fun setDefaultFrontCamera(useFrontCamera: Boolean)

    /**
     * Set comment overlay visibility preference
     */
    suspend fun setShowCommentOverlay(show: Boolean)

    /**
     * Check if user can start a new stream
     */
    suspend fun canStartNewStream(): Boolean

    /**
     * Get remaining free streams
     */
    suspend fun getRemainingFreeStreams(): Int
}
