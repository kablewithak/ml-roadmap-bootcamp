package com.kable.multistream.domain.repository

import com.kable.multistream.domain.model.Comment
import com.kable.multistream.domain.model.Platform
import kotlinx.coroutines.flow.Flow

/**
 * Repository interface for comment aggregation
 */
interface CommentRepository {

    /**
     * Start listening for comments from specified platforms
     */
    suspend fun startListening(platforms: Set<Platform>)

    /**
     * Stop listening for comments from all platforms
     */
    suspend fun stopListening()

    /**
     * Stop listening for comments from a specific platform
     */
    suspend fun stopListeningToPlatform(platform: Platform)

    /**
     * Get unified comment stream from all platforms
     */
    fun getCommentStream(): Flow<List<Comment>>

    /**
     * Get comments from a specific platform
     */
    fun getPlatformComments(platform: Platform): Flow<List<Comment>>

    /**
     * Clear all comments
     */
    suspend fun clearComments()
}
