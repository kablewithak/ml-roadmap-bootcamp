package com.kable.multistream.domain.model

/**
 * Represents a comment from any platform
 */
data class Comment(
    val id: String,
    val platform: Platform,
    val username: String,
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val userId: String? = null,
    val avatarUrl: String? = null
) {
    /**
     * Formatted display text for the comment overlay
     */
    fun getDisplayText(): String = "[$username]: $text"
}

/**
 * Comment queue for managing incoming comments
 */
class CommentQueue(private val maxSize: Int = 50) {
    private val comments = mutableListOf<Comment>()

    @Synchronized
    fun add(comment: Comment) {
        comments.add(comment)
        // Remove oldest comments if we exceed max size
        while (comments.size > maxSize) {
            comments.removeAt(0)
        }
    }

    @Synchronized
    fun addAll(newComments: List<Comment>) {
        newComments.forEach { add(it) }
    }

    @Synchronized
    fun getAll(): List<Comment> = comments.toList()

    @Synchronized
    fun clear() {
        comments.clear()
    }

    @Synchronized
    fun size(): Int = comments.size
}
